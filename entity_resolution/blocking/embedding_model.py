"""Two-tower embedding blocker: hashed character 3-gram + word bags for names and addresses, encoded by EmbeddingBag
towers (IDF-weighted sum, L2-normalized) and trained contrastively on the training pairs.

One model per S1 fold: training candidates are always produced by the model that did not see their labels.
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import HashingVectorizer

from .. import config as C
from ..utils import LOG, format_duration, progress, s1_fold

NB = C.HASH_BUCKETS


# ---------------------------------------------------------------------------------------------- hashed features
def name_analyzer(core):
    ns = core.replace(" ", "")
    p = "<" + ns + ">"
    return [p[i:i + 3] for i in range(len(p) - 2)] + ["w:" + w for w in core.split()]


def addr_analyzer(addr):
    out = []
    for w in addr.split():
        out.append("w:" + w)
        if not w.isdigit() and len(w) > 2:
            p = "<" + w + ">"
            out.extend(p[i:i + 3] for i in range(len(p) - 2))
    return out


def _hash_chunk(args):
    texts, which = args
    hv = HashingVectorizer(analyzer=name_analyzer if which == "name" else addr_analyzer, n_features=NB,
                           alternate_sign=False, norm=None, binary=True, dtype=np.float32)
    X = hv.transform(texts).tocsr()
    X.sort_indices()
    return np.diff(X.indptr).astype(np.int32), X.indices.astype(np.int32)


def hashed_path(ws, split, tag, field):
    return ws.path("hashed", f"{split}_{tag}_{field}.npz")


def compute_hashed_features(ws):
    """Stage: hashed sparse features (CSR) per split, table and field."""
    with Pool(6) as pool:
        for split in ["train", "test"]:
            for tag in ["s1", "r"]:
                for field, col in [("name", "name_core"), ("addr", "addr_clean")]:
                    texts = pl.read_parquet(ws.table(split, tag), columns=[col])[col].to_list()
                    ch = 50000
                    gen = ((texts[i:i + ch], field) for i in range(0, len(texts), ch))
                    lens, idx = [], []
                    for a, b in progress(pool.imap(_hash_chunk, gen, chunksize=1), desc=f"hash {split} {tag} {field}",
                                         total=-(-len(texts) // ch), unit="chunk"):
                        lens.append(a)
                        idx.append(b)
                    del texts
                    lens = np.concatenate(lens)
                    indptr = np.zeros(len(lens) + 1, np.int64)
                    np.cumsum(lens, out=indptr[1:])
                    idx = np.concatenate(idx)
                    np.savez(hashed_path(ws, split, tag, field), indptr=indptr, indices=idx)
                    LOG.info("hashed %s_%s_%s: %s rows, %.1f features per row", split, tag, field,
                             f"{len(lens):,}", len(idx) / len(lens))
                    del lens, idx, indptr


def load_csr(ws, split, tag, field):
    z = np.load(hashed_path(ws, split, tag, field))
    return z["indptr"], z["indices"]


def idf_table(ws, split):
    """IDF per hashed feature, from the split's own records (S1 + S2/S3)."""
    res = {}
    for field in ["name", "addr"]:
        df = np.zeros(NB, dtype=np.float64)
        n = 0
        for tag in ["s1", "r"]:
            ip, ix = load_csr(ws, split, tag, field)
            df += np.bincount(ix, minlength=NB)
            n += len(ip) - 1
        res[field] = np.log((n + 1) / (df + 1)).astype(np.float32) + 1.0
    return res


def gather(indptr, indices, rows):
    starts = indptr[rows]
    lens = indptr[rows + 1] - starts
    offsets = np.zeros(len(rows), dtype=np.int64)
    np.cumsum(lens[:-1], out=offsets[1:])
    flat = np.repeat(starts - offsets, lens) + np.arange(lens.sum())
    return indices[flat], offsets


# ---------------------------------------------------------------------------------------------- model
class Towers(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.name = nn.EmbeddingBag(NB, dim, mode="sum", sparse=True)
        self.addr = nn.EmbeddingBag(NB, dim, mode="sum", sparse=True)
        for e in (self.name, self.addr):
            nn.init.normal_(e.weight, 0, 0.1)

    def encode(self, field, flat, offsets, w):
        emb = self.name if field == "name" else self.addr
        return F.normalize(emb(flat, offsets, per_sample_weights=w), dim=-1)


class Encoder:
    """Hashed features (CPU) + IDF (GPU) of one split; encodes arbitrary rows with a model."""

    def __init__(self, ws, split, dev):
        self.dev = dev
        self.csr = {(t, f): load_csr(ws, split, t, f) for t in ["s1", "r"] for f in ["name", "addr"]}
        self.idf = {f: torch.from_numpy(v).to(dev) for f, v in idf_table(ws, split).items()}

    def batch(self, model, tag, field, rows):
        ip, ix = self.csr[(tag, field)]
        flat, off = gather(ip, ix, rows)
        flat = torch.from_numpy(flat.astype(np.int64)).to(self.dev, non_blocking=True)
        off = torch.from_numpy(off).to(self.dev, non_blocking=True)
        return model.encode(field, flat, off, self.idf[field][flat])


def contrastive_loss(q, c, pos_mask, r_is_matched, r_owner, tau):
    """Multi-positive InfoNCE from S1 to records + cross-entropy from each matched record to its S1."""
    logits = q @ c.T / tau
    lse_all = torch.logsumexp(logits, dim=1)
    lse_pos = torch.logsumexp(logits.masked_fill(~pos_mask, -1e4), dim=1)
    has_pos = pos_mask.any(1)
    l1 = (lse_all - lse_pos)[has_pos].mean()
    lt = logits.T[r_is_matched]
    l2 = F.cross_entropy(lt, r_owner[r_is_matched])
    return l1 + l2


def model_path(ws, tag):
    return ws.path("blocker", f"model_{tag}.pt")


def train_blocker(ws, fold_id):
    """Stage: train the blocker on the ground-truth pairs of the S1 entities in one fold."""
    dev = torch.device("cuda")
    s1 = pl.read_parquet(ws.table("train", "s1"), columns=["country", "name_core"])
    rdf = pl.read_parquet(ws.table("train", "r"), columns=["country"])
    gt = pl.read_parquet(ws.table("train", "gt"))
    n1 = s1.height
    keep = s1_fold(n1) == fold_id
    gs = gt["s1_idx"].to_numpy()
    gr = gt["r_idx"].to_numpy()
    m = keep[gs]
    gs, gr = gs[m], gr[m]
    order = np.argsort(gs, kind="stable")
    gs, gr = gs[order], gr[order]
    matched_s1 = np.unique(gs)
    starts = np.searchsorted(gs, matched_s1)
    ends = np.searchsorted(gs, matched_s1, side="right")
    has_match = np.zeros(n1, bool)
    has_match[np.unique(gt["s1_idx"].to_numpy())] = True
    single_s1 = np.where(keep & ~has_match)[0]
    r_matched = np.zeros(rdf.height, bool)
    r_matched[gt["r_idx"].to_numpy()] = True
    distract_r = np.where(~r_matched)[0]
    country = s1["country"].to_numpy()
    rcountry = rdf["country"].to_numpy()
    cnames = np.unique(country)
    by_c = {c: np.where(country[matched_s1] == c)[0] for c in cnames}  # positions into matched_s1
    single_by_c = {c: single_s1[country[single_s1] == c] for c in cnames}
    distr_by_c = {c: distract_r[rcountry[distract_r] == c] for c in cnames}
    core = s1["name_core"].to_numpy()
    hard_order = {c: v[np.argsort(core[matched_s1[v]], kind="stable")] for c, v in by_c.items()}  # hard negatives
    LOG.info("blocker fold %d: %s matched S1, %s pairs", fold_id, f"{len(matched_s1):,}", f"{len(gs):,}")

    enc = Encoder(ws, "train", dev)
    model = Towers(C.EMBED_DIM).to(dev)
    opt = torch.optim.SparseAdam(model.parameters(), lr=C.BLOCKER_LR)
    rng = np.random.default_rng(0)
    bs, tau = C.BLOCKER_BATCH, C.BLOCKER_TAU
    for ep in range(C.BLOCKER_EPOCHS):
        batches = []
        for c in cnames:
            for src in ("rand", "hard"):
                if src == "rand":
                    v = rng.permutation(by_c[c])
                else:
                    v = hard_order[c]
                    v = np.roll(v, rng.integers(len(v)))       # random offset so batches differ across epochs
                bl = [v[i:i + bs] for i in range(0, len(v), bs)]
                if src == "hard":
                    bl = bl[: int(len(bl) * C.BLOCKER_HARD_FRACTION)]
                batches += [(c, b) for b in bl]
        rng.shuffle(batches)
        t0 = time.time()
        tot = 0.0
        bar = progress(batches, desc=f"blocker fold {fold_id}, epoch {ep + 1}/{C.BLOCKER_EPOCHS}", unit="batch")
        for bi, (c, b) in enumerate(bar):
            sidx = matched_s1[b]
            lens = ends[b] - starts[b]
            ridx = gr[np.repeat(starts[b], lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))]
            owner = np.repeat(np.arange(len(b)), lens)
            ns = min(len(single_by_c[c]), bs // 8)
            nd = min(len(distr_by_c[c]), bs)
            extra_s = rng.choice(single_by_c[c], ns, replace=False) if ns else np.zeros(0, np.int64)
            extra_r = rng.choice(distr_by_c[c], nd, replace=False) if nd else np.zeros(0, np.int64)
            q_rows = np.concatenate([sidx, extra_s])
            c_rows = np.concatenate([ridx, extra_r])
            qn = enc.batch(model, "s1", "name", q_rows)
            qa = enc.batch(model, "s1", "addr", q_rows)
            cn = enc.batch(model, "r", "name", c_rows)
            ca = enc.batch(model, "r", "addr", c_rows)
            nq, nc = len(q_rows), len(c_rows)
            pos = torch.zeros(nq, nc, dtype=torch.bool, device=dev)
            own_t = torch.from_numpy(owner).to(dev)
            pos[own_t, torch.arange(len(ridx), device=dev)] = True
            r_is_m = torch.zeros(nc, dtype=torch.bool, device=dev)
            r_is_m[: len(ridx)] = True
            r_owner = torch.zeros(nc, dtype=torch.long, device=dev)
            r_owner[: len(ridx)] = own_t
            loss = (contrastive_loss(qn, cn, pos, r_is_m, r_owner, tau)
                    + contrastive_loss(qa, ca, pos, r_is_m, r_owner, tau)
                    + contrastive_loss(torch.cat([qn, qa], 1), torch.cat([cn, ca], 1), pos, r_is_m, r_owner, 2 * tau))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            bar.set_postfix(loss=f"{tot / (bi + 1):.4f}", refresh=False)
        LOG.info("blocker fold %d, epoch %d/%d: %d batches, mean loss %.4f (%s)", fold_id, ep + 1, C.BLOCKER_EPOCHS,
                 len(batches), tot / len(batches), format_duration(time.time() - t0))
    torch.save(model.state_dict(), model_path(ws, f"fold{fold_id}"))
