"""exp01_block: learned hashed-ngram embedding blocker (GPU) -> candidate pairs.

Stages (each cached):
  feats  : hashed feature CSR per split/field       -> data/cache/emb_v1/{split}_{s1|r}_{name|addr}.npz
  train  : train two-tower EmbeddingBag model        -> data/cache/emb_v1/model_{tag}.pt
           tag = fold0 / fold1 (trained on GT pairs of S1 in that fold) or full (all train pairs; unused in exp01)
  search : GPU IVF top-k per country (ivf_search)    -> data/cache/cand_v1/{split}/{country}_{tag}.parquet
           train: S1 of fold f are queried with the model trained on the *other* fold (honest OOF)
           test : model given by --test_tag (default fold0, same cos distribution as the OOF train candidates)
Channels: name (k_name), addr (k_addr), joint = cos_name + cos_addr (k_joint), reverse joint (R->S1, k_rev).
`search()` (exact brute force) is kept only for small-sample evaluation (quick_eval); it does not scale.
"""
import os
import sys
import time
import argparse
import glob

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, Logger, save_json, s1_fold, set_determinism  # noqa: E402

PREP = os.path.join(CACHE, "prep_v1")
EMB = os.path.join(CACHE, "emb_v1")
CAND = os.path.join(CACHE, "cand_v1")
NB = 1 << 19  # hash buckets per field


# ------------------------------------------------------------------------------ features
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


def stage_feats(log):
    from multiprocessing import Pool
    os.makedirs(EMB, exist_ok=True)
    with Pool(6) as pool:
        for split in ["train", "test"]:
            for tag in ["s1", "r"]:
                for field, col in [("name", "name_core"), ("addr", "addr_clean")]:
                    out = os.path.join(EMB, f"{split}_{tag}_{field}.npz")
                    if os.path.exists(out):
                        continue
                    texts = pl.read_parquet(os.path.join(PREP, f"{split}_{tag}.parquet"), columns=[col])[col].to_list()
                    ch = 50000
                    gen = ((texts[i:i + ch], field) for i in range(0, len(texts), ch))
                    lens, idx = [], []
                    for a, b in pool.imap(_hash_chunk, gen, chunksize=1):
                        lens.append(a)
                        idx.append(b)
                    del texts
                    lens = np.concatenate(lens)
                    indptr = np.zeros(len(lens) + 1, np.int64)
                    np.cumsum(lens, out=indptr[1:])
                    idx = np.concatenate(idx)
                    np.savez(out, indptr=indptr, indices=idx)
                    log(f"feats {split}_{tag}_{field}: rows={len(lens)} nnz={len(idx)} ({len(idx) / len(lens):.1f}/row)")
                    del lens, idx, indptr


def load_csr(split, tag, field):
    z = np.load(os.path.join(EMB, f"{split}_{tag}_{field}.npz"))
    return z["indptr"], z["indices"]


def idf_table(split):
    """IDF per hashed feature, from the split's own records (S1 + R)."""
    res = {}
    for field in ["name", "addr"]:
        df = np.zeros(NB, dtype=np.float64)
        n = 0
        for tag in ["s1", "r"]:
            ip, ix = load_csr(split, tag, field)
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


# ------------------------------------------------------------------------------ model
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
    """Holds CSR features (CPU) + idf (GPU) for one split and encodes arbitrary rows."""

    def __init__(self, split, dev):
        self.dev = dev
        self.csr = {(t, f): load_csr(split, t, f) for t in ["s1", "r"] for f in ["name", "addr"]}
        self.idf = {f: torch.from_numpy(v).to(dev) for f, v in idf_table(split).items()}

    def batch(self, model, tag, field, rows):
        ip, ix = self.csr[(tag, field)]
        flat, off = gather(ip, ix, rows)
        flat = torch.from_numpy(flat.astype(np.int64)).to(self.dev, non_blocking=True)
        off = torch.from_numpy(off).to(self.dev, non_blocking=True)
        return model.encode(field, flat, off, self.idf[field][flat])

    @torch.no_grad()
    def encode_all(self, model, tag, rows, bs=131072):
        outs = []
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            e = torch.cat([self.batch(model, tag, "name", r), self.batch(model, tag, "addr", r)], dim=1)
            outs.append(e.half())
        return torch.cat(outs)


def mnrl_losses(q, c, pos_mask, r_is_matched, r_owner, tau):
    """q: (nq,d) S1 embeddings, c: (nc,d) R embeddings, pos_mask: (nq,nc) bool.
    r->s CE over S1 for matched R; s->r multi-positive NCE over R."""
    logits = q @ c.T / tau
    # s -> r (multi-positive)
    lse_all = torch.logsumexp(logits, dim=1)
    lse_pos = torch.logsumexp(logits.masked_fill(~pos_mask, -1e4), dim=1)
    has_pos = pos_mask.any(1)
    l1 = (lse_all - lse_pos)[has_pos].mean()
    # r -> s
    lt = logits.T[r_is_matched]
    l2 = F.cross_entropy(lt, r_owner[r_is_matched])
    return l1 + l2


def stage_train(args, log):
    dev = torch.device("cuda")
    s1 = pl.read_parquet(os.path.join(PREP, "train_s1.parquet"), columns=["country", "name_core"])
    rdf = pl.read_parquet(os.path.join(PREP, "train_r.parquet"), columns=["country"])
    gt = pl.read_parquet(os.path.join(PREP, "train_gt.parquet"))
    n1 = s1.height
    fold = s1_fold(n1)
    if args.tag in ("fold0", "fold1"):
        keep = fold == int(args.tag[-1])
    else:
        keep = np.ones(n1, bool)
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
    # hard-negative ordering: sort matched S1 by name core within country
    core = s1["name_core"].to_numpy()
    hard_order = {c: v[np.argsort(core[matched_s1[v]], kind="stable")] for c, v in by_c.items()}
    log(f"train tag={args.tag}: matched S1={len(matched_s1)} pairs={len(gs)} singles={len(single_s1)} "
        f"distractors={len(distract_r)}")

    enc = Encoder("train", dev)
    model = Towers(args.dim).to(dev)
    opt = torch.optim.SparseAdam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    bs = args.bs
    for ep in range(args.epochs):
        batches = []
        for c in cnames:
            for src in ("rand", "hard"):
                if src == "rand":
                    v = rng.permutation(by_c[c])
                else:
                    v = hard_order[c]
                    # random offset so batches differ across epochs
                    v = np.roll(v, rng.integers(len(v)))
                bl = [v[i:i + bs] for i in range(0, len(v), bs)]
                if src == "hard":
                    bl = bl[: int(len(bl) * args.hard_frac)]
                batches += [(c, b) for b in bl]
        rng.shuffle(batches)
        t0 = time.time()
        tot = 0.0
        for bi, (c, b) in enumerate(batches):
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
            loss = (mnrl_losses(qn, cn, pos, r_is_m, r_owner, args.tau)
                    + mnrl_losses(qa, ca, pos, r_is_m, r_owner, args.tau)
                    + mnrl_losses(torch.cat([qn, qa], 1), torch.cat([cn, ca], 1), pos, r_is_m, r_owner,
                                  2 * args.tau))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            if bi % 200 == 0:
                log(f"ep{ep} batch {bi}/{len(batches)} loss {tot / (bi + 1):.4f} ({time.time() - t0:.0f}s)")
        log(f"epoch {ep} done: mean loss {tot / len(batches):.4f} time {time.time() - t0:.0f}s")
        torch.save(model.state_dict(), os.path.join(EMB, f"model_{args.tag}.pt"))
        if args.eval_q:
            quick_eval(args, model, enc, s1, gt, fold, log)


# ------------------------------------------------------------------------------ search
@torch.no_grad()
def search(enc, model, q_tag, q_rows, c_tag, c_rows, ks, d, q_mega=600000, q_bs=512, c_bs=262144):
    """Exact top-k search. Queries/corpus are encoded on the fly in chunks (corpus never fully on GPU).
    Channels: name (cos on name half), addr (cos on addr half), joint (sum). Returns
    dict ch -> (scores float16 np (nq,k), local corpus idx int32 np (nq,k), -1 = none)."""
    dev = enc.dev
    out = {ch: (np.zeros((len(q_rows), k), np.float16), np.full((len(q_rows), k), -1, np.int32))
           for ch, k in ks.items() if k > 0}
    for qm in range(0, len(q_rows), q_mega):
        Q = enc.encode_all(model, q_tag, q_rows[qm:qm + q_mega])
        nq = Q.shape[0]
        res = {ch: (torch.full((nq, k), -9.0, dtype=torch.float16, device=dev),
                    torch.full((nq, k), -1, dtype=torch.int32, device=dev)) for ch, k in ks.items() if k > 0}
        for c0 in range(0, len(c_rows), c_bs):
            Cc = enc.encode_all(model, c_tag, c_rows[c0:c0 + c_bs])
            Cn, Ca = Cc[:, :d], Cc[:, d:]
            for q0 in range(0, nq, q_bs):
                Qb = Q[q0:q0 + q_bs]
                Sn = Qb[:, :d] @ Cn.T
                Sa = Qb[:, d:] @ Ca.T
                for ch in res:
                    S = Sn if ch == "name" else (Sa if ch == "addr" else Sn + Sa)
                    k = ks[ch]
                    v, i = torch.topk(S, min(k, S.shape[1]), dim=1)
                    rs, ri = res[ch]
                    allv = torch.cat([rs[q0:q0 + q_bs], v], 1)
                    alli = torch.cat([ri[q0:q0 + q_bs], (i + c0).int()], 1)
                    v2, j = torch.topk(allv, k, dim=1)
                    rs[q0:q0 + q_bs] = v2
                    ri[q0:q0 + q_bs] = torch.gather(alli, 1, j)
                del Sn, Sa, S
            del Cc, Cn, Ca
        for ch, (rs, ri) in res.items():
            out[ch][0][qm:qm + nq] = rs.cpu().numpy()
            out[ch][1][qm:qm + nq] = ri.cpu().numpy()
        del Q, res
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def encode_cpu(enc, model, tag, rows, bs=131072):
    """Encode rows -> CPU fp16 torch tensor (n, 2d) (name half | addr half)."""
    out = torch.empty((len(rows), 2 * model.name.embedding_dim), dtype=torch.float16)
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        e = torch.cat([enc.batch(model, tag, "name", r), enc.batch(model, tag, "addr", r)], dim=1)
        out[i:i + len(r)] = e.half().cpu()
    return out


@torch.no_grad()
def kmeans_gpu(X, k, dev, iters=8, sample=400000, seed=0):
    """Spherical k-means on a sample of CPU fp16 rows X -> (k, d) fp16 centroids on GPU."""
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n, generator=g)[:min(sample, n)]
    S = X[idx].to(dev).float()
    S = F.normalize(S, dim=1)
    C = S[torch.randperm(S.shape[0], generator=g)[:k].to(dev)].clone()
    for _ in range(iters):
        a = torch.cat([(S[i:i + 65536] @ C.T).argmax(1) for i in range(0, S.shape[0], 65536)])
        Cn = torch.zeros_like(C).index_add_(0, a, S)
        cnt = torch.bincount(a, minlength=k)
        empty = cnt == 0
        if empty.any():
            Cn[empty] = S[torch.randint(0, S.shape[0], (int(empty.sum()),), device=dev)]
        C = F.normalize(Cn, dim=1)
    return C.half()


@torch.no_grad()
def ivf_topk(Q, X, k, dev, nprobe=32, per_cluster=1000, q_mega=1000000, max_elems=60_000_000):
    """Approximate top-k inner-product search.
    Q: CPU fp16 (nq, d) queries, X: CPU fp16 (nc, d) corpus. Returns (scores fp16 np, idx int32 np) (nq, k)."""
    nq, nc = Q.shape[0], X.shape[0]
    K = max(1, nc // per_cluster)
    C = kmeans_gpu(X, K, dev)
    a = torch.cat([(X[i:i + 262144].to(dev) @ C.T).argmax(1) for i in range(0, nc, 262144)])
    order = torch.argsort(a)
    counts = torch.bincount(a, minlength=K).cpu().numpy()
    starts = np.zeros(K + 1, np.int64)
    np.cumsum(counts, out=starts[1:])
    Xs = X[order.cpu()]  # corpus sorted by cluster (CPU)
    order_g = order.int()
    kk_all = min(k, nc)
    out_v = np.full((nq, k), -9.0, np.float16)
    out_i = np.full((nq, k), -1, np.int32)
    npb = min(nprobe, K)
    for q0 in range(0, nq, q_mega):
        Qg = Q[q0:q0 + q_mega].to(dev)
        m = Qg.shape[0]
        probes = torch.cat([torch.topk(Qg[i:i + 65536] @ C.T, npb, dim=1).indices
                            for i in range(0, m, 65536)])  # (m, npb)
        pc = probes.reshape(-1)
        pq = torch.arange(m, device=dev).repeat_interleave(npb)
        srt = torch.argsort(pc)
        pc, pq = pc[srt], pq[srt]
        qcounts = torch.bincount(pc, minlength=K).cpu().numpy()
        qstarts = np.zeros(K + 1, np.int64)
        np.cumsum(qcounts, out=qstarts[1:])
        rv = torch.full((m, k), -9.0, dtype=torch.float16, device=dev)
        ri = torch.full((m, k), -1, dtype=torch.int32, device=dev)
        for c in range(K):
            nm, nqc = counts[c], qcounts[c]
            if nm == 0 or nqc == 0:
                continue
            M = Xs[starts[c]:starts[c + 1]].to(dev, non_blocking=True)
            gidx = order_g[starts[c]:starts[c + 1]]
            ql_all = pq[qstarts[c]:qstarts[c + 1]]
            step = max(256, int(max_elems // nm))
            for j in range(0, nqc, step):
                ql = ql_all[j:j + step]
                S = Qg[ql] @ M.T
                v, i = torch.topk(S, min(kk_all, nm), dim=1)
                allv = torch.cat([rv[ql], v], 1)
                alli = torch.cat([ri[ql], gidx[i]], 1)
                v2, jj = torch.topk(allv, k, dim=1)
                rv[ql] = v2
                ri[ql] = torch.gather(alli, 1, jj)
        out_v[q0:q0 + m] = rv.cpu().numpy()
        out_i[q0:q0 + m] = ri.cpu().numpy()
        del Qg, rv, ri, probes, pc, pq
    del Xs, C, order_g
    torch.cuda.empty_cache()
    return out_v, out_i


@torch.no_grad()
def ivf_search(enc, model, q_tag, q_rows, c_tag, c_rows, ks, d, nprobe=32, per_cluster=1000):
    """Same output as search() but with GPU IVF per channel (name half / addr half / joint = full vector)."""
    dev = enc.dev
    Qf = encode_cpu(enc, model, q_tag, q_rows)
    Xf = encode_cpu(enc, model, c_tag, c_rows)
    out = {}
    for ch, k in ks.items():
        if k <= 0:
            continue
        sl = slice(0, d) if ch == "name" else (slice(d, 2 * d) if ch == "addr" else slice(0, 2 * d))
        Q = Qf[:, sl].contiguous()
        X = Xf[:, sl].contiguous()
        out[ch] = ivf_topk(Q, X, k, dev, nprobe=nprobe, per_cluster=per_cluster)
        del Q, X
    return out


@torch.no_grad()
def pair_cos(enc, model, s1_rows, r_rows, bs=400000):
    """Exact cos_name / cos_addr for given (s1,r) pairs."""
    cn = np.zeros(len(s1_rows), np.float32)
    ca = np.zeros(len(s1_rows), np.float32)
    for i in range(0, len(s1_rows), bs):
        a, b = s1_rows[i:i + bs], r_rows[i:i + bs]
        cn[i:i + bs] = (enc.batch(model, "s1", "name", a) * enc.batch(model, "r", "name", b)).sum(1).cpu().numpy()
        ca[i:i + bs] = (enc.batch(model, "s1", "addr", a) * enc.batch(model, "r", "addr", b)).sum(1).cpu().numpy()
    return cn, ca


def quick_eval(args, model, enc, s1, gt, fold, log):
    """Recall on held-out-fold S1 queries (sample) against the full same-country R pool (US + India)."""
    rng = np.random.default_rng(1)
    rdf_c = pl.read_parquet(os.path.join(PREP, "train_r.parquet"), columns=["country"])["country"].to_numpy()
    country = s1["country"].to_numpy()
    other = 1 - int(args.tag[-1]) if args.tag.startswith("fold") else 1
    gs = gt["s1_idx"].to_numpy()
    gr = gt["r_idx"].to_numpy()
    model.eval()
    for c in np.unique(country):
        qs = np.where((country == c) & (fold == other))[0]
        qs = rng.choice(qs, min(args.eval_q, len(qs)), replace=False)
        rr = np.where(rdf_c == c)[0]
        ks = {"name": 50, "addr": 50, "joint": 50}
        res = search(enc, model, "s1", qs, "r", rr, ks, args.dim)
        qpos = {s: i for i, s in enumerate(qs)}
        m = np.isin(gs, qs)
        gq = np.array([qpos[s] for s in gs[m]])
        # map R global -> local col
        rloc = np.full(len(rdf_c), -1, np.int64)
        rloc[rr] = np.arange(len(rr))
        gc = rloc[gr[m]]
        out = []
        for ch, (sc, ix) in res.items():
            for k in (5, 10, 20, 30, 50):
                hit = (ix[gq, :k] == gc[:, None]).any(1).mean()
                out.append(f"{ch}@{k}={hit:.4f}")
        # union of channels at k=20 each
        u = np.concatenate([res[ch][1][:, :20] for ch in res], 1)
        hit = (u[gq] == gc[:, None]).any(1).mean()
        log(f"[eval {c}] q={len(qs)} pairs={len(gq)} " + " ".join(out) + f" union20x3={hit:.4f}")
        del res
        torch.cuda.empty_cache()
    model.train()


def stage_search(args, log):
    dev = torch.device("cuda")
    split = args.split
    s1c = pl.read_parquet(os.path.join(PREP, f"{split}_s1.parquet"), columns=["country"])["country"].to_numpy()
    rc = pl.read_parquet(os.path.join(PREP, f"{split}_r.parquet"), columns=["country"])["country"].to_numpy()
    n1 = len(s1c)
    enc = Encoder(split, dev)
    if split == "train":
        fold = s1_fold(n1)
        jobs = [("fold1", np.where(fold == 0)[0]), ("fold0", np.where(fold == 1)[0])]  # OOF: other fold's model
    else:
        jobs = [(args.test_tag, np.arange(n1))]  # fold model: same cos-feature distribution as OOF train
    ks = {"name": args.k_name, "addr": args.k_addr, "joint": args.k_joint}
    cands = []
    for tag, qrows_all in jobs:
        model = Towers(args.dim).to(dev)
        model.load_state_dict(torch.load(os.path.join(EMB, f"model_{tag}.pt")))
        model.eval()
        for c in np.unique(s1c):
            t0 = time.time()
            job_path = os.path.join(CAND, split, f"{c}_{tag}.parquet")
            if os.path.exists(job_path):
                log(f"skip existing {job_path}")
                continue
            parts = []
            qs = qrows_all[s1c[qrows_all] == c]
            rr = np.where(rc == c)[0]
            if len(qs) == 0 or len(rr) == 0:
                continue
            res = ivf_search(enc, model, "s1", qs, "r", rr, ks, args.dim, nprobe=args.nprobe)
            for ch, (sc, ix) in res.items():
                k = ix.shape[1]
                valid = ix >= 0
                parts.append(pl.DataFrame({
                    "s1": np.repeat(qs, k)[valid.ravel()].astype(np.int32),
                    "r": rr[ix[valid]].astype(np.int32),
                    "ch": np.full(valid.sum(), {"name": 0, "addr": 1, "joint": 2}[ch], np.int8),
                    "rank": np.tile(np.arange(k, dtype=np.int16), len(qs))[valid.ravel()],
                }))
            del res
            # reverse: each R retrieves its top-k_rev S1 (joint)
            if args.k_rev > 0:
                rres = ivf_search(enc, model, "r", rr, "s1", qs, {"joint": args.k_rev}, args.dim, nprobe=args.nprobe)
                sc, ix = rres["joint"]
                valid = ix >= 0
                parts.append(pl.DataFrame({
                    "s1": qs[ix[valid]].astype(np.int32),
                    "r": np.repeat(rr, args.k_rev)[valid.ravel()].astype(np.int32),
                    "ch": np.full(valid.sum(), 3, np.int8),
                    "rank": np.tile(np.arange(args.k_rev, dtype=np.int16), len(rr))[valid.ravel()],
                }))
                del rres
            job = (pl.concat(parts).group_by(["s1", "r"]).agg([
                pl.col("rank").filter(pl.col("ch") == i).min().fill_null(999).cast(pl.Int16).alias(f"rk_{nm}")
                for i, nm in enumerate(["name", "addr", "joint", "rev"])]))
            del parts
            cn, ca = pair_cos(enc, model, job["s1"].to_numpy(), job["r"].to_numpy())
            job = job.with_columns(pl.Series("cos_n", cn), pl.Series("cos_a", ca)).sort(["s1", "r"])
            os.makedirs(os.path.dirname(job_path), exist_ok=True)
            job.write_parquet(job_path)
            torch.cuda.empty_cache()
            log(f"search {split} tag={tag} country={c}: q={len(qs)} corpus={len(rr)} pairs={job.height} "
                f"{time.time() - t0:.0f}s")
        del model
    del enc
    torch.cuda.empty_cache()
    # stats (per country, lazily)
    parts = sorted(glob.glob(os.path.join(CAND, split, "*.parquet")))
    tot = pl.scan_parquet(parts).select(pl.len()).collect().item()
    log(f"candidates {split}: {tot} pairs, {tot / n1:.1f} per S1, files {len(parts)}")
    if split == "train":
        gt = pl.read_parquet(os.path.join(PREP, "train_gt.parquet")).rename({"s1_idx": "s1", "r_idx": "r"})
        gt = gt.with_columns(pl.lit(1, pl.Int8).alias("y"))
        hits = {nm: 0 for nm in ["all", "name", "addr", "joint", "rev"]}
        for pth in parts:
            j = pl.read_parquet(pth).join(gt, on=["s1", "r"], how="inner")
            hits["all"] += j.height
            for nm in ["name", "addr", "joint", "rev"]:
                hits[nm] += j.filter(pl.col(f"rk_{nm}") < 999).height
        stats = {"pairs": tot, "per_s1": tot / n1, "pair_recall": hits["all"] / gt.height}
        for nm in ["name", "addr", "joint", "rev"]:
            stats[f"recall_only_{nm}"] = hits[nm] / gt.height
        log("blocking stats:", stats)
        save_json(stats, os.path.join(CAND, "train_stats.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "train", "search"])
    ap.add_argument("--tag", default="fold0")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--hard_frac", type=float, default=0.5)
    ap.add_argument("--eval_q", type=int, default=20000)
    ap.add_argument("--k_name", type=int, default=20)
    ap.add_argument("--k_addr", type=int, default=20)
    ap.add_argument("--k_joint", type=int, default=30)
    ap.add_argument("--k_rev", type=int, default=3)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--test_tag", default="fold0")
    args = ap.parse_args()
    log = Logger(os.path.join(EMB if args.stage != "search" else CAND, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    set_determinism(0)  # reproducible reruns (IVF k-means index_add_ is otherwise nondeterministic)
    torch.manual_seed(0)
    if args.stage == "feats":
        stage_feats(log)
    elif args.stage == "train":
        stage_train(args, log)
    else:
        stage_search(args, log)


if __name__ == "__main__":
    main()
