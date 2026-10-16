"""Cross-encoders: pretrained transformers that read both records as text ("name ; address" vs "name ; address") and
re-score the uncertain band (0.01 < p < 0.99). The 250K-token embedding matrix is frozen (faster on a 4 GB GPU, and
unseen French / Hindi words keep their pretrained meaning). 2-fold cross-fitting by S1 entity, like the matchers.

  ce1 : multilingual-e5-small, trained on the uncertain band of the token matcher, then filled in for the band of the
        stage-2 matcher with its saved fold models
  ce2 : multilingual-e5-small, trained on the uncertain band of the stage-2 matcher
  ce3 : bert-base-uncased, same band (architecture diversity for the stacker)
The scores are used for the training countries only.
"""
import os
import math
import time

import numpy as np
import polars as pl
import torch

from .. import config as C
from ..utils import LOG, format_duration, progress, s1_fold
from .gradient_boosting import model_dir


def band_dir(ws, name):
    return ws.dir("cross_encoders", name)


def build_band(ws, name, base_model):
    """Stage: uncertain-band pairs of a matcher's predictions (train: out-of-fold, test) with their texts."""
    run_dir = band_dir(ws, name)
    lo, hi = C.CE_BAND
    for split in ["train", "test"]:
        src = os.path.join(model_dir(ws, base_model), "oof.parquet" if split == "train" else "test_pred.parquet")
        t = pl.read_parquet(src)
        band = t.filter((pl.col("p") > lo) & (pl.col("p") < hi))
        s1 = pl.read_parquet(ws.table(split, "s1"), columns=["name_raw", "addr_raw", "country"])
        r = pl.read_parquet(ws.table(split, "r"), columns=["name_raw", "addr_raw"])
        a = s1[band["s1"].to_numpy()]
        b = r[band["r"].to_numpy()]
        out = band.with_columns(
            pl.Series("a_name", a["name_raw"]), pl.Series("a_addr", a["addr_raw"]),
            pl.Series("b_name", b["name_raw"]), pl.Series("b_addr", b["addr_raw"]),
            pl.Series("country", a["country"]))
        out = out.with_columns(
            ((pl.col("a_name").str.to_lowercase() + " ; " + pl.col("a_addr").str.to_lowercase())).alias("ta"),
            ((pl.col("b_name").str.to_lowercase() + " ; " + pl.col("b_addr").str.to_lowercase())).alias("tb"),
        ).drop("a_name", "a_addr", "b_name", "b_addr")
        if split == "train":
            out = out.with_columns(pl.Series("fold", s1_fold(len(s1))[out["s1"].to_numpy()]))
        out.write_parquet(os.path.join(run_dir, f"{split}_band.parquet"))
        LOG.info("%s %s band: %s pairs", name, split, f"{out.height:,}")


def _quiet_transformers():
    """Hide the weight-loading reports of transformers (the new classification head is expected to be missing)."""
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()


def load_model(model_id, revision, dev):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _quiet_transformers()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    # eager attention: the default memory-efficient attention backward has no deterministic kernel
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, num_labels=1,
                                                               attn_implementation="eager").to(dev)
    for prm in model.base_model.embeddings.parameters():
        prm.requires_grad = False
    return tok, model


def encode(tok, ta, tb, max_len):
    enc = tok(ta, tb, truncation="longest_first", max_length=max_len, padding=False)
    return enc["input_ids"]


def batches(ids, idx, bs, pad_id, dev):
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        seqs = [ids[j] for j in sel]
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        for k, s in enumerate(seqs):
            x[k, :len(s)] = torch.tensor(s)
        yield sel, x.to(dev, non_blocking=True), (x != pad_id).to(dev, non_blocking=True)


def train_one(tok, model, ids, y, idx, epochs, bs, dev, desc="training"):
    from transformers import get_linear_schedule_with_warmup
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=C.CE_LR, weight_decay=0.01)
    steps = epochs * math.ceil(len(idx) / bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    yt = torch.tensor(y, dtype=torch.float32)
    model.train()
    for ep in range(epochs):
        # length-bucketed shuffling: shuffle, then sort within chunks of 100 batches -> less padding
        perm = rng.permutation(idx)
        chunk = bs * 100
        perm = np.concatenate([sorted(perm[i:i + chunk], key=lambda j: len(ids[j])) for i in range(0, len(perm), chunk)])
        bl = [perm[i:i + bs] for i in range(0, len(perm), bs)]
        rng.shuffle(bl)
        bar = progress(bl, desc=f"{desc}, epoch {ep + 1}/{epochs}", unit="batch")
        for step, sel in enumerate(bar):
            _, x, m = next(batches(ids, sel, len(sel), tok.pad_token_id, dev))
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(input_ids=x, attention_mask=m).logits.squeeze(-1)
            loss = lossf(logit.float(), yt[sel].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            if step % 25 == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
    model.eval()


@torch.no_grad()
def score(tok, model, ids, idx, bs, dev, desc="scoring"):
    out = np.zeros(len(idx), np.float32)
    order = np.argsort([len(ids[j]) for j in idx], kind="stable")
    idx_sorted = np.asarray(idx)[order]
    pos = 0
    for sel, x, m in progress(batches(ids, idx_sorted, bs, tok.pad_token_id, dev), desc=desc,
                              total=-(-len(idx_sorted) // bs), unit="batch"):
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(input_ids=x, attention_mask=m).logits.squeeze(-1).float().cpu().numpy()
        out[order[pos:pos + len(sel)]] = lg
        pos += len(sel)
    return out


def train(ws, name):
    """Stage: fine-tune one cross-encoder per fold; out-of-fold scores for the training band, fold-average for test."""
    model_id, revision, epochs, bs = C.CROSS_ENCODERS[name]
    dev = torch.device("cuda")
    torch.manual_seed(0)
    run_dir = band_dir(ws, name)
    tr = pl.read_parquet(os.path.join(run_dir, "train_band.parquet"))
    te = pl.read_parquet(os.path.join(run_dir, "test_band.parquet"))
    tok, _ = load_model(model_id, revision, "cpu")
    ids_tr = encode(tok, tr["ta"].to_list(), tr["tb"].to_list(), C.CE_MAX_LEN)
    ids_te = encode(tok, te["ta"].to_list(), te["tb"].to_list(), C.CE_MAX_LEN)
    y = tr["y"].to_numpy().astype(np.float32)
    fold = tr["fold"].to_numpy()
    ce_tr = np.zeros(len(y), np.float32)
    ce_te = np.zeros(len(ids_te), np.float32)
    LOG.info("%s: %s, %d epochs, batch %d; %s training-band pairs, %s test-band pairs", name, model_id, epochs, bs,
             f"{len(y):,}", f"{len(ids_te):,}")
    for k in (0, 1):
        t0 = time.time()
        _, model = load_model(model_id, revision, dev)
        tr_idx = np.where(fold == k)[0]
        ev_idx = np.where(fold != k)[0]
        train_one(tok, model, ids_tr, y, tr_idx, epochs, bs, dev, desc=f"{name} fold {k}: training")
        ce_tr[ev_idx] = score(tok, model, ids_tr, ev_idx, C.CE_SCORE_BATCH, dev, desc=f"{name} fold {k}: scoring train")
        ce_te += score(tok, model, ids_te, np.arange(len(ids_te)), C.CE_SCORE_BATCH, dev,
                       desc=f"{name} fold {k}: scoring test") / 2
        model.save_pretrained(os.path.join(run_dir, f"ce_fold{k}"))
        LOG.info("%s fold %d: trained on %s pairs, scored %s out-of-fold + %s test pairs (%s)", name, k,
                 f"{len(tr_idx):,}", f"{len(ev_idx):,}", f"{len(ids_te):,}", format_duration(time.time() - t0))
        del model
        torch.cuda.empty_cache()
    tr.select("s1", "r").with_columns(pl.Series("ce", ce_tr)).write_parquet(os.path.join(run_dir, "train_ce.parquet"))
    te.select("s1", "r").with_columns(pl.Series("ce", ce_te)).write_parquet(os.path.join(run_dir, "test_ce.parquet"))


def _texts(ws, pairs, split):
    s1 = pl.read_parquet(ws.table(split, "s1"), columns=["name_raw", "addr_raw"])
    r = pl.read_parquet(ws.table(split, "r"), columns=["name_raw", "addr_raw"])
    a, b = s1[pairs["s1"].to_numpy()], r[pairs["r"].to_numpy()]
    ta = (a["name_raw"].str.to_lowercase() + " ; " + a["addr_raw"].str.to_lowercase()).to_list()
    tb = (b["name_raw"].str.to_lowercase() + " ; " + b["addr_raw"].str.to_lowercase()).to_list()
    return ta, tb


def fill_ce1(ws):
    """Stage: score the stage-2 band pairs that cross-encoder 1 has not scored, with its saved fold models (training
    pairs out-of-fold, test pairs of training countries as the fold average)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _quiet_transformers()
    model_id, revision, _, _ = C.CROSS_ENCODERS["ce1"]
    dev = torch.device("cuda")
    ce_dir = band_dir(ws, "ce1")
    out_dir = band_dir(ws, "ce1_filled")
    lo, hi = C.CE_BAND
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    models = []
    for k in (0, 1):
        models.append(AutoModelForSequenceClassification.from_pretrained(
            os.path.join(ce_dir, f"ce_fold{k}"), attn_implementation="eager").to(dev).eval())
    for split, src in [("train", "oof.parquet"), ("test", "test_pred.parquet")]:
        base = pl.read_parquet(os.path.join(model_dir(ws, "stage2"), src)).select("s1", "r", "p")
        have = pl.read_parquet(os.path.join(ce_dir, f"{split}_ce.parquet"))
        band = base.filter((pl.col("p") > lo) & (pl.col("p") < hi))
        if split == "test":
            country = pl.read_parquet(ws.table("test", "s1"), columns=["country"])["country"].to_numpy()
            band = band.filter(pl.Series(np.isin(country[band["s1"].to_numpy()], list(C.TRAINING_COUNTRIES))))
        miss = band.join(have.select("s1", "r"), on=["s1", "r"], how="anti").select("s1", "r")
        ta, tb = _texts(ws, miss, split)
        ids = encode(tok, ta, tb, C.CE_MAX_LEN)
        ce = np.zeros(len(ids), np.float32)
        with torch.no_grad():
            if split == "train":
                fold = s1_fold(pl.read_parquet(ws.table("train", "s1"), columns=["country"]).height)[miss["s1"].to_numpy()]
                for k in (0, 1):                       # the fold-k model was trained on S1 fold k -> scores fold != k
                    idx = np.where(fold != k)[0]
                    if len(idx):
                        ce[idx] = score(tok, models[k], ids, idx, C.CE_SCORE_BATCH, dev, desc=f"ce1 fold {k}: scoring train")
            else:
                for k in (0, 1):
                    ce += score(tok, models[k], ids, np.arange(len(ids)), C.CE_SCORE_BATCH, dev,
                                desc=f"ce1 fold {k}: scoring test") / 2
        new = miss.with_columns(pl.Series("ce", ce))
        allce = pl.concat([have.select("s1", "r", "ce"),
                           new.with_columns(pl.col("s1").cast(have["s1"].dtype), pl.col("r").cast(have["r"].dtype))])
        allce.write_parquet(os.path.join(out_dir, f"{split}_ce.parquet"))
        LOG.info("ce1 %s: %s scores, %s added for the stage-2 band", split, f"{allce.height:,}", f"{new.height:,}")
