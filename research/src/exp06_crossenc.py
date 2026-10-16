"""exp06_crossenc: multilingual cross-encoder re-scoring of the *uncertain band* on top of exp05.

Why: after exp05 the remaining matcher errors are "family" look-alikes whose resolution is semantic - noise swaps in
generic business words (Center/Services/Group/Cie...) while a swapped *content* word means a different business
(Solidair Sportive vs Solidair Loisirs). France has 2.5x more uncertain pairs than the US, and its words were never
seen in training. A pretrained multilingual encoder (intfloat/multilingual-e5-small, MIT, 118M) can learn the
noise-vs-entity distinction on US/India and transfer it through its multilingual token semantics.
Only pairs with exp05 stage-2 p in (lo, hi) are re-scored: 1.7% of train pairs; oracle on that band = 0.9947 vs 0.9876.

Stages:
  pairs   : band pairs + raw text (train from exp05 OOF, test from exp05 test_pred)  -> runs/<run>/{train,test}_band.parquet
  train   : 2-fold cross-fit fine-tuning (S1 folds) -> OOF logits for train band, fold-avg logits for test band
            --loco: train on US band only, score India band (transfer check)
  stack   : logistic stacker on [logit(p2), ce] replaces p inside the band; decode + dense-world F0.5
  predict : test band re-scored -> submission (in-country thr / unseen-country thr)
"""
import os
import sys
import json
import math
import time
import argparse

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, save_json, s1_fold, write_outputs, validate_outputs  # noqa: E402
import exp03_dense as M3  # noqa: E402

MODEL = "intfloat/multilingual-e5-small"
MODEL_REV = "614241f622f53c4eeff9890bdc4f31cfecc418b3"  # pinned HF revision (exact weights on a fresh download)
PREP = {"train": os.path.join(CACHE, "prep_v1"), "test": os.path.join(CACHE, "prep_v2")}
BASE = "exp05"
TRAIN_COUNTRIES = {"US", "India"}


def stage_pairs(args, log):
    run_dir = os.path.join(RUNS, args.run)
    os.makedirs(run_dir, exist_ok=True)
    for split in ["train", "test"]:
        src = os.path.join(RUNS, BASE, "oof.parquet" if split == "train" else "test_pred.parquet")
        t = pl.read_parquet(src)
        band = t.filter((pl.col("p") > args.lo) & (pl.col("p") < args.hi))
        s1 = pl.read_parquet(os.path.join(PREP[split], f"{split}_s1.parquet"), columns=["name_raw", "addr_raw", "country"])
        r = pl.read_parquet(os.path.join(PREP[split], f"{split}_r.parquet"), columns=["name_raw", "addr_raw"])
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
        elif args.test_countries:
            # CE scores are only used for training countries: skip scoring the other test pairs (saves ~20% time)
            out = out.filter(pl.col("country").is_in(args.test_countries.split(",")))
        out.write_parquet(os.path.join(run_dir, f"{split}_band.parquet"))
        log(f"{split}: band ({args.lo},{args.hi}) pairs {out.height} of {t.height}"
            + (f", pos rate {out['y'].mean():.3f}" if split == "train" else ""))


# ---------------------------------------------------------------------------- cross-encoder
def load_model(dev):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    # eager attention: the default memory-efficient attention backward has no deterministic kernel
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=MODEL_REV, num_labels=1,
                                                               attn_implementation="eager").to(dev)
    # freeze the 250K-token multilingual embedding matrix (96M of 118M params): ~4x faster steps, and French/Hindi
    # tokens (never seen in our training pairs) keep their pretrained meaning instead of drifting
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


def train_one(tok, model, ids, y, idx, args, log, dev):
    from transformers import get_linear_schedule_with_warmup
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(idx) / args.bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    yt = torch.tensor(y, dtype=torch.float32)
    model.train()
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        # length-bucketed shuffling: shuffle, then sort within chunks of 100 batches -> less padding
        perm = rng.permutation(idx)
        chunk = args.bs * 100
        perm = np.concatenate([sorted(perm[i:i + chunk], key=lambda j: len(ids[j])) for i in range(0, len(perm), chunk)])
        bl = [perm[i:i + args.bs] for i in range(0, len(perm), args.bs)]
        rng.shuffle(bl)
        run = 0.0
        for bi, sel in enumerate(bl):
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
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if bi else loss.item()
            if bi % 500 == 0:
                log(f"  ep{ep} step {bi}/{len(bl)} loss {run:.4f} ({time.time() - t0:.0f}s)")
    model.eval()


@torch.no_grad()
def score(tok, model, ids, idx, bs, dev):
    out = np.zeros(len(idx), np.float32)
    order = np.argsort([len(ids[j]) for j in idx], kind="stable")
    idx_sorted = np.asarray(idx)[order]
    pos = 0
    for sel, x, m in batches(ids, idx_sorted, bs, tok.pad_token_id, dev):
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(input_ids=x, attention_mask=m).logits.squeeze(-1).float().cpu().numpy()
        out[order[pos:pos + len(sel)]] = lg
        pos += len(sel)
    return out


def stage_train(args, log):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    run_dir = os.path.join(RUNS, args.run)
    tr = pl.read_parquet(os.path.join(run_dir, "train_band.parquet"))
    te = pl.read_parquet(os.path.join(run_dir, "test_band.parquet"))
    tok, _ = load_model("cpu")
    t0 = time.time()
    ids_tr = encode(tok, tr["ta"].to_list(), tr["tb"].to_list(), args.max_len)
    ids_te = encode(tok, te["ta"].to_list(), te["tb"].to_list(), args.max_len)
    log(f"tokenized train {len(ids_tr)} test {len(ids_te)} in {time.time() - t0:.0f}s; "
        f"mean len {np.mean([len(x) for x in ids_tr]):.1f}")
    y = tr["y"].to_numpy().astype(np.float32)
    fold = tr["fold"].to_numpy()
    country = tr["country"].to_numpy()
    if args.loco:
        tr_idx = np.where(country == "US")[0]
        ev_idx = np.where(country == "India")[0]
        _, model = load_model(dev)
        log(f"LOCO: train on US band ({len(tr_idx)}), score India band ({len(ev_idx)})")
        train_one(tok, model, ids_tr, y, tr_idx, args, log, dev)
        ce = np.full(len(y), np.nan, np.float32)
        ce[ev_idx] = score(tok, model, ids_tr, ev_idx, 128, dev)
        tr.select("s1", "r").with_columns(pl.Series("ce", ce)).write_parquet(os.path.join(run_dir, "loco_ce.parquet"))
        return
    ce_tr = np.zeros(len(y), np.float32)
    ce_te = np.zeros(len(ids_te), np.float32)
    for k in (0, 1):
        _, model = load_model(dev)
        tr_idx = np.where(fold == k)[0]
        ev_idx = np.where(fold != k)[0]
        log(f"fold {k}: train {len(tr_idx)} (pos {y[tr_idx].mean():.3f}), score {len(ev_idx)}")
        train_one(tok, model, ids_tr, y, tr_idx, args, log, dev)
        ce_tr[ev_idx] = score(tok, model, ids_tr, ev_idx, 128, dev)
        ce_te += score(tok, model, ids_te, np.arange(len(ids_te)), 128, dev) / 2
        from sklearn.metrics import roc_auc_score
        log(f"fold {k}: OOF AUC ce {roc_auc_score(y[ev_idx], ce_tr[ev_idx]):.4f} vs stage-2 p "
            f"{roc_auc_score(y[ev_idx], tr['p'].to_numpy()[ev_idx]):.4f}")
        model.save_pretrained(os.path.join(run_dir, f"ce_fold{k}"))
        del model
        torch.cuda.empty_cache()
    tr.select("s1", "r").with_columns(pl.Series("ce", ce_tr)).write_parquet(os.path.join(run_dir, "train_ce.parquet"))
    te.select("s1", "r").with_columns(pl.Series("ce", ce_te)).write_parquet(os.path.join(run_dir, "test_ce.parquet"))


def stage_score_test(args, log):
    """Re-score the test band from the SAVED fold cross-encoders (predict-only reproduction path)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = torch.device("cuda")
    run_dir = os.path.join(RUNS, args.run)
    te = pl.read_parquet(os.path.join(run_dir, "test_band.parquet"))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    ids = encode(tok, te["ta"].to_list(), te["tb"].to_list(), args.max_len)
    ce = np.zeros(len(ids), np.float32)
    for k in (0, 1):
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(run_dir, f"ce_fold{k}"), attn_implementation="eager").to(dev).eval()
        ce += score(tok, model, ids, np.arange(len(ids)), 128, dev) / 2
        del model
        torch.cuda.empty_cache()
    out = os.path.join(run_dir, args.out_name)
    te.select("s1", "r").with_columns(pl.Series("ce", ce)).write_parquet(out)
    log(f"scored {len(ids)} test band pairs -> {out}")


# ---------------------------------------------------------------------------- stacking / decoding
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_features(df):
    lp = logit(df["p"].to_numpy())
    ce = df["ce"].to_numpy()
    return np.stack([lp, ce, lp * ce], 1)


def fit_stacker(df):
    from sklearn.linear_model import LogisticRegression
    X = stack_features(df)
    return LogisticRegression(C=1.0, max_iter=1000).fit(X, df["y"].to_numpy())


def stage_stack(args, log):
    run_dir = os.path.join(RUNS, args.run)
    keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
    oof = pl.read_parquet(os.path.join(RUNS, BASE, "oof.parquet"))
    band = pl.read_parquet(os.path.join(run_dir, "train_band.parquet"), columns=["s1", "r", "y", "p", "fold", "country"]) \
             .join(pl.read_parquet(os.path.join(run_dir, "train_ce.parquet")), on=["s1", "r"])
    # cross-fitted stacker: fit on fold k band rows, apply to fold 1-k
    newp = np.zeros(band.height, np.float32)
    f = band["fold"].to_numpy()
    for k in (0, 1):
        lr = fit_stacker(band.filter(pl.Series(f == k)))
        newp[f != k] = lr.predict_proba(stack_features(band.filter(pl.Series(f != k))))[:, 1]
        log(f"stacker fit on fold {k}: coef {np.round(lr.coef_[0], 3)} intercept {lr.intercept_[0]:.3f}")
    band = band.with_columns(pl.Series("p_new", newp))
    base = M3.decode_eval(oof, keep, log, "exp05 (reference)")
    new = oof.join(band.select("s1", "r", "p_new"), on=["s1", "r"], how="left") \
             .with_columns(pl.coalesce(["p_new", "p"]).alias("p")).drop("p_new")
    m = M3.decode_eval(new, keep, log, "exp06 stacked")
    # CE alone inside the band (sanity): p := sigmoid(ce)
    alone = oof.join(band.select("s1", "r", pl.Series("pc", 1 / (1 + np.exp(-band["ce"].to_numpy())))), on=["s1", "r"], how="left") \
               .with_columns(pl.coalesce(["pc", "p"]).alias("p")).drop("pc")
    M3.decode_eval(alone, keep, log, "exp06 CE alone in band")
    lr_all = fit_stacker(band)
    import pickle
    with open(os.path.join(run_dir, "stacker.pkl"), "wb") as fh:
        pickle.dump(lr_all, fh)
    save_json({"reference": base, "stacked": m}, os.path.join(run_dir, "metrics.json"))


def stage_loco_eval(args, log):
    """Transfer check: CE trained on US band, stacker fit on US band (with in-country OOF CE), applied to India band."""
    run_dir = os.path.join(RUNS, args.run)
    keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0))
    country = pl.read_parquet(os.path.join(PREP["train"], "train_s1.parquet"), columns=["country"])["country"].to_numpy()
    oof = pl.read_parquet(os.path.join(RUNS, BASE, "oof.parquet"))
    band = pl.read_parquet(os.path.join(run_dir, "train_band.parquet"), columns=["s1", "r", "y", "p", "country"])
    cross = band.join(pl.read_parquet(os.path.join(run_dir, "train_ce.parquet")), on=["s1", "r"])
    us = cross.filter(pl.col("country") == "US")
    lr = fit_stacker(us)
    ind = band.join(pl.read_parquet(os.path.join(run_dir, "loco_ce.parquet")), on=["s1", "r"]).filter(pl.col("country") == "India")
    ind = ind.with_columns(pl.Series("p_new", lr.predict_proba(stack_features(ind))[:, 1]))
    kin = keep & (country == "India")
    oin = oof.filter(pl.Series(country[oof["s1"].to_numpy()] == "India"))
    M3.decode_eval(oin, kin, log, "India exp05 (reference)")
    new = oin.join(ind.select("s1", "r", "p_new"), on=["s1", "r"], how="left") \
             .with_columns(pl.coalesce(["p_new", "p"]).alias("p")).drop("p_new")
    M3.decode_eval(new, kin, log, "India with US-trained CE")


def stage_predict(args, log):
    import pickle
    run_dir = os.path.join(RUNS, args.run)
    with open(os.path.join(run_dir, "stacker.pkl"), "rb") as fh:
        lr = pickle.load(fh)
    with open(os.path.join(run_dir, "metrics.json")) as fh:
        thr = args.thr if args.thr is not None else json.load(fh)["stacked"]["best_thr"]
    tp = pl.read_parquet(os.path.join(RUNS, BASE, "test_pred.parquet"))
    band = pl.read_parquet(os.path.join(run_dir, "test_band.parquet"), columns=["s1", "r", "p"]) \
             .join(pl.read_parquet(os.path.join(run_dir, "test_ce.parquet")), on=["s1", "r"])
    band = band.with_columns(pl.Series("p_new", lr.predict_proba(stack_features(band))[:, 1].astype(np.float32)))
    res = tp.join(band.select("s1", "r", "p_new"), on=["s1", "r"], how="left") \
            .with_columns(pl.coalesce(["p_new", "p"]).alias("p")).drop("p_new")
    res.write_parquet(os.path.join(run_dir, "test_pred.parquet"))
    s1 = pl.read_parquet(os.path.join(PREP["test"], "test_s1.parquet"), columns=["eid", "country"])
    country = s1["country"].to_numpy()
    t = np.where(np.isin(country, list(TRAIN_COUNTRIES)), thr, args.unseen_thr).astype(np.float32)
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.with_columns(pl.Series("t", t[a["s1"].to_numpy()])).filter(pl.col("p") > pl.col("t"))
    r_ids = pl.read_parquet(os.path.join(PREP["test"], "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(run_dir, "output")
    write_outputs(out_dir, s1["eid"].to_numpy(), a["s1"].to_numpy(), a["r"].to_numpy(),
                  res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    n = np.bincount(a["s1"].to_numpy(), minlength=len(country))
    for c in np.unique(country):
        m = country == c
        log(f"test {c}: thr {thr if c in TRAIN_COUNTRIES else args.unseen_thr} mean matches {n[m].mean():.3f} "
            f"empty {np.mean(n[m] == 0):.4f}")
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def stage_predict2(args, log):
    """Country-aware variant. Seen countries (in training): in-country stacker (validated +0.0019 dense).
    Unseen countries (France): --unseen_mode exp05 = keep exp05 p (CE not used), or
    ood = stacker fitted in the out-of-country regime (US-trained CE scored on India, cross-fit weight ~0.22)."""
    import pickle
    run_dir = os.path.join(RUNS, args.run)
    with open(os.path.join(run_dir, "stacker.pkl"), "rb") as fh:
        lr_in = pickle.load(fh)
    lr_ood = None
    if args.unseen_mode == "ood":  # needs the LOCO cross-encoder scores (train --loco)
        band_tr = pl.read_parquet(os.path.join(run_dir, "train_band.parquet"), columns=["s1", "r", "y", "p", "country"])
        ood = band_tr.join(pl.read_parquet(os.path.join(run_dir, "loco_ce.parquet")), on=["s1", "r"])                      .filter(pl.col("country") == "India")
        lr_ood = fit_stacker(ood)
        log(f"OOD stacker coef {np.round(lr_ood.coef_[0], 3)} intercept {lr_ood.intercept_[0]:.3f}")
    log(f"in-country stacker coef {np.round(lr_in.coef_[0], 3)}")
    tp = pl.read_parquet(os.path.join(RUNS, BASE, "test_pred.parquet"))
    s1 = pl.read_parquet(os.path.join(PREP["test"], "test_s1.parquet"), columns=["eid", "country"])
    country = s1["country"].to_numpy()
    band = pl.read_parquet(os.path.join(run_dir, "test_band.parquet"), columns=["s1", "r", "p"])              .join(pl.read_parquet(os.path.join(run_dir, "test_ce.parquet")), on=["s1", "r"])
    seen = np.isin(country[band["s1"].to_numpy()], list(TRAIN_COUNTRIES))
    X = stack_features(band)
    p_new = np.where(seen, lr_in.predict_proba(X)[:, 1], band["p"].to_numpy() if args.unseen_mode == "exp05"
                     else lr_ood.predict_proba(X)[:, 1]).astype(np.float32)
    band = band.with_columns(pl.Series("p_new", p_new))
    res = tp.join(band.select("s1", "r", "p_new"), on=["s1", "r"], how="left")             .with_columns(pl.coalesce(["p_new", "p"]).alias("p")).drop("p_new")
    out_run = os.path.join(RUNS, f"{args.run}_{args.unseen_mode}")
    os.makedirs(out_run, exist_ok=True)
    res.write_parquet(os.path.join(out_run, "test_pred.parquet"))
    thr = args.thr if args.thr is not None else 0.7
    t = np.where(np.isin(country, list(TRAIN_COUNTRIES)), thr, args.unseen_thr).astype(np.float32)
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.with_columns(pl.Series("t", t[a["s1"].to_numpy()])).filter(pl.col("p") > pl.col("t"))
    r_ids = pl.read_parquet(os.path.join(PREP["test"], "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(out_run, "output")
    write_outputs(out_dir, s1["eid"].to_numpy(), a["s1"].to_numpy(), a["r"].to_numpy(),
                  res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    n = np.bincount(a["s1"].to_numpy(), minlength=len(country))
    for c in np.unique(country):
        m = country == c
        log(f"[{args.unseen_mode}] test {c}: mean matches {n[m].mean():.3f} empty {np.mean(n[m] == 0):.4f}")
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pairs", "train", "score_test", "stack", "loco_eval", "predict", "predict2"])
    ap.add_argument("--run", default="exp06")
    ap.add_argument("--base", default="exp05", help="stage-2 run whose OOF/test probabilities define the band")
    ap.add_argument("--lo", type=float, default=0.01)
    ap.add_argument("--hi", type=float, default=0.99)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    ap.add_argument("--unseen_mode", default="exp05", choices=["exp05", "ood"])
    ap.add_argument("--out_name", default="test_ce.parquet")
    ap.add_argument("--model", default="", help="HF model id (default intfloat/multilingual-e5-small)")
    ap.add_argument("--model_rev", default="", help="pinned HF revision for --model")
    ap.add_argument("--test_countries", default="", help="pairs: keep only these countries' test band (comma list)")
    args = ap.parse_args()
    global BASE, MODEL, MODEL_REV
    BASE = args.base
    if args.model:                      # e.g. exp23: bert-base-uncased (Apache-2.0) for architecture diversity
        MODEL, MODEL_REV = args.model, args.model_rev or None
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}{'_loco' if args.loco else ''}.txt"))
    log("args:", vars(args))
    from er_common import set_determinism
    set_determinism(0)  # reproducible reruns
    {"pairs": stage_pairs, "train": stage_train, "score_test": stage_score_test, "stack": stage_stack, "loco_eval": stage_loco_eval,
     "predict": stage_predict, "predict2": stage_predict2}[args.stage](args, log)


if __name__ == "__main__":
    main()
