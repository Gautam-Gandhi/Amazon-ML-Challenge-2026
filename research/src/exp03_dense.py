"""exp03_dense: train & validate in a *test-density* world via S1 dropout.

Why: test has ~5.75 S2/S3 records per S1 vs ~4.68 in train (distractor share ~40% vs 26%), and the per-country
expected-F analysis showed the model is less certain on test. Dropping a random (1-keep_frac) of train S1 entities
turns their S2/S3 records into distractors - the same mechanism that seems to create test distractors - and all
density-dependent features (ranks, competition margins, name-frequency counts) are recomputed in that world.

Stages (train world = kept S1 only; test world is untouched and reuses feat_v1/test):
  feats   : cand_v1 raw -> drop S1 -> ranks/prune/relational -> exp01 pair features -> feat_v3_<w>/train
  train1  : stage-1 XGB (2-fold cross-fit) on the dense world   -> runs/<run>/s1_oof.parquet
            also scores the exp01 fold models on the same world (reference)
  feats2  : stage-2 features (exp02 style) from stage-1 p, for train (dense) and test
  train2  : stage-2 XGB cross-fit + decoding search            -> runs/<run>/oof.parquet, metrics.json
  predict : test stage-1 -> stage-2 -> submission TSVs
"""
import os
import sys
import time
import glob
import json
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, save_json, s1_fold, macro_f05, write_outputs, validate_outputs  # noqa
import exp01_match as M1   # frozen exp01 feature code
import exp02_stack as M2   # frozen exp02 stage-2 feature code

PREP = os.path.join(CACHE, "prep_v1")


def world_tag(args):
    return f"k{int(round(args.keep_frac * 100))}s{args.world_seed}"


def feat_dir(args, split, stage):
    if split == "test" and stage == 1:
        return os.path.join(CACHE, "feat_v1", "test")  # test world unchanged
    base = "feat_v3" if stage == 1 else "feat_v3s2"
    tag = world_tag(args) if split == "train" else "test"
    if stage == 2:
        tag = f"{world_tag(args)}_{split}"  # stage-2 test features depend on the stage-1 models of this world
    return os.path.join(CACHE, base, tag)


def keep_mask(args):
    n1 = pl.read_parquet(os.path.join(PREP, "train_s1.parquet"), columns=["country"]).height
    rng = np.random.default_rng(1000 + args.world_seed)
    return rng.random(n1) < args.keep_frac


def tables_kept(keep):
    """exp01 tables, but name-frequency counts among S1 use only the kept S1 (the world the model sees)."""
    s1 = pl.read_parquet(os.path.join(PREP, "train_s1.parquet"), columns=M1.S1_COLS).with_row_index("i")
    r = pl.read_parquet(os.path.join(PREP, "train_r.parquet"), columns=M1.R_COLS).with_row_index("i")
    s1k = s1.filter(pl.Series(keep))
    s1_cnt = s1k.group_by(["country", "name_core"]).len("s1_core_cnt")
    r_cnt = r.group_by(["country", "name_core"]).len("r_core_cnt")
    s1 = s1.join(s1_cnt, on=["country", "name_core"], how="left").join(r_cnt, on=["country", "name_core"], how="left").sort("i")
    r = r.join(s1_cnt, on=["country", "name_core"], how="left").join(r_cnt, on=["country", "name_core"], how="left").sort("i")
    fill = [pl.col("s1_core_cnt").fill_null(0).cast(pl.Int32), pl.col("r_core_cnt").fill_null(0).cast(pl.Int32)]
    return s1.with_columns(fill), r.with_columns(fill)


def stage_feats(args, log):
    keep = keep_mask(args)
    out_dir = feat_dir(args, "train", 1)
    os.makedirs(out_dir, exist_ok=True)
    s1, r = tables_kept(keep)
    keep_idx = pl.Series("s1", np.where(keep)[0].astype(np.int32))
    log(f"world {world_tag(args)}: kept S1 {keep.sum()} / {len(keep)}")
    pi = 0
    for country in ["India", "US"]:
        cand = M1.load_cand("train", country)
        cand = cand.filter(pl.col("s1").is_in(keep_idx))
        cand = M1.add_ranks(cand)
        n0 = cand.height
        cand = M1.relational(M1.prune(cand, args)).sort(["s1", "r"])
        log(f"{country}: {n0} candidates in world -> {cand.height} after pruning")
        pi = M1._feats_country(cand, s1, r, out_dir, pi, log)


def labels_kept(keep):
    gt = pl.read_parquet(os.path.join(PREP, "train_gt.parquet"))
    return gt.filter(pl.Series(keep[gt["s1_idx"].to_numpy()]))


def f05_world(s, r, gt, keep, return_per=False):
    per = macro_f05(s, r, gt["s1_idx"].to_numpy(), gt["r_idx"].to_numpy(), len(keep), return_per_entity=True)
    return per[keep] if return_per else float(per[keep].mean())


def cross_fit(args, log, parts_iter, run_dir, name, keep):
    """Generic 2-fold cross-fit XGB over an iterator factory of feature parts. Returns OOF DataFrame."""
    import xgboost as xgb
    n1 = len(keep)
    fold = s1_fold(n1)
    rng = np.random.default_rng(0)
    tr, va = {0: [], 1: []}, {0: [], 1: []}
    feats = None
    for part in parts_iter():
        part = M1.add_labels(part)
        if feats is None:
            feats = [c for c in part.columns if c not in M1.NON_FEATS]
        f = fold[part["s1"].to_numpy()]
        y = part["y"].to_numpy()
        u = rng.random(len(y))
        for k in (0, 1):
            tr[k].append(part.filter(pl.Series((f == k) & ((y == 1) | (u < args.neg_frac)))).select(feats + ["y"]))
            va[k].append(part.filter(pl.Series((f == k) & (u > 0.97))).select(feats + ["y"]))
    log(f"[{name}] features {len(feats)}")
    boosters = {}
    for k in (0, 1):
        T = pl.concat(tr[k]); tr[k] = None
        V = pl.concat(va[1 - k])
        ytr = T["y"].to_numpy()
        wtr = np.where(ytr == 1, 1.0, 1.0 / args.neg_frac).astype(np.float32)
        # cast in polars: to_numpy() of mixed columns would build a float64 copy first (2x the RAM)
        dtr = xgb.QuantileDMatrix(T.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(), ytr, weight=wtr,
                                  feature_names=feats)
        del T
        dva = xgb.QuantileDMatrix(V.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(), V["y"].to_numpy(), ref=dtr,
                                  feature_names=feats)
        t0 = time.time()
        bst = xgb.train(M1.xgb_params(args), dtr, num_boost_round=args.rounds, evals=[(dva, "va")],
                        early_stopping_rounds=50, verbose_eval=200)
        log(f"[{name}] fold {k}: rows {len(ytr)}, best iter {bst.best_iteration}, logloss {bst.best_score:.5f}, "
            f"{time.time() - t0:.0f}s")
        bst.save_model(os.path.join(run_dir, f"{name}_fold{k}.json"))
        boosters[k] = bst
        del dtr, dva
    imp = boosters[0].get_score(importance_type="gain")
    imp = sorted(imp.items(), key=lambda t: -t[1])[:20]
    log(f"[{name}] top features:", [(a, round(b, 1)) for a, b in imp])
    outs = []
    for part in parts_iter():
        part = M1.add_labels(part)
        f = fold[part["s1"].to_numpy()]
        X = part.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
        p = np.zeros(len(X), np.float32)
        for k in (0, 1):
            m = f == k
            b = boosters[1 - k]
            p[m] = b.inplace_predict(X[m], iteration_range=(0, b.best_iteration + 1))
        outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"], "y": part["y"], "p": p, "src": part["src"]}))
    return pl.concat(outs)


def decode_eval(res, keep, log, tag, thrs=(0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    gt = labels_kept(keep)
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    out = {}
    best = (None, -1)
    for thr in thrs:
        d = a.filter(pl.col("p") > thr)
        f = f05_world(d["s1"].to_numpy(), d["r"].to_numpy(), gt, keep)
        out[f"assign@{thr}"] = f
        if f > best[1]:
            best = (thr, f)
    log(f"[{tag}] assign decoding: " + " ".join(f"{k.split('@')[1]}:{v:.5f}" for k, v in out.items()))
    d = a.filter(pl.col("p") > best[0])
    per = f05_world(d["s1"].to_numpy(), d["r"].to_numpy(), gt, keep, return_per=True)
    country = pl.read_parquet(os.path.join(PREP, "train_s1.parquet"), columns=["country"])["country"].to_numpy()[keep]
    ngt = np.bincount(gt["s1_idx"].to_numpy(), minlength=len(keep))[keep]
    bd = {c: round(float(per[country == c].mean()), 5) for c in np.unique(country)}
    bd["single"] = round(float(per[ngt == 0].mean()), 5)
    bd["ngt1"] = round(float(per[ngt == 1].mean()), 5)
    bd["ngt2"] = round(float(per[ngt == 2].mean()), 5)
    bd["ngt3+"] = round(float(per[ngt >= 3].mean()), 5)
    log(f"[{tag}] BEST thr={best[0]} F0.5={best[1]:.5f} breakdown {bd}")
    return {"best_thr": best[0], "best_f05": best[1], "curve": out, "breakdown": bd}


def iter_feat1(args, split):
    d = feat_dir(args, split, 1)
    return lambda: (pl.read_parquet(p) for p in sorted(glob.glob(os.path.join(d, "part*.parquet"))))


def stage_train1(args, log):
    keep = keep_mask(args)
    run_dir = os.path.join(RUNS, args.run)
    os.makedirs(run_dir, exist_ok=True)
    metrics = {}
    # reference: exp01 fold models (trained in the normal world) scored on this dense world
    import xgboost as xgb
    fold = s1_fold(len(keep))
    ref = []
    bs = []
    for k in (0, 1):
        b = xgb.Booster(); b.load_model(os.path.join(RUNS, "exp01", f"xgb_fold{k}.json")); bs.append(b)
    for part in iter_feat1(args, "train")():
        part = M1.add_labels(part)
        f = fold[part["s1"].to_numpy()]
        X = part.select(bs[0].feature_names).to_numpy().astype(np.float32)
        p = np.zeros(len(X), np.float32)
        for k in (0, 1):
            p[f == k] = bs[1 - k].inplace_predict(X[f == k])
        ref.append(pl.DataFrame({"s1": part["s1"], "r": part["r"], "y": part["y"], "p": p}))
    metrics["exp01_models_on_dense_world"] = decode_eval(pl.concat(ref), keep, log, "exp01 models, dense world")
    del ref
    res = cross_fit(args, log, iter_feat1(args, "train"), run_dir, "stage1", keep)
    res.write_parquet(os.path.join(run_dir, "s1_oof.parquet"))
    metrics["stage1"] = decode_eval(res, keep, log, "stage1 dense-trained")
    save_json(metrics, os.path.join(run_dir, "metrics_stage1.json"))


def stage_feats2(args, log):
    """exp02-style stage-2 features; train from stage-1 OOF p, test from the stage-1 fold-model average."""
    import xgboost as xgb
    run_dir = os.path.join(RUNS, args.run)
    for split in ["train", "test"]:
        if split == "train":
            t = pl.read_parquet(os.path.join(run_dir, "s1_oof.parquet"), columns=["s1", "r", "p"])
        else:
            bs = []
            for k in (0, 1):
                b = xgb.Booster(); b.load_model(os.path.join(run_dir, f"stage1_fold{k}.json")); bs.append(b)
            outs = []
            for part in iter_feat1(args, "test")():
                X = part.select(bs[0].feature_names).to_numpy().astype(np.float32)
                outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"],
                                          "p": np.mean([b.inplace_predict(X) for b in bs], axis=0).astype(np.float32)}))
            t = pl.concat(outs)
            t.write_parquet(os.path.join(run_dir, "test_stage1_pred.parquet"))
        t = M2.group_feats(t.with_row_index("row"))
        r_tab = pl.read_parquet(os.path.join(PREP, f"{split}_r.parquet"), columns=["name_core", "addr_clean", "addr_nums", "src"])
        cons = M2.consistency_feats(t, r_tab, args.n_anchor, log)
        full = pl.concat([t.drop("row"), cons.drop("row")], how="horizontal").rename({"p": "p1"})
        out = feat_dir(args, split, 2)
        os.makedirs(out, exist_ok=True)
        lens = [pl.scan_parquet(p).select(pl.len()).collect().item()
                for p in sorted(glob.glob(os.path.join(feat_dir(args, split, 1), "part*.parquet")))]
        st = 0
        for i, n in enumerate(lens):
            full.slice(st, n).drop("s1", "r").write_parquet(os.path.join(out, f"part{i:03d}.parquet"))
            st += n
        assert st == full.height
        log(f"stage-2 feats {split}: {full.height} rows")


def iter_feat12(args, split):
    d1, d2 = feat_dir(args, split, 1), feat_dir(args, split, 2)
    p1 = sorted(glob.glob(os.path.join(d1, "part*.parquet")))
    p2 = sorted(glob.glob(os.path.join(d2, "part*.parquet")))
    assert len(p1) == len(p2)
    return lambda: (pl.concat([pl.read_parquet(a), pl.read_parquet(b)], how="horizontal") for a, b in zip(p1, p2))


def stage_train2(args, log):
    keep = keep_mask(args)
    run_dir = os.path.join(RUNS, args.run)
    res = cross_fit(args, log, iter_feat12(args, "train"), run_dir, "stage2", keep)
    res.write_parquet(os.path.join(run_dir, "oof.parquet"))
    m = decode_eval(res, keep, log, "stage2 dense-trained")
    save_json(m, os.path.join(run_dir, "metrics.json"))


def stage_predict(args, log):
    import xgboost as xgb
    run_dir = os.path.join(RUNS, args.run)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        thr = json.load(f)["best_thr"] if args.thr is None else args.thr
    bs = []
    for k in (0, 1):
        b = xgb.Booster(); b.load_model(os.path.join(run_dir, f"stage2_fold{k}.json")); bs.append(b)
    outs = []
    for part in iter_feat12(args, "test")():
        X = part.select(bs[0].feature_names).to_numpy().astype(np.float32)
        outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"],
                                  "p": np.mean([b.inplace_predict(X) for b in bs], axis=0).astype(np.float32)}))
    res = pl.concat(outs)
    res.write_parquet(os.path.join(run_dir, "test_pred.parquet"))
    ms, mr = M1.decode(res, thr, "assign")
    s1_ids = pl.read_parquet(os.path.join(PREP, "test_s1.parquet"), columns=["eid"])["eid"].to_numpy()
    r_ids = pl.read_parquet(os.path.join(PREP, "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(run_dir, "output")
    write_outputs(out_dir, s1_ids, ms, mr, res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    log(f"decoded with assign thr={thr}")
    M2.summarize_test(ms, s1_ids, log)
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "train1", "feats2", "train2", "predict"])
    ap.add_argument("--run", default="exp03")
    ap.add_argument("--keep_frac", type=float, default=0.8)
    ap.add_argument("--world_seed", type=int, default=0)
    ap.add_argument("--max_cand", type=int, default=15)
    ap.add_argument("--keep_rrank", type=int, default=2)
    ap.add_argument("--n_anchor", type=int, default=3)
    ap.add_argument("--neg_frac", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    from er_common import set_determinism
    set_determinism(0)  # reproducible reruns
    {"feats": stage_feats, "train1": stage_train1, "feats2": stage_feats2, "train2": stage_train2,
     "predict": stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
