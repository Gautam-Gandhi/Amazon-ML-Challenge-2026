"""exp02_stack: stage-2 stacking on top of exp01 (stage-1 probabilities + group-consistency features).

Idea: the true S2/S3 records of one S1 entity resemble *each other*. An empty-address record with the same name
as another confident candidate (that has an address) is very likely a match, and a record that looks like none of
the confident candidates is suspicious. Stage-1 OOF probabilities make these features honest.

Stages:
  feats --split {train,test} : stage-1 p (train: runs/<base>/oof.parquet, test: runs/<base>/test_pred.parquet)
                               -> data/cache/feat_v2_<base>/{split}/partNNN.parquet (row-aligned with feat_v1 parts)
  train                      : 2-fold cross-fit XGB on feat_v1 + feat_v2 -> OOF + decoding search
  predict                    : test -> submission
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
from er_common import CACHE, RUNS, Logger, save_json, s1_fold, write_outputs, validate_outputs  # noqa
import exp01_match as M1  # reuse decoding/eval helpers (frozen exp01 code)

PREP = os.path.join(CACHE, "prep_v1")
FEAT1 = os.path.join(CACHE, "feat_v1")


def feat2_dir(base, split):
    return os.path.join(CACHE, f"feat_v2_{base}", split)


def part_lengths(split):
    return [pl.scan_parquet(p).select(pl.len()).collect().item()
            for p in sorted(glob.glob(os.path.join(FEAT1, split, "part*.parquet")))]


def stage1_table(args, split):
    if split == "train":
        t = pl.read_parquet(os.path.join(RUNS, args.base, "oof.parquet"), columns=["s1", "r", "p"])
    else:
        t = pl.read_parquet(os.path.join(RUNS, args.base, "test_pred.parquet"), columns=["s1", "r", "p"])
    return t.with_row_index("row")


def group_feats(t):
    """Probability-based competition features."""
    return t.with_columns([
        pl.col("p").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("p_rank_s1"),
        (pl.col("p") - pl.col("p").max().over("s1")).alias("p_gap_s1"),
        pl.col("p").sum().over("s1").alias("p_sum_s1"),
        (pl.col("p") > 0.5).sum().over("s1").cast(pl.Int16).alias("p_nhi_s1"),
        pl.col("p").rank("ordinal", descending=True).over("r").cast(pl.Int16).alias("p_rank_r"),
        (pl.col("p") - pl.col("p").sort(descending=True).slice(1, 1).first().over("r")).alias("p_margin_r"),
        pl.col("p").sort(descending=True).slice(1, 1).first().over("r").alias("p_2nd_r"),
    ]).with_columns([
        # expected number of other matches in this S1 besides this pair
        (pl.col("p_sum_s1") - pl.col("p")).alias("p_sum_other_s1"),
    ])


def consistency_feats(t, r_tab, n_anchor, log, chunk_s1=400000):
    """For each candidate (s1, r): similarity of r to the top-n_anchor other confident candidates of s1."""
    from rapidfuzz import process, fuzz
    anchors = (t.filter(pl.col("p") > 0.5)
                .with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("ar"))
                .filter(pl.col("ar") <= n_anchor + 1)
                .select("s1", pl.col("r").alias("a"), pl.col("p").alias("pa")))
    s1_vals = t["s1"].unique().sort().to_numpy()
    out = []
    names = r_tab["name_core"]
    addrs = r_tab["addr_clean"]
    nums = r_tab["addr_nums"]
    srcs = r_tab["src"].to_numpy()
    for i in range(0, len(s1_vals), chunk_s1):
        lo, hi = s1_vals[i], s1_vals[min(i + chunk_s1, len(s1_vals)) - 1]
        tc = t.filter((pl.col("s1") >= lo) & (pl.col("s1") <= hi)).select("row", "s1", "r")
        ac = anchors.filter((pl.col("s1") >= lo) & (pl.col("s1") <= hi))
        j = tc.join(ac, on="s1", how="inner").filter(pl.col("r") != pl.col("a"))
        ri = j["r"].to_numpy()
        ai = j["a"].to_numpy()
        nm = process.cpdist(names.gather(ri).to_list(), names.gather(ai).to_list(), scorer=fuzz.token_sort_ratio,
                            workers=-1, dtype=np.uint8)
        ad = process.cpdist(addrs.gather(ri).to_list(), addrs.gather(ai).to_list(), scorer=fuzz.token_set_ratio,
                            workers=-1, dtype=np.uint8)
        nu = process.cpdist(nums.gather(ri).to_list(), nums.gather(ai).to_list(), scorer=fuzz.token_set_ratio,
                            workers=-1, dtype=np.uint8)
        a_empty = (addrs.gather(ai) == "").to_numpy()
        ad = np.where(a_empty, 255, ad)  # 255 = anchor has no address
        same_src = (srcs[ri] == srcs[ai]).astype(np.int8)
        j = j.with_columns(pl.Series("an_name", nm), pl.Series("an_addr", ad), pl.Series("an_num", nu),
                           pl.Series("an_same_src", same_src))
        j = j.with_columns([
            pl.when(pl.col("an_addr") == 255).then(None).otherwise(pl.col("an_addr")).alias("an_addr_v"),
        ])
        agg = j.group_by("row").agg([
            pl.col("an_name").max().alias("cons_name_max"),
            pl.col("an_name").mean().alias("cons_name_mean"),
            pl.col("an_addr_v").max().alias("cons_addr_max"),
            pl.col("an_num").max().alias("cons_num_max"),
            (pl.col("an_name") * pl.col("pa")).max().alias("cons_name_pw"),
            pl.col("pa").filter(pl.col("an_name") == pl.col("an_name").max()).first().alias("cons_name_pa"),
            ((pl.col("an_name") >= 90) & (pl.col("an_same_src") == 0)).sum().cast(pl.Int8).alias("cons_n_name90_xsrc"),
            ((pl.col("an_name") >= 90) & (pl.col("an_same_src") == 1)).sum().cast(pl.Int8).alias("cons_n_name90_ssrc"),
            pl.len().cast(pl.Int8).alias("cons_n_anchor"),
        ])
        out.append(agg)
        log(f"consistency s1 {lo}..{hi}: pairs {j.height}")
    agg = pl.concat(out)
    return t.select("row").join(agg, on="row", how="left").sort("row")


def stage_feats(args, log):
    split = args.split
    t = stage1_table(args, split)
    t = group_feats(t)
    r_tab = pl.read_parquet(os.path.join(PREP, f"{split}_r.parquet"), columns=["name_core", "addr_clean", "addr_nums", "src"])
    cons = consistency_feats(t, r_tab, args.n_anchor, log)
    full = pl.concat([t.drop("row"), cons.drop("row")], how="horizontal").rename({"p": "p1"})
    out = feat2_dir(args.base, split)
    os.makedirs(out, exist_ok=True)
    st = 0
    for i, n in enumerate(part_lengths(split)):
        full.slice(st, n).drop("s1", "r").write_parquet(os.path.join(out, f"part{i:03d}.parquet"))
        st += n
    assert st == full.height, (st, full.height)
    log(f"feat_v2 {split}: {full.height} rows, cols {full.width}")


def iter_parts(args, split):
    p1 = sorted(glob.glob(os.path.join(FEAT1, split, "part*.parquet")))
    p2 = sorted(glob.glob(os.path.join(feat2_dir(args.base, split), "part*.parquet")))
    assert len(p1) == len(p2)
    for a, b in zip(p1, p2):
        yield pl.concat([pl.read_parquet(a), pl.read_parquet(b)], how="horizontal")


def stage_train(args, log):
    import xgboost as xgb
    run_dir = os.path.join(RUNS, args.run)
    os.makedirs(run_dir, exist_ok=True)
    n1 = pl.read_parquet(os.path.join(PREP, "train_s1.parquet"), columns=["country"]).height
    fold = s1_fold(n1)
    rng = np.random.default_rng(0)
    tr = {0: [], 1: []}
    va = {0: [], 1: []}
    feats = None
    for part in iter_parts(args, "train"):
        part = M1.add_labels(part)
        if feats is None:
            feats = [c for c in part.columns if c not in M1.NON_FEATS]
        f = fold[part["s1"].to_numpy()]
        y = part["y"].to_numpy()
        u = rng.random(len(y))
        for k in (0, 1):
            tr[k].append(part.filter(pl.Series((f == k) & ((y == 1) | (u < args.neg_frac)))).select(feats + ["y"]))
            va[k].append(part.filter(pl.Series((f == k) & (u > 0.97))).select(feats + ["y"]))
    log(f"features {len(feats)}")
    boosters = {}
    imps = {}
    for k in (0, 1):
        T = pl.concat(tr[k]); tr[k] = None
        V = pl.concat(va[1 - k])
        ytr = T["y"].to_numpy()
        wtr = np.where(ytr == 1, 1.0, 1.0 / args.neg_frac).astype(np.float32)
        dtr = xgb.QuantileDMatrix(T.select(feats).to_numpy().astype(np.float32), ytr, weight=wtr, feature_names=feats)
        del T
        dva = xgb.QuantileDMatrix(V.select(feats).to_numpy().astype(np.float32), V["y"].to_numpy(), ref=dtr,
                                  feature_names=feats)
        t0 = time.time()
        bst = xgb.train(M1.xgb_params(args), dtr, num_boost_round=args.rounds, evals=[(dva, "va")],
                        early_stopping_rounds=50, verbose_eval=100)
        log(f"fold {k}: rows {len(ytr)}, best iter {bst.best_iteration}, logloss {bst.best_score:.5f}, "
            f"{time.time() - t0:.0f}s")
        bst.save_model(os.path.join(run_dir, f"xgb_fold{k}.json"))
        boosters[k] = bst
        imps[k] = bst.get_score(importance_type="gain")
        del dtr, dva
    outs = []
    for part in iter_parts(args, "train"):
        part = M1.add_labels(part)
        f = fold[part["s1"].to_numpy()]
        X = part.select(feats).to_numpy().astype(np.float32)
        p = np.zeros(len(X), np.float32)
        for k in (0, 1):
            m = f == k
            b = boosters[1 - k]
            p[m] = b.inplace_predict(X[m], iteration_range=(0, b.best_iteration + 1))
        outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"], "y": part["y"], "p": p, "src": part["src"]}))
    res = pl.concat(outs)
    res.write_parquet(os.path.join(run_dir, "oof.parquet"))
    imp = pl.DataFrame({"feature": list(imps[0].keys()), "gain": list(imps[0].values())}).sort("gain", descending=True)
    imp.write_csv(os.path.join(run_dir, "feature_importance.csv"))
    log("top features:", [(d["feature"], round(d["gain"], 1)) for d in imp.head(25).to_dicts()])
    evaluate(res, n1, run_dir, log)


def evaluate(res, n1, run_dir, log):
    from er_common import macro_f05
    gt = pl.read_parquet(os.path.join(PREP, "train_gt.parquet"))
    gs, gr = gt["s1_idx"].to_numpy(), gt["r_idx"].to_numpy()
    a = res.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    results = {}
    best = (None, -1)
    for thr in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        d = a.filter(pl.col("p") > thr)
        f = macro_f05(d["s1"].to_numpy(), d["r"].to_numpy(), gs, gr, n1)
        results[f"assign@{thr}"] = f
        log(f"decode assign thr={thr}: F0.5={f:.5f} (pred pairs {d.height})")
        if f > best[1]:
            best = (thr, f)
    for thr in [0.6, 0.7, 0.8]:
        ps, pr = M1.decode(res, thr, "thr")
        results[f"thr@{thr}"] = macro_f05(ps, pr, gs, gr, n1)
        log(f"decode thr thr={thr}: F0.5={results[f'thr@{thr}']:.5f}")
    log(f"BEST assign thr={best[0]} F0.5={best[1]:.5f}")
    results["best"] = {"mode": "assign", "thr": best[0], "f05": best[1]}
    d = a.filter(pl.col("p") > best[0])
    per = macro_f05(d["s1"].to_numpy(), d["r"].to_numpy(), gs, gr, n1, return_per_entity=True)
    country = pl.read_parquet(os.path.join(PREP, "train_s1.parquet"), columns=["country"])["country"].to_numpy()
    ngt = np.bincount(gs, minlength=n1)
    for c in np.unique(country):
        results[f"f05_{c}"] = float(per[country == c].mean())
    results["f05_singletons"] = float(per[ngt == 0].mean())
    for k in range(1, 8):
        results[f"f05_ngt{k}"] = float(per[ngt == k].mean())
    log("breakdown:", {k: round(v, 5) for k, v in results.items() if k.startswith("f05")})
    save_json(results, os.path.join(run_dir, "metrics.json"))


def stage_predict(args, log):
    import xgboost as xgb
    run_dir = os.path.join(RUNS, args.run)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        best = json.load(f)["best"]
    thr = args.thr if args.thr is not None else best["thr"]
    log(f"decoding assign thr={thr}")
    boosters = []
    for k in (0, 1):
        b = xgb.Booster()
        b.load_model(os.path.join(run_dir, f"xgb_fold{k}.json"))
        boosters.append(b)
    feats = boosters[0].feature_names
    outs = []
    for part in iter_parts(args, "test"):
        X = part.select(feats).to_numpy().astype(np.float32)
        p = np.mean([b.inplace_predict(X) for b in boosters], axis=0).astype(np.float32)
        outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"], "p": p}))
    res = pl.concat(outs)
    res.write_parquet(os.path.join(run_dir, "test_pred.parquet"))
    ms, mr = M1.decode(res, thr, "assign")
    s1_ids = pl.read_parquet(os.path.join(PREP, "test_s1.parquet"), columns=["eid"])["eid"].to_numpy()
    r_ids = pl.read_parquet(os.path.join(PREP, "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(run_dir, "output")
    write_outputs(out_dir, s1_ids, ms, mr, res["s1"].to_numpy(), res["r"].to_numpy(), r_ids)
    summarize_test(ms, s1_ids, log)
    log("VALID" if validate_outputs(out_dir, log) else "INVALID")


def summarize_test(ms, s1_ids, log):
    country = pl.read_parquet(os.path.join(PREP, "test_s1.parquet"), columns=["country"])["country"].to_numpy()
    npred = np.bincount(ms, minlength=len(s1_ids))
    for c in np.unique(country):
        m = country == c
        log(f"test {c}: S1={m.sum()} mean matches {npred[m].mean():.3f}, empty {np.mean(npred[m] == 0):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "train", "predict"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--base", default="exp01", help="stage-1 run providing probabilities")
    ap.add_argument("--run", default="exp02")
    ap.add_argument("--n_anchor", type=int, default=3)
    ap.add_argument("--neg_frac", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()
    d = os.path.join(RUNS, args.run)
    log = Logger(os.path.join(d, f"log_{args.stage}_{args.split if args.stage == 'feats' else ''}.txt"))
    log("args:", vars(args))
    {"feats": stage_feats, "train": stage_train, "predict": stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
