"""exp01_match: pair features + cross-fitted XGBoost (GPU) + F0.5-aware decoding -> submission.

Stages:
  feats   --split {train,test} : candidate pairs (cand_v1) -> feature parquet parts (data/cache/feat_v1/{split}/)
  train                        : 2-fold cross-fit on train (S1 folds, same as blocker), OOF preds + decoding search
  predict                      : test features -> avg of fold models -> decode -> runs/<run>/output/*.tsv + validate
"""
import os
import sys
import time
import glob
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, save_json, s1_fold, macro_f05, write_outputs, validate_outputs  # noqa

PREP = os.path.join(CACHE, "prep_v1")
CAND = os.path.join(CACHE, "cand_v1")
FEAT = os.path.join(CACHE, "feat_v1")

S1_COLS = ["name_core", "name_clean", "name_sfx", "addr_clean", "addr_nums", "addr_ncomp", "name_flags", "country"]
R_COLS = S1_COLS + ["name_alias", "src"]


# ------------------------------------------------------------------------------ features
def load_tables_ordered(split):
    s1 = pl.read_parquet(os.path.join(PREP, f"{split}_s1.parquet"), columns=S1_COLS).with_row_index("i")
    r = pl.read_parquet(os.path.join(PREP, f"{split}_r.parquet"), columns=R_COLS).with_row_index("i")
    s1_cnt = s1.group_by(["country", "name_core"]).len("s1_core_cnt")
    r_cnt = r.group_by(["country", "name_core"]).len("r_core_cnt")
    s1 = (s1.join(s1_cnt, on=["country", "name_core"], how="left")
            .join(r_cnt, on=["country", "name_core"], how="left").sort("i"))
    r = (r.join(s1_cnt, on=["country", "name_core"], how="left")
           .join(r_cnt, on=["country", "name_core"], how="left").sort("i"))
    fill = [pl.col("s1_core_cnt").fill_null(0).cast(pl.Int32), pl.col("r_core_cnt").fill_null(0).cast(pl.Int32)]
    return s1.with_columns(fill), r.with_columns(fill)


def fuzz_feats(a, b, prefix):
    """rapidfuzz pairwise scores (multithreaded C++) -> dict of uint8 arrays."""
    from rapidfuzz import process, fuzz
    from rapidfuzz.distance import JaroWinkler
    out = {}
    for nm, sc in [("ratio", fuzz.ratio), ("tsort", fuzz.token_sort_ratio), ("tset", fuzz.token_set_ratio),
                   ("partial", fuzz.partial_ratio)]:
        out[f"{prefix}_{nm}"] = process.cpdist(a, b, scorer=sc, workers=-1, dtype=np.uint8)
    if prefix.startswith("n"):
        out[f"{prefix}_jw"] = (process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=-1,
                                              dtype=np.float32) * 100).astype(np.uint8)
    return out


def add_ranks(cand):
    """cos_j + rank of each pair within its S1 and within its R (used for pruning)."""
    return cand.with_columns((pl.col("cos_n") + pl.col("cos_a")).alias("cos_j")).with_columns([
        pl.col("cos_j").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("s1_rank_j"),
        pl.col("cos_j").rank("ordinal", descending=True).over("r").cast(pl.Int16).alias("r_rank_j"),
    ])


def relational(cand):
    """Competition features over the (pruned) candidate table (by s1 and by r)."""
    return cand.with_columns([
        pl.len().over("s1").cast(pl.Int16).alias("s1_ncand"),
        (pl.col("cos_j") - pl.col("cos_j").max().over("s1")).alias("s1_gap_j"),
        (pl.col("cos_a") - pl.col("cos_a").max().over("s1")).alias("s1_gap_a"),
        (pl.col("cos_n") - pl.col("cos_n").max().over("s1")).alias("s1_gap_n"),
        pl.len().over("r").cast(pl.Int16).alias("r_ns1"),
        (pl.col("cos_j") - pl.col("cos_j").max().over("r")).alias("r_gap_j"),
        # margin to the best *other* S1 competing for this r
        (pl.col("cos_j") - pl.col("cos_j").sort(descending=True).slice(1, 1).first().over("r")).alias("r_margin2"),
    ])


def load_cand(split, country):
    parts = sorted(glob.glob(os.path.join(CAND, split, f"{country}_*.parquet")))
    return pl.concat([pl.read_parquet(p) for p in parts])


def prune(cand, args):
    if args.max_cand > 0:
        cand = cand.filter((pl.col("s1_rank_j") <= args.max_cand) | (pl.col("r_rank_j") <= args.keep_rrank))
    return cand


def stage_feats(args, log):
    split = args.split
    out_dir = os.path.join(FEAT, split)
    os.makedirs(out_dir, exist_ok=True)
    s1, r = load_tables_ordered(split)
    log("tables loaded")
    countries = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(CAND, split, "*.parquet"))})
    pi = 0
    for country in countries:
        cand = add_ranks(load_cand(split, country))
        n0 = cand.height
        cand = relational(prune(cand, args)).sort(["s1", "r"])
        log(f"{split} {country}: {n0} candidate pairs -> {cand.height} after pruning")
        pi = _feats_country(cand, s1, r, out_dir, pi, log)


def _feats_country(cand, s1, r, out_dir, pi, log):
    chunk = 3_000_000
    n = cand.height
    for st in range(0, n, chunk):
        t0 = time.time()
        c = cand.slice(st, chunk)
        si = c["s1"].to_numpy()
        ri = c["r"].to_numpy()
        A = s1[si]
        B = r[ri]
        f = {}
        f.update(fuzz_feats(A["name_core"].to_list(), B["name_core"].to_list(), "nc"))
        f.update(fuzz_feats(A["name_clean"].to_list(), B["name_clean"].to_list(), "ncl"))
        a_ns = A["name_core"].str.replace_all(" ", "").to_list()
        b_ns = B["name_core"].str.replace_all(" ", "").to_list()
        from rapidfuzz import process, fuzz
        f["nns_ratio"] = process.cpdist(a_ns, b_ns, scorer=fuzz.ratio, workers=-1, dtype=np.uint8)
        f["nns_partial"] = process.cpdist(a_ns, b_ns, scorer=fuzz.partial_ratio, workers=-1, dtype=np.uint8)
        f.update(fuzz_feats(A["addr_clean"].to_list(), B["addr_clean"].to_list(), "ad"))
        an = A["addr_nums"].to_list()
        bn = B["addr_nums"].to_list()
        f["num_tset"] = process.cpdist(an, bn, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.uint8)
        a1 = A["addr_nums"].str.split(" ").list.first().fill_null("")
        b1 = B["addr_nums"].str.split(" ").list.first().fill_null("")
        f["num1_ratio"] = process.cpdist(a1.to_list(), b1.to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.uint8)
        df = pl.DataFrame({
            "an": A["addr_nums"], "bn": B["addr_nums"], "a1": a1, "b1": b1,
            "at": A["addr_clean"], "bt": B["addr_clean"], "nt": A["name_core"], "mt": B["name_core"],
            "asf": A["name_sfx"], "bsf": B["name_sfx"],
        })
        sets = df.select(
            pl.when((pl.col("a1") == "") | (pl.col("b1") == "")).then(-1)
            .otherwise((pl.col("a1") == pl.col("b1")).cast(pl.Int8)).cast(pl.Int8).alias("num1_eq"),
            _jacc("an", "bn").alias("num_jacc"),
            _inter("an", "bn").alias("num_inter"),
            _jacc("at", "bt").alias("ad_jacc"),
            _jacc("nt", "mt").alias("nc_jacc"),
            (pl.col("nt").str.split(" ").list.first() == pl.col("mt").str.split(" ").list.first())
            .cast(pl.Int8).alias("nc_first_eq"),
            (pl.col("asf") == pl.col("bsf")).cast(pl.Int8).alias("sfx_eq"),
            _jacc("asf", "bsf").alias("sfx_jacc"),
            pl.col("an").str.count_matches(r"\S+").cast(pl.Int8).alias("a_nnum"),
            pl.col("bn").str.count_matches(r"\S+").cast(pl.Int8).alias("b_nnum"),
            pl.col("nt").str.count_matches(r"\S+").cast(pl.Int8).alias("a_ntok"),
            pl.col("mt").str.count_matches(r"\S+").cast(pl.Int8).alias("b_ntok"),
            pl.col("at").str.count_matches(r"\S+").cast(pl.Int16).alias("a_natok"),
            pl.col("bt").str.count_matches(r"\S+").cast(pl.Int16).alias("b_natok"),
            (pl.col("asf") != "").cast(pl.Int8).alias("a_hassfx"),
            (pl.col("bsf") != "").cast(pl.Int8).alias("b_hassfx"),
        )
        fl = B["name_flags"].to_numpy()
        extra = pl.DataFrame({
            "b_native": (fl & 1).astype(np.int8), "b_domain": ((fl >> 1) & 1).astype(np.int8),
            "b_hashtag": ((fl >> 2) & 1).astype(np.int8), "b_alias": ((fl >> 3) & 1).astype(np.int8),
            "b_junk": ((fl >> 4) & 1).astype(np.int8),
            "src": B["src"].to_numpy(),
            "b_addr_empty": (B["addr_clean"] == "").cast(pl.Int8).to_numpy(),
            "a_ncomp": A["addr_ncomp"].to_numpy(), "b_ncomp": B["addr_ncomp"].to_numpy(),
            "a_s1cnt": A["s1_core_cnt"].to_numpy(), "a_rcnt": A["r_core_cnt"].to_numpy(),
            "b_s1cnt": B["s1_core_cnt"].to_numpy(), "b_rcnt": B["r_core_cnt"].to_numpy(),
            "a_nlen": A["name_core"].str.len_chars().cast(pl.Int16).to_numpy(),
            "b_nlen": B["name_core"].str.len_chars().cast(pl.Int16).to_numpy(),
        })
        feat = pl.concat([c, pl.DataFrame(f), sets, extra], how="horizontal")
        feat.write_parquet(os.path.join(out_dir, f"part{pi:03d}.parquet"))
        log(f"part {pi}: rows {feat.height} cols {feat.width} {time.time() - t0:.0f}s")
        pi += 1
    return pi


def _toks(c):
    return pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _inter(a, b):
    return _toks(a).list.set_intersection(_toks(b)).list.len().cast(pl.Int8)


def _jacc(a, b):
    i = _toks(a).list.set_intersection(_toks(b)).list.len()
    u = _toks(a).list.set_union(_toks(b)).list.len()
    return pl.when(u > 0).then(i / u).otherwise(0.0).cast(pl.Float32)


# ------------------------------------------------------------------------------ model
NON_FEATS = {"s1", "r", "y", "fold"}


def load_feats(split, cols=None):
    parts = sorted(glob.glob(os.path.join(FEAT, split, "part*.parquet")))
    return pl.concat([pl.read_parquet(p, columns=cols) for p in parts])


def add_labels(df):
    gt = pl.read_parquet(os.path.join(PREP, "train_gt.parquet")).rename({"s1_idx": "s1", "r_idx": "r"})
    gt = gt.with_columns(pl.lit(1, pl.Int8).alias("y"))
    return df.join(gt, on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))


def xgb_params(args):
    return {"objective": "binary:logistic", "eval_metric": "logloss", "device": args.device, "tree_method": "hist",
            "max_depth": args.depth, "eta": args.eta, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "lambda": 1.0, "max_bin": 256}


def iter_parts(split, cols=None):
    for p in sorted(glob.glob(os.path.join(FEAT, split, "part*.parquet"))):
        yield pl.read_parquet(p, columns=cols)


def stage_train(args, log):
    import xgboost as xgb
    run_dir = os.path.join(RUNS, args.run)
    os.makedirs(run_dir, exist_ok=True)
    n1 = pl.read_parquet(os.path.join(PREP, "train_s1.parquet"), columns=["country"]).height
    fold = s1_fold(n1)
    rng = np.random.default_rng(0)
    # one pass: sampled training rows per fold + small early-stopping slices
    tr = {0: [], 1: []}
    va = {0: [], 1: []}
    feats = None
    n_rows = n_pos = 0
    for part in iter_parts("train"):
        part = add_labels(part)
        if feats is None:
            feats = [c for c in part.columns if c not in NON_FEATS]
        f = fold[part["s1"].to_numpy()]
        y = part["y"].to_numpy()
        n_rows += len(y)
        n_pos += int(y.sum())
        u = rng.random(len(y))
        for k in (0, 1):
            m = (f == k) & ((y == 1) | (u < args.neg_frac))
            tr[k].append(part.filter(pl.Series(m)).select(feats + ["y"]))
            mv = (f == k) & (u > 0.97)
            va[k].append(part.filter(pl.Series(mv)).select(feats + ["y"]))
    log(f"train rows {n_rows}, positives {n_pos}, features {len(feats)}")
    boosters = {}
    imps = {}
    for k in (0, 1):
        T = pl.concat(tr[k]); tr[k] = None
        V = pl.concat(va[1 - k])  # early stopping on the other fold
        ytr = T["y"].to_numpy()
        wtr = np.where(ytr == 1, 1.0, 1.0 / args.neg_frac).astype(np.float32)
        dtr = xgb.QuantileDMatrix(T.select(feats).to_numpy().astype(np.float32), ytr, weight=wtr,
                                  feature_names=feats)
        del T
        dva = xgb.QuantileDMatrix(V.select(feats).to_numpy().astype(np.float32), V["y"].to_numpy(), ref=dtr,
                                  feature_names=feats)
        t0 = time.time()
        bst = xgb.train(xgb_params(args), dtr, num_boost_round=args.rounds, evals=[(dva, "va")],
                        early_stopping_rounds=50, verbose_eval=100)
        log(f"fold {k}: trained on {len(ytr)} rows, best iter {bst.best_iteration}, "
            f"best logloss {bst.best_score:.5f}, {time.time() - t0:.0f}s")
        bst.save_model(os.path.join(run_dir, f"xgb_fold{k}.json"))
        boosters[k] = bst
        imps[k] = bst.get_score(importance_type="gain")
        del dtr, dva
    # OOF predictions: rows of fold k scored by the model trained on fold 1-k
    outs = []
    for part in iter_parts("train"):
        part = add_labels(part)
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
    evaluate_decoding(res, n1, run_dir, log)


# ------------------------------------------------------------------------------ decoding
def decode(res, thr, mode, n1=None):
    """res: DataFrame(s1, r, p). Returns (s1, r) arrays of predicted matches.
    mode 'thr': p > thr
    mode 'assign': each r kept only for its argmax-p S1, then p > thr
    mode 'efb': assign, then per S1 choose the prefix of sorted candidates maximizing expected F0.5
                (thr acts as a minimum p for inclusion)."""
    df = res.select("s1", "r", "p")
    if mode in ("assign", "efb"):
        df = df.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    if mode in ("thr", "assign"):
        df = df.filter(pl.col("p") > thr)
        return df["s1"].to_numpy(), df["r"].to_numpy()
    # expected F-beta prefix selection
    b2 = 0.25
    df = df.sort(["s1", "p"], descending=[False, True]).with_columns([
        pl.col("p").cum_sum().over("s1").alias("cp"),
        pl.col("p").sum().over("s1").alias("sp"),
        (pl.int_range(pl.len()).over("s1") + 1).alias("j"),
        (1 - pl.col("p").clip(1e-6, 1 - 1e-6)).log().sum().over("s1").exp().alias("p_empty"),
    ]).with_columns(((1 + b2) * pl.col("cp") / (pl.col("j") + b2 * pl.col("sp"))).alias("ef"))
    best = df.group_by("s1").agg(pl.col("ef").max().alias("best_ef"), pl.col("p_empty").first())
    df = df.join(best, on="s1").with_columns(
        pl.col("ef").rank("ordinal", descending=True).over("s1").alias("ef_rank"))
    jstar = df.filter(pl.col("ef_rank") == 1).select("s1", pl.col("j").alias("jstar"))
    df = df.join(jstar, on="s1")
    df = df.filter((pl.col("j") <= pl.col("jstar")) & (pl.col("best_ef") > pl.col("p_empty"))
                   & (pl.col("p") > thr))
    return df["s1"].to_numpy(), df["r"].to_numpy()


def evaluate_decoding(res, n1, run_dir, log):
    gt = pl.read_parquet(os.path.join(PREP, "train_gt.parquet"))
    gs, gr = gt["s1_idx"].to_numpy(), gt["r_idx"].to_numpy()
    # candidate-set upper bound
    pos = res.filter(pl.col("y") == 1)
    ub = macro_f05(pos["s1"].to_numpy(), pos["r"].to_numpy(), gs, gr, n1)
    log(f"upper bound F0.5 (perfect matcher on candidates): {ub:.5f}; pair recall {pos.height / len(gs):.5f}")
    results = {"upper_bound": ub}
    best = (None, -1)
    for mode in ["thr", "assign", "efb"]:
        for thr in ([0.3, 0.4, 0.5, 0.6, 0.7] if mode != "efb" else [0.0, 0.1, 0.2, 0.3, 0.4]):
            ps, pr = decode(res, thr, mode)
            f = macro_f05(ps, pr, gs, gr, n1)
            results[f"{mode}@{thr}"] = f
            log(f"decode {mode} thr={thr}: F0.5={f:.5f} (pred pairs {len(ps)})")
            if f > best[1]:
                best = ((mode, thr), f)
    log(f"BEST decoding {best[0]} F0.5={best[1]:.5f}")
    results["best"] = {"mode": best[0][0], "thr": best[0][1], "f05": best[1]}
    # breakdown for best
    ps, pr = decode(res, best[0][1], best[0][0])
    per = macro_f05(ps, pr, gs, gr, n1, return_per_entity=True)
    country = pl.read_parquet(os.path.join(PREP, "train_s1.parquet"), columns=["country"])["country"].to_numpy()
    ngt = np.bincount(gs, minlength=n1)
    for c in np.unique(country):
        results[f"f05_{c}"] = float(per[country == c].mean())
    results["f05_singletons"] = float(per[ngt == 0].mean())
    results["f05_nonsingle"] = float(per[ngt > 0].mean())
    for k in range(1, 8):
        results[f"f05_ngt{k}"] = float(per[ngt == k].mean())
    log("breakdown:", {k: round(v, 5) for k, v in results.items() if k.startswith("f05")})
    save_json(results, os.path.join(run_dir, "metrics.json"))


def stage_predict(args, log):
    import xgboost as xgb
    run_dir = os.path.join(RUNS, args.run)
    import json
    with open(os.path.join(run_dir, "metrics.json")) as f:
        best = json.load(f)["best"]
    mode = args.mode or best["mode"]
    thr = args.thr if args.thr is not None else best["thr"]
    log(f"decoding with mode={mode} thr={thr}")
    boosters = []
    for f in (0, 1):
        b = xgb.Booster()
        b.load_model(os.path.join(run_dir, f"xgb_fold{f}.json"))
        boosters.append(b)
    feats = boosters[0].feature_names
    outs = []
    for part in iter_parts("test"):
        X = part.select(feats).to_numpy().astype(np.float32)
        p = np.mean([b.inplace_predict(X) for b in boosters], axis=0).astype(np.float32)
        outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"], "p": p}))
    res = pl.concat(outs)
    df = res
    res.write_parquet(os.path.join(run_dir, "test_pred.parquet"))
    ms, mr = decode(res, thr, mode)
    s1_ids = pl.read_parquet(os.path.join(PREP, "test_s1.parquet"), columns=["eid"])["eid"].to_numpy()
    r_ids = pl.read_parquet(os.path.join(PREP, "test_r.parquet"), columns=["eid"])["eid"].to_numpy()
    out_dir = os.path.join(run_dir, "output")
    write_outputs(out_dir, s1_ids, ms, mr, df["s1"].to_numpy(), df["r"].to_numpy(), r_ids)
    n1 = len(s1_ids)
    npred = np.bincount(ms, minlength=n1)
    log(f"test: {len(ms)} matches, {(npred == 0).mean():.4f} S1 empty, mean {npred.mean():.2f} per S1")
    ok = validate_outputs(out_dir, log)
    log("VALID" if ok else "INVALID")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats", "train", "predict"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--run", default="exp01")
    ap.add_argument("--max_cand", type=int, default=0)
    ap.add_argument("--keep_rrank", type=int, default=1)
    ap.add_argument("--neg_frac", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--thr", type=float, default=None)
    args = ap.parse_args()
    d = os.path.join(RUNS, args.run) if args.stage != "feats" else os.path.join(FEAT)
    log = Logger(os.path.join(d, f"log_{args.stage}_{args.split if args.stage == 'feats' else ''}.txt"))
    log("args:", vars(args))
    {"feats": stage_feats, "train": stage_train, "predict": stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
