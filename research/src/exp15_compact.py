"""exp15_compact: compact candidate set = ANN blocking + learned pair filter (stage 1), matcher (stage 2) on the filtered set.

Why (organisers' update 09-26): candidate_pairs.tsv counts in the final ranking and a SMALLER candidate set per Source 1
entity ranks higher. Our candidate set was the pruned ANN output: 18.56 pairs per S1 (32.2M test pairs) for ~3.4 true
matches. Measured on dense validation (EXPERIMENTS.md "exp15"): filtering the ANN candidates with the stage-1 model at
p1 > 0.01 keeps recall 0.9842 (vs 0.9843) at 3.65 pairs per S1 with unchanged F0.5; tighter ANN cut-offs alone lose
recall (top-3: 6.0 per S1, recall 0.9756, F -0.0006).
Pipeline: ANN blocking (exp01_block, ~19 per S1) -> stage-1 GBDT filter (exp13 stage-1 models, cross-fitted OOF on
train) keeps p1 > --thr_seen (training countries) / > --thr_unseen (other countries: stage 1 is less calibrated out of
country) -> candidate set (~4 per S1) -> stage 2 retrained on the filtered set, with its competition / consistency
features computed ONLY over the kept pairs -> exp10 CE stack (seen countries) -> exp11 rule (unseen countries).
candidate_pairs.tsv = the filtered set = exactly the pairs fed to the stage-2 matcher.
Stages: feats2 | train2 | predict   (stage-1 features/models reused from exp13)
"""
import os
import sys
import glob
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger  # noqa: E402
import exp02_stack as M2  # noqa: E402
import exp05_tokfeat as E5  # noqa: E402
import exp13_rolefeat as E13  # noqa: E402

S2DIR15 = os.path.join(CACHE, "feat_v15s2")
SEEN = {"US", "India"}


def stage_feats2(args, log):
    """p1 = exp13 stage-1 probability (cross-fitted OOF on train, fold average on test), stored as column p1 of the
    exp13 stage-2 parts. Keep p1 > thr, recompute stage-2 features on the kept pairs only."""
    for split in ["train", "test"]:
        base = sorted(glob.glob(os.path.join(E5.FEAT3 if split == "train" else E5.FEAT4T, "part*.parquet")))
        old = sorted(glob.glob(os.path.join(E13.S2DIR13, split, "part*.parquet")))
        assert len(base) == len(old)
        country = pl.read_parquet(os.path.join(E5.PREP[split], f"{split}_s1.parquet"), columns=["country"])["country"].to_numpy()
        seen = np.isin(country, list(SEEN))
        parts, n_all = [], 0
        for i, (b, o) in enumerate(zip(base, old)):
            d = pl.concat([pl.read_parquet(b, columns=["s1", "r"]), pl.read_parquet(o, columns=["p1"])], how="horizontal")
            thr = np.where(seen[d["s1"].to_numpy()], args.thr_seen, args.thr_unseen)
            m = d["p1"].to_numpy() > thr
            parts.append(d.with_row_index("_row").filter(pl.Series(m)).with_columns(pl.lit(i).alias("_part")))
            n_all += d.height
        t = pl.concat(parts).rename({"p1": "p"})
        n1 = len(country)
        log(f"{split}: kept {t.height:,} of {n_all:,} pairs ({t.height / n_all:.1%}); {t.height / n1:.2f} per S1 "
            f"(was {n_all / n1:.2f})")
        t = M2.group_feats(t.with_row_index("row"))
        r_tab = pl.read_parquet(os.path.join(E5.PREP[split], f"{split}_r.parquet"),
                                columns=["name_core", "addr_clean", "addr_nums", "src"])
        cons = M2.consistency_feats(t, r_tab, 3, log)
        full = pl.concat([t.drop("row"), cons.drop("row")], how="horizontal").rename({"p": "p1"}).drop("s1", "r")
        out = os.path.join(S2DIR15, split)
        os.makedirs(out, exist_ok=True)
        for i, b in enumerate(base):
            full.filter(pl.col("_part") == i).drop("_part").write_parquet(os.path.join(out, os.path.basename(b)))
        log(f"stage-2 feats {split}: {full.height:,}")


def use_layers(args):
    """exp13 layers for stage 1; stage-2 iterator = exp13 layers restricted to the kept rows + exp15 stage-2 features."""
    E13.use_layers(args)
    E5.S2DIR = S2DIR15

    def iter_s2(split):
        base = E5.iter_s1(split)
        p2 = sorted(glob.glob(os.path.join(S2DIR15, split, "part*.parquet")))

        def gen():
            for a, b in zip(base(), p2):
                s = pl.read_parquet(b)
                rows = s["_row"].to_numpy()
                yield pl.concat([a[rows], s.drop("_row")], how="horizontal")
        return gen
    E5.iter_s2 = iter_s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["feats2", "train2", "predict"])
    ap.add_argument("--run", default="exp15")
    ap.add_argument("--test_feat", default="feat_v1/test")
    ap.add_argument("--thr_seen", type=float, default=0.01)
    ap.add_argument("--thr_unseen", type=float, default=0.003)
    ap.add_argument("--neg_frac", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--unseen_thr", type=float, default=0.9)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    from er_common import set_determinism
    set_determinism(0)
    use_layers(args)
    if args.stage == "feats2":
        stage_feats2(args, log)
    else:
        {"train2": E5.stage_train2, "predict": E5.stage_predict}[args.stage](args, log)


if __name__ == "__main__":
    main()
