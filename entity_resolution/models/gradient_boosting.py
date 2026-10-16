"""Cross-fitted XGBoost matchers.

All models are trained in the dense training world with 2-fold cross-fitting by S1 entity: fold-k pairs are scored
by the model trained on the other fold (out-of-fold predictions for training data), test pairs by the average of both.
Each fold trains on all positives plus 30% of the negatives (re-weighted), with early stopping on a small slice.

Three matchers:
  token matcher : base + token features; its uncertain band is the training data of cross-encoder 1
  stage 1       : base + token + consensus + role features (124); its probability p1 filters the candidates
  stage 2       : the 124 features + p1-competition and consistency features, trained only on the kept candidates
"""
import os
import glob
import time

import numpy as np
import polars as pl

from .. import config as C
from ..utils import LOG, format_duration, progress, s1_fold
from ..features.pair_features import layer_dir
from ..features.competition_features import group_feats, consistency_feats
from .validation import add_labels, train_keep_mask

NON_FEATS = {"s1", "r", "y", "fold"}


def xgb_params():
    return {"objective": "binary:logistic", "eval_metric": "logloss", "device": C.XGB_DEVICE, "tree_method": "hist",
            "max_depth": C.XGB_DEPTH, "eta": C.XGB_ETA, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "lambda": 1.0, "max_bin": 256}


def model_dir(ws, name):
    return ws.dir("models", name)


def xgb_progress(rounds, desc):
    """XGBoost callback that shows the boosting rounds (and the validation loss) as a progress bar."""
    import xgboost as xgb

    class _Progress(xgb.callback.TrainingCallback):
        def before_training(self, model):
            self.bar = progress(total=rounds, desc=desc, unit="round")
            return model

        def after_iteration(self, model, epoch, evals_log):
            self.bar.update(1)
            for data, metrics in evals_log.items():
                for metric, values in metrics.items():
                    self.bar.set_postfix_str(f"{data} {metric} {float(values[-1]):.5f}", refresh=False)
            return False

        def after_training(self, model):
            self.bar.close()
            return model
    return _Progress()


class FeatureParts:
    """Iterator factory over row-aligned feature parts: each call starts a new pass; len() is the number of parts."""

    def __init__(self, n_parts, make):
        self.n_parts, self.make = n_parts, make

    def __len__(self):
        return self.n_parts

    def __call__(self):
        return self.make()


def _layer_parts(ws, split, layers):
    files = [sorted(glob.glob(os.path.join(layer_dir(ws, layer, split), "part*.parquet"))) for layer in layers]
    assert len({len(f) for f in files}) == 1, [len(f) for f in files]
    return files


def iter_layers(ws, split, layers):
    """Factory of an iterator over feature parts: the given layers concatenated column-wise (row-aligned)."""
    files = _layer_parts(ws, split, layers)
    return FeatureParts(len(files[0]), lambda: (pl.concat([pl.read_parquet(f) for f in fs], how="horizontal")
                                                for fs in zip(*files)))


def iter_with_stage2(ws, split, layers, stage2_layer):
    """Layers + a stage-2 feature layer covering all candidates (row-aligned)."""
    base = iter_layers(ws, split, layers)
    p2 = sorted(glob.glob(os.path.join(layer_dir(ws, stage2_layer, split), "part*.parquet")))
    return FeatureParts(len(base), lambda: (pl.concat([a, pl.read_parquet(b)], how="horizontal") for a, b in zip(base(), p2)))


def iter_kept(ws, split, layers, stage2_layer):
    """Layers restricted to the kept candidates (_row index of each part) + the stage-2 feature layer."""
    base = iter_layers(ws, split, layers)
    p2 = sorted(glob.glob(os.path.join(layer_dir(ws, stage2_layer, split), "part*.parquet")))

    def gen():
        for a, b in zip(base(), p2):
            s = pl.read_parquet(b)
            rows = s["_row"].to_numpy()
            yield pl.concat([a[rows], s.drop("_row")], how="horizontal")
    return FeatureParts(len(base), gen)


def cross_fit(ws, parts_iter, out_dir, name, keep):
    """2-fold cross-fit over an iterator factory of feature parts. Saves the fold models, returns the OOF table."""
    import xgboost as xgb
    n1 = len(keep)
    fold = s1_fold(n1)
    rng = np.random.default_rng(0)
    tr, va = {0: [], 1: []}, {0: [], 1: []}
    feats = None
    for part in progress(parts_iter(), desc=f"{name}: sampling training rows", total=len(parts_iter), unit="part"):
        part = add_labels(ws, part)
        if feats is None:
            feats = [c for c in part.columns if c not in NON_FEATS]
        f = fold[part["s1"].to_numpy()]
        y = part["y"].to_numpy()
        u = rng.random(len(y))
        for k in (0, 1):
            tr[k].append(part.filter(pl.Series((f == k) & ((y == 1) | (u < C.NEG_FRACTION)))).select(feats + ["y"]))
            va[k].append(part.filter(pl.Series((f == k) & (u > 0.97))).select(feats + ["y"]))
    LOG.info("[%s] %d features", name, len(feats))
    boosters = {}
    for k in (0, 1):
        T = pl.concat(tr[k]); tr[k] = None
        V = pl.concat(va[1 - k])
        ytr = T["y"].to_numpy()
        wtr = np.where(ytr == 1, 1.0, 1.0 / C.NEG_FRACTION).astype(np.float32)
        # cast in polars: to_numpy() of mixed columns would first build a float64 copy (2x the RAM)
        dtr = xgb.QuantileDMatrix(T.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(), ytr, weight=wtr,
                                  feature_names=feats)
        del T
        dva = xgb.QuantileDMatrix(V.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(), V["y"].to_numpy(),
                                  ref=dtr, feature_names=feats)
        t0 = time.time()
        bst = xgb.train(xgb_params(), dtr, num_boost_round=C.XGB_ROUNDS, evals=[(dva, "va")],
                        early_stopping_rounds=50, verbose_eval=False,
                        callbacks=[xgb_progress(C.XGB_ROUNDS, f"{name} fold {k}: boosting")])
        LOG.info("[%s] fold %d: %s training rows, best iteration %d, validation logloss %.5f (%s)", name, k,
                 f"{len(ytr):,}", bst.best_iteration, bst.best_score, format_duration(time.time() - t0))
        bst.save_model(os.path.join(out_dir, f"{name}_fold{k}.json"))
        boosters[k] = bst
        del dtr, dva
    outs = []
    for part in progress(parts_iter(), desc=f"{name}: out-of-fold predictions", total=len(parts_iter), unit="part"):
        part = add_labels(ws, part)
        f = fold[part["s1"].to_numpy()]
        X = part.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
        p = np.zeros(len(X), np.float32)
        for k in (0, 1):
            m = f == k
            b = boosters[1 - k]
            p[m] = b.inplace_predict(X[m], iteration_range=(0, b.best_iteration + 1))
        outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"], "y": part["y"], "p": p, "src": part["src"]}))
    return pl.concat(outs)


def predict_folds(out_dir, name, parts_iter):
    """Test predictions: mean of the two fold models."""
    import xgboost as xgb
    bs = []
    for k in (0, 1):
        b = xgb.Booster(); b.load_model(os.path.join(out_dir, f"{name}_fold{k}.json")); bs.append(b)
    outs = []
    for part in progress(parts_iter(), desc=f"{name}: test predictions", total=len(parts_iter), unit="part"):
        X = part.select(bs[0].feature_names).to_numpy().astype(np.float32)
        outs.append(pl.DataFrame({"s1": part["s1"], "r": part["r"],
                                  "p": np.mean([b.inplace_predict(X) for b in bs], axis=0).astype(np.float32)}))
    return pl.concat(outs)


def _stage2_layer_from_p1(ws, t, split, out_layer, part_layer):
    """p1 -> competition + consistency features for every candidate, written row-aligned with the part files."""
    t = group_feats(t.with_row_index("row"))
    r_tab = pl.read_parquet(ws.table(split, "r"), columns=["name_core", "addr_clean", "addr_nums", "src"])
    cons = consistency_feats(t, r_tab, C.N_ANCHORS)
    full = pl.concat([t.drop("row"), cons.drop("row")], how="horizontal").rename({"p": "p1"}).drop("s1", "r")
    out = layer_dir(ws, out_layer, split)
    st = 0
    for p in sorted(glob.glob(os.path.join(layer_dir(ws, part_layer, split), "part*.parquet"))):
        n = pl.scan_parquet(p).select(pl.len()).collect().item()
        full.slice(st, n).write_parquet(os.path.join(out, os.path.basename(p)))
        st += n
    assert st == full.height


# ---------------------------------------------------------------------------------------------- token matcher
TOKEN_LAYERS = ["base", "tokens"]


def train_token_matcher_stage1(ws):
    """Stage: token matcher, first stage."""
    out = model_dir(ws, "token_matcher")
    res = cross_fit(ws, iter_layers(ws, "train", TOKEN_LAYERS), out, "stage1", train_keep_mask(ws))
    res.write_parquet(os.path.join(out, "s1_oof.parquet"))


def build_token_matcher_stage2_features(ws):
    """Stage: token matcher, p1-based features (train: out-of-fold p1, test: fold average)."""
    out = model_dir(ws, "token_matcher")
    for split in ["train", "test"]:
        if split == "train":
            t = pl.read_parquet(os.path.join(out, "s1_oof.parquet"), columns=["s1", "r", "p"])
        else:
            t = predict_folds(out, "stage1", iter_layers(ws, "test", TOKEN_LAYERS))
        _stage2_layer_from_p1(ws, t, split, "token_matcher_s2", "tokens")
        LOG.info("token matcher stage-2 features %s: %s rows", split, f"{t.height:,}")


def train_token_matcher_stage2(ws):
    """Stage: token matcher, second stage (out-of-fold) + test predictions."""
    out = model_dir(ws, "token_matcher")
    res = cross_fit(ws, iter_with_stage2(ws, "train", TOKEN_LAYERS, "token_matcher_s2"), out, "stage2", train_keep_mask(ws))
    res.write_parquet(os.path.join(out, "oof.parquet"))
    predict_folds(out, "stage2", iter_with_stage2(ws, "test", TOKEN_LAYERS, "token_matcher_s2")).write_parquet(
        os.path.join(out, "test_pred.parquet"))


# ---------------------------------------------------------------------------------------------- stage 1 / candidate filter
STAGE1_LAYERS = ["base", "tokens", "consensus", "roles"]


def train_stage1(ws):
    """Stage: stage-1 matcher on the 124 pair features."""
    out = model_dir(ws, "stage1")
    res = cross_fit(ws, iter_layers(ws, "train", STAGE1_LAYERS), out, "stage1", train_keep_mask(ws))
    res.write_parquet(os.path.join(out, "s1_oof.parquet"))


def predict_stage1(ws):
    """Stage: p1 for every candidate (train: out-of-fold, test: fold average), stored row-aligned."""
    out = model_dir(ws, "stage1")
    for split in ["train", "test"]:
        if split == "train":
            t = pl.read_parquet(os.path.join(out, "s1_oof.parquet"), columns=["s1", "r", "p"])
        else:
            t = predict_folds(out, "stage1", iter_layers(ws, "test", STAGE1_LAYERS))
        dst = layer_dir(ws, "p1", split)
        st = 0
        for p in sorted(glob.glob(os.path.join(layer_dir(ws, "tokens", split), "part*.parquet"))):
            n = pl.scan_parquet(p).select(pl.len()).collect().item()
            t.slice(st, n).select(pl.col("p").alias("p1")).write_parquet(os.path.join(dst, os.path.basename(p)))
            st += n
        assert st == t.height
        LOG.info("stage-1 p1 %s: %s pairs", split, f"{t.height:,}")


# ---------------------------------------------------------------------------------------------- stage 2
def build_stage2_features(ws):
    """Stage: candidate filter (p1 > 0.01 training countries / 0.003 other countries) and stage-2 features computed
    over the kept candidates only. The kept set is the candidate set of the final model."""
    for split in ["train", "test"]:
        base = sorted(glob.glob(os.path.join(layer_dir(ws, "base", split), "part*.parquet")))
        old = sorted(glob.glob(os.path.join(layer_dir(ws, "p1", split), "part*.parquet")))
        assert len(base) == len(old)
        country = pl.read_parquet(ws.table(split, "s1"), columns=["country"])["country"].to_numpy()
        seen = np.isin(country, list(C.TRAINING_COUNTRIES))
        parts, n_all = [], 0
        for i, (b, o) in enumerate(zip(base, old)):
            d = pl.concat([pl.read_parquet(b, columns=["s1", "r"]), pl.read_parquet(o, columns=["p1"])], how="horizontal")
            thr = np.where(seen[d["s1"].to_numpy()], C.CANDIDATE_FILTER["seen"], C.CANDIDATE_FILTER["unseen"])
            m = d["p1"].to_numpy() > thr
            parts.append(d.with_row_index("_row").filter(pl.Series(m)).with_columns(pl.lit(i).alias("_part")))
            n_all += d.height
        t = pl.concat(parts).rename({"p1": "p"})
        LOG.info("%s: candidate filter keeps %s of %s pairs (%.2f per S1 entity)", split, f"{t.height:,}", f"{n_all:,}",
                 t.height / len(country))
        t = group_feats(t.with_row_index("row"))
        r_tab = pl.read_parquet(ws.table(split, "r"), columns=["name_core", "addr_clean", "addr_nums", "src"])
        cons = consistency_feats(t, r_tab, C.N_ANCHORS)
        full = pl.concat([t.drop("row"), cons.drop("row")], how="horizontal").rename({"p": "p1"}).drop("s1", "r")
        out = layer_dir(ws, "stage2", split)
        for i, b in enumerate(base):
            full.filter(pl.col("_part") == i).drop("_part").write_parquet(os.path.join(out, os.path.basename(b)))


def train_stage2(ws):
    """Stage: stage-2 matcher on the kept candidates (out-of-fold) + test predictions."""
    out = model_dir(ws, "stage2")
    res = cross_fit(ws, iter_kept(ws, "train", STAGE1_LAYERS, "stage2"), out, "stage2", train_keep_mask(ws))
    res.write_parquet(os.path.join(out, "oof.parquet"))
    predict_folds(out, "stage2", iter_kept(ws, "test", STAGE1_LAYERS, "stage2")).write_parquet(
        os.path.join(out, "test_pred.parquet"))
