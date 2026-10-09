"""Why XGBoost and not a conventional logistic regression for p1? Same 124 features, same dense world, same S1 folds
and weighting (all positives + 30% negatives x 1/0.3), trained on a 10% sample of the stage-1 training rows:
  lr     : logistic regression; NaN -> median + missing indicator; standardized
  lrbin  : logistic regression on one-hot quantile bins (<= 32 per feature + a NaN bin): any curve per feature, no interactions
  xgb10  : the production XGBoost settings on the same 10% sample (control for data size)
  p      : production stage-1 XGBoost OOF (exp13, 7.1M rows per fold)
usage: lr_vs_xgb.py train | eval"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))                       # src/analysis
_SRC = _os.path.dirname(_HERE)                                              # src (pipeline modules)
_REPO = _os.path.dirname(_SRC)
_WORK = _os.environ.get("ER_WORK_DIR", _os.path.join(_REPO, "work"))       # caches (data/cache) and runs/
_DATA = _os.environ.get("ER_DATA_DIR", _os.path.join(_REPO, "dataset"))    # train/ and test/ TSVs
import os, sys, glob, argparse, time, pickle
import numpy as np, polars as pl, xgboost as xgb
WT = _SRC
sys.path.insert(0, WT)
os.environ["ER_WORK_DIR"] = _WORK
import exp01_match as M1
import exp03_dense as M3
from er_common import s1_fold
C = _os.path.join(_WORK, 'data', 'cache')
R = _os.path.join(_WORK, 'runs')
SP = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

keep = M3.keep_mask(argparse.Namespace(keep_frac=0.8, world_seed=0)); n1 = len(keep); fold = s1_fold(n1)
b0 = xgb.Booster(); b0.load_model(os.path.join(R, "exp13", "stage1_fold0.json")); F = b0.feature_names
oof = pl.read_parquet(os.path.join(R, "exp13", "s1_oof.parquet"), columns=["s1", "r", "y", "p"])
LAYERS = [sorted(glob.glob(os.path.join(d, "part*.parquet"))) for d in
          [os.path.join(C, "feat_v3", "k80s0"), os.path.join(C, "feat_v7", "train"), os.path.join(C, "feat_v9", "train"), os.path.join(C, "feat_v13", "train")]]


def load_part(i):
    X = pl.concat([pl.read_parquet(f) for f in (L[i] for L in LAYERS)], how="horizontal")
    return X["s1"].to_numpy(), X.select([pl.col(c).cast(pl.Float32) for c in F]).to_numpy()


class Plain:
    def fit(self, X, y, w):
        from sklearn.linear_model import LogisticRegression
        self.nan_cols = np.where(np.isnan(X).any(0))[0]
        med = np.nanmedian(X, axis=0); self.med = np.where(np.isnan(med), 0, med).astype(np.float32)
        Z = np.where(np.isnan(X), self.med, X)
        self.mu = Z.mean(0); self.sd = Z.std(0) + 1e-6
        self.m = LogisticRegression(C=1.0, max_iter=1000).fit(self._z(X), y, sample_weight=w)
        self.iters = int(self.m.n_iter_[0]); return self

    def _z(self, X):
        ind = np.isnan(X[:, self.nan_cols]).astype(np.float32)
        Z = (np.where(np.isnan(X), self.med, X) - self.mu) / self.sd
        return np.hstack([Z.astype(np.float32), ind])

    def predict(self, X):
        return self.m.predict_proba(self._z(X))[:, 1].astype(np.float32)


class Binned:
    def fit(self, X, y, w, nb=32):
        from sklearn.linear_model import LogisticRegression
        import scipy.sparse as sp
        self.edges = []
        for j in range(X.shape[1]):
            x = X[:, j]; x = x[~np.isnan(x)]
            self.edges.append(np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1])).astype(np.float32) if len(x) else np.zeros(0, np.float32))
        self.nbin = np.array([len(e) + 2 for e in self.edges])          # bins 0..len(e), NaN bin len(e)+1
        self.off = np.concatenate([[0], np.cumsum(self.nbin)[:-1]]).astype(np.int32)
        B = self._bins(X); n, d = B.shape
        A = sp.csr_matrix((np.ones(n * d, np.float32), B.ravel(), np.arange(0, n * d + 1, d)), shape=(n, int(self.nbin.sum())))
        self.m = LogisticRegression(C=1.0, max_iter=1000).fit(A, y, sample_weight=w)
        self.coef = self.m.coef_[0].astype(np.float32); self.b = float(self.m.intercept_[0])
        self.iters = int(self.m.n_iter_[0]); return self

    def _bins(self, X):
        B = np.empty(X.shape, np.int32)
        for j, e in enumerate(self.edges):
            x = X[:, j]; bj = np.searchsorted(e, x, side="right")
            bj[np.isnan(x)] = len(e) + 1
            B[:, j] = bj + self.off[j]
        return B

    def predict(self, X):
        z = self.coef[self._bins(X)].sum(1) + self.b
        return (1 / (1 + np.exp(-z))).astype(np.float32)


def train():
    rng = np.random.default_rng(0); S = 0.10
    Xs, ys, Xv, yv = ({0: [], 1: []} for _ in range(4))
    st = 0
    for i in range(len(LAYERS[0])):
        s1, M = load_part(i)
        y = oof["y"].slice(st, len(s1)).to_numpy(); st += len(s1)
        assert (oof["s1"].slice(st - len(s1), len(s1)).to_numpy() == s1).all()
        f = fold[s1]; u = rng.random(len(y))
        for k in (0, 1):
            m = (f == k) & (((y == 1) & (u < S)) | ((y == 0) & (u < S * 0.3)))
            Xs[k].append(M[m]); ys[k].append(y[m])
            mv = (f == k) & (u > 0.985)
            Xv[k].append(M[mv]); yv[k].append(y[mv])
        del M
    for d in (Xs, ys, Xv, yv):
        for k in (0, 1):
            d[k] = np.concatenate(d[k])
    log(f"sample rows per fold: {len(ys[0]):,} / {len(ys[1]):,} (positives {int(ys[0].sum()):,} / {int(ys[1].sum()):,})")
    models = {}
    for k in (0, 1):
        w = np.where(ys[k] == 1, 1.0, 1 / 0.3).astype(np.float32)
        lr = Plain().fit(Xs[k], ys[k], w); log(f"fold {k}: plain LR fitted ({lr.iters} iterations)")
        lb = Binned().fit(Xs[k], ys[k], w); log(f"fold {k}: binned LR fitted ({lb.iters} iterations, {int(lb.nbin.sum())} columns)")
        dtr = xgb.QuantileDMatrix(Xs[k], ys[k], weight=w, feature_names=F)
        dva = xgb.QuantileDMatrix(Xv[1 - k], yv[1 - k], ref=dtr, feature_names=F)
        bst = xgb.train(M1.xgb_params(argparse.Namespace(device="cuda", depth=8, eta=0.1)), dtr, num_boost_round=2000,
                        evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=False)
        bst.save_model(os.path.join(SP, f"xgb10_fold{k}.json"))
        log(f"fold {k}: XGBoost-10% best iteration {bst.best_iteration}")
        models[k] = {"lr": lr, "lrbin": lb}
    with open(os.path.join(SP, "lr_models.pkl"), "wb") as fh:
        pickle.dump(models, fh)


def evaluate():
    from sklearn.metrics import roc_auc_score
    with open(os.path.join(SP, "lr_models.pkl"), "rb") as fh:
        models = pickle.load(fh)
    for k in (0, 1):
        b = xgb.Booster(); b.load_model(os.path.join(SP, f"xgb10_fold{k}.json")); models[k]["xgb10"] = b
    P = {"lr": [], "lrbin": [], "xgb10": []}
    for i in range(len(LAYERS[0])):
        s1, M = load_part(i)
        f = fold[s1]; out = {n: np.zeros(len(s1), np.float32) for n in P}
        for k in (0, 1):
            idx = np.where(f == k)[0]; mdl = models[1 - k]
            for a in range(0, len(idx), 750_000):
                sl = idx[a:a + 750_000]; Xc = M[sl]
                out["lr"][sl] = mdl["lr"].predict(Xc)
                out["lrbin"][sl] = mdl["lrbin"].predict(Xc)
                out["xgb10"][sl] = mdl["xgb10"].inplace_predict(Xc, iteration_range=(0, mdl["xgb10"].best_iteration + 1))
        for n in P:
            P[n].append(out[n])
        del M
        log(f"part {i} scored")
    res = oof.with_columns([pl.Series(n, np.concatenate(v)) for n, v in P.items()])
    res.select("s1", "r", "y", "p", "lr", "lrbin", "xgb10").write_parquet(os.path.join(SP, "lr_vs_xgb_oof.parquet"))
    gt = M3.labels_kept(keep); npos = gt.height
    y = res["y"].to_numpy()
    budget = int((res["p"] > 0.01).sum())
    rows = []
    for name, lab in [("p", "XGBoost (production, 7.1M rows/fold)"), ("xgb10", "XGBoost, same 10% sample"),
                      ("lrbin", "logistic regression on 32 bins/feature"), ("lr", "logistic regression (plain)")]:
        p = res[name].to_numpy().astype(np.float64)
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        ll = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
        auc = float(roc_auc_score(y, p))
        cut = np.partition(p, len(p) - budget)[len(p) - budget]
        rec_budget = float(y[p >= cut].sum()) / npos
        miss_budget = int(y.sum() - y[p >= cut].sum())
        a = res.select("s1", "r", pl.col(name).alias("q"))
        a = a.filter(pl.col("q") == pl.col("q").max().over("r")).unique(subset=["r"], keep="first")
        best = (None, -1.0)
        for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
            d = a.filter(pl.col("q") > thr)
            fv = M3.f05_world(d["s1"].to_numpy(), d["r"].to_numpy(), gt, keep)
            if fv > best[1]:
                best = (thr, fv)
        rows.append({"model": lab, "logloss": round(ll, 5), "AUC": round(auc, 6),
                     "filter recall @ same #pairs": round(rec_budget, 5), "true pairs lost by filter": miss_budget,
                     "stage-1 F0.5": round(best[1], 5), "best thr": best[0]})
        log(f"{lab}: done")
    pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(10)
    print(f"true pairs {npos:,}; in candidates {int(y.sum()):,}; filter budget = production p1 > 0.01 = {budget:,} pairs")
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1]]()
