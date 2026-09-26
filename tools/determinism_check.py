"""Run each GPU/randomized training component twice on a small sample and report whether results are bit-identical.
Usage: python tools/determinism_check.py [--deterministic]
  --deterministic : enable er_common.set_determinism() (seeds + torch deterministic algorithms + cuBLAS config)
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    if args.deterministic:
        from er_common import set_determinism
        set_determinism(0)
    import torch
    import polars as pl
    import exp01_block as B
    dev = torch.device("cuda")

    # 1) k-means (IVF) on random fp16 data
    X = torch.randn(200000, 128, generator=torch.Generator().manual_seed(1)).half()
    c1 = B.kmeans_gpu(X, 512, dev).float().cpu().numpy()
    c2 = B.kmeans_gpu(X, 512, dev).float().cpu().numpy()
    print(f"kmeans_gpu identical: {np.array_equal(c1, c2)} (max diff {np.abs(c1 - c2).max():.2e})", flush=True)

    # 2) EmbeddingBag towers: a few SparseAdam steps on random bags
    def tower_run():
        torch.manual_seed(0)
        m = B.Towers(64).to(dev)
        opt = torch.optim.SparseAdam(m.parameters(), lr=0.01)
        g = torch.Generator(device="cpu").manual_seed(2)
        for _ in range(30):
            flat = torch.randint(0, B.NB, (40000,), generator=g).to(dev)
            off = torch.arange(0, 40000, 20, device=dev)
            w = torch.ones(40000, device=dev)
            e = m.encode("name", flat, off, w)
            loss = (e @ e.T).logsumexp(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return m.name.weight.detach().cpu().numpy()
    w1, w2 = tower_run(), tower_run()
    print(f"embedding towers identical: {np.array_equal(w1, w2)} (max diff {np.abs(w1 - w2).max():.2e})", flush=True)

    # 3) XGBoost GPU hist on a real feature part
    import xgboost as xgb
    import exp01_match as M1
    d = M1.add_labels(pl.read_parquet(os.path.join(ROOT, "data/cache/feat_v3/k80s0/part000.parquet")))
    feats = [c for c in d.columns if c not in M1.NON_FEATS]
    Xn = d.select(feats).to_numpy().astype(np.float32)[:1000000]
    y = d["y"].to_numpy()[:1000000]
    prm = {"objective": "binary:logistic", "device": "cuda", "tree_method": "hist", "max_depth": 8, "eta": 0.1,
           "subsample": 0.8, "colsample_bytree": 0.8, "seed": 0}
    p = []
    for _ in range(2):
        b = xgb.train(prm, xgb.QuantileDMatrix(Xn, y), 100)
        p.append(b.inplace_predict(Xn))
    print(f"xgboost gpu identical: {np.array_equal(p[0], p[1])} (max diff {np.abs(p[0] - p[1]).max():.2e})", flush=True)

    # 4) cross-encoder: 150 training steps on band pairs
    import exp06_crossenc as E
    tr = pl.read_parquet(os.path.join(ROOT, "runs/exp06/train_band.parquet")).head(9600)
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        tok, model = E.load_model(dev)
        ids = E.encode(tok, tr["ta"].to_list(), tr["tb"].to_list(), 96)
        E.train_one(tok, model, ids, tr["y"].to_numpy().astype(np.float32), np.arange(len(ids)),
                    argparse.Namespace(lr=3e-5, bs=64, epochs=1), lambda *a: None, dev)
        outs.append(E.score(tok, model, ids, np.arange(2000), 128, dev))
        del model
        torch.cuda.empty_cache()
    print(f"cross-encoder identical: {np.array_equal(outs[0], outs[1])} (max diff {np.abs(outs[0] - outs[1]).max():.2e})",
          flush=True)


if __name__ == "__main__":
    main()
