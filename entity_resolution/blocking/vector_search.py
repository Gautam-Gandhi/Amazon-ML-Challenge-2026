"""Approximate nearest-neighbour search on the GPU: an inverted-file (IVF) index built with spherical k-means.

The corpus is clustered (about 1,000 vectors per cluster); each query scans only its NPROBE closest clusters, and the
top-k is merged cluster by cluster. Vectors are kept on the CPU in fp16 and moved to the GPU cluster by cluster, so
millions of records fit a 4 GB card.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..utils import progress


@torch.no_grad()
def encode_cpu(enc, model, tag, rows, bs=131072):
    """Encode rows -> CPU fp16 tensor (n, 2d): name half | address half."""
    out = torch.empty((len(rows), 2 * model.name.embedding_dim), dtype=torch.float16)
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        e = torch.cat([enc.batch(model, tag, "name", r), enc.batch(model, tag, "addr", r)], dim=1)
        out[i:i + len(r)] = e.half().cpu()
    return out


@torch.no_grad()
def kmeans_gpu(X, k, dev, iters=8, sample=400000, seed=0):
    """Spherical k-means on a sample of CPU fp16 rows X -> (k, d) fp16 centroids on the GPU."""
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
def ivf_topk(Q, X, k, dev, nprobe=32, per_cluster=1000, q_mega=1000000, max_elems=60_000_000, desc="IVF search"):
    """Approximate top-k inner product. Q: CPU fp16 (nq, d) queries, X: CPU fp16 (nc, d) corpus.
    Returns (scores fp16 (nq, k), corpus indices int32 (nq, k); -1 = none)."""
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
    bar = progress(total=K * (-(-nq // q_mega)), desc=desc, unit="cluster")
    for q0 in range(0, nq, q_mega):
        Qg = Q[q0:q0 + q_mega].to(dev)
        m = Qg.shape[0]
        probes = torch.cat([torch.topk(Qg[i:i + 65536] @ C.T, npb, dim=1).indices for i in range(0, m, 65536)])
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
            bar.update(1)
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
    bar.close()
    del Xs, C, order_g
    torch.cuda.empty_cache()
    return out_v, out_i


@torch.no_grad()
def ivf_search(enc, model, q_tag, q_rows, c_tag, c_rows, ks, d, nprobe=32, per_cluster=1000, desc=""):
    """Top-k per channel: name (name half), addr (address half), joint (full vector)."""
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
        out[ch] = ivf_topk(Q, X, k, dev, nprobe=nprobe, per_cluster=per_cluster, desc=f"{desc}{ch} channel")
        del Q, X
    return out


@torch.no_grad()
def pair_cosines(enc, model, s1_rows, r_rows, bs=400000):
    """Exact name and address cosines for given (s1, r) pairs."""
    cn = np.zeros(len(s1_rows), np.float32)
    ca = np.zeros(len(s1_rows), np.float32)
    for i in range(0, len(s1_rows), bs):
        a, b = s1_rows[i:i + bs], r_rows[i:i + bs]
        cn[i:i + bs] = (enc.batch(model, "s1", "name", a) * enc.batch(model, "r", "name", b)).sum(1).cpu().numpy()
        ca[i:i + bs] = (enc.batch(model, "s1", "addr", a) * enc.batch(model, "r", "addr", b)).sum(1).cpu().numpy()
    return cn, ca
