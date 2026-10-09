"""exp25_llm: zero-shot multilingual LLM judge (Qwen2.5-1.5B-Instruct, Apache-2.0) for record pairs.

Why (09-27): France (unseen) is limited by the matcher's knowledge of French naming, and the fine-tuned
cross-encoders learned the US/India vocabulary (they score LB-validated French noise-word swaps strongly negative).
A zero-shot instruct LLM is not fine-tuned on our countries: it judges a pair from general (multilingual) knowledge,
e.g. "Et Fils" is a decoration, "Sport" vs "Amis" is a different organisation, "Ander"/"Andre" is a typo.
Score = logit("Yes") - logit("No") for the first answer token of a chat prompt (one forward pass per pair).
Stages:
  valid : (a) labeled in-country uncertain pairs (dense OOF of --base_oof): AUC of the LLM score vs the model p and
              gain of a cross-validated logistic stack [logit p, llm];
          (b) LB-established French truths: noise-word swaps (true, exp11 LB +0.0014) vs descriptor swaps (mostly
              false, exp14rcd LB -0.0031): the LLM must rank the first above the second
  score : score the unseen-country pairs of --base with p in (--lo, --hi) -> runs/<run>/test_llm.parquet
"""
import os
import sys
import time
import argparse

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er_common import CACHE, RUNS, Logger, save_json  # noqa: E402

LLM = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SYSTEM = ("You compare two business records from different databases and decide whether they describe the same "
          "business entity. Formatting differences, abbreviations, typos, legal-form variations and missing address "
          "parts are normal for the same business. Different businesses can have similar names or share an address.")


def prompt(tok, a_name, a_addr, b_name, b_addr):
    msg = [{"role": "system", "content": SYSTEM},
           {"role": "user", "content": f"Record A: {a_name} | {a_addr}\nRecord B: {b_name} | {b_addr}\n"
                                       "Same business? Answer Yes or No."}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


class Judge:
    def __init__(self, dev="cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(LLM, revision=LLM_REV)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(LLM, revision=LLM_REV, torch_dtype=torch.float16).to(dev).eval()
        self.dev = dev
        self.yes = self.tok.encode("Yes", add_special_tokens=False)[0]
        self.no = self.tok.encode("No", add_special_tokens=False)[0]

    @torch.no_grad()
    def score(self, rows, bs=24, log=None):
        """rows: list of (a_name, a_addr, b_name, b_addr) -> np.array of logit(Yes) - logit(No)."""
        texts = [prompt(self.tok, *r) for r in rows]
        order = np.argsort([len(t) for t in texts])
        out = np.zeros(len(texts), np.float32)
        t0 = time.time()
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True).to(self.dev)
            lg = self.model(**enc).logits[:, -1, :].float()
            out[idx] = (lg[:, self.yes] - lg[:, self.no]).cpu().numpy()
            if log and (i // bs) % 200 == 0:
                log(f"  scored {i + len(idx):,}/{len(texts):,} ({(i + len(idx)) / max(time.time() - t0, 1e-6):.1f} pairs/s)")
        return out


def text_rows(pairs, split):
    P = os.path.join(CACHE, "prep_v1" if split == "train" else "prep_v2")
    s1 = pl.read_parquet(os.path.join(P, f"{split}_s1.parquet"), columns=["name_raw", "addr_raw"])
    r = pl.read_parquet(os.path.join(P, f"{split}_r.parquet"), columns=["name_raw", "addr_raw"])
    a, b = s1[pairs["s1"].to_numpy()], r[pairs["r"].to_numpy()]
    return list(zip(a["name_raw"].to_list(), a["addr_raw"].fill_null("").to_list(),
                    b["name_raw"].to_list(), b["addr_raw"].fill_null("").to_list()))


def stage_valid(args, log):
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    judge = Judge()
    # (a) in-country labeled uncertain assigned pairs
    oof = pl.read_parquet(os.path.join(RUNS, args.base_oof, "oof_combined.parquet"))
    gt = pl.read_parquet(os.path.join(CACHE, "prep_v1", "train_gt.parquet")).select(
        pl.col("s1_idx").alias("s1"), pl.col("r_idx").alias("r"), pl.lit(1).alias("y"))
    a = oof.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")
    a = a.filter((pl.col("p") > args.lo) & (pl.col("p") < args.hi)).sample(args.n_valid, seed=0)
    a = a.join(gt.with_columns(pl.col("s1").cast(a["s1"].dtype), pl.col("r").cast(a["r"].dtype)), on=["s1", "r"], how="left") \
         .with_columns(pl.col("y").fill_null(0))
    llm = judge.score(text_rows(a, "train"), log=log)
    y, p = a["y"].to_numpy(), a["p"].to_numpy()
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    X1, X2 = lp[:, None], np.stack([lp, llm, lp * llm], 1)
    q1 = cross_val_predict(LogisticRegression(max_iter=1000), X1, y, cv=5, method="predict_proba")[:, 1]
    q2 = cross_val_predict(LogisticRegression(max_iter=1000), X2, y, cv=5, method="predict_proba")[:, 1]
    ll = lambda q: float(-np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1))))
    res = {"n": int(len(y)), "pos_rate": float(y.mean()), "auc_p": float(roc_auc_score(y, p)),
           "auc_llm": float(roc_auc_score(y, llm)), "auc_stack": float(roc_auc_score(y, q2)),
           "logloss_p": ll(q1), "logloss_stack": ll(q2)}
    log("in-country uncertain pairs:", res)
    # (b) French LB-established truths
    import exp11_rolerule as R
    s1 = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_s1.parquet"), columns=["country", "name_core", "addr_nums"])
    r = pl.read_parquet(os.path.join(CACHE, "prep_v2", "test_r.parquet"), columns=["country", "name_core", "addr_nums"])
    tp = pl.read_parquet(os.path.join(RUNS, args.base_fr, "test_pred.parquet")).select("s1", "r", "p")
    fr = pl.Series(s1["country"].to_numpy() == "France")
    tp = tp.filter(fr.gather(tp["s1"].to_numpy()))
    d = R.role_pairs(R.assigned(tp), s1, r).filter(pl.col("swap") & pl.col("hn_eq") & (pl.col("wn") >= 200))
    groups = {"noise-word swap (LB: true)": d.filter(pl.col("w").is_in(["fils", "associes"])),
              "descriptor swap (LB: mostly false)": d.filter(pl.col("nsc") <= 0.1)}
    out = {}
    for name, g in groups.items():
        g = g.sample(min(args.n_fr, g.height), seed=0)
        sc = judge.score(text_rows(g, "test"), log=log)
        out[name] = {"n": int(len(sc)), "mean": float(sc.mean()), "share_yes": float((sc > 0).mean()),
                     "model_p_mean": float(g["p"].mean())}
        log(f"France {name}:", out[name])
    save_json({"in_country": res, "france_lb_groups": out}, os.path.join(RUNS, args.run, "valid.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["valid"])
    ap.add_argument("--run", default="exp25")
    ap.add_argument("--base_oof", default="exp20c")
    ap.add_argument("--base_fr", default="exp15c", help="France predictions BEFORE the exp11 rule")
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--n_valid", type=int, default=3000)
    ap.add_argument("--n_fr", type=int, default=800)
    args = ap.parse_args()
    os.makedirs(os.path.join(RUNS, args.run), exist_ok=True)
    log = Logger(os.path.join(RUNS, args.run, f"log_{args.stage}.txt"))
    log("args:", vars(args))
    torch.manual_seed(0)
    {"valid": stage_valid}[args.stage](args, log)


if __name__ == "__main__":
    main()
