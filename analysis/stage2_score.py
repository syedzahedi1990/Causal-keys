"""Score preregistered predictions P-2026-10-03-C (docs/PREREGISTRATION.md) on stage-2 results.

Inputs: results/paper1_frames/{qwen,mistral}.json from experiments/paper1_frames.py, and Paper 1's released
native study outputs (data/mechanism/native_{qwen,mistral}.jsonl.gz) for the reproduction check.

Primary population: native cores whose base B, source S and target T = pair-swap(S) are all distinct.
m(run) = logp(T) - logp(S) over full-vocabulary log-probs of the candidate tokens (letters for LETTER).
Fits (seeds 101-103) are averaged within core before a core bootstrap (10,000 resamples, ratio of means).
  phi_f    = [m(M) - m(P)] / [m(T) - m(S)]     learned remap's S->T shift relative to the natural shift
  psi_f    = [m(P+K_M) - m(P)] / [m(M) - m(P)] key-only addition relative to the full remap
  rho_f    = [m(M+K_P) - m(M)] / [m(P) - m(M)] key-only removal relative to the full remap
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

LOCS = ["box", "basket", "shelf", "drawer", "cabinet", "closet"]
SEEDS = (101, 102, 103)
KAPPA_V = {"qwen": 0.73, "mistral": 0.53}
RNG = np.random.default_rng(0)
B = 10000


def per_core(results, arm):
    rows = {}
    for r in results:
        if r["arm"] != arm:
            continue
        c = r["core"]
        if len({c["base"], c["source"], c["target"]}) < 3:
            continue
        iS, iT = LOCS.index(c["source"]), LOCS.index(c["target"])
        m = {k: v["cand"][iT] - v["cand"][iS] for k, v in r["runs"].items()}
        avg = lambda pre: float(np.mean([m[f"{pre}_{s}"] for s in SEEDS]))
        rows[c["id"]] = {"S": m["S"], "T": m["T"], "B": m["B"], "M": avg("m3"), "P": avg("pca"), "F": avg("f_star"),
                         "add": avg("addition"), "rem": avg("removal"),
                         "addv": avg("addition_v") if "addition_v_101" in m else float("nan"),
                         "remv": avg("removal_v") if "removal_v_101" in m else float("nan"),
                         "M_T_rate": float(np.mean([r["runs"][f"m3_{s}"]["cand"].index(max(r["runs"][f"m3_{s}"]["cand"])) == iT for s in SEEDS])),
                         "P_S_rate": float(np.mean([r["runs"][f"pca_{s}"]["cand"].index(max(r["runs"][f"pca_{s}"]["cand"])) == iS for s in SEEDS]))}
    return rows


def ratio(rows, num, den, ids=None):
    ids = sorted(rows) if ids is None else ids
    a = np.array([num(rows[i]) for i in ids]); b = np.array([den(rows[i]) for i in ids])
    idx = RNG.integers(0, len(ids), (B, len(ids)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5), r


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


def reproduction(model, res, p1_root):
    meta = json.load(open(Path(p1_root) / f"data/mechanism/native_{model}.json"))
    F = meta["row_fields"]
    ids = [c["id"] for c in meta["cores"]]
    saved = {}
    for line in gzip.open(Path(p1_root) / f"data/mechanism/native_{model}.jsonl.gz"):
        r = dict(zip(F, json.loads(line)))
        if r["endpoint"] in ("M", "P") and r["view"] == 0:
            saved[(ids[r["core"]], r["endpoint"], r["seed"])] = r["global_token_id"]
    agree = {"M": [], "P": []}
    for r in res:
        if r["arm"] != "P1":
            continue
        for e, pre in (("M", "m3"), ("P", "pca")):
            for s in SEEDS:
                k = (r["core"]["id"], e, s)
                if k in saved:
                    agree[e].append(saved[k] == r["runs"][f"{pre}_{s}"]["argmax"])
    return {e: (float(np.mean(v)) if v else float("nan"), len(v)) for e, v in agree.items()}


def main(root="results/paper1_frames", p1_root=None):
    for model in ("mistral", "qwen"):
        f = Path(root) / f"{model}.json"
        if not f.exists():
            print(f"{model}: no results")
            continue
        d = json.load(open(f))
        res = d["results"]
        arms = sorted({r["arm"] for r in res}, key=["P1", "NONE", "BEFORE", "POST", "LETTER"].index)
        rows = {a: per_core(res, a) for a in arms}
        print(f"\n######## {model}  ({d['provenance']['repo']} @ {d['provenance']['revision']}, "
              f"{d['provenance']['device']} x{d['provenance']['n_gpus']}, transformers {d['provenance']['transformers']})")
        if p1_root:
            rep = reproduction(model, res, p1_root)
            print(f"  reproduction vs Paper 1 saved native outputs (P1, direct, argmax token): "
                  f"M {rep['M'][0]:.3f} (n={rep['M'][1]}), P {rep['P'][0]:.3f} (n={rep['P'][1]})")
        phis = {}
        for a in arms:
            R = rows[a]
            phi = ratio(R, lambda x: x["M"] - x["P"], lambda x: x["T"] - x["S"])
            psi = ratio(R, lambda x: x["add"] - x["P"], lambda x: x["M"] - x["P"])
            rho = ratio(R, lambda x: x["rem"] - x["M"], lambda x: x["P"] - x["M"])
            phis[a] = phi
            vinfo = ""
            if not np.isnan(next(iter(R.values()))["addv"]):
                psiv = ratio(R, lambda x: x["addv"] - x["P"], lambda x: x["M"] - x["P"])
                rhov = ratio(R, lambda x: x["remv"] - x["M"], lambda x: x["P"] - x["M"])
                vinfo = f"  psiV(add) {fmt(psiv)}  rhoV(rem) {fmt(rhov)}"
            print(f"  {a:6s} n={len(R):3d}  phi(M) {fmt(phi)}  psi(add) {fmt(psi)}  rho(rem) {fmt(rho)}{vinfo}  "
                  f"M T-rate {np.mean([x['M_T_rate'] for x in R.values()]):.3f}  "
                  f"P S-rate {np.mean([x['P_S_rate'] for x in R.values()]):.3f}  "
                  f"natural span {np.mean([x['T'] - x['S'] for x in R.values()]):+.2f}")
        if "P1" in rows and "NONE" in rows:
            ids = sorted(set(rows["P1"]) & set(rows["NONE"]))
            idx = RNG.integers(0, len(ids), (B, len(ids)))
            def phi_of(R, sel):
                a = np.array([R[i]["M"] - R[i]["P"] for i in ids]); b = np.array([R[i]["T"] - R[i]["S"] for i in ids])
                return a[sel].mean(-1) / b[sel].mean(-1)
            diff_bs = phi_of(rows["P1"], idx) - phi_of(rows["NONE"], idx)
            diff = phi_of(rows["P1"], np.arange(len(ids))) - phi_of(rows["NONE"], np.arange(len(ids)))
            psiP = ratio(rows["P1"], lambda x: x["add"] - x["P"], lambda x: x["M"] - x["P"])
            psiN = ratio(rows["NONE"], lambda x: x["add"] - x["P"], lambda x: x["M"] - x["P"])
            kv = KAPPA_V[model]
            pN = phis["NONE"]
            print(f"  PREDICTIONS ({model}):")
            print(f"    C1 transfer law: phi_NONE {fmt(pN)} within kappa_V {kv:.2f} +/- 0.15 -> "
                  f"{'MET' if abs(pN[0] - kv) <= 0.15 else 'NOT MET'}")
            print(f"    C2 phi_P1 - phi_NONE = {diff:+.3f} [{np.percentile(diff_bs, 2.5):+.3f},{np.percentile(diff_bs, 97.5):+.3f}] "
                  f"-> {'MET' if np.percentile(diff_bs, 2.5) > 0 else 'NOT MET'}")
            print(f"    C3 psi_NONE {psiN[0]:+.3f} < 0.5 * psi_P1 ({psiP[0]:+.3f}) -> "
                  f"{'MET' if psiN[0] < 0.5 * psiP[0] else 'NOT MET'}")


if __name__ == "__main__":
    main(*(sys.argv[1:] + [None] * (2 - len(sys.argv[1:]))))
