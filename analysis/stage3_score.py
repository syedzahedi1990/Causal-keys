"""Score preregistered predictions P-2026-10-03-D (docs/PREREGISTRATION.md) on stage-3 results.

D1 generality: for each new task, a model counts if identity(K) in P1 > 0 with CI excluding 0 AND the paired
   identity(K)(P1) - identity(K)(NONE) > 0 with CI excluding 0. Met if >= 4 of 5 models per task.
D2 structural control: mean BEFORE identity(K) <= 0.5 nats in >= 9 of 10 task x model cells.
D3 localisation (belief task, exact row-restricted key swap, all layers): in P1, the choice-word rows recover >= 0.7
   of the full key effect while the question rows and the remaining tail rows each recover <= 0.15, in all 3 models.
   In POST (where the full key effect is positive), re-mention-word rows recover >= 0.5 in >= 2 of 3 models.
Ratios use a core bootstrap of the ratio of means (10,000 resamples).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_prereg import ci, fmt, per_core  # noqa: E402

MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen3-8B", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]
LOC_MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3"]
RNG = np.random.default_rng(1)


def ratio(a, b, B=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = RNG.integers(0, len(a), (B, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def main(root="results/gpu_stage3"):
    print("== D1/D2 generality: identity(K) by task, model and arm (all items, 95% core bootstrap)")
    d1, d2 = {}, []
    for task in ("paint", "schedule"):
        d1[task] = 0
        for m in MODELS:
            f = Path(root) / "task_factorial" / f"{task}_{m}_s0.json"
            if not f.exists():
                print(f"  {task:8s} {m:26s} MISSING")
                continue
            res = json.load(open(f))["results"]
            arms = {a: per_core(res, a) for a in ("P1", "NONE", "BEFORE", "POST", "LETTER")}
            idk = {a: ci([v["idK"] for v in arms[a].values()]) for a in arms}
            shared = sorted(set(arms["P1"]) & set(arms["NONE"]))
            diff = ci([arms["P1"][c]["idK"] - arms["NONE"][c]["idK"] for c in shared])
            share = ratio([v["d"]["K_S@0"] for v in arms["P1"].values()],
                          [v["d"]["K_S@0"] + v["d"]["V_S@0"] for v in arms["P1"].values()])
            ok = idk["P1"][1] > 0 and diff[1] > 0
            d1[task] += ok
            d2.append(idk["BEFORE"][0] <= 0.5)
            print(f"  {task:8s} {m:26s} P1 {fmt(idk['P1'])}  LETTER {fmt(idk['LETTER'])}  POST {fmt(idk['POST'])}  "
                  f"NONE {fmt(idk['NONE'])}  BEFORE {fmt(idk['BEFORE'])}  P1-NONE {fmt(diff)}  P1 key share {share[0]:.2f}  -> {'ok' if ok else 'no'}")
    for task, k in d1.items():
        print(f"  D1 [{task}]: {k}/5 models -> {'MET' if k >= 4 else 'NOT MET'}")
    print(f"  D2: BEFORE identity(K) <= 0.5 in {sum(d2)}/{len(d2)} cells -> {'MET' if sum(d2) >= 9 and len(d2) == 10 else 'NOT MET'}")

    print("\n== D3 localisation: fraction of the full key effect recovered when only these rows see the swapped key")
    p1_ok, post_ok = 0, 0
    for m in LOC_MODELS:
        fs = glob.glob(f"{root}/row_restricted/{m}/*_direct.json")
        if not fs:
            print(f"  {m}: MISSING")
            continue
        res = json.load(open(fs[0]))
        for arm in ("P1", "POST", "LETTER"):
            R = [r for r in res if r["arm"] == arm]
            full = [r["m"]["all"] - r["m_B"] for r in R]
            line = f"  {m:26s} {arm:6s} n={len(R):3d} full {np.mean(full):+6.2f} nats |"
            fr = {}
            for g in ("choice_words", "remention_words", "question", "story_tail", "rest_after_p"):
                if not R or not R[0]["sizes"].get(g):
                    continue
                fr[g] = ratio([r["m"][g] - r["m_B"] for r in R], full)
                line += f" {g} {fr[g][0]:+.2f} [{fr[g][1]:+.2f},{fr[g][2]:+.2f}]"
            print(line)
            if arm == "P1" and fr:
                p1_ok += fr["choice_words"][0] >= 0.7 and fr["question"][0] <= 0.15 and fr["rest_after_p"][0] <= 0.15
            if arm == "POST" and fr and np.mean(full) > 0:
                post_ok += fr["remention_words"][0] >= 0.5
    print(f"  D3 [P1]: {p1_ok}/3 models -> {'MET' if p1_ok == 3 else 'NOT MET'}")
    print(f"  D3 [POST]: {post_ok}/3 models -> {'MET' if post_ok >= 2 else 'NOT MET'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
