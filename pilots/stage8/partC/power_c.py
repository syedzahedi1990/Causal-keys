"""Power of the Part C law criteria (C3: MAD <= 0.12 with bootstrap upper bound <= 0.17, max gap <= 0.25) using the
committed stage-1 per-story natural rows (Qwen2.5-7B, Mistral-7B, OLMo-2-7B; four formats at l0 = 0) as the noise model.

Each replicate draws n stories for sigma and an INDEPENDENT n-story sample for the edit (worst case: no within-story
correlation between edit and natural rows), scales the edit's per-story ID_K / ID_V by an efficacy factor, then moves
the edit's key share by a scenario-specific shift (applied as a reallocation between ID_K and ID_V that keeps ID_K+ID_V).
"""
import json

import numpy as np

ROOT = "/home/user/Causal-keys/results/gpu_stage1/"
MODELS = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]
ARMS = ["LETTER", "P1", "POST", "NONE"]
rng = np.random.default_rng(1)


def rows(m, arm):
    d = json.load(open(f"{ROOT}{m}_s0.json"))["results"]
    R = [r for r in d if r["arm"] == arm and r["view"] == "direct"]

    def dl(r, k, t):
        return r["m"][k]["lp"][t] - r["m"]["ID@0"]["lp"][t]
    K = np.array([0.5 * ((dl(r, "K_S@0", "S") - dl(r, "K_X@0", "S")) + (dl(r, "K_X@0", "X") - dl(r, "K_S@0", "X"))) for r in R])
    V = np.array([0.5 * ((dl(r, "V_S@0", "S") - dl(r, "V_X@0", "S")) + (dl(r, "V_X@0", "X") - dl(r, "V_S@0", "X"))) for r in R])
    return K, V


DATA = {(m, a): rows(m, a) for m in MODELS for a in ARMS}


def share(K, V):
    return K.mean() / (K.mean() + V.mean())


def scenario_shift(name, sig, arm):
    if name == "law":
        return 0.0
    if name == "copy_only":          # R1: kappa ~ 0.05 everywhere
        return 0.05 - sig
    if name == "key_flat":           # R3: kappa >= 0.65 everywhere (BIND-like)
        return max(0.0, 0.65 - sig)
    if name == "partial_0.15":       # systematic 0.15 under-read in the two intermediate formats
        return -0.15 if arm in ("P1", "POST") else 0.0
    if name == "partial_0.25":
        return -0.25 if arm in ("P1", "POST") else 0.0
    raise ValueError(name)


def one(n, name, eff=0.9, nboot=200):
    gaps, cells = [], []
    for m in MODELS:
        for a in ARMS:
            K, V = DATA[(m, a)]
            i = rng.integers(0, len(K), n)
            j = rng.integers(0, len(K), n)
            sig = share(K[i], V[i])
            Ke, Ve = eff * K[j], eff * V[j]
            tot = Ke + Ve
            s0 = share(Ke, Ve)
            target = min(1.0, max(0.0, s0 + scenario_shift(name, share(K, V), a)))
            # reallocate per story keeping K+V: K' = K + (target - s0) * tot
            Ke2 = Ke + (target - s0) * tot
            Ve2 = tot - Ke2
            cells.append((K[i], V[i], Ke2, Ve2))
            gaps.append(abs(share(Ke2, Ve2) - sig))
    mad, mx = float(np.mean(gaps)), float(np.max(gaps))
    # bootstrap upper bound of MAD (resample stories within cell, both sides)
    bs = []
    for _ in range(nboot):
        g = []
        for (K, V, Ke, Ve) in cells:
            i = rng.integers(0, n, n)
            j = rng.integers(0, n, n)
            g.append(abs(share(Ke[j], Ve[j]) - share(K[i], V[i])))
        bs.append(np.mean(g))
    ub = float(np.percentile(bs, 97.5))
    return mad, mx, ub


for name in ("law", "partial_0.15", "partial_0.25", "copy_only", "key_flat"):
    for n in (60, 120):
        res = [one(n, name) for _ in range(150)]
        met = np.mean([(mad <= 0.12) and (ub <= 0.17) and (mx <= 0.25) for mad, mx, ub in res])
        print(f"{name:13s} n={n:3d}  P(C3 met per model-family set of 12 cells) = {met:.2f}   "
              f"mean MAD {np.mean([r[0] for r in res]):.3f}  mean max {np.mean([r[1] for r in res]):.3f}  mean UB {np.mean([r[2] for r in res]):.3f}")
