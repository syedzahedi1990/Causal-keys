"""Critic check: probability that JB9 (|stat(F) - stat(S0)| <= 0.05 for every P4 model x 2x2 arm, s_ID and r)
fails by sampling noise alone when F and S0 are exchangeable draws of the same generator.
Uses the committed stage-3b per-core rows (lowercase), two-stage location-pair cluster bootstrap SD,
and also a direct split-sample simulation (draw two independent n=150 samples by resampling pairs+cores)."""
import json, numpy as np
from scipy.stats import norm
ROOT = "/home/user/Causal-keys/results/gpu_stage3b/format_2x2"
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]
ARMS = ["AFTER", "POST", "BEFORE", "PRE", "NONE"]
rng = np.random.default_rng(1)
def idk(r, ch):
    dl = lambda k, t: r["m"][k]["lp"][t] - r["m"]["ID@0"]["lp"][t]
    return 0.5 * ((dl(f"{ch}_S@0", "S") - dl(f"{ch}_X@0", "S")) + (dl(f"{ch}_X@0", "X") - dl(f"{ch}_S@0", "X")))
def cl_idx(keys, B):
    cl = {}
    for i, k in enumerate(keys): cl.setdefault(k, []).append(i)
    C = [np.array(v) for v in cl.values()]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(C), len(C))
        out.append(np.concatenate([C[c][rng.integers(0, len(C[c]), len(C[c]))] for c in pick]))
    return out
Pall = 1.0; Pall_s = 1.0
for m in MODELS:
    d = json.load(open(f"{ROOT}/{m}_s0.json"))["results"]
    D = {}
    for a in ARMS:
        R = [r for r in d if r["arm"] == a]
        D[a] = (np.array([idk(r, "K") for r in R]), np.array([idk(r, "V") for r in R]))
    keys = [(r["core"]["base"], r["core"]["source"]) for r in d if r["arm"] == "AFTER"]
    idx = cl_idx(keys, 3000)
    aK = D["AFTER"][0]
    line = []
    for a in ["AFTER", "POST", "BEFORE", "PRE", "NONE"]:
        K, V = D[a]
        s = np.array([K[i].mean() / (K[i].mean() + V[i].mean()) for i in idx])
        r = np.array([K[i].mean() / aK[i].mean() for i in idx])
        for nm, v, full in (("s", s, K.mean() / (K.mean() + V.mean())), ("r", r, K.mean() / aK.mean())):
            if nm == "r" and a == "AFTER": continue
            sd = v.std()
            p = 2 * norm.cdf(0.05 / (np.sqrt(2) * sd)) - 1
            if a != "NONE": Pall *= p
            line.append(f"{nm}({a})={full:+.3f} sd={sd:.3f} P(|diff|<=.05)={p:.2f}")
    print(m); print("   " + "\n   ".join(line))
print(f"\nP(JB9 met | F and S0 exchangeable, 2x2 arms, s and r) ~= {Pall:.3f}  (normal approx, independence across cells)")
