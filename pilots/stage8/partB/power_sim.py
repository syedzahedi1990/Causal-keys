"""Part B pilot 3 (CPU, committed stage-3b per-core rows only): Monte-Carlo power of the proposed JB criteria at n = 150
with the two-stage location-pair cluster bootstrap at the Bonferroni level 1 - 0.05/8 (99.375 %).
Templates: the per-core (ID_K, ID_V) vectors of the four stage-3b models; a synthetic model is drawn by resampling
template cores (with their location-pair labels) and rescaling one arm so that its true ratio equals the target;
noise inflation 'x2' doubles each core's deviation from the template mean (a noisier new family)."""
import json
import numpy as np

ROOT = "/home/user/Causal-keys/results/gpu_stage3b/format_2x2"
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]
ARMS = ["AFTER", "POST", "BEFORE", "PRE", "NONE"]
rng = np.random.default_rng(7)
LEVEL = 1 - 0.05 / 8
NB, NREP, N = 600, 100, 150


def idk(r, ch):
    dl = lambda k, t: r["m"][k]["lp"][t] - r["m"]["ID@0"]["lp"][t]
    return 0.5 * ((dl(f"{ch}_S@0", "S") - dl(f"{ch}_X@0", "S")) + (dl(f"{ch}_X@0", "X") - dl(f"{ch}_S@0", "X")))


def load(m):
    d = json.load(open(f"{ROOT}/{m}_s0.json"))["results"]
    out = {}
    for arm in ARMS:
        R = [r for r in d if r["arm"] == arm]
        out[arm] = (np.array([idk(r, "K") for r in R]), np.array([idk(r, "V") for r in R]))
    keys = [(r["core"]["base"], r["core"]["source"]) for r in d if r["arm"] == "AFTER"]
    return out, keys


def cluster_idx(keys, nb):
    cl = {}
    for i, k in enumerate(keys):
        cl.setdefault(k, []).append(i)
    C = [np.array(v) for v in cl.values()]
    res = []
    for _ in range(nb):
        pick = rng.integers(0, len(C), len(C))
        res.append(np.concatenate([C[c][rng.integers(0, len(C[c]), len(C[c]))] for c in pick]))
    return res


def ci(vals):
    a = (1 - LEVEL) / 2
    return np.quantile(vals, a), np.quantile(vals, 1 - a)


def synth(T, keys, arm, target, kind, infl):
    """Resample N template cores; rescale arm `arm` so that the population statistic equals target."""
    ii = rng.integers(0, len(keys), N)
    k = [keys[i] for i in ii]
    D = {a: [T[a][0][ii].copy(), T[a][1][ii].copy()] for a in ARMS}
    if infl != 1:
        for a in ARMS:
            for c in (0, 1):
                mu = T[a][c].mean(); D[a][c] = mu + infl * (D[a][c] - mu)
    K, V = T[arm][0], T[arm][1]
    if kind == "r":       # scale K(arm) so that mean K(arm)/mean K(AFTER) = target (template means)
        s = target * T["AFTER"][0].mean() / K.mean() if abs(K.mean()) > 1e-6 else 0
        D[arm][0] = D[arm][0] * s if abs(K.mean()) > 1e-6 else D[arm][0] - D[arm][0].mean() + target * T["AFTER"][0].mean()
    elif kind == "s":     # shift K(arm) so that mean K / (mean K + mean V) = target, V fixed
        want = target * V.mean() / (1 - target)
        D[arm][0] = D[arm][0] - K.mean() + want
    return D, k


def rep(T, keys, arm, target, kind, infl, crit):
    hits = 0
    for _ in range(NREP):
        D, k = synth(T, keys, arm, target, kind, infl)
        idx = cluster_idx(k, NB)
        if kind == "r":
            f = lambda i: D[arm][0][i].mean() / D["AFTER"][0][i].mean()
            g = lambda i: (D["POST"][0][i].mean() - D["PRE"][0][i].mean()) / D["AFTER"][0][i].mean()
        else:
            f = lambda i: D[arm][0][i].mean() / (D[arm][0][i].mean() + D[arm][1][i].mean())
            g = None
        full = np.arange(N)
        est = f(full)
        lo, hi = ci(np.array([f(i) for i in idx]))
        extra = None
        if g is not None:
            extra = ci(np.array([g(i) for i in idx]))
        hits += crit(est, lo, hi, extra)
    return hits / NREP


def main():
    T, keys = {}, {}
    for m in MODELS:
        T[m], keys[m] = load(m)
    jb3 = lambda e, lo, hi, x: e >= 0.15 and lo >= 0.10 and e <= 0.75 and hi < 1.0 and x[0] > 0
    jb2 = lambda e, lo, hi, x: hi <= 0.10
    jb4 = lambda e, lo, hi, x: e <= 0.10 and hi <= 0.15
    jb1 = lambda e, lo, hi, x: e >= 0.50 and lo >= 0.40
    tm = ["Qwen2.5-7B-Instruct", "OLMo-2-1124-7B-Instruct"]   # tightest and noisiest POST templates
    print(f"level {LEVEL:.4f}, n={N}, {NREP} replicates x {NB} cluster-bootstrap resamples")
    for infl in (1, 2):
        print(f"\n### noise inflation x{infl}")
        for m in tm:
            print(f"  template {m}")
            print("   JB3 r(POST) true:", "  ".join(f"{t:.2f}->{rep(T[m], keys[m], 'POST', t, 'r', infl, jb3):.2f}" for t in (0.12, 0.15, 0.18, 0.22)))
            print("   JB2 r(BEFORE) true:", "  ".join(f"{t:.2f}->{rep(T[m], keys[m], 'BEFORE', t, 'r', infl, jb2):.2f}" for t in (0.0, 0.05, 0.08)))
            print("   JB4 s_ID(NONE) true:", "  ".join(f"{t:.2f}->{rep(T[m], keys[m], 'NONE', t, 's', infl, jb4):.2f}" for t in (0.06, 0.09, 0.11)))
            print("   JB1 s_ID(AFTER) true:", "  ".join(f"{t:.2f}->{rep(T[m], keys[m], 'AFTER', t, 's', infl, jb1):.2f}" for t in (0.50, 0.55, 0.60)))


if __name__ == "__main__":
    main()
