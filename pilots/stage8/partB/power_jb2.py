"""JB2 power with an additive shift of the per-core ID_K(BEFORE / PRE) (the multiplicative rescale of power_sim.py
explodes the spread when the template mean is near zero)."""
import numpy as np, power_sim as P
P.NREP, P.NB = 60, 400
def synth_shift(T, keys, arm, target, infl):
    ii = P.rng.integers(0, len(keys), P.N); k = [keys[i] for i in ii]
    D = {a: [T[a][0][ii].copy(), T[a][1][ii].copy()] for a in P.ARMS}
    if infl != 1:
        for a in P.ARMS:
            for c in (0, 1):
                mu = T[a][c].mean(); D[a][c] = mu + infl * (D[a][c] - mu)
    D[arm][0] = D[arm][0] - T[arm][0].mean() + target * T["AFTER"][0].mean()
    return D, k
def rep(T, keys, arm, target, infl):
    hits = 0
    for _ in range(P.NREP):
        D, k = synth_shift(T, keys, arm, target, infl)
        idx = P.cluster_idx(k, P.NB)
        f = lambda i: D[arm][0][i].mean() / D["AFTER"][0][i].mean()
        s = lambda i: D[arm][0][i].mean() / (D[arm][0][i].mean() + D[arm][1][i].mean())
        _, hi = P.ci(np.array([f(i) for i in idx])); _, hs = P.ci(np.array([s(i) for i in idx]))
        hits += hi <= 0.10 and hs <= 0.10
    return hits / P.NREP
for m in ["Qwen2.5-7B-Instruct", "OLMo-2-1124-7B-Instruct"]:
    T, keys = P.load(m)
    for infl in (1, 2):
        print(m, f"x{infl}", "JB2 BEFORE:", "  ".join(f"{t:.2f}->{rep(T, keys, 'BEFORE', t, infl):.2f}" for t in (0.0, 0.05, 0.08)),
              "| PRE:", "  ".join(f"{t:.2f}->{rep(T, keys, 'PRE', t, infl):.2f}" for t in (0.0, 0.05)), flush=True)
