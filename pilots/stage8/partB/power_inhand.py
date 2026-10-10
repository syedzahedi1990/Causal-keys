"""Part B pilot 2 (CPU, committed stage-3b rows only): scale-free statistics of the 2x2 on the seed-0 cores,
core bootstrap vs location-pair cluster bootstrap, and a Monte-Carlo power check of the proposed J-B thresholds.
Data: results/gpu_stage3b/format_2x2/{model}_s0.json (lowercase scoring, n = 150 per arm)."""
import json, sys
import numpy as np

ROOT = "/home/user/Causal-keys/results/gpu_stage3b/format_2x2"
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]
ARMS = ["AFTER", "POST", "BEFORE", "PRE", "NONE"]
B = 4000
rng = np.random.default_rng(20261010)


def idk(r, ch):
    dl = lambda k, t: r["m"][k]["lp"][t] - r["m"]["ID@0"]["lp"][t]
    return 0.5 * ((dl(f"{ch}_S@0", "S") - dl(f"{ch}_X@0", "S")) + (dl(f"{ch}_X@0", "X") - dl(f"{ch}_S@0", "X")))


def load(m):
    d = json.load(open(f"{ROOT}/{m}_s0.json"))["results"]
    out = {}
    for arm in ARMS:
        R = [r for r in d if r["arm"] == arm]
        out[arm] = dict(K=np.array([idk(r, "K") for r in R]), V=np.array([idk(r, "V") for r in R]),
                        key=[(r["core"]["base"], r["core"]["source"]) for r in R],
                        core=[json.dumps(r["core"], sort_keys=True) for r in R])
    # same cores across arms, same order?
    assert all(out[a]["core"] == out["AFTER"]["core"] for a in ARMS)
    return out


def idx_core(n):
    return rng.integers(0, n, (B, n))


def idx_cluster(keys):
    """two-stage: resample clusters (location pairs) with replacement, then cores within each drawn cluster."""
    cl = {}
    for i, k in enumerate(keys):
        cl.setdefault(k, []).append(i)
    C = list(cl.values())
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(C), len(C))
        ii = []
        for c in pick:
            mem = C[c]
            ii.extend(np.array(mem)[rng.integers(0, len(mem), len(mem))])
        out.append(np.array(ii))
    return out


def stat_ci(fn, idx):
    vals = np.array([fn(i) for i in idx])
    return np.percentile(vals, [2.5, 97.5]), vals.std()


def main():
    allrows = {}
    for m in MODELS:
        D = load(m)
        n = len(D["AFTER"]["K"])
        keys = D["AFTER"]["key"]
        ic, il = idx_core(n), idx_cluster(keys)
        print(f"== {m}  n={n}  clusters={len(set(keys))}")
        aK = D["AFTER"]["K"]
        for arm in ARMS:
            K, V = D[arm]["K"], D[arm]["V"]
            r = lambda i, K=K: K[i].mean() / aK[i].mean()
            s = lambda i, K=K, V=V: K[i].mean() / (K[i].mean() + V[i].mean())
            full = np.arange(n)
            (rc, rsd), (rl, rsdl) = stat_ci(r, ic), stat_ci(r, il)
            (sc, ssd), (sl, ssdl) = stat_ci(s, ic), stat_ci(s, il)
            print(f"  {arm:6s} ID_K {K.mean():+6.2f} ID_V {V.mean():+6.2f} | r=ID_K/ID_K(AFTER) {r(full):+.3f} core[{rc[0]:+.3f},{rc[1]:+.3f}] "
                  f"pair[{rl[0]:+.3f},{rl[1]:+.3f}] (sd {rsd:.3f}/{rsdl:.3f}) | s_ID {s(full):+.3f} core[{sc[0]:+.3f},{sc[1]:+.3f}] pair[{sl[0]:+.3f},{sl[1]:+.3f}]"
                  f" | per-core CV of ID_K {K.std() / max(abs(K.mean()), 1e-9):.2f}")
        # paired contrasts
        for a, b in (("POST", "PRE"), ("AFTER", "BEFORE"), ("POST", "NONE")):
            dlt = D[a]["K"] - D[b]["K"]
            rr = lambda i: dlt[i].mean() / aK[i].mean()
            (c1, _), (c2, _) = stat_ci(rr, ic), stat_ci(rr, il)
            print(f"  ({a}-{b})/AFTER {rr(np.arange(n)):+.3f} core[{c1[0]:+.3f},{c1[1]:+.3f}] pair[{c2[0]:+.3f},{c2[1]:+.3f}]")
        allrows[m] = D
    np.save("/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partB/inhand_cv.npy",
            {m: {a: (allrows[m][a]["K"], allrows[m][a]["V"]) for a in ARMS} for m in MODELS}, allow_pickle=True)


if __name__ == "__main__":
    main()
