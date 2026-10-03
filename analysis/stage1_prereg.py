"""Score preregistered predictions P-2026-10-03-B (docs/PREREGISTRATION.md) on GPU stage-1 results.

Written and committed before the stage-1 numbers were inspected. All items are primary (n = 150 per arm).

identity(K) = 0.5 * [(dlogp_S(K_S) - dlogp_S(K_X)) + (dlogp_X(K_X) - dlogp_X(K_S))], keys clamped from layer 0,
changes measured against the batched identity row. CIs are 95% core-bootstrap (10,000 resamples). The
P1 - NONE contrast is paired by story core (the same cores appear in every arm).

Prediction 1 (strict reading): a model counts only if identity(K) > 0 with CI excluding 0 in BOTH P1 and LETTER.
Prediction 2: paired identity(K)(P1) - identity(K)(NONE) > 0 with CI excluding 0.
Both are judged over the five preregistered open models and need >= 4 of 5.
Prediction 3 is exploratory: key share dK/(dK+dV) in P1 at l0 = 0 and at l0 = round(0.0625 * n_layers).
"""
import glob
import json
import sys

import numpy as np

PREREG_MODELS = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-8B", "Mistral-7B-Instruct-v0.3",
                 "OLMo-2-1124-7B-Instruct"]
QWEN_SCALE = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"]
RNG = np.random.default_rng(0)
B = 10000


def ci(x):
    x = np.asarray(x, float)
    bs = x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def ratio_ci(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = RNG.integers(0, len(a), (B, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def per_core(results, arm):
    out = {}
    for r in results:
        if r["arm"] != arm or r["view"] != "direct":
            continue
        m, idr = r["m"], r["m"]["ID@0"]
        dl = {k: {t: m[k]["lp"][t] - idr["lp"][t] for t in ("S", "B", "X")} for k in m}
        idk = 0.5 * ((dl["K_S@0"]["S"] - dl["K_X@0"]["S"]) + (dl["K_X@0"]["X"] - dl["K_S@0"]["X"]))
        idv = 0.5 * ((dl["V_S@0"]["S"] - dl["V_X@0"]["S"]) + (dl["V_X@0"]["X"] - dl["V_S@0"]["X"]))
        d = {k: m[k]["m"] - idr["m"] for k in m}
        out[json.dumps(r["core"], sort_keys=True)] = {"idK": idk, "idV": idv, "d": d, "L": r["n_layers"]}
    return out


def fmt(t):
    return f"{t[0]:+6.2f} [{t[1]:+6.2f},{t[2]:+6.2f}]"


def main(root):
    rows, verdict1, verdict2 = {}, {}, {}
    for f in sorted(glob.glob(f"{root}/*_s0.json")):
        name = f.split("/")[-1][: -len("_s0.json")]
        res = json.load(open(f))["results"]
        arms = {a: per_core(res, a) for a in ("P1", "LETTER", "NONE", "BEFORE", "POST")}
        L = next(iter(arms["P1"].values()))["L"]
        l0m = round(0.0625 * L)
        p1, let, none = arms["P1"], arms["LETTER"], arms["NONE"]
        idk = {a: ci([v["idK"] for v in arms[a].values()]) for a in arms}
        shared = sorted(set(p1) & set(none))
        diff = ci([p1[c]["idK"] - none[c]["idK"] for c in shared])
        share0 = ratio_ci([v["d"]["K_S@0"] for v in p1.values()], [v["d"]["K_S@0"] + v["d"]["V_S@0"] for v in p1.values()])
        sharem = ratio_ci([v["d"][f"K_S@{l0m}"] for v in p1.values()],
                          [v["d"][f"K_S@{l0m}"] + v["d"][f"V_S@{l0m}"] for v in p1.values()])
        rows[name] = dict(idk=idk, diff=diff, n_pair=len(shared), share0=share0, sharem=sharem, l0m=l0m, L=L)
        verdict1[name] = idk["P1"][1] > 0 and idk["LETTER"][1] > 0
        verdict2[name] = diff[1] > 0

    print("identity(K) by arm, all items (95% core bootstrap)")
    print(f"{'model':28s} {'P1':>22s} {'LETTER':>22s} {'NONE':>22s} {'BEFORE':>22s} {'POST':>22s}")
    for n, r in rows.items():
        print(f"{n:28s} " + " ".join(f"{fmt(r['idk'][a]):>22s}" for a in ("P1", "LETTER", "NONE", "BEFORE", "POST")))
    print("\nP1 - NONE identity(K), paired by core; P1 key share at l0=0 and depth-matched l0")
    for n, r in rows.items():
        print(f"{n:28s} diff {fmt(r['diff'])} (n={r['n_pair']})   share@0 {fmt(r['share0'])}   "
              f"share@{r['l0m']}/{r['L']} {fmt(r['sharem'])}")
    k1 = sum(verdict1.get(m, False) for m in PREREG_MODELS)
    k2 = sum(verdict2.get(m, False) for m in PREREG_MODELS)
    missing = [m for m in PREREG_MODELS if m not in rows]
    print("\nPREREGISTERED VERDICTS (P-2026-10-03-B), five open models:", ", ".join(PREREG_MODELS))
    if missing:
        print("  missing models:", missing)
    print(f"  Prediction 1 (P1 and LETTER identity(K) > 0, CI excl. 0): {k1}/5 -> {'MET' if k1 >= 4 else 'NOT MET'}"
          f"   per model: {[(m, verdict1.get(m)) for m in PREREG_MODELS]}")
    print(f"  Prediction 2 (P1 - NONE > 0, CI excl. 0):                 {k2}/5 -> {'MET' if k2 >= 4 else 'NOT MET'}"
          f"   per model: {[(m, verdict2.get(m)) for m in PREREG_MODELS]}")
    print("  Prediction 3 (exploratory) Qwen2.5 P1 key share @l0=0 / depth-matched:")
    for m in QWEN_SCALE:
        if m in rows:
            print(f"    {m:24s} {fmt(rows[m]['share0'])}   {fmt(rows[m]['sharem'])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/gpu_stage1")
