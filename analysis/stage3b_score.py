"""Score preregistered predictions P-2026-10-04-E (docs/PREREGISTRATION.md) on stage-3b results.

E1 instruction-matched 2x2 (one-word instruction in every arm; Qwen2.5-7B/14B, Mistral-7B, OLMo-2-7B):
   a) ID_K(AFTER) > 0, CI excluding 0, in 4/4 models
   b) ID_K(BEFORE) <= 0.5 in >= 3/4 and ID_K(PRE) <= 0.5 in >= 3/4
   c) paired ID_K(AFTER) - ID_K(BEFORE) > 0 (CI excl. 0) in 4/4; paired ID_K(POST) - ID_K(PRE) > 0 (CI excl. 0) in >= 3/4
E2 role control (direct view; P1 and NONE): |f_K| <= 0.15 and |f_V| <= 0.15 in >= 3/4 models for both arms, where
   f_C is the fraction of the role-swap effect carried by the critical token's channel C
E3 environment: |s_K(Qwen2.5-14B, transformers 5.9.0, 2 GPUs) - 0.91| <= 0.03 and |s_K(Qwen2.5-32B, 5.18.0, 1 GPU) - 0.86| <= 0.03
E4 value-only exchange in Paper 1 frames (Mistral-24B, Qwen-72B): psi_V(NONE) >= 0.5 and psi_V(NONE) > psi_K(NONE)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_prereg import ci, fmt, per_core  # noqa: E402
import stage2_score as s2  # noqa: E402

M4 = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3", "OLMo-2-1124-7B-Instruct"]
RNG = np.random.default_rng(2)


def ratio(a, b, B=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = RNG.integers(0, len(a), (B, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def paired(A, B):
    ids = sorted(set(A) & set(B))
    return ci([A[i]["idK"] - B[i]["idK"] for i in ids])


def main(root="results/gpu_stage3b"):
    print("== E1 instruction-matched 2x2 (identity carried by keys, nats)")
    a_ok = b1 = b2 = c1 = c2 = 0
    for m in M4:
        f = Path(root) / "format_2x2" / f"{m}_s0.json"
        if not f.exists():
            print(f"  {m}: MISSING"); continue
        res = json.load(open(f))["results"]
        arms = {a: per_core(res, a) for a in ("AFTER", "BEFORE", "POST", "PRE", "NONE")}
        idk = {a: ci([v["idK"] for v in arms[a].values()]) for a in arms}
        dAB, dPP = paired(arms["AFTER"], arms["BEFORE"]), paired(arms["POST"], arms["PRE"])
        a_ok += idk["AFTER"][1] > 0; b1 += idk["BEFORE"][0] <= 0.5; b2 += idk["PRE"][0] <= 0.5
        c1 += dAB[1] > 0; c2 += dPP[1] > 0
        print(f"  {m:26s} " + "  ".join(f"{a} {fmt(idk[a])}" for a in arms) + f"  AFTER-BEFORE {fmt(dAB)}  POST-PRE {fmt(dPP)}")
    print(f"  E1a {a_ok}/4 -> {'MET' if a_ok == 4 else 'NOT MET'};  E1b BEFORE {b1}/4, PRE {b2}/4 -> "
          f"{'MET' if b1 >= 3 and b2 >= 3 else 'NOT MET'};  E1c AFTER-BEFORE {c1}/4, POST-PRE {c2}/4 -> "
          f"{'MET' if c1 == 4 and c2 >= 3 else 'NOT MET'}")

    print("\n== E2 role control: fraction of the role-swap effect carried by the critical token's key / value")
    ok = {"P1": 0, "NONE": 0}
    for m in M4 + ["Qwen2.5-72B-Instruct"]:
        f = Path(root) / "role_factorial" / f"{m}_s0.json"
        if not f.exists():
            print(f"  {m}: MISSING"); continue
        res = json.load(open(f))["results"]
        for arm in ("P1", "NONE", "LETTER"):
            R = [r for r in res if r["arm"] == arm and r["view"] == "direct"]
            full = [r["clean"]["R"]["m"] - r["m"]["ID"] for r in R]
            fk = ratio([r["m"]["K_R"] - r["m"]["ID"] for r in R], full)
            fv = ratio([r["m"]["V_R"] - r["m"]["ID"] for r in R], full)
            fkv = ratio([r["m"]["KV_R"] - r["m"]["ID"] for r in R], full)
            print(f"  {m:26s} {arm:6s} n={len(R):3d} role effect {np.mean(full):+6.2f}  f_K {fmt(fk)}  f_V {fmt(fv)}  f_KV {fmt(fkv)}")
            if arm in ok and m in M4:
                ok[arm] += abs(fk[0]) <= 0.15 and abs(fv[0]) <= 0.15
    print(f"  E2: P1 {ok['P1']}/4, NONE {ok['NONE']}/4 -> {'MET' if ok['P1'] >= 3 and ok['NONE'] >= 3 else 'NOT MET'}")

    print("\n== E3 environment deconfound (P1 key share)")
    e3 = []
    for sub, m, ref in (("tf59_2gpu", "Qwen2.5-14B-Instruct", 0.91), ("tf518_1gpu", "Qwen2.5-32B-Instruct", 0.86)):
        f = Path(root) / "env_check" / sub / f"{m}_s0.json"
        if not f.exists():
            print(f"  {sub} {m}: MISSING"); e3.append(False); continue
        p1 = per_core(json.load(open(f))["results"], "P1")
        sh = ratio([v["d"]["K_S@0"] for v in p1.values()], [v["d"]["K_S@0"] + v["d"]["V_S@0"] for v in p1.values()])
        e3.append(abs(sh[0] - ref) <= 0.03)
        print(f"  {sub:11s} {m:22s} s_K {fmt(sh)} vs original {ref:.2f} -> {'within 0.03' if e3[-1] else 'DIFFERS'}")
    print(f"  E3 -> {'MET' if all(e3) and len(e3) == 2 else 'NOT MET'}")

    print("\n== E4 value-only vs key-only exchange in Paper 1 frames")
    e4 = []
    for model in ("mistral", "qwen"):
        f = Path(root) / "paper1_frames_v" / f"{model}.json"
        if not f.exists():
            print(f"  {model}: MISSING"); e4.append(False); continue
        res = json.load(open(f))["results"]
        for a in ("P1", "LETTER", "POST", "NONE", "BEFORE"):
            R = s2.per_core(res, a)
            phi = s2.ratio(R, lambda x: x["M"] - x["P"], lambda x: x["T"] - x["S"])
            psk = s2.ratio(R, lambda x: x["add"] - x["P"], lambda x: x["M"] - x["P"])
            psv = s2.ratio(R, lambda x: x["addv"] - x["P"], lambda x: x["M"] - x["P"])
            rhv = s2.ratio(R, lambda x: x["remv"] - x["M"], lambda x: x["P"] - x["M"])
            print(f"  {model:8s} {a:6s} phi {s2.fmt(phi)}  psi_K {s2.fmt(psk)}  psi_V {s2.fmt(psv)}  rho_V {s2.fmt(rhv)}")
            if a == "NONE":
                e4.append(psv[0] >= 0.5 and psv[0] > psk[0])
    print(f"  E4 -> {'MET' if all(e4) and len(e4) == 2 else 'NOT MET'} ({e4})")


if __name__ == "__main__":
    main(*sys.argv[1:])
