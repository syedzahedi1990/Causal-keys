"""Part (b) of P-2026-10-05-G (docs/PREREGISTRATION.md): Gate b and predictions G5-G8 on experiments/attention_knockout.py
outputs ({root}/{model}_s0.json), exactly as drafted. Called by analysis/stage5_score.py (``score(root)``) or standalone.

Per model m, arm f, mask M (means over cores; 95 % percentile CIs from 10,000 core-bootstrap resamples with one fixed
index set per n, so every ratio of means and paired difference is recomputed within the same resamples):
  r_K(M) = mean ID_K^M / mean ID_K^M0;  q_V^M = mean ID_V^M / mean d_KV^M;  rho_V(f, M) = mean ID_V^M(f) / mean ID_V^M0(NONE)
  acc = restricted argmax on target; on = full-vocabulary top-1 on target; loc = location mass (self-clamp row)
Gate b (every arm and model): under M8 |mean ID_K|, |mean ID_V| <= 0.1 nats, acc_B <= 0.5, on_B <= 0.5.
G5  list: r_K(M1) <= 0.20, CI upper <= 0.25, AFTER and P1, 3/3;  sentence: r_K(M1) <= 0.40 under POST in >= 2/3.
G6  AFTER: r_K(M2) in [0.80, 1.20]; |mean ID_V^M2 - mean ID_V^M0| <= 0.15 x mean ID_V^M0(NONE); paired ID_K^M1 - ID_K^M2 < 0
    with CI excluding 0; each in 3/3.
G7  AFTER and P1 with M1: (a) q_V^M1 >= 0.5 x q_V^M0(NONE) and paired ID_V^M1 - ID_V^M0 > 0 (CI excl. 0) in >= 2/3 (both
    formats); (b) acc_B, acc_S, on_B, on_S >= 0.90 and loc ratio >= 0.50 with lower bound >= 0.40 in 3/3 (both formats),
    span^M1 >= 0.5 x span^M0 in >= 2/3. H_replaced: paired ID_V diff upper <= 1.0 and q_V^M1 <= 0.5 x q_V(NONE) in >= 2/3,
    with on_B^M1 <= 0.60 or loc ratio <= 0.50 in >= 2/3 (a model counts when it shows the pattern in both formats).
G8  (a) NONE with M3: mean ID_V^M3 <= 0.60 x mean ID_V^M0, paired diff < 0 (CI excl. 0), >= 2/3; (b) AFTER with M3:
    r_K(M3) >= 0.80, acc_B and on_B >= 0.90, 3/3; (c) AFTER and P1, M4 vs M1: mean ID_V^M4 <= 0.60 x mean ID_V^M1,
    paired diff < 0 (CI excl. 0), in >= 2/3 per format, a model counting only if it meets G7a (both formats, as in G7).
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

np.seterr(all="ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1_prereg import per_core as per_core_3b  # noqa: E402

from experiments.attention_knockout import MASKS, NONE_MASKS, masks_of, per_core  # noqa: E402

MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3"]
Q_V_NONE_3B = {"Qwen2.5-7B-Instruct": 0.391, "Qwen2.5-14B-Instruct": 0.335, "Mistral-7B-Instruct-v0.3": 0.433}
SEED, B = 20261005, 10000
_idx = {}


def IDX(n):
    if n not in _idx:
        _idx[n] = np.random.default_rng(SEED).integers(0, n, (B, n))
    return _idx[n]


NAN = (float("nan"),) * 3


def boot(x):
    x = np.asarray(x, float)
    if not len(x):
        return NAN
    bs = x[IDX(len(x))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a):
        return NAN
    r = a[IDX(len(a))].mean(1) / b[IDX(len(a))].mean(1)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


def verdict(ok):
    return "MET" if ok else "NOT MET"


class Cell:
    """One model's results: ``self[f, M]`` = per-core dict; ``v(f, M, key)`` the per-core vector (shared core order)."""

    def __init__(self, res):
        self.res, self.P = res, {}
        for f in ("AFTER", "P1", "POST", "NONE"):
            for M in masks_of(f):
                P = per_core(res, f, M)
                if P:
                    self.P[f, M] = P

    def has(self, f, M):
        return (f, M) in self.P

    def v(self, f, M, key, ids=None):
        P = self.P.get((f, M), {})
        return np.array([P[i][key] for i in (sorted(P) if ids is None else ids)], float)

    def ids(self, *cells):
        return sorted(set.intersection(*(set(self.P.get(c, {})) for c in cells)))

    def paired(self, f, M1, M2, key="idK"):
        ids = self.ids((f, M1), (f, M2))
        return boot(self.v(f, M1, key, ids) - self.v(f, M2, key, ids))

    def rat(self, f, M, key, M0="M0"):
        ids = self.ids((f, M), (f, M0))
        return ratio(self.v(f, M, key, ids), self.v(f, M0, key, ids))

    def mean(self, f, M, key):
        return self.v(f, M, key).mean() if self.has(f, M) else float("nan")

    def q_V(self, f, M):
        return ratio(self.v(f, M, "idV"), self.v(f, M, "dKV"))


def score(root, models=MODELS, stage3b="results/gpu_stage3b/format_2x2", out=print):
    cells, prov = {}, {}
    for f in sorted(glob.glob(f"{root}/*_s0.json")):
        name = f.split("/")[-1][: -len("_s0.json")]
        d = json.load(open(f))
        cells[name], prov[name] = Cell(d["results"]), d["provenance"]
    extra = [m for m in cells if m not in models]
    out(f"== Part (b) attention knockout (P-2026-10-05-G, Gate b and G5-G8): root {root}")
    out(f"   primary models: {models}" + (f"; exploratory: {extra}" if extra else ""))
    for m in models:
        if m not in cells:
            out(f"   {m}: MISSING (counts as not met everywhere)")
    for m, p in prov.items():
        out(f"   {m}: n cores {len({json.dumps(r['core'], sort_keys=True) for r in cells[m].res})}, skipped {p.get('skipped_items')}, "
            f"attn {p.get('attn_implementation')}, dtype {p.get('dtype')}, transformers {p.get('transformers')}, torch {p.get('torch')}, "
            f"commit {str(p.get('git_commit'))[:10]}, C_init counts {p.get('c_init_counts')}, len core 0 {p.get('len_core0')}")

    out("\n-- per arm x mask (all items): ID_K, r_K, ID_V, span = mean d_KV, q_V, acc_B/acc_S, on_B/on_S, loc mass, noise floor (within-batch duplicate row; under M0 also / batch vs clean unmasked run; under M1-M8 that difference is the mask's own effect, printed as 'mask effect')")
    for m, c in cells.items():
        for f in ("AFTER", "P1", "POST", "NONE"):
            for M in masks_of(f):
                if not c.has(f, M):
                    out(f"   {m:26s} {f:5s} {M:5s} MISSING"); continue
                out(f"   {m:26s} {f:5s} {M:5s} n={len(c.P[f, M]):3d} ID_K {fmt(boot(c.v(f, M, 'idK')))} r_K {fmt(c.rat(f, M, 'idK'))}  "
                    f"ID_V {fmt(boot(c.v(f, M, 'idV')))} span {c.mean(f, M, 'dKV'):+7.2f} q_V {fmt(c.q_V(f, M))}  "
                    f"acc {c.mean(f, M, 'accB'):.2f}/{c.mean(f, M, 'accS'):.2f} on {c.mean(f, M, 'onB'):.2f}/{c.mean(f, M, 'onS'):.2f} "
                    f"loc {c.mean(f, M, 'loc'):.3f} floor {c.mean(f, M, 'floor'):.3f}" + (f"/{c.mean(f, M, 'floor_clean'):.3f}" if M == "M0" else f" mask effect {c.mean(f, M, 'floor_clean'):.3f}"))

    out("\n== Gate b (sanity, M8 in every arm): |mean ID_K| <= 0.1, |mean ID_V| <= 0.1 nats, acc_B <= 0.5, on_B <= 0.5")
    gate = {}  # True / False / None (file MISSING: not measured, counts as not met in the k/k lines but is not an M8 failure)
    for m in models:
        if m not in cells:
            gate[m] = None; out(f"   {m:26s} MISSING"); continue
        c, ok = cells[m], True
        for f in ("AFTER", "P1", "POST", "NONE"):
            if not c.has(f, "M8"):
                out(f"   {m:26s} {f:5s} M8 MISSING"); ok = False; continue
            k, v, a, o = c.mean(f, "M8", "idK"), c.mean(f, "M8", "idV"), c.mean(f, "M8", "accB"), c.mean(f, "M8", "onB")
            cell_ok = bool(abs(k) <= 0.1 and abs(v) <= 0.1 and a <= 0.5 and o <= 0.5)
            ok &= cell_ok
            out(f"   {m:26s} {f:5s} ID_K {k:+.3f} ID_V {v:+.3f} acc_B {a:.2f} on_B {o:.2f} floor {c.mean(f, 'M8', 'floor'):.3f} -> {verdict(cell_ok)}"
                + ("  (kernel-noise failure: within-batch floor > 0.05)" if not cell_ok and c.mean(f, "M8", "floor") > 0.05 else ""))
        gate[m] = ok
    n_gate, N = sum(v is True for v in gate.values()), len(models)
    g_fail, g_miss = [m for m in models if gate[m] is False], [m for m in models if gate[m] is None]
    TWO = max(1, round(2 * N / 3))
    out(f"   Gate b -> {verdict(False if g_fail else None if g_miss else True)} ({n_gate}/{N} models" + (f"; MISSING {g_miss}" if g_miss else "")
        + "; a failing model's part is uninterpretable until fixed and re-run)")

    out("\n== G5 necessity of the candidate-word edges: r_K(M1)")
    g5l, g5s = {}, {}
    for m in models:
        if m not in cells:
            g5l[m] = g5s[m] = False; continue
        c = cells[m]
        rA, rP, rS = c.rat("AFTER", "M1", "idK"), c.rat("P1", "M1", "idK"), c.rat("POST", "M1", "idK")
        g5l[m] = rA[0] <= 0.20 and rA[2] <= 0.25 and rP[0] <= 0.20 and rP[2] <= 0.25
        g5s[m] = rS[0] <= 0.40
        out(f"   {m:26s} AFTER {fmt(rA)}  P1 {fmt(rP)}  (<= 0.20, upper <= 0.25) -> {verdict(g5l[m])};  POST {fmt(rS)} (<= 0.40) -> {verdict(g5s[m])}")
    kl, ks = sum(g5l.values()), sum(g5s.values())
    G5 = kl == N and ks >= TWO
    out(f"   G5 list (AFTER and P1) {kl}/{N} -> {verdict(kl == N)};  G5 sentence (POST) {ks}/{N} -> {verdict(ks >= TWO)};  G5 -> {verdict(G5)}")

    out("\n== G6 matched control column (AFTER, M2 = R_cand x C_init)")
    g6 = {}
    for m in models:
        if m not in cells:
            g6[m] = False; continue
        c = cells[m]
        r2 = c.rat("AFTER", "M2", "idK")
        dv = abs(c.mean("AFTER", "M2", "idV") - c.mean("AFTER", "M0", "idV"))
        lim = 0.15 * c.mean("NONE", "M0", "idV")
        d12 = c.paired("AFTER", "M1", "M2")
        g6[m] = 0.80 <= r2[0] <= 1.20 and dv <= lim and d12[2] < 0
        out(f"   {m:26s} r_K(M2) {fmt(r2)} in [0.80, 1.20]; |dID_V| {dv:.3f} <= {lim:.3f}; paired ID_K M1-M2 {fmt(d12)} < 0 -> {verdict(g6[m])}")
        for k in (1, 2):
            ids = [i for i in c.ids(("AFTER", "M2"), ("AFTER", "M0")) if c.P["AFTER", "M2"][i]["n_init"] == k]
            if len(ids) >= 2:
                out(f"      {len(ids):3d} cores with {k} init column(s): r_K(M2) {fmt(ratio(c.v('AFTER', 'M2', 'idK', ids), c.v('AFTER', 'M0', 'idK', ids)))}")
    k6 = sum(g6.values())
    G6 = k6 == N
    out(f"   G6 -> {verdict(G6)} ({k6}/{N})")

    out("\n== G7 the copy takes over and the answer stays (AFTER and P1 with M1); H_redundant vs H_replaced")
    g7a, g7b, g7span, hrep, hbeh = {}, {}, {}, {}, {}
    for m in models:
        if m not in cells:
            g7a[m] = g7b[m] = g7span[m] = hrep[m] = hbeh[m] = False; continue
        c = cells[m]
        qN = c.q_V("NONE", "M0")
        out(f"   {m:26s} q_V^M0(NONE) {fmt(qN)} (stage-3b reference {Q_V_NONE_3B.get(m, float('nan')):.3f}); threshold 0.5 x = {0.5 * qN[0]:.3f}")
        A, Bk, sp, rep, beh = True, True, True, True, True
        for f in ("AFTER", "P1"):
            q1 = c.q_V(f, "M1")
            dV = c.paired(f, "M1", "M0", "idV")
            nid = c.ids((f, "M1"), ("NONE", "M0"))
            rho = ratio(c.v(f, "M1", "idV", nid), c.v("NONE", "M0", "idV", nid))
            a_ok = q1[0] >= 0.5 * qN[0] and dV[1] > 0
            acc, on = (c.mean(f, "M1", "accB"), c.mean(f, "M1", "accS")), (c.mean(f, "M1", "onB"), c.mean(f, "M1", "onS"))
            lr = c.rat(f, "M1", "loc")
            sr = c.rat(f, "M1", "dKV")
            b_ok = min(acc + on) >= 0.90 and lr[0] >= 0.50 and lr[1] >= 0.40
            A &= a_ok; Bk &= b_ok; sp &= sr[0] >= 0.5
            rep &= dV[2] <= 1.0 and q1[0] <= 0.5 * qN[0]
            beh &= on[0] <= 0.60 or lr[0] <= 0.50
            out(f"      {f:5s} (a) q_V^M1 {fmt(q1)} >= {0.5 * qN[0]:.3f}; paired ID_V M1-M0 {fmt(dV)} > 0 -> {verdict(a_ok)};  rho_V {fmt(rho)}")
            out(f"      {f:5s} (b) acc_B/S {acc[0]:.2f}/{acc[1]:.2f} on_B/S {on[0]:.2f}/{on[1]:.2f} >= 0.90; loc ratio {fmt(lr)} (>= 0.50, lower >= 0.40) -> {verdict(b_ok)};"
                f"  span ratio {fmt(sr)} (>= 0.5) -> {verdict(sr[0] >= 0.5)}")
        g7a[m], g7b[m], g7span[m], hrep[m], hbeh[m] = A, Bk, sp, rep, beh
        out(f"      (a) both formats -> {verdict(A)};  (b) both formats -> {verdict(Bk)};  span both -> {verdict(sp)};  "
            f"H_replaced ID_V/q_V pattern {verdict(rep)}, behaviour pattern (on_B <= 0.60 or loc ratio <= 0.50) {verdict(beh)}")
    ka, kb, ksp, kr, kbeh = (sum(d.values()) for d in (g7a, g7b, g7span, hrep, hbeh))
    G7 = ka >= TWO and kb == N and ksp >= TWO
    HR = kr >= TWO and kbeh >= TWO
    out(f"   G7 (a) {ka}/{N} (>= {TWO}) -> {verdict(ka >= TWO)};  (b) {kb}/{N} ({N}/{N}) -> {verdict(kb == N)};  span {ksp}/{N} (>= {TWO}) -> {verdict(ksp >= TWO)}")
    out(f"   G7 H_redundant -> {verdict(G7)};  H_replaced ({kr}/{N} ID_V/q_V, {kbeh}/{N} behaviour, each >= {TWO}) -> {verdict(HR)}"
        f"{';  neither: partial takeover' if not G7 and not HR else ';  both patterns: contradictory, reported as such' if G7 and HR else ''}")

    out("\n== G8 routes at the answer position")
    g8a, g8b, g8c = {}, {}, {}
    for m in models:
        if m not in cells:
            g8a[m] = g8b[m] = False; g8c[m] = {"AFTER": False, "P1": False}; continue
        c = cells[m]
        r3 = c.rat("NONE", "M3", "idV")
        d3 = c.paired("NONE", "M3", "M0", "idV")
        g8a[m] = r3[0] <= 0.60 and d3[2] < 0
        out(f"   {m:26s} (a) NONE M3: ID_V^M3/ID_V^M0 {fmt(r3)} <= 0.60; paired {fmt(d3)} < 0 -> {verdict(g8a[m])}")
        rk3 = c.rat("AFTER", "M3", "idK")
        acc3, on3 = c.mean("AFTER", "M3", "accB"), c.mean("AFTER", "M3", "onB")
        g8b[m] = rk3[0] >= 0.80 and acc3 >= 0.90 and on3 >= 0.90
        out(f"   {m:26s} (b) AFTER M3: r_K(M3) {fmt(rk3)} >= 0.80; acc_B {acc3:.2f} on_B {on3:.2f} >= 0.90 -> {verdict(g8b[m])}")
        g8c[m] = {}
        for f in ("AFTER", "P1"):
            r4 = c.rat(f, "M4", "idV", "M1")
            d4 = c.paired(f, "M4", "M1", "idV")
            ok = r4[0] <= 0.60 and d4[2] < 0
            g8c[m][f] = ok and g7a[m]  # "a model meeting G7a" = the model's G7a verdict (both formats), as counted in G7
            out(f"   {m:26s} (c) {f:5s} M4 vs M1: ID_V^M4/ID_V^M1 {fmt(r4)} <= 0.60; paired {fmt(d4)} < 0 -> {verdict(ok)}; "
                f"G7a (both formats) {verdict(g7a[m])}; on_B^M4 {c.mean(f, 'M4', 'onB'):.2f} loc^M4 {c.mean(f, 'M4', 'loc'):.3f} -> {verdict(g8c[m][f])}")
    k8a, k8b = sum(g8a.values()), sum(g8b.values())
    k8c = {f: sum(d[f] for d in g8c.values()) for f in ("AFTER", "P1")}
    G8a, G8b, G8c = k8a >= TWO, k8b == N, all(k >= TWO for k in k8c.values())
    out(f"   G8 (a) {k8a}/{N} (>= {TWO}) -> {verdict(G8a)};  (b) {k8b}/{N} ({N}/{N}) -> {verdict(G8b)};  (c) AFTER {k8c['AFTER']}/{N}, P1 {k8c['P1']}/{N} (each >= {TWO}) -> {verdict(G8c)}")

    out("\n== VERDICTS, part (b)")
    if g_fail:
        out(f"   GATE b FAILED in {g_fail}: G5-G8 are uninterpretable until the knockout is fixed and re-run")
    if g_miss:
        out(f"   Gate b not measured in {g_miss} (file MISSING: not met in every k/k line, not an M8 failure)")
    if not g_fail and not g_miss:
        out("   ALL GATES MET: the predictions are interpreted")
    out(f"   Gate b -> {verdict(False if g_fail else None if g_miss else True)}")
    out(f"   G5 -> {verdict(G5)}   (list {kl}/{N}, sentence {ks}/{N})")
    out(f"   G6 -> {verdict(G6)}   ({k6}/{N})")
    out(f"   G7 -> H_redundant {verdict(G7)}; H_replaced {verdict(HR)}" + ("; partial takeover" if not G7 and not HR else ""))
    out(f"   G8 -> (a) {verdict(G8a)}, (b) {verdict(G8b)}, (c) {verdict(G8c)}")

    out("\n== Exploratory")
    for m, c in cells.items():
        out(f"   {m}: r_K / r_V (1 - r_V = share of the value read removed) by mask")
        for f in ("AFTER", "P1", "POST", "NONE"):
            rk = " ".join(f"{M} {c.rat(f, M, 'idK')[0]:+.2f}/{c.rat(f, M, 'idV')[0]:+.2f}" for M in masks_of(f) if M != "M0" and c.has(f, M))
            out(f"      {f:5s} {rk}")
        if c.has("AFTER", "M5"):
            out(f"      row specificity AFTER: r_K(M5 trk) {fmt(c.rat('AFTER', 'M5', 'idK'))}, r_K(M6 oth) {fmt(c.rat('AFTER', 'M6', 'idK'))}, "
                f"r_K(M1) {fmt(c.rat('AFTER', 'M1', 'idK'))};  second control M2b: r_K {fmt(c.rat('AFTER', 'M2b', 'idK'))} r_V {fmt(c.rat('AFTER', 'M2b', 'idV'))}")
        if c.has("POST", "M1"):
            out(f"      POST takeover: q_V(M1) {fmt(c.q_V('POST', 'M1'))} vs q_V(M0) {fmt(c.q_V('POST', 'M0'))} vs q_V(NONE) {fmt(c.q_V('NONE', 'M0'))}")
        for f in ("AFTER", "P1", "POST", "NONE"):
            parts = []
            for M in masks_of(f):
                P = c.P.get((f, M))
                if not P:
                    continue
                dK, dV, dKV = (c.v(f, M, k) for k in ("dK", "dV", "dKV"))
                comp = [i for i in sorted(P) if P[i]["competent"]]
                parts.append(f"{M}: s_K {dK.mean() / (dK.mean() + dV.mean()):+.2f} s_ID {c.v(f, M, 'idK').mean() / (c.v(f, M, 'idK').mean() + c.v(f, M, 'idV').mean()):+.2f} "
                             f"int {(dKV - dK - dV).mean():+.2f} competent n={len(comp)} ID_K {c.v(f, M, 'idK', comp).mean() if comp else float('nan'):+.2f}")
            out(f"      {f:5s} " + "; ".join(parts))
        voc = prov[m].get("vocab", {})
        for f in ("AFTER", "P1", "POST", "NONE"):
            for M in ("M0", "M1", "M4", "M8"):
                if c.has(f, M):
                    cnt = Counter(voc.get(str(P["top1"]["ID"]), P["top1"]["ID"]) for P in c.P[f, M].values())
                    out(f"      top-1 at a, {f:5s} {M:3s} self-clamp row: " + ", ".join(f"{t!r} {k}" for t, k in cnt.most_common(5)))
        em = [r["emit"] for r in c.res if r["arm"] == "NONE" and "emit" in r]
        if em:
            ok, steps = [e for e in em if e["a_emit"] is not None], prov[m].get("args", {}).get("emit_steps", 12)
            if ok:
                rows = {"M0": [], "M3e": []}
                for e in ok:
                    for M in rows:
                        mm = e["m"][M]
                        dl = {k: {t: mm[k]["lp"][t] - mm["ID"]["lp"][t] for t in ("S", "X")} for k in mm}
                        rows[M].append(0.5 * ((dl["V_S"]["S"] - dl["V_X"]["S"]) + (dl["V_X"]["X"] - dl["V_S"]["X"])))
                out(f"      emission-position pass (NONE): {len(ok)}/{len(em)} items reach an on-target top-1 within {steps} steps; mean steps "
                    f"{np.mean([len(e['tokens']) for e in ok]):.2f}; ID_V at a_emit: M0 {np.mean(rows['M0']):+.2f}, M3' {np.mean(rows['M3e']):+.2f}, "
                    f"ratio {fmt(ratio(rows['M3e'], rows['M0']))}")
            else:
                out(f"      emission-position pass (NONE): 0/{len(em)} items reach an on-target top-1 within {steps} steps")
        f3 = Path(stage3b) / f"{m}_s0.json"
        r3 = json.load(open(f3))["results"] if f3.exists() else []
        if r3 and "ID@0" in r3[0]["m"]:  # a format_factorial 2x2 file
            for f in ("AFTER", "POST", "NONE"):
                ref = per_core_3b(r3, f)
                ids = [i for i in sorted(c.P[f, "M0"]) if i in ref]
                if ids:
                    gk = np.mean([c.P[f, "M0"][i]["idK"] - ref[i]["idK"] for i in ids]); gv = np.mean([c.P[f, "M0"][i]["idV"] - ref[i]["idV"] for i in ids])
                    out(f"      consistency with stage 3b ({f}, {len(ids)} shared cores): ID_K gap {gk:+.3f}, ID_V gap {gv:+.3f} nats (M0 self-clamp reference vs the 2x2 run)")
    return {"gate": gate, "G5": bool(G5), "G6": bool(G6), "G7": bool(G7), "H_replaced": bool(HR), "G8": (bool(G8a), bool(G8b), bool(G8c))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/gpu_stage5/knockout")
    ap.add_argument("--models", default=None, help="comma-separated model tags to score as primary (default: the three preregistered); 'auto' = every file found")
    ap.add_argument("--stage3b", default="results/gpu_stage3b/format_2x2")
    a = ap.parse_args(argv)
    models = MODELS if a.models is None else [f.split("/")[-1][: -len("_s0.json")] for f in sorted(glob.glob(f"{a.root}/*_s0.json"))] \
        if a.models == "auto" else a.models.split(",")
    if a.models is not None:
        print(f"NOTE: primary models overridden ({models}); the preregistered verdicts are over {MODELS}")
    score(a.root, models, a.stage3b)


if __name__ == "__main__":
    main()
