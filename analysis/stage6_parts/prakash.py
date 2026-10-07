"""Part (b) of P-2026-10-05-H (docs/PREREGISTRATION.md): Gates b0-b3, the kappa evaluability rule and predictions H7-H12
on experiments/prakash_swap.py outputs ({root}/{model}/{preflight,filter,sweep_*,lstar,exchange,clamp}.json), exactly as
drafted. Called by analysis/stage6_score.py (``score(root)``) or standalone.

Per model, arm (BIND, ID), depth l_patch and format f, over the population pairs (95 % percentile CIs from 10,000 pair-
bootstrap resamples with one fixed index set per n, every ratio of means recomputed within the same resamples):
  psi_K = mean[m(r2) - m(r0)] / mean[m(r1) - m(r0)], psi_V likewise with r3; interaction = 1 - psi_K - psi_V;
  kappa = psi_K / (psi_K + psi_V); Phi = mean[m(r1) - m(r0)]; kappa_w from r2w/r3w normalised by r4w (secondary);
  s_ID(f, l0) = mean ID_K / (mean ID_K + mean ID_V) from the natural clamp at p, ID_K = 1/2[(D_S(K_S) - D_S(K_X)) +
  (D_X(K_X) - D_X(K_S))] against the self-clamp row.
kappa evaluability: psi_K + psi_V >= 0.5 and psi_K, psi_V >= -0.1 on the point estimates; a resample failing the rule is
dropped from kappa's CI (and from a paired contrast's), and a dropped fraction > 5 % fails the CI. A cell that is not
evaluable reports psi_K, psi_V and the interaction, labelled interaction-carried when the interaction >= 0.5.
A cell is usable (f evaluable) for H7, H9-H11 and Gate b3 when Gate b0 and Gate b2 pass in it, the rule holds and its
arm's edit reproduces at the cell's depth: IIA(l*) >= 0.7 for BIND, IIA_ID(l*_ID) >= 0.7 for ID at l*_ID (Gate b1), and
IIA_ID(l*) >= 0.7 for ID at l* (the H11 Part 2 condition; Gate b3(b) fails without it). H7 and H11 also need s_ID(f, l0)
defined (mean ID_K + mean ID_V > 0); H10 is stated on kappa alone (no s_ID condition) and, like H7 and H9, needs
NO-MENTION and OPTIONS-AFTER usable (otherwise NOT EVALUABLE). H7's r needs all three formats; otherwise it counts as not
met. "flat-high" labels kappa >= 0.75 in every usable format once H10 is evaluable (reported, no verdict). H12
compares each named 14B verdict that is MET or NOT MET with the same prediction at 70B; a 14B NOT EVALUABLE is left out,
and H12 is NOT EVALUABLE when none remains.
l* and l*_ID are re-derived from the NO-MENTION sweep files (earliest layer with IIA >= max - 0.01) and asserted equal
to lstar.json. Verdicts: MET / NOT MET / NOT EVALUABLE (None); H12 is NOT RUN without a Llama-3-70B directory.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

np.seterr(all="ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ckeys.causaltom import POOL_SHA256  # noqa: E402

PRIMARY, OPTIONAL = "Qwen2.5-14B-Instruct", "Meta-Llama-3-70B-Instruct"
NM, QN, OA, LA, Q2 = "NO-MENTION", "QNAMES", "OPTIONS-AFTER", "LETTERS-AFTER", "QNAMES2"
LAW = (NM, QN, OA)
SEED, B = 20261005, 10000
B0_TOL, B0_TOL_TEST, B1, B2, BOUND, CROSS, FLAT, FLAT_HIGH, R_MIN = 0.3, 1e-3, 0.7, 3.0, 0.25, 0.4, 0.25, 0.75, 0.9
H10_NEED = (NM, OA)   # as H7/H9: the discriminating format (OPTIONS-AFTER) and NO-MENTION must be usable (spec G-P4)
_idx = {}
WORDING = {
    "H7": "for every evaluable f in {NO-MENTION, QNAMES, OPTIONS-AFTER}, with NO-MENTION and OPTIONS-AFTER evaluable: "
          "|kappa(f) - s_ID(f, l*+1)| <= 0.25, and Pearson r(kappa, s_ID(., l*+1)) >= 0.9 over the three",
    "H8": "under NO-MENTION psi_V >= 0.5 and psi_K <= 0.25, with the CI of psi_V - psi_K excluding 0",
    "H9": "kappa(OPTIONS-AFTER) - kappa(NO-MENTION) >= 0.4 with the paired CI excluding 0; kappa(QNAMES) between the two if evaluable",
    "H10": "kappa(f) <= 0.25 for every evaluable f in {NO-MENTION, QNAMES, OPTIONS-AFTER}, with NO-MENTION and OPTIONS-AFTER "
           "evaluable (dissociation reading only if Gate b3 passes)",
    "H11": "Part 1: every evaluable f |kappa_ID(f) - s_ID(f, l*_ID+1)| <= 0.25 and kappa_ID(OPTIONS-AFTER) - kappa_ID(NO-MENTION) >= 0.4 "
           "with CI excluding 0; Part 2 (if IIA_ID(l*) >= 0.7): every evaluable f |kappa_ID at l*(f) - s_ID(f, l*+1)| <= 0.25; "
           "met if Part 1 holds and Part 2 holds or is not evaluable",
    "H12": "if run: Gate b1 with l* in 30..40, and the 14B verdicts of H8, H11 Part 1 and whichever of {H7, H9} or H10 was met "
           "reproduced at the same thresholds (each 14B verdict that is MET or NOT MET recurs at 70B)"}


def IDX(n):
    if n not in _idx:
        _idx[n] = np.random.default_rng(SEED).integers(0, n, (B, n))
    return _idx[n]


def V(ok):
    return "NOT EVALUABLE" if ok is None else "MET" if ok else "NOT MET"


def f3(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def fci(t):
    return f"{f3(t[0])} [{f3(t[1])},{f3(t[2])}]"


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)


def boot(x):
    x = np.asarray(x, float)
    return (x.mean(), *pct(x[IDX(len(x))].mean(1))) if len(x) else (np.nan,) * 3


def rule(pk, pv):
    return (pk + pv >= 0.5) & (pk >= -0.1) & (pv >= -0.1)


def tie_rule(iia):
    mx = max(iia.values())
    return min(int(l) for l, v in iia.items() if v >= mx - 0.01 - 1e-9)


# --------------------------------------------------------------------------- cells
class Ex:
    """One (arm, depth, format) exchange cell over the population pairs (pool order)."""

    def __init__(self, cells, test):
        self.n, self.test = len(cells), test
        self.m = {r: np.array([c["m"][r] for c in cells], float) for r in cells[0]["m"]}
        self.mB = np.array([c["m_B"] for c in cells], float)
        self.ok1 = np.array([c["ok_r1"] for c in cells], float)
        self.sec = np.array([c["ok_B"] and c["ok_C"] for c in cells], bool)
        self.q = np.array([c["q"] for c in cells])
        d = lambda r: self.m[r] - self.m["r0"]  # noqa: E731
        self.d = {r: d(r) for r in self.m}
        I = IDX(self.n)
        self.M = {r: self.d[r].mean() for r in self.d}
        self.R = {r: self.d[r][I].mean(1) for r in self.d}
        self.pk, self.pv = self.M["r2"] / self.M["r1"], self.M["r3"] / self.M["r1"]
        self.PK, self.PV = self.R["r2"] / self.R["r1"], self.R["r3"] / self.R["r1"]
        self.inter = 1 - self.pk - self.pv
        self.kappa = self.pk / (self.pk + self.pv)
        self.KAP = self.PK / (self.PK + self.PV)
        self.valid = rule(self.PK, self.PV)
        self.drop = 1 - self.valid.mean()
        self.rule = bool(rule(self.pk, self.pv))
        self.kci = pct(self.KAP[self.valid])
        self.phi = boot(self.d["r1"])
        self.iia = self.ok1.mean()
        self.b0a = np.abs(self.m["r4"] - self.m["r1"]).mean()
        self.b0b = np.abs(self.m["r0"] - self.mB).mean()
        self.b0 = bool(self.b0a <= (B0_TOL_TEST if test else B0_TOL) and self.b0b <= (B0_TOL_TEST if test else B0_TOL))
        self.b2 = bool(self.phi[0] >= B2)
        self.rho_k = (self.m["r5"] - self.m["r1"]).mean() / (self.m["r0"] - self.m["r1"]).mean()
        self.rho_v = (self.m["r6"] - self.m["r1"]).mean() / (self.m["r0"] - self.m["r1"]).mean()
        if "r4w" in self.M:
            self.pkw, self.pvw = self.M["r2w"] / self.M["r4w"], self.M["r3w"] / self.M["r4w"]
            self.kappa_w = self.pkw / (self.pkw + self.pvw)

    def usable(self, b1):
        return bool(self.b0 and self.b2 and b1 and self.rule)

    def state(self):
        if self.rule:
            return f"kappa {f3(self.kappa)} [{f3(self.kci[0])},{f3(self.kci[1])}] (dropped {self.drop:.1%}" + (", CI FAILED: > 5 %)" if self.drop > 0.05 else ")")
        return (f"kappa NOT EVALUABLE (psi_K {f3(self.pk)}, psi_V {f3(self.pv)}, interaction {f3(self.inter)}"
                + ("; interaction-carried" if self.inter >= 0.5 else "") + ")")


def diff_ci(a, b):
    """Paired kappa(a) - kappa(b) on shared resamples; a resample counts only when both satisfy the rule.
    Returns (point, lo, hi, dropped fraction)."""
    ok = a.valid & b.valid
    dlt = (a.KAP - b.KAP)[ok]
    return (a.kappa - b.kappa, *pct(dlt), 1 - ok.mean())


def excl0(lo, hi):
    return bool(np.isfinite(lo) and (lo > 0 or hi < 0))


class Clamp:
    """Natural clamp at p, one (format, onset) cell: per-pair ID_K, ID_V, ID_KV against the self-clamp row."""

    def __init__(self, cells):
        def D(c, row, t):
            return c["lp"][row][t] - c["lp"]["ID"][t]

        def idx(c, ch):
            return 0.5 * ((D(c, f"{ch}_S", "S") - D(c, f"{ch}_X", "S")) + (D(c, f"{ch}_X", "X") - D(c, f"{ch}_S", "X")))
        self.n = len(cells)
        self.k, self.v, self.kv = (np.array([idx(c, ch) for c in cells], float) for ch in ("K", "V", "KV"))
        I = IDX(self.n)
        K, Vv = self.k[I].mean(1), self.v[I].mean(1)
        self.s = self.k.mean() / (self.k.mean() + self.v.mean()) if self.k.mean() + self.v.mean() > 0 else np.nan
        self.s_ci = pct(K / (K + Vv))
        self.idk = boot(self.k)
        self.inter = 1 - (self.k.mean() + self.v.mean()) / self.kv.mean()

    def line(self):
        return (f"ID_K {fci(self.idk)}  ID_V {f3(self.v.mean())}  ID_KV {f3(self.kv.mean())}  s_ID {f3(self.s)} "
                f"[{f3(self.s_ci[0])},{f3(self.s_ci[1])}]  interaction/ID_KV {f3(self.inter)}")


class Model:
    def __init__(self, d, test, out):
        self.d, self.test, self.name = d, test, d.name
        J = lambda f: json.load(open(d / f))  # noqa: E731
        self.pf, self.lj = J("preflight.json"), J("lstar.json")
        self.filt = J("filter.json") if (d / "filter.json").exists() else {}
        self.pop = self.lj["population"]
        self.sweeps = {}
        for f in sorted(d.glob("sweep_*.json")):
            s = J(f.name)
            if s.get("label"):
                continue
            iia = {int(l): np.mean([r["ok"] for r in v]) for l, v in s["rows"].items()}
            phi = {int(l): np.mean([r["m_patch"] - r["m_self"] for r in v]) for l, v in s["rows"].items()}
            assert [r["i"] for r in next(iter(s["rows"].values()))] == self.pop, f"{f.name}: not the population of lstar.json"
            self.sweeps[s["arm"], s["format"]] = (iia, phi)
        self.ls, self.li = (tie_rule(self.sweeps[a, NM][0]) for a in ("BIND", "ID"))
        assert (self.ls, self.li) == (self.lj["lstar"], self.lj["lstar_ID"]), \
            f"l* re-derived from the sweeps ({self.ls}, {self.li}) != lstar.json ({self.lj['lstar']}, {self.lj['lstar_ID']})"
        self.iia_b = self.sweeps["BIND", NM][0][self.ls]
        self.iia_i = self.sweeps["ID", NM][0][self.li]
        self.iia_i_at_ls = self.sweeps["ID", NM][0].get(self.ls)
        self.b1 = {"BIND": bool(self.iia_b >= B1), "ID": bool(self.iia_i >= B1)}
        ex = J("exchange.json") if (d / "exchange.json").exists() else {"cells": []}
        cl = J("clamp.json") if (d / "clamp.json").exists() else {"cells": []}
        assert ex.get("lstar", self.ls) == self.ls and cl.get("lstar", self.ls) == self.ls and ex.get("lstar_ID", self.li) == self.li
        g = {}
        for c in ex["cells"]:
            g.setdefault((c["arm"], c["depth"], c["format"]), []).append(c)
        for k, v in g.items():
            assert [c["i"] for c in v] == self.pop, f"exchange cell {k}: not the population"
        self.E = {k: Ex(v, test) for k, v in g.items()}
        h = {}
        for c in cl["cells"]:
            h.setdefault((c["format"], c["l0"]), []).append(c)
        self.C = {k: Clamp(v) for k, v in h.items()}

    def ex(self, arm, depth, f):
        return self.E.get((arm, depth, f))

    def sid(self, f, l0):
        c = self.C.get((f, l0))
        return np.nan if c is None else c.s

    def b1_at(self, arm, depth):
        """The arm's edit reproduces at this depth: Gate b1 (BIND at l*, ID at l*_ID), or IIA_ID(l*) >= 0.7 for ID at l*."""
        if arm == "BIND" or depth == self.li:
            return self.b1[arm]
        return bool(self.sweeps["ID", NM][0].get(depth, -1) >= B1)

    def cell_ok(self, arm, depth, f):
        e = self.ex(arm, depth, f)
        return e is not None and e.usable(self.b1_at(arm, depth))


# --------------------------------------------------------------------------- gates and predictions
def gates(M, out):
    out(f"   Gate b1 reproduction (NO-MENTION sweep): BIND IIA(l* = {M.ls}) = {M.iia_b:.3f} (>= 0.7) -> {V(M.b1['BIND'])}; "
        f"ID IIA_ID(l*_ID = {M.li}) = {M.iia_i:.3f} (>= 0.7) -> {V(M.b1['ID'])}; IIA_ID(l*) = "
        + ("not swept" if M.iia_i_at_ls is None else f"{M.iia_i_at_ls:.3f}") + " (H11 Part 2 and Gate b3(b) need >= 0.7)")
    tol = B0_TOL_TEST if M.test else B0_TOL
    for (arm, dep, f), e in sorted(M.E.items()):
        out(f"   {arm:4s} @{dep:2d} {f:13s} n={e.n:3d} Gate b0 mean|m(r4)-m(r1)| {e.b0a:.2e} mean|m(r0)-m(B)| {e.b0b:.2e} (<= {tol:g}) -> {V(e.b0)};  "
            f"Gate b2 Phi {fci(e.phi)} (>= 3) -> {V(e.b2)};  rule {'holds' if e.rule else 'fails'};  {e.state()}")
    c = M.C.get((OA, M.ls + 1))
    e = M.ex("ID", M.ls, OA)
    a_ok = None if c is None else bool(c.idk[0] > 0 and excl0(c.idk[1], c.idk[2]) and np.isfinite(c.s) and c.s >= 0.5)
    b_ok = None if e is None else bool(M.cell_ok("ID", M.ls, OA) and e.kappa >= 0.5)
    b3 = None if a_ok is None or b_ok is None else bool(a_ok and b_ok)
    out(f"   Gate b3 (H10 as a dissociation): (a) ID_K(OPTIONS-AFTER, l*+1) " + ("MISSING" if c is None else f"{fci(c.idk)} > 0, CI excl. 0; s_ID {f3(c.s)} >= 0.5")
        + f" -> {V(a_ok)};  (b) kappa_ID at l* (OPTIONS-AFTER) " + ("MISSING" if e is None else f"{e.state()}, >= 0.5 and evaluable (IIA_ID(l*) >= 0.7)")
        + f" -> {V(b_ok)};  Gate b3 -> {V(b3)}")
    return b3


def law(M, arm, depth, l0, need=(NM, OA)):
    """{f: (usable, gap)} over the law formats; usable needs the gates, the rule and a defined s_ID(f, l0)."""
    res = {}
    for f in LAW:
        e = M.ex(arm, depth, f)
        s = M.sid(f, l0)
        u = M.cell_ok(arm, depth, f) and np.isfinite(s)
        res[f] = (u, abs(e.kappa - s) if u else np.nan, e.kappa if e else np.nan, s)
    ev = [f for f in LAW if res[f][0]]
    return res, ev, all(res[f][0] for f in need)


def law_text(res):
    return "; ".join(f"{f} " + (f"kappa {f3(k)} s_ID {f3(s)} gap {gap:.3f}" if u else f"not evaluable (kappa {f3(k)}, s_ID {f3(s)})") for f, (u, gap, k, s) in res.items())


def predictions(M, b3, out):
    R = {}
    ls, li = M.ls, M.li
    # H7
    res, ev, need = law(M, "BIND", ls, ls + 1)
    if not need:
        h7 = None
    else:
        gaps = all(res[f][1] <= BOUND for f in ev)
        r = np.corrcoef([res[f][2] for f in LAW], [res[f][3] for f in LAW])[0, 1] if len(ev) == 3 else np.nan
        h7 = bool(gaps and np.isfinite(r) and r >= R_MIN)
    R["H7"] = (h7, f"{law_text(res)}; r = " + (f"{np.corrcoef([res[f][2] for f in LAW], [res[f][3] for f in LAW])[0, 1]:+.3f}" if len(ev) == 3 else "not evaluable (needs all three formats; counts as not met)"))
    res0, ev0, need0 = law(M, "BIND", ls, 0)
    out(f"   (H7 secondary, against s_ID(f, 0): {law_text(res0)})")
    # H8
    e = M.ex("BIND", ls, NM)
    if e is None or not (e.b0 and e.b2 and M.b1["BIND"]):
        R["H8"] = (None, "BIND NO-MENTION cell missing or failing Gate b0/b1/b2")
    else:
        dv = e.PV - e.PK
        lo, hi = pct(dv)
        R["H8"] = (bool(e.pv >= 0.5 and e.pk <= 0.25 and excl0(lo, hi)), f"psi_V {f3(e.pv)} psi_K {f3(e.pk)} psi_V - psi_K {f3(e.pv - e.pk)} [{f3(lo)},{f3(hi)}]")
    # H9
    a, b, q = (M.ex("BIND", ls, f) for f in (OA, NM, QN))
    if not (M.cell_ok("BIND", ls, OA) and M.cell_ok("BIND", ls, NM)):
        R["H9"] = (None, "NO-MENTION or OPTIONS-AFTER not evaluable")
    else:
        dpt, lo, hi, dr = diff_ci(a, b)
        between = True
        if M.cell_ok("BIND", ls, QN):
            between = bool(min(a.kappa, b.kappa) <= q.kappa <= max(a.kappa, b.kappa))
        R["H9"] = (bool(dpt >= CROSS and excl0(lo, hi) and dr <= 0.05 and between),
                   f"kappa(OA) - kappa(NM) {f3(dpt)} [{f3(lo)},{f3(hi)}] dropped {dr:.1%}" + (" (CI FAILED)" if dr > 0.05 else "")
                   + (f"; kappa(QNAMES) {f3(q.kappa)} between: {between}" if M.cell_ok("BIND", ls, QN) else "; QNAMES not evaluable"))
    # H10 (kappa alone: usability from the gates and the rule, no s_ID condition)
    ev10 = [f for f in LAW if M.cell_ok("BIND", ls, f)]
    k10 = {f: M.ex("BIND", ls, f).kappa for f in ev10}
    if not ev10 or not all(f in ev10 for f in H10_NEED):
        R["H10"] = (None, "no format evaluable" if not ev10 else f"{', '.join(H10_NEED)} must be evaluable")
    else:
        R["H10"] = (bool(all(k <= FLAT for k in k10.values())), "; ".join(f"{f} kappa {f3(k)}" for f, k in k10.items())
                    + "".join(f"; {f} not evaluable" for f in LAW if f not in ev10)
                    + ("; flat-high (kappa >= 0.75 in every evaluable format: address read by the answer position)" if all(k >= FLAT_HIGH for k in k10.values()) else "")
                    + ("" if b3 else "; Gate b3 " + ("not passed" if b3 is False else "not evaluable") + ": not evaluable as a dissociation at this depth"))
    # H11
    r1, ev1, need1 = law(M, "ID", li, li + 1)
    p1 = None
    if M.b1["ID"] and need1:
        a, b = M.ex("ID", li, OA), M.ex("ID", li, NM)
        dpt, lo, hi, dr = diff_ci(a, b)
        p1 = bool(all(r1[f][1] <= BOUND for f in ev1) and dpt >= CROSS and excl0(lo, hi) and dr <= 0.05)
        t1 = f"{law_text(r1)}; kappa_ID(OA) - kappa_ID(NM) {f3(dpt)} [{f3(lo)},{f3(hi)}] dropped {dr:.1%}"
    else:
        t1 = "not evaluable (" + ("Gate b1 ID failed" if not M.b1["ID"] else "NO-MENTION or OPTIONS-AFTER not evaluable") + f"): {law_text(r1)}"
    p2e = M.b1_at("ID", ls)
    r2, ev2, _ = law(M, "ID", ls, ls + 1)
    p2 = bool(all(r2[f][1] <= BOUND for f in ev2)) if p2e and ev2 else None
    R["H11"] = (None if p1 is None else bool(p1 and p2 is not False),
                f"Part 1 {V(p1)} ({t1}); Part 2 {V(p2)} (" + (law_text(r2) if p2e else "IIA_ID(l*) < 0.7 or not swept") + ")")
    R["H11 Part 1"] = (p1, t1)
    return R


def score(root, out=print, test=False):
    root = Path(root)
    dirs = {d.name: d for d in sorted(root.iterdir()) if d.is_dir()} if root.is_dir() else {}
    out(f"== Part (b) the exchange on Prakash et al.'s intervention (P-2026-10-05-H, Gates b0-b3, H7-H12): root {root}")
    prim = PRIMARY if PRIMARY in dirs else (next((m for m in dirs if m != OPTIONAL and (dirs[m] / "lstar.json").exists()), None) if test else None)
    if test and prim and prim != PRIMARY:
        out(f"   TEST: {prim} stands in for {PRIMARY}; the verdicts are plumbing checks, not results")
    models = {}
    for m in [x for x in (prim, OPTIONAL) if x in dirs]:
        if (dirs[m] / "SKIPPED.txt").exists():
            out(f"   {m}: SKIPPED ({(dirs[m] / 'SKIPPED.txt').read_text().strip()})")
            continue
        if not all((dirs[m] / f).exists() for f in ("preflight.json", "lstar.json", "sweep_BIND_NO-MENTION.json", "sweep_ID_NO-MENTION.json")):
            out(f"   {m}: INCOMPLETE (no preflight, sweeps or lstar.json; its gates and predictions are NOT EVALUABLE)")
            continue
        models[m] = Model(dirs[m], test, out)
    if prim not in models:
        out(f"   {PRIMARY}: MISSING (every gate and prediction NOT EVALUABLE)")
        out("   GATES\n" + "\n".join(f"   {g} NOT EVALUABLE (no results)" for g in ("Gate b0", "Gate b1", "Gate b2", "Gate b3")))
    res = {}
    for m, M in models.items():
        p = M.pf
        out(f"\n-- {m}: release {p['release']['sha'][:10]} files " + ", ".join(f"{k.split('/')[-1]} {v[:8]}" for k, v in p["release"]["files"].items())
            + f"; pool sha256 {p['pool_sha256'][:16]} ({'OK' if p['pool_sha256'] == POOL_SHA256 else 'MISMATCH'}), n_pool {p['n_pool']}, n_bos {p['n_bos']}; "
            + ", ".join(f"{f} T={v['length']}" for f, v in p["formats"].items()))
        fl = M.filt
        if fl.get("pairs"):
            passing = [x["i"] for x in fl["pairs"] if x["ok"]]
            theirs = set(passing[80:160])
            out(f"   LM filter (NO-MENTION, both prompts): {fl['n_pass']}/{len(fl['pairs'])} pass (accuracy {fl['accuracy']:.3f}); population n = {len(M.pop)}"
                + (" (fewer than 150 pass: n is the number that pass)" if fl["n_pass"] < 150 else "")
                + f", the first {len(M.pop)} passing pairs in pool order: {'OK' if M.pop == passing[:len(M.pop)] else 'MISMATCH'}"
                + f"; overlap with the 81st-160th passing pairs (their validation split, under our filter): {len(theirs & set(M.pop))}/{len(theirs)}")
        else:
            out(f"   no LM filter (TEST): population = the first {len(M.pop)} pool pairs")
        out(f"   l* = {M.ls} (lstar.json {M.lj['lstar']}), l*_ID = {M.li} (lstar.json {M.lj['lstar_ID']}): re-derived from the sweep files, equal")
        for arm in ("BIND", "ID"):
            iia, phi = M.sweeps[arm, NM]
            out(f"   sweep {arm:4s} NO-MENTION IIA/Phi: " + " ".join(f"{l}:{iia[l]:.2f}/{phi[l]:+.1f}" for l in sorted(iia)))
        out("   GATES")
        b3 = gates(M, out)
        R = predictions(M, b3, out)
        res[m] = {"b1": M.b1, "b3": b3, "R": R, "lstar": M.ls, "lstar_ID": M.li,
                  "b0": {f"{a}@{d} {f}": e.b0 for (a, d, f), e in M.E.items()}, "b2": {f"{a}@{d} {f}": e.b2 for (a, d, f), e in M.E.items()}}
    out("\n   PREDICTIONS (" + (prim or PRIMARY) + ")")
    P = res.get(prim, {}).get("R", {})
    verd = {}
    for h in ("H7", "H8", "H9", "H10", "H11"):
        ok, txt = P.get(h, (None, "no results"))
        verd[h] = (ok, txt)
        out(f"   {h:4s} {WORDING[h]}\n        {txt} -> {V(ok)}")
    run70 = OPTIONAL in dirs and not (dirs[OPTIONAL] / "SKIPPED.txt").exists()
    if OPTIONAL in res and prim in res:
        L = res[OPTIONAL]
        named = ["H8", "H11 Part 1"] + [h for h in ("H7", "H9", "H10") if P[h][0] is True]
        ev14 = [h for h in named if P[h][0] is not None]
        rep = {h: L["R"][h][0] is P[h][0] for h in ev14}
        g70 = bool(L["b1"]["BIND"] and 30 <= L["lstar"] <= 40)
        verd["H12"] = (None if not ev14 else bool(g70 and all(rep.values())),
                       f"70B Gate b1 {V(L['b1']['BIND'])}, l* = {L['lstar']} (30..40): {'OK' if g70 else 'FAILED'}; "
                       + "".join(f"{h} NOT EVALUABLE at 14B (left out); " for h in named if h not in ev14)
                       + ("reproduced (14B verdict, 70B verdict): " + ", ".join(f"{h} {V(P[h][0])}, {V(L['R'][h][0])}: {'yes' if v else 'no'}" for h, v in rep.items())
                          if ev14 else "no named 14B verdict is MET or NOT MET"))
    else:
        verd["H12"] = (None, "NOT RUN (optional; no Llama-3-70B results)" if not run70 else
                       "70B results incomplete" if OPTIONAL not in res else "no Qwen2.5-14B results to reproduce")
    out(f"   H12  {WORDING['H12']}\n        {verd['H12'][1]} -> {'NOT RUN' if not run70 else V(verd['H12'][0])}")
    for m, M in models.items():
        out(f"\n-- {m}: EXPLORATORY")
        exploratory(M, out)
    gate_sum = {m: {"b1": r["b1"], "b3": r["b3"], "b0": r["b0"], "b2": r["b2"]} for m, r in res.items()}
    return {"gates": gate_sum, "verdicts": verd, "primary": prim, "models": list(models)}


def exploratory(M, out):
    for (arm, dep, f), e in sorted(M.E.items()):
        sec = int(e.sec.sum())
        line = (f"   {arm:4s} @{dep:2d} {f:13s} psi_K {f3(e.pk)} psi_V {f3(e.pv)} interaction {f3(e.inter)} rho_K {f3(e.rho_k)} rho_V {f3(e.rho_v)} "
                f"IIA {e.iia:.2f}" + (f" kappa_w {f3(e.kappa_w)} (|kappa_w - kappa| {abs(e.kappa_w - e.kappa):.3f}" + (" > 0.1: punctuation rows)" if abs(e.kappa_w - e.kappa) > 0.1 else ")") if hasattr(e, "kappa_w") else "")
                + f"; secondary population {sec}/{e.n}" + (" (FLAG: loses > 30 %)" if sec < 0.7 * e.n else ""))
        for q in (0, 1):
            s = e.q == q
            if s.any() and s.sum() < e.n:
                line += f"; q={q}: psi_K {f3(e.d['r2'][s].mean() / e.d['r1'][s].mean())} psi_V {f3(e.d['r3'][s].mean() / e.d['r1'][s].mean())} (n={int(s.sum())})"
        out(line)
    for (f, l0), c in sorted(M.C.items()):
        out(f"   clamp {f:13s} l0={l0:2d} n={c.n:3d} {c.line()}")
    if all(M.cell_ok("BIND", M.ls, f) for f in LAW + (LA,)) and all(np.isfinite(M.sid(f, M.ls + 1)) for f in LAW + (LA,)):
        k = [M.ex("BIND", M.ls, f).kappa for f in LAW + (LA,)]
        out(f"   BIND law over four formats (LETTERS-AFTER added): r = {np.corrcoef(k, [M.sid(f, M.ls + 1) for f in LAW + (LA,)])[0, 1]:+.3f}")
    for (arm, f), (iia, phi) in sorted(M.sweeps.items()):
        if f != NM:
            out(f"   sweep {arm:4s} {f:13s} IIA/Phi: " + " ".join(f"{l}:{iia[l]:.2f}/{phi[l]:+.1f}" for l in sorted(iia)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/gpu_stage6/prakash")
    ap.add_argument("--test", action="store_true", help="TEST_MODE outputs: the model found stands in for Qwen2.5-14B, b0 at 1e-3")
    a = ap.parse_args(argv)
    score(a.root, test=a.test)


if __name__ == "__main__":
    main()
