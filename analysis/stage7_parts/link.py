"""Step 2 of P-2026-10-08-I (docs/PREREGISTRATION.md): Gates I-G1, I-G3, I-G4 and predictions I1-I7 on link/<tag>.json of
experiments/stage7_link.py (stage link), the family run frames/<tag>.json (experiments/paper1_frames.py) and the
committed stage-3b reference, as worded in the entry.

Per core, seeds averaged within core (as analysis/stage2_score.per_core), m = lp[T] - lp[S] over the arm's candidates:
  a(S) = m(x_S) - m(P) (B_x batch); D^O, K^O, V^O = m(M) - m(P), m(P + K_M) - m(P), m(P + V_M) - m(P) in the family
  batch of condition O ("0" = the unblocked batch); every ratio, kappa, closure and t recomputed within each resample of
  the fixed core bootstrap (common.W; the formats share the cores and are resampled jointly).
"""
import numpy as np

from .common import NAMES, V, comb, est, f3, kappa, lower_ok, nan, why_undefined

SEEDS = (101, 102, 103)
ARMS = ("P1", "LETTER", "POST", "NONE", "BEFORE")
CONF = ("P1", "LETTER")
CTRL = ("rand0", "rand1", "rand2", "active")


def m_of(run, rec):
    return run["cand"][rec["iT"]] - run["cand"][rec["iS"]]


class Link:
    def __init__(self, J):
        self.J, self.P = J, J["provenance"]
        self.arms = {a: J["arms"][a] for a in ARMS if a in J["arms"]}
        ids = {a: [r["core"]["id"] for r in R] for a, R in self.arms.items()}
        ref = next(iter(ids.values()), [])
        self.same_cores = all(v == ref for v in ids.values())
        self.n = len(ref)

    def R(self, arm):
        return self.arms.get(arm, [])

    def has(self, arm, cond):
        R = self.R(arm)
        return bool(R) and all(f"{cond}/P_101" in r["runs"] or f"{cond}/B" in r["runs"] for r in R)

    def sm(self, arm, cond, name):
        """Per core: mean over seeds of m(cond/name_seed)."""
        return np.array([np.mean([m_of(r["runs"][f"{cond}/{name}_{s}"], r) for s in SEEDS]) for r in self.R(arm)], float)

    def nat(self, arm, cond, name):
        return np.array([m_of(r["runs"][f"{cond}/{name}"], r) for r in self.R(arm)], float)

    def fam(self, arm, cond):
        P = self.sm(arm, cond, "P")
        return dict(D=self.sm(arm, cond, "M") - P, K=self.sm(arm, cond, "PK") - P, V=self.sm(arm, cond, "PV") - P)

    def a(self, arm, name, cond="x"):
        return self.sm(arm, cond, name) - self.sm(arm, cond, "P")

    def Dx(self, arm):
        return self.sm(arm, "x", "M") - self.sm(arm, "x", "P")

    # ---- argmax rates over the six candidates
    def rates(self, arm, cond):
        R = self.R(arm)
        top = lambda run: int(np.argmax(run["cand"]))  # noqa: E731
        Mr = [top(r["runs"][f"{cond}/M_{s}"]) for r in R for s in SEEDS]
        Pr = [top(r["runs"][f"{cond}/P_{s}"]) for r in R for s in SEEDS]
        Br = [top(r["runs"][f"{cond}/B"]) for r in R]
        tg = lambda key: [r[key] for r in R for _ in SEEDS]  # noqa: E731
        init3 = tg("init")
        return dict(M_T=np.mean(np.array(Mr) == tg("iT")), P_S=np.mean(np.array(Pr) == tg("iS")),
                    B_B=np.mean(np.array(Br) == [r["iB"] for r in R]), M_init=np.mean(np.array(Mr) == init3),
                    P_init=np.mean(np.array(Pr) == init3), B_init=np.mean(np.array(Br) == [r["init"] for r in R]))


def rates_txt(L, arm, conds=("0", "A:H", "A+:H", "N:H")):
    parts = []
    for c in conds:
        if L.has(arm, c):
            r = L.rates(arm, c)
            parts.append(f"{c}: M->T {r['M_T']:.2f} P->S {r['P_S']:.2f} B->B {r['B_B']:.2f} (initial location M {r['M_init']:.2f} "
                         f"P {r['P_init']:.2f} B {r['B_init']:.2f})")
    return "; ".join(parts)


# ---------------------------------------------------------------- statistics of a condition
def cond_stats(L, arm, cond):
    """Every step-2 statistic of condition ``cond`` in format ``arm`` against the unblocked batch and NO-MENTION."""
    f0, fO, fN = L.fam(arm, "0"), L.fam(arm, cond), L.fam("NONE", "0")
    xs = [f0["D"], f0["K"], f0["V"], fO["D"], fO["K"], fO["V"], fN["D"], fN["K"], fN["V"]]
    k0 = lambda D0, K0, V0, *_: kappa(K0, V0, D0)  # noqa: E731
    kO = lambda D0, K0, V0, D, K, V, *_: kappa(K, V, D)  # noqa: E731
    kN = lambda D0, K0, V0, D, K, V, DN, KN, VN: kappa(KN, VN, DN)  # noqa: E731
    S = dict(
        t=est(lambda D0, K0, V0, D, *_: D / D0, *xs),
        psiK_t=est(lambda D0, K0, V0, D, K, *_: K / D0, *xs), psiV_t=est(lambda D0, K0, V0, D, K, V, *_: V / D0, *xs),
        psiK_O=est(lambda D0, K0, V0, D, K, *_: K / D, *xs), psiV_O=est(lambda D0, K0, V0, D, K, V, *_: V / D, *xs),
        nonadd_O=est(lambda D0, K0, V0, D, K, V, *_: 1 - (K + V) / D, *xs),
        kappa0=est(k0, *xs), kappaO=est(kO, *xs), kappaN=est(kN, *xs),
        c_kappa=est(lambda *x: (k0(*x) - kO(*x)) / (k0(*x) - kN(*x)), *xs),
        c_K=est(lambda D0, K0, V0, D, K, V, DN, KN, VN: (K0 / D0 - K / D0) / (K0 / D0 - KN / DN), *xs),
        dV=est(lambda D0, K0, V0, D, K, V, *_: V - V0, *xs),
        FV=0.25 * (fN["V"].mean() - f0["V"].mean()),
        dV_per_lostK=est(lambda D0, K0, V0, D, K, V, *_: (V - V0) / (K0 - K), *xs),
        dkappa=est(lambda *x: kO(*x) - k0(*x), *xs),
        dpsiV=est(lambda D0, K0, V0, D, K, V, *_: (V - V0) / D0, *xs),
        D0=f0["D"].mean(), K0=f0["K"].mean(), V0=f0["V"].mean(), D=fO["D"].mean(), K=fO["K"].mean(), V=fO["V"].mean())
    return S


def stats_txt(S):
    return (f"t {f3(S['t'])}; psi~_K {S['psiK_t'][0]:+.3f}, psi~_V {S['psiV_t'][0]:+.3f}; blocked scale psi_K {S['psiK_O'][0]:+.3f}, "
            f"psi_V {S['psiV_O'][0]:+.3f}, non-additive {S['nonadd_O'][0]:+.3f}; kappa {f3(S['kappaO'])} (unblocked {S['kappa0'][0]:+.3f}); "
            f"c_kappa {f3(S['c_kappa'])}; c~_K {f3(S['c_K'])}; dV {f3(S['dV'])} nats (floor {S['FV']:.2f}); dV / lost K {S['dV_per_lostK'][0]:+.3f}; "
            f"nats D {S['D']:.2f} K {S['K']:.2f} V {S['V']:.2f} (unblocked {S['D0']:.2f} / {S['K0']:.2f} / {S['V0']:.2f})")


# ---------------------------------------------------------------- gates
def frames_rows(results, arm):
    import stage2_score as s2
    return s2.per_core(results, arm)


def phi_psi(rows, ids=None):
    ids = sorted(rows) if ids is None else ids
    g = lambda f: np.mean([f(rows[i]) for i in ids])  # noqa: E731
    d = g(lambda x: x["M"] - x["P"])
    return dict(phi=d / g(lambda x: x["T"] - x["S"]), psiK=g(lambda x: x["add"] - x["P"]) / d, psiV=g(lambda x: x["addv"] - x["P"]) / d, D=d)


def gate_g1a(frames, ref, arm, test):
    """|d phi|, |d psi_K|, |d psi_V| <= 0.03 against the stage-3b reference."""
    if frames is None:
        return None, "no family run (frames)", {}
    F = phi_psi(frames_rows(frames["results"], arm))
    if ref is None:
        return None, (f"phi {F['phi']:+.3f}, psi_K {F['psiK']:+.3f}, psi_V {F['psiV']:+.3f}; no stage-3b reference"
                      + (" (TEST_MODE: none exists for this model)" if test else "")), F
    Rf = phi_psi(frames_rows(ref["results"], arm))
    d = {k: F[k] - Rf[k] for k in ("phi", "psiK", "psiV")}
    ok = all(abs(x) <= 0.03 for x in d.values())
    return ok, ", ".join(f"{k} {F[k]:+.3f} vs {Rf[k]:+.3f} (diff {d[k]:+.3f})" for k in d) + " each within 0.03", F


def gate_g1b(L, frames, arm):
    """The unblocked family batch reproduces the family's psi_K, psi_V (0.03) and D (3 %); B_x x_all reproduces psi_K."""
    if frames is None or not L.has(arm, "0"):
        return None, "no family run or no unblocked batch", {}
    rows = frames_rows(frames["results"], arm)
    ids = [r["core"]["id"] for r in L.R(arm)]
    if not set(ids) <= set(rows):
        return None, "the family run lacks cores of the link run", {}
    F = phi_psi(rows, ids)
    f0 = L.fam(arm, "0")
    D0 = f0["D"].mean()
    pk, pv = f0["K"].mean() / D0, f0["V"].mean() / D0
    ok = abs(pk - F["psiK"]) <= 0.03 and abs(pv - F["psiV"]) <= 0.03 and abs(D0 / F["D"] - 1) <= 0.03
    txt = (f"unblocked batch psi_K {pk:+.3f} vs {F['psiK']:+.3f}, psi_V {pv:+.3f} vs {F['psiV']:+.3f} (within 0.03), "
           f"D {D0:.2f} vs {F['D']:.2f} ({100 * (D0 / F['D'] - 1):+.1f} %, within 3 %)")
    if arm in ("P1", "LETTER", "POST"):
        px = L.a(arm, "x_all").mean() / L.Dx(arm).mean()
        okx = abs(px - F["psiK"]) <= 0.03
        ok = ok and okx
        txt += f"; B_x x_all psi_K {px:+.3f} vs {F['psiK']:+.3f} (within 0.03): {V(okx)}"
    return ok, txt, F


def gate_g1c(L, arm):
    if not (L.has(arm, "N:null") and L.has(arm, "0")):
        return None, "no null knockout"
    S = cond_stats(L, arm, "N:null")
    dk = S["kappaO"][0] - S["kappa0"][0]
    dv = S["dpsiV"][0]
    dt = S["t"][0] - 1
    ok = not nan(dk) and abs(dk) <= 0.02 and abs(dv) <= 0.02 and abs(dt) <= 0.02
    return ok, f"|kappa_null - kappa| {abs(dk):.4f}, |psi~_V null - psi_V| {abs(dv):.4f}, |t - 1| {abs(dt):.4f} (each <= 0.02)"


def gate_g1d(L):
    R = L.R("BEFORE")
    if not R or not L.has("BEFORE", "N:H_P1"):
        return None, "no LIST-BEFORE results", float("nan")
    mx_b = max(abs(a - b) for r in R for s in SEEDS for a, b in zip(r["runs"][f"before/x_HP1_{s}"]["cand"], r["runs"][f"before/x_all_{s}"]["cand"]))
    keys = [k[len("N:H_P1/"):] for k in R[0]["runs"] if k.startswith("N:H_P1/")]
    mx_n = max(abs(a - b) for r in R for k in keys for a, b in zip(r["runs"][f"N:H_P1/{k}"]["cand"], r["runs"][f"0/{k}"]["cand"]))
    mx = max(mx_b, mx_n)
    return mx <= 0.1, f"max |B_x(H*_OPTIONS-AFTER at the list rows) - x_all| {mx_b:.4g}, max |N(H*_OPTIONS-AFTER) - unblocked| {mx_n:.4g} nats (<= 0.1)", mx


def gate_g3(L, arm):
    if not L.has(arm, "0"):
        return None, None, "no unblocked batch"
    D0 = L.fam(arm, "0")["D"].mean()
    c1 = D0 >= 3
    txt = f"(1) D {D0:.2f} >= 3: {V(c1)}"
    c2 = None
    if arm in ("P1", "LETTER", "POST"):
        g = L.a(arm, "x_all").mean() - L.a(arm, "x_notG").mean()
        c2 = g >= 1
        txt += f"; (2) a(all) - a(all@G) {g:.2f} >= 1: {V(c2)}"
    return c1, c2, txt


def gate_g4(L):
    if not L.has("P1", "A+:H"):
        return None, "no A+ results"
    S = cond_stats(L, "P1", "A+:H")
    ok = S["t"][0] >= 0.6 and not nan(S["dV"][1]) and S["dV"][1] > 0
    return ok, f"A+(H*): t {f3(S['t'])} >= 0.6, dV {f3(S['dV'])} > 0 with CI excluding 0"


# ---------------------------------------------------------------- predictions (one format)
def i1(L, arm, KN=None):
    aa, an = L.a(arm, "x_all"), L.a(arm, "x_notG")
    g = est(lambda n, a: 1 - n / a, an, aa)
    gt = est(lambda a, n, k: (a - n) / (a - k), aa, an, KN) if KN is not None else (float("nan"),) * 4
    if arm == "POST":
        ok = g[0] >= 0.5
        return ok, f"g_K {f3(g)} >= 0.5 (reported): {V(ok)}; g~_K {f3(gt)}", g
    ok = g[0] >= 0.7 and lower_ok(g, 0.6)
    return ok, f"g_K {f3(g)} (>= 0.7, lower >= 0.6); g~_K {f3(gt)}", g


def ko_x(L, arm, name, cond="x"):
    """KO_x(S) = [mean a(all) - mean a(S)] / [mean a(all) - mean a(all@G)]; a(all), a(S) from the batch ``cond`` (x or
    curve, each against its own P rows), a(all@G) from the B_x batch."""
    aa, an, aS = L.a(arm, "x_all", cond), L.a(arm, "x_notG"), L.a(arm, name, cond)
    return est(lambda a, n, s: (a - s) / (a - n), aa, an, aS)


def i2(L, arm, psiK_none):
    k = ko_x(L, arm, "x_H")
    aa, aH = L.a(arm, "x_all"), L.a(arm, "x_H")
    rK = est(lambda s, a: s / a, aH, aa)
    pK = est(lambda s, d: s / d, aH, L.Dx(arm))
    ok = k[0] >= 0.8 and lower_ok(k, 0.7)
    return ok, f"KO_x(H*) {f3(k)} (>= 0.8, lower >= 0.7); r_K(H*) {rK[0]:+.3f}, psi_K^x(H*) {pK[0]:+.3f} vs psi_K(NO-MENTION) {psiK_none:+.3f}", k


def i3(L, arm):
    ks = {s: ko_x(L, arm, f"x_{s}") for s in CTRL}
    subs = [t[0] <= 0.25 for t in ks.values()]
    return comb(subs), "KO_x " + ", ".join(f"{s} {f3(t)}" for s, t in ks.items()) + " each <= 0.25", ks


def i4(L, arm):
    S = cond_stats(L, arm, "A:H")
    if arm == "P1":
        t = S["c_kappa"]
        why = why_undefined(S["kappaO"]) or why_undefined(t, "c_kappa")
        ok = (not why) and t[0] >= 0.5 and lower_ok(t, 0.3)
        txt = f"c_kappa {f3(t)} (>= 0.5, lower >= 0.3); kappa^A {f3(S['kappaO'])} vs {S['kappa0'][0]:+.3f} (NO-MENTION {S['kappaN'][0]:+.3f})"
        return ok, txt + (f"; NOT MET: {why}" if why else "") + "  [consistency check: follows from I1 x I2 without a takeover]", S
    t = S["c_K"]
    ok = t[0] >= 0.5 and lower_ok(t, 0.3)
    return ok, (f"c~_K {f3(t)} (>= 0.5, lower >= 0.3); two-sided: kappa^A {f3(S['kappaO'])}, c_kappa {f3(S['c_kappa'])}"
                "  [consistency check: follows from I1 x I2 without a takeover]"), S


def i5(L, arm="P1"):
    S = cond_stats(L, arm, "A:H")
    d = S["dV"]
    ok = not nan(d[1]) and d[1] > 0 and d[0] >= S["FV"]
    return ok, (f"dV {f3(d)} nats > 0 (CI excl. 0) and >= F_V {S['FV']:.2f}; psi~_V {S['psiV_t'][0]:+.3f}, psi_V^A {S['psiV_O'][0]:+.3f}, "
                f"dV / (K - K^A) {S['dV_per_lostK'][0]:+.3f}"), S


def i6(L, arm="P1"):
    S = cond_stats(L, arm, "A:H")
    t = S["t"]
    ok = t[0] >= 0.75 and lower_ok(t, 0.6)
    return ok, f"t {f3(t)} (>= 0.75, lower >= 0.6); argmax rates {rates_txt(L, arm, ('0', 'A:H'))}", S


def i7(L, arm):
    parts, subs = [], []
    for s in CTRL:
        S = cond_stats(L, arm, f"A:{s}")
        dk, dv, dt = S["dkappa"][0], S["dpsiV"][0], S["t"][0] - 1
        ok = not nan(dk) and abs(dk) <= 0.10 and abs(dv) <= 0.10 and abs(dt) <= 0.10
        subs.append(ok)
        parts.append(f"{s}: |d kappa| {abs(dk):.3f}" + (" (kappa undefined after blocking)" if nan(dk) else "") + f", |d psi~_V| {abs(dv):.3f}, |t - 1| {abs(dt):.3f}")
    return comb(subs), "; ".join(parts) + " (each <= 0.10)", subs
