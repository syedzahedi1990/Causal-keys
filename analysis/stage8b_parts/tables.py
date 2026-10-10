"""Paper tables of stage-8 part B as LaTeX fragments in the results directory (tabular environments only; the paper
wraps them): tab_fresh.tex (fresh stories, eight families: r, s_ID, beta_K, beta_V per arm), tab_mass.tex (candidate
mass of the clean B run under L, Sigma and E, and the coverage, per model and arm, F and S0), tab_invariance.tex (S0:
s_ID and r under L and E with the paired difference and its 95 % interval); and fig_crossover.csv, the data of the
behavioural crossover figure (beta_K and beta_V per model and F arm with 95 % cluster intervals, and the flip rate)."""
from __future__ import annotations

from pathlib import Path

from . import lines as ln
from . import stats as st

NAMES = ("tab_fresh.tex", "tab_mass.tex", "tab_invariance.tex", "fig_crossover.csv")


def _ci(e, nd=2):
    if e.undefined():
        return "--"
    return f"{e.pt:.{nd}f} [{e.bound(0.95, 'lo'):.{nd}f}, {e.bound(0.95, 'hi'):.{nd}f}]"


def _tex(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def write_tables(root, models, slots):
    root = Path(root)
    keys = list(ln.P4) + [slots.get(k, k) for k in ln.N4] + [ln.SMALL]
    # tab_fresh
    arms = ("AFTER", "POST", "POST-NULL", "PRE", "BEFORE", "NONE")
    L = [r"\begin{tabular}{ll" + "c" * 4 + "}", r"\toprule",
         r"Model & Arm & $r$ & $s_{\mathrm{ID}}$ & $\beta_K$ & $\beta_V$ \\", r"\midrule"]
    for k in keys:
        m = models.get(k)
        if m is None or m.F is None:
            continue
        v = m.F.view()
        for a in arms:
            if not m.F.has(a):
                continue
            sg = "E" if m.cov(a)[0] >= ln.COV_MIN and m.cov("AFTER")[0] >= ln.COV_MIN else "beta"
            r = "1" if a == "AFTER" else _ci(ln.r_est(v, a, sg))
            s = _ci(ln.s_est(v, a, sg))
            bk, bv = st.est(v.boot, st.mean, v.k(a, "beta")), st.est(v.boot, st.mean, v.v(a, "beta"))
            mark = "" if sg == "E" else r"$^\beta$"
            L.append(f"{_tex(ln.FAMILY.get(k, k))} & {_tex(a)}{mark} & {r} & {s} & {_ci(bk)} & {_ci(bv)} " + r"\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    (root / "tab_fresh.tex").write_text("\n".join(L) + "\n")
    # tab_mass
    M = [r"\begin{tabular}{lllcccc}", r"\toprule", r"Model & Population & Arm & $M^L$ & $M^\Sigma$ & $M^E$ & coverage \\", r"\midrule"]
    for k in keys:
        m = models.get(k)
        if m is None:
            continue
        for name, p in (("F", m.F), ("S0", m.S0)):
            if p is None:
                continue
            v = p.view()
            for a in p.arms:
                ms = [float(v.mass(a, "B", sg).mean()) for sg in ("L", "sigma", "E")]
                cov = m.cov(a, p)[0]
                M.append(f"{_tex(ln.FAMILY.get(k, k))} & {name} & {_tex(a)} & {ms[0]:.3f} & {ms[1]:.3f} & {ms[2]:.3f} & {cov:.3f} \\\\")
    M += [r"\bottomrule", r"\end{tabular}"]
    (root / "tab_mass.tex").write_text("\n".join(M) + "\n")
    # tab_invariance
    T = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Model & Arm & $M^L$ & $s^L_{\mathrm{ID}}$ & $s^E_{\mathrm{ID}}$ & $s^E - s^L$ & $r^E - r^L$ \\", r"\midrule"]
    for k in ln.P4:
        m = models.get(k)
        if m is None or m.S0 is None:
            continue
        v = m.S0.view()
        for a in m.S0.arms:
            sL, sE = ln.s_est(v, a, "L"), ln.s_est(v, a, "E")
            dr = "--" if a == "AFTER" else _ci(ln.eq_est(v, a, "r"), 3)
            T.append(f"{_tex(ln.FAMILY.get(k, k))} & {_tex(a)} & {ln.ml_mass(v, a):.3f} & {_ci(sL)} & {_ci(sE)} & "
                     f"{_ci(ln.eq_est(v, a, 's'), 3)} & {dr} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    (root / "tab_invariance.tex").write_text("\n".join(T) + "\n")
    # fig_crossover.csv
    C = ["model,family,arm,beta_K,beta_K_lo,beta_K_hi,beta_V,beta_V_lo,beta_V_hi,flip_rate,n"]
    for k in keys:
        m = models.get(k)
        if m is None or m.F is None:
            continue
        v = m.F.view()
        for a in m.F.arms:
            bk, bv = st.est(v.boot, st.mean, v.k(a, "beta")), st.est(v.boot, st.mean, v.v(a, "beta"))
            C.append(",".join([k, ln.FAMILY.get(k, k), a] + [f"{x:.4f}" for x in (bk.pt, bk.bound(0.95, "lo"), bk.bound(0.95, "hi"),
                                                                                 bv.pt, bv.bound(0.95, "lo"), bv.bound(0.95, "hi"),
                                                                                 ln.flip_rate(v, a))] + [str(v.n)]))
    (root / "fig_crossover.csv").write_text("\n".join(C) + "\n")
