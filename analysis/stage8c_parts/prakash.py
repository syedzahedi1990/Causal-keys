"""J-C6, J-C-SCREEN and J-C-WIN of P-2026-10-10-J part C on Prakash et al.'s material: jc6/<tag>/exchange.json
(experiments/prakash_caa.py), overlap/<tag>/screen.json and window.json (experiments/stage8_overlap.py). H's definitions
(stage 6, analysis/stage6_parts/prakash.py), pair bootstrap with this part's seed:
  psi_K = mean[m(r2) - m(r0)] / mean[m(r1) - m(r0)], psi_V likewise with r3; kappa = psi_K / (psi_K + psi_V);
  Phi = mean[m(r1) - m(r0)]; IIA (or flip rate) = the share of pairs whose r1 argmax is the target word;
  the kappa rule: psi_K + psi_V >= 0.5 and psi_K, psi_V >= -0.1 (resamples failing it are dropped; > 5 % dropped fails a CI);
  Gate b0: mean |m(r4) - m(r1)| and mean |m(r0) - m(B)| <= 0.3 nats (1e-3 under TEST); Gate b2: Phi >= 3 nats;
  s_ID(f, l0) = mean ID_K / (mean ID_K + mean ID_V) of the natural clamp at p from block l0 (against the self-clamp row).
"""
from __future__ import annotations

import numpy as np

from .stats import NAN, Q, boot_of, component

B0, B0_TEST, B2 = 0.3, 1e-3, 3.0
NM, QN, OA = "NO-MENTION", "QNAMES", "OPTIONS-AFTER"
LAW = (NM, QN, OA)
WIN_PHI, WIN_IIA, WIN_SID = 3.0, 0.5, 0.4
SCREEN_IIA, SCREEN_N = 0.7, 50
H7_GAP, H7_R, H9_DIFF = 0.25, 0.9, 0.4
J6_KAPPA, J6_DIFF, J6_FLIP = 0.30, 0.30, 0.5
G7_IIA, G7_TOL = 0.95, 0.05
ANCHOR = {NM: 0.618, QN: 0.906, OA: 0.864}


def rule(pk, pv):
    return (pk + pv >= 0.5) & (pk >= -0.1) & (pv >= -0.1)


class Ex:
    """One exchange cell (arm, format) over the population pairs."""

    def __init__(self, cells, test):
        self.n = len(cells)
        boot = boot_of(None, self.n)
        m = {r: np.array([c["m"][r] for c in cells], float) for r in cells[0]["m"]}
        d = {r: m[r] - m["r0"] for r in m}
        M = {r: Q(*boot.mean(d[r])) for r in ("r1", "r2", "r3")}
        with np.errstate(all="ignore"):
            self.pk = Q(M["r2"].pt / M["r1"].pt, M["r2"].bs / M["r1"].bs)
            self.pv = Q(M["r3"].pt / M["r1"].pt, M["r3"].bs / M["r1"].bs)
            ok = rule(self.pk.bs, self.pv.bs)
            self.kappa = Q(self.pk.pt / (self.pk.pt + self.pv.pt), np.where(ok, self.pk.bs / (self.pk.bs + self.pv.bs), np.nan))
        self.rule = bool(rule(self.pk.pt, self.pv.pt))
        self.phi = M["r1"]
        self.iia = float(np.mean([c["ok_r1"] for c in cells]))
        self.b0a = float(np.abs(m["r4"] - m["r1"]).mean())
        self.b0b = float(np.abs(m["r0"] - np.array([c["m_B"] for c in cells], float)).mean())
        tol = B0_TEST if test else B0
        self.b0 = bool(self.b0a <= tol and self.b0b <= tol)
        self.b2 = bool(self.phi.pt >= B2)
        self.inter = 1 - self.pk.pt - self.pv.pt
        self.nu = float(np.mean([c["nu_CAA_ID"] for c in cells])) if "nu_CAA_ID" in cells[0] else NAN

    def usable(self):
        return bool(self.b0 and self.b2 and self.rule)

    def txt(self):
        return (f"kappa {self.kappa.txt()} psi_K {self.pk.pt:+.3f} psi_V {self.pv.pt:+.3f} Phi {self.phi.pt:+.2f} IIA {self.iia:.2f}"
                + f" b0 {self.b0a:.3f}/{self.b0b:.3f}" + ("" if self.rule else " [kappa rule fails]"))


def kdiff(a: Ex, b: Ex) -> Q:
    """Paired kappa(a) - kappa(b) over the resamples where both satisfy the rule."""
    return Q(a.kappa.pt - b.kappa.pt, a.kappa.bs - b.kappa.bs)


def s_id(cells, l0, fmt=None):
    """(point, Q of s_ID) of the natural clamp cells at onset l0 (and format)."""
    cs = [c for c in cells if c["l0"] == l0 and (fmt is None or c.get("format") == fmt)]
    if not cs:
        return None

    def D(c, row, t):
        return c["lp"][row][t] - c["lp"]["ID"][t]

    def idx(c, ch):
        return 0.5 * ((D(c, f"{ch}_S", "S") - D(c, f"{ch}_X", "S")) + (D(c, f"{ch}_X", "X") - D(c, f"{ch}_S", "X")))
    boot = boot_of(None, len(cs))
    K = Q(*boot.mean([idx(c, "K") for c in cs]))
    V = Q(*boot.mean([idx(c, "V") for c in cs]))
    with np.errstate(all="ignore"):
        den = K.pt + V.pt
        return Q(K.pt / den if den > 0 else NAN, np.where(K.bs + V.bs > 0, K.bs / (K.bs + V.bs), np.nan))


# --------------------------------------------------------------------------- J-C6
def jc6(J, test, comps=None):
    """(verdict, gate J-C-G7, lines)."""
    lines = []
    fm = J["formats"]
    by = {(c["arm"], c["format"]): [] for c in J["cells"]}
    for c in J["cells"]:
        by[(c["arm"], c["format"])].append(c)
    E = {k: Ex(v, test) for k, v in by.items()}
    g7 = bool(all(abs(E[("BIND", f)].kappa.pt - ANCHOR[f]) <= G7_TOL for f in fm if f in ANCHOR) and E[("BIND", NM)].iia >= G7_IIA)
    flip = E[("CAA", NM)].iia
    lines.append(f"J-C-G7 anchor: BIND IIA(NO-MENTION) {E[('BIND', NM)].iia:.3f} (>= {G7_IIA}); kappa_BIND "
                 + ", ".join(f"{f} {E[('BIND', f)].kappa.pt:+.3f} vs {ANCHOR.get(f, NAN):.3f}" for f in fm) + f" (within {G7_TOL}) -> {'MET' if g7 else 'NOT MET'}")
    lines.append(f"CAA flip rate under NO-MENTION {flip:.3f} (gate >= {J6_FLIP}); nu(CAA vs ID@{J['lstar']}) "
                 + ", ".join(f"{f} {E[('CAA', f)].nu:.3f}" for f in fm))
    per = {}
    for f in fm:
        caa, bind = E[("CAA", f)], E[("BIND", f)]
        lines.append(f"  {f}: CAA {caa.txt()}; BIND {bind.txt()}; ID {E[('ID', f)].txt()}")
        if not (caa.usable() and bind.usable() and flip >= J6_FLIP):   # both arms usable (b0, b2, kappa rule) decide evaluability
            per[f] = None
            lines.append(f"    not evaluable: " + ", ".join(x for x, ok in (
                ("CAA Gate b0", caa.b0), ("CAA Phi >= 3", caa.b2), ("CAA kappa rule", caa.rule), ("BIND Gate b0", bind.b0),
                ("BIND Phi >= 3", bind.b2), ("BIND kappa rule", bind.rule), ("flip >= 0.5", flip >= J6_FLIP)) if not ok))
            continue
        dk = kdiff(bind, caa)
        ok = bool(caa.kappa.pt <= J6_KAPPA and dk.pt >= J6_DIFF and dk.lower() > 0 and dk.drop <= 0.05)
        per[f] = ok
        if comps is not None:
            comps.append(component("J-C6", f"{f} H0: kappa_BIND - kappa_CAA <= 0", dk, 0.0, ">", dk.lower() > 0))
        lines.append(f"    kappa_CAA {caa.kappa.pt:+.3f} (<= {J6_KAPPA}); kappa_BIND - kappa_CAA {dk.txt()} (>= {J6_DIFF}; H0: <= 0 rejected) -> {'met' if ok else 'not met'}")
    if not g7 or per.get(NM) is None or per.get(OA) is None:
        v = None
    else:
        v = all(x for x in per.values() if x is not None)
    return v, g7, lines


# --------------------------------------------------------------------------- the screen
def screen(J, test):
    """(evaluable, window list, l_w, lines) recomputed from the screen file; MISMATCH noted if it differs from the stored."""
    lines = []
    rows = J["sweep_rows"]
    iia = {int(l): float(np.mean([r["ok"] for r in v])) for l, v in rows.items()}
    phi = {int(l): float(np.mean([r["m_patch"] - r["m_self"] for r in v])) for l, v in rows.items()}
    WB = sorted(l for l in iia if phi[l] >= WIN_PHI and iia[l] >= WIN_IIA)
    sid = {}
    for l0 in J["onsets"]:
        q = s_id(J["clamp_cells"], l0)
        sid[l0] = q
    W = [l for l in WB if (l + 1) in sid and sid[l + 1] is not None and sid[l + 1].pt >= WIN_SID]
    lw = min(W) if W else None
    n = len(J["population"])
    ev = bool((test or n >= SCREEN_N) and max(iia.values()) >= SCREEN_IIA)
    same = W == J["window"] and lw == J["l_w"]
    lines.append(f"population {n} pairs; BIND max IIA {max(iia.values()):.3f} at block {max(iia, key=iia.get)} (screen evaluable: "
                 f"max IIA >= {SCREEN_IIA}{'' if test else f' and n >= {SCREEN_N}'}) -> {'yes' if ev else 'no'}")
    lines.append("  BIND IIA/Phi per block: " + " ".join(f"{l}:{iia[l]:.2f}/{phi[l]:+.1f}" for l in sorted(iia)))
    lines.append("  s_ID(OPTIONS-AFTER, l0): " + " ".join(f"{l0}:{sid[l0].pt:+.3f}" for l0 in sorted(sid)))
    lines.append(f"  W_B (Phi >= {WIN_PHI}, IIA >= {WIN_IIA}) {WB}; window (s_ID(l+1) >= {WIN_SID}) {W}; l_w {lw}"
                 + ("" if same else "  MISMATCH with the stored window"))
    return ev, W, lw, lines, same


def window(J, test, comps=None):
    """H7 and H9 (H's thresholds) at the window depth. Returns (met or None, lines)."""
    lines = []
    lw = J["l_w"]
    by = {}
    for c in J["cells"]:
        by.setdefault(c["format"], []).append(c)
    E = {f: Ex(v, test) for f, v in by.items()}
    sid = {f: s_id(J["clamp_cells"], lw + 1, f) for f in by}
    use = {f: E[f].usable() and sid[f] is not None and np.isfinite(sid[f].pt) for f in E}
    for f in E:
        lines.append(f"  {f}: BIND@{lw} {E[f].txt()}; s_ID(f, {lw + 1}) {sid[f].txt() if sid[f] else 'nan'}; usable {use[f]}")
    if not (use.get(NM) and use.get(OA)):
        return None, lines + ["  H7/H9 not evaluable: NO-MENTION and OPTIONS-AFTER must be usable"]
    ev = [f for f in LAW if use.get(f)]
    gaps = {f: abs(E[f].kappa.pt - sid[f].pt) for f in ev}
    r = np.corrcoef([E[f].kappa.pt for f in LAW], [sid[f].pt for f in LAW])[0, 1] if len(ev) == 3 else NAN
    h7 = bool(all(g <= H7_GAP for g in gaps.values()) and np.isfinite(r) and r >= H7_R)
    d = kdiff(E[OA], E[NM])
    between = True
    if use.get(QN):
        lo, hi = sorted((E[OA].kappa.pt, E[NM].kappa.pt))
        between = lo <= E[QN].kappa.pt <= hi
    h9 = bool(d.pt >= H9_DIFF and d.lower() > 0 and d.drop <= 0.05 and between)
    if comps is not None:
        comps.append(component("J-C-WIN", f"l_w={lw} H0: kappa(OA) - kappa(NM) <= 0", d, 0.0, ">", d.lower() > 0))
    lines.append(f"  H7: gaps {', '.join(f'{f} {g:.3f}' for f, g in gaps.items())} (<= {H7_GAP}); r {r:+.3f} (>= {H7_R}) -> {'met' if h7 else 'not met'}")
    lines.append(f"  H9: kappa(OA) - kappa(NM) {d.txt()} (>= {H9_DIFF}, H0: <= 0 rejected); QNAMES between: {between} -> {'met' if h9 else 'not met'}")
    return bool(h7 and h9), lines
