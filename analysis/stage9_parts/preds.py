"""Predictions P1-P5 and POST, Direction B (B1-B3) and the falsifiers (F1-F3, B-F1, B-F2) of the stage-9 scorer, as
docs/STAGE9_PREDICTIONS.md sections 2, 3 and 5 define them (statistics: docs/stage9/SPEC.md sections 9-10).

Per model a prediction combines its parts (formats, onsets, cells) with the rule of analysis/stage9_parts/stats.combine;
over models: SUPPORTED if SUPPORTED in all three, SUPPORTED AT THE POINT if at least that in all three, NOT SUPPORTED as
soon as one model is (named), else NOT EVALUABLE. P2 (passages) is "2 of 3" at the point (stats.k_of); P5b is read at
qwen7 and mistral7 (the models with stored E4b bases). A prediction with a stories and a passages part is scored as two
lines (P1s, P1p, ...) and the line that names both (P1, ...) combines them per model. Falsifiers are read at the point
in each model (TRIGGERED / not), with the interval printed; a falsifier triggers when it triggers in any model.
Primary analysis (note section 5): generated answers, competent items (C9(f)), containment for passages; the
E-argmax / all-items / strict versions are computed by the same functions with ``how``, ``sub`` and ``rule``.
"""
from __future__ import annotations

import numpy as np

from .data import EDIT_FORMATS, nearest_onset
from .stats import (NE, NOT, PT, SUP, Q, V, auc, band, check, combine, diff, first_below, fn, k_of, missing, rate,
                    ratio, trig, wmedian)

KEYS = ("mistral7", "llama8", "qwen7")
E4B_KEYS = ("qwen7", "mistral7")
EDIT_BLOCKS, DOSE_BLOCKS = (3, 7, 11, 15), (3, 7)
P2_MAX = {"mistral7": 0.35, "llama8": 0.35, "qwen7": 0.55}
AUC_ROWS = ("E1x0.5", "E1x0.75", "E1", "E1x1.5", "E4b", "J0.25", "J0.5", "J0.65", "J0.8", "J0.9", "nat", "J1.25")
RK_RULE = 0.8

TITLES = {
    "P1": "free form stays open: late flip >= 0.95 at every l <= 0.5 L in NONE, BEFORE, POST-NULL (stories); >= 0.8 in "
          "NOM and OPTB at x = 0.15, 0.3, 0.45 (passages)",
    "P1s": "stories: late flip >= 0.95 at every l <= 0.5 L in NONE, BEFORE and POST-NULL, each model",
    "P1p": "passages: late flip >= 0.8 in NOM and OPTB at x = 0.15, 0.3, 0.45, each model",
    "P2": "listed closes: l_c(AFTER)/L, l_c(LETTER)/L <= 0.35 (Mistral, Llama) / 0.55 (Qwen), [l_c(NONE) - l_c(AFTER)]/L "
          ">= 0.15; passages OPTA late flip at round(0.3 L) <= 0.4 in 2 of 3 models",
    "P2s": "stories: l_c(AFTER)/L and l_c(LETTER)/L <= 0.35 (mistral7, llama8) / 0.55 (qwen7); [l_c(NONE) - l_c(AFTER)]/L "
           ">= 0.15, each model",
    "P2p": "passages: OPTA late flip at round(0.3 L) <= 0.4 in at least 2 of 3 models (counted at the point)",
    "P3": "the crossover: early flip at l_b >= 0.7 in AFTER and LETTER, <= 0.05 in NONE and BEFORE (stories); early "
          "window to round(0.3 L): OPTA >= 0.5, NOM <= 0.1, OPTB <= 0.1 (passages)",
    "P3s": "stories, at l_b: early flip >= 0.7 in AFTER and in LETTER, <= 0.05 in NONE and in BEFORE, each model",
    "P3p": "passages, early window to round(0.3 L): OPTA >= 0.5, NOM <= 0.1, OPTB <= 0.1, each model",
    "P4": "silent: at l_b in AFTER >= 0.8 of the non-flipped answers are B, each model; at Mistral the median absolute "
          "p(B) of the B answers >= 0.8",
    "P5a": "single-site E1 tracks the natural late patch: |flip(E1) - flip(nat)| <= 0.1 in every format x block cell, "
           "each model",
    "P5b": "E4b fails with the list where the window is open: tau = flip(AFTER)/flip(NONE) <= 0.7 at every block where "
           "E4b's NONE flip >= 0.5 (qwen7, mistral7; precondition tau_nat(3) >= 0.9)",
    "POST": "scope: POST late flip >= 0.8 at every l <= 0.5 L, each model",
    "B1": "block 3: joint interpolation at alpha = 0.5 flips <= 0.35 of AFTER and >= 0.9 of NONE answers, each model",
    "B2": "block 3: E1 at 0.5x flips >= 0.9 of NONE and <= 0.5 of AFTER answers, each model",
    "B3": "r_K predicts AFTER transfer: AUC >= 0.8 over the E1 multipliers, E4b and the joint-alpha rows in the cells "
          "where the natural patch transfers, each model",
}
FTITLES = {
    "F1": "early flip at l_b >= 0.2 in NONE or in BEFORE",
    "F2s": "[l_c(BEFORE) - l_c(AFTER)]/L < 0.1",
    "F2p": "passages: late flip OPTB - OPTA at round(0.3 L) < 0.1",
    "F3": "[l_c(NONE) - l_c(AFTER)]/L < 0.1",
    "B-F1": "joint alpha = 0.5 AFTER flip >= 0.6 at block 3",
    "B-F2": "psi_V(NONE, 0.5) - psi_K(AFTER, 0.5) <= 0.1 at block 3 (E, nats)",
}


class Model:
    """One model's runs (any may be None) and its recomputed l_b."""

    def __init__(self, key, st=None, lb=None, ed=None, ps=None, leta=None, l_b=None):
        self.key, self.st, self.lb, self.ed, self.ps, self.leta, self.l_b = key, st, lb, ed, ps, leta, l_b


class Ctx:
    """The models, the check log, the pytest gate and the registry of every number printed."""

    def __init__(self, models: dict, log, gate, test: bool, fetch_failed: str = ""):
        self.models, self.log, self.gate, self.test = models, log, gate, test
        self.fetch_failed = fetch_failed
        self.nums: dict = {}
        self.res: dict = {}       # id -> (combined level, {model: V})
        self.fal: dict = {}       # id -> (True / False / None, {model: (True / False / None, text)})
        self.p5c: dict = {}
        self._lc: dict = {}
        self._q: dict = {}

    # ------------------------------------------------------------------ bookkeeping
    def ne(self, key: str, arm: str) -> str:
        why = []
        if self.log.run:   # the run-level checks, the FP32 gate among them (outside TEST)
            why.append("run-level check failed (" + "; ".join(self.log.run)[:200] + ")")
        m = self.models.get(key)
        f = None if m is None else {"stories": m.st, "edits": m.ed, "passages": m.ps}[arm]
        if f is None:
            ff = f" (FETCH_FAILED.txt: {key})" if key in self.fetch_failed else ""
            why.append(f"no {arm} results for {key}{ff}")
        bad = self.log.why(key, arm)
        if bad:
            why.append("a check failed: " + "; ".join(bad)[:300])
        part = self.log.why_partial(key, arm)
        if part:
            why.append("partial at the deadline (not a MISMATCH): " + "; ".join(part)[:300])
        return "; ".join(why)

    def rec(self, arm: str, key: str, stat: str, q: Q) -> Q:
        self.nums[f"{arm}/{key}/{stat}"] = q.as_dict()
        return q

    # ------------------------------------------------------------------ stories
    def _memo(self, k, make):
        if k not in self._q:
            self._q[k] = make()
        return self._q[k]

    def sflip(self, m, fmt, row, who="S", sub="comp", how="gen") -> Q:
        return self._memo(("s", m.key, fmt, row, who, sub, how), lambda: self._sflip(m, fmt, row, who, sub, how))

    def _sflip(self, m, fmt, row, who, sub, how) -> Q:
        S = m.st
        if S is None or not S.has(fmt):
            return missing(f"{fmt} not run")
        if row not in S.rows:
            return missing(f"no row {row}")
        q = rate(S.B, S.is_(fmt, row, who, how), S.mask(fmt, sub))
        tag = ("flip" if who == "S" else "stale" if who == "B" else who) + ("" if how == "gen" else "_E") + ("" if sub == "comp" else "_all")
        return self.rec("stories", m.key, f"{fmt}:{row}:{tag}", q)

    def lc(self, m, fmt, sub="comp", how="gen"):
        """(l_c / L as a Q, the point l_c in blocks or None): the first onset with late flip < 0.5, L if none."""
        k = (m.key, fmt, sub, how)
        if k in self._lc:
            return self._lc[k]
        S = m.st
        if S is None or not S.has(fmt):
            r = (missing(f"{fmt} not run"), None)
        else:
            on, L = S.onsets, S.L
            msk = S.mask(fmt, sub)
            if msk.sum() == 0:
                r = (missing(f"no {'competent ' if sub == 'comp' else ''}items in {fmt}"), None)
            else:
                X = np.stack([S.is_(fmt, f"late@{l}", "S", how) for l in on], 1) * msk[:, None]
                pt = X.sum(0) / msk.sum()
                with np.errstate(all="ignore"):
                    den = S.B.W @ msk
                    F = np.where(den[:, None] > 0, (S.B.W @ X) / np.where(den > 0, den, 1.0)[:, None], np.nan)
                lpt = float(first_below(pt, on, L))
                q = Q(lpt / L, first_below(F, on, L) / L, int(msk.sum()))
                tag = "lc_frac" + ("" if how == "gen" else "_E") + ("" if sub == "comp" else "_all")
                r = (self.rec("stories", m.key, f"{fmt}:{tag}", q), int(lpt))
        self._lc[k] = r
        return r

    def silent(self, m, fmt, row, sub="comp"):
        """(share B among the non-S answers, median absolute p(B) of the B answers, median six-way p(B) of them)."""
        S = m.st
        if S is None or not S.has(fmt) or row not in S.rows:
            x = missing(f"{fmt} {row} not run")
            return x, x, x
        msk = S.mask(fmt, sub)
        notS = msk * (1 - S.is_(fmt, row, "S"))
        isB = S.is_(fmt, row, "B")
        a = self.rec("stories", m.key, f"{fmt}:{row}:silent_share", rate(S.B, isB, notS))
        selB = (msk * isB) > 0
        b = self.rec("stories", m.key, f"{fmt}:{row}:median_abs_pB", wmedian(S.B, S.absp(fmt, row, "B"), selB))
        c = self.rec("stories", m.key, f"{fmt}:{row}:median_six_pB", wmedian(S.B, S.sixp(fmt, row, "B"), selB))
        return a, b, c

    # ------------------------------------------------------------------ passages
    def pflip(self, m, split, fmt, row, Y="S", sub="comp", rule="contain", run=None) -> Q:
        return self._memo(("p", m.key, id(run), split, fmt, row, Y, sub, rule),
                          lambda: self._pflip(m, split, fmt, row, Y, sub, rule, run))

    def _pflip(self, m, split, fmt, row, Y, sub, rule, run) -> Q:
        R = (run or m.ps)
        sp = R.split(split) if R is not None else None
        if sp is None or not sp.has(fmt):
            return missing(f"{split}/{fmt} not run")
        if row not in R.rows:
            return missing(f"no row {row}")
        q = rate(sp.B, sp.is_(fmt, row, Y, rule), sp.mask(fmt, sub, rule))
        tag = ("flip" if Y == "S" else "stale" if Y == "B" else Y) + {"contain": "", "strict": "_strict", "lp": "_lp"}[rule] \
            + ("" if sub == "comp" else "_all")
        return self.rec("passages", m.key, f"{split}:{fmt}:{row}:{tag}", q)

    # ------------------------------------------------------------------ edits
    def eflip(self, m, fmt, l, Z, who="t", how="gen") -> Q:
        return self._memo(("e", m.key, fmt, l, Z, who, how), lambda: self._eflip(m, fmt, l, Z, who, how))

    def _eflip(self, m, fmt, l, Z, who, how) -> Q:
        E = m.ed
        if E is None or not E.has_row(fmt, l, Z):
            return missing(f"{Z} at {fmt}@{l} not run")
        q = rate(E.B, E.hit(fmt, l, Z, who, how))
        tag = ("flip" if who == "t" else "stale") + ("" if how == "gen" else "_E")
        return self.rec("edits", m.key, f"{fmt}@{l}:{Z}:{tag}", q)

    def tau(self, m, l, Z) -> Q:
        return self._memo(("t", m.key, l, Z), lambda: self._tau(m, l, Z))

    def _tau(self, m, l, Z) -> Q:
        E = m.ed
        if E is None or not (E.has_row("AFTER", l, Z) and E.has_row("NONE", l, Z)):
            return missing(f"{Z} at block {l} not run in AFTER and NONE")
        return self.rec("edits", m.key, f"{l}:{Z}:tau", ratio(E.B, E.hit("AFTER", l, Z), E.hit("NONE", l, Z)))

    def psi(self, m, fmt, l, kind, a, sg="E") -> Q:
        return self._memo(("psi", m.key, fmt, l, kind, a, sg), lambda: self._psi(m, fmt, l, kind, a, sg))

    def _psi(self, m, fmt, l, kind, a, sg="E") -> Q:
        """psi_K = ID(K{a}) / ID(K1), psi_V = ID(V{a}) / ID(V1), psi_J = ID(J{a}) / ID(nat) (ratios of means)."""
        E = m.ed
        num, den = {"K": (f"K{a}", "K1"), "V": (f"V{a}", "V1"), "J": (f"J{a}", "nat")}[kind]
        if E is None or not (E.has_row(fmt, l, num) and E.has_row(fmt, l, den)):
            return missing(f"{num} / {den} at {fmt}@{l} not run")
        q = ratio(E.B, E.ident(fmt, l, num, sg), E.ident(fmt, l, den, sg))
        return self.rec("dose", m.key, f"{fmt}@{l}:psi_{kind}({a}):{'nats' if sg == 'E' else 'logit'}", q)


# --------------------------------------------------------------------------- the lines
def _model_v(parts, label="") -> V:
    lv = combine([p.level for p in parts])
    why = "; ".join(sorted({p.why for p in parts if p.level == NE and p.why}))
    return V(lv, label, parts, why)


def _guard(ctx, key, arm) -> V | None:
    why = ctx.ne(key, arm)
    return V(NE, "", why=why) if why else None


def p1s(ctx, m, how="gen", sub="comp", fmts=("NONE", "BEFORE", "POST-NULL"), thr=0.95) -> V:
    g = _guard(ctx, m.key, "stories")
    if g:
        return g
    S = m.st
    parts = []
    for f in fmts:
        for l in [l for l in S.onsets if 2 * l <= S.L]:
            parts.append(check(ctx.sflip(m, f, f"late@{l}", sub=sub, how=how), ">=", thr, f"{f} late@{l}"))
    return _model_v(parts)


def post(ctx, m, how="gen", sub="comp") -> V:
    return p1s(ctx, m, how, sub, fmts=("POST",), thr=0.8)


def p1p(ctx, m, split="E", rule="contain", sub="comp") -> V:
    g = _guard(ctx, m.key, "passages")
    if g:
        return g
    R = m.ps
    parts = []
    for f in ("NOM", "OPTB"):
        for l in R.onsets[:3]:
            parts.append(check(ctx.pflip(m, split, f, f"late@{l}", sub=sub, rule=rule), ">=", 0.8, f"{split} {f} late@{l}"))
    return _model_v(parts)


def p2s(ctx, m, how="gen", sub="comp") -> V:
    g = _guard(ctx, m.key, "stories")
    if g:
        return g
    thr = P2_MAX.get(m.key, 0.35)
    qa, la = ctx.lc(m, "AFTER", sub, how)
    ql, ll = ctx.lc(m, "LETTER", sub, how)
    qn, ln = ctx.lc(m, "NONE", sub, how)
    gap = ctx.rec("stories", m.key, "NONE-AFTER:lc_gap_frac" + ("" if how == "gen" else "_E") + ("" if sub == "comp" else "_all"),
                  diff(qn, qa))
    L = m.st.L
    cen = lambda l: "" if l is None else (f" (l_c {l}" + (", censored: > last onset" if l == L else "") + ")")  # noqa: E731
    parts = [check(qa, "<=", thr, "l_c(AFTER)/L" + cen(la)), check(ql, "<=", thr, "l_c(LETTER)/L" + cen(ll)),
             check(gap, ">=", 0.15, "[l_c(NONE) - l_c(AFTER)]/L" + cen(ln).replace("l_c", "l_c(NONE)"))]
    return _model_v(parts)


def p2p(ctx, m, split="E", rule="contain", sub="comp") -> V:
    g = _guard(ctx, m.key, "passages")
    if g:
        return g
    l = m.ps.onsets[1]
    return _model_v([check(ctx.pflip(m, split, "OPTA", f"late@{l}", sub=sub, rule=rule), "<=", 0.4, f"{split} OPTA late@{l}")])


def p3s(ctx, m, how="gen", sub="comp") -> V:
    g = _guard(ctx, m.key, "stories")
    if g:
        return g
    if m.l_b is None:
        return V(NE, "", why="l_b is None (no onset with late AFTER flip <= 0.2 on C, or no lb file)")
    r = f"early@{m.l_b}"
    parts = [check(ctx.sflip(m, "AFTER", r, sub=sub, how=how), ">=", 0.7, f"AFTER {r}"),
             check(ctx.sflip(m, "LETTER", r, sub=sub, how=how), ">=", 0.7, f"LETTER {r}"),
             check(ctx.sflip(m, "NONE", r, sub=sub, how=how), "<=", 0.05, f"NONE {r}"),
             check(ctx.sflip(m, "BEFORE", r, sub=sub, how=how), "<=", 0.05, f"BEFORE {r}")]
    return _model_v(parts)


def p3p(ctx, m, split="E", rule="contain", sub="comp") -> V:
    g = _guard(ctx, m.key, "passages")
    if g:
        return g
    r = f"early@{m.ps.onsets[1]}"
    return _model_v([check(ctx.pflip(m, split, "OPTA", r, sub=sub, rule=rule), ">=", 0.5, f"{split} OPTA {r}"),
                     check(ctx.pflip(m, split, "NOM", r, sub=sub, rule=rule), "<=", 0.1, f"{split} NOM {r}"),
                     check(ctx.pflip(m, split, "OPTB", r, sub=sub, rule=rule), "<=", 0.1, f"{split} OPTB {r}")])


def p4(ctx, m, sub="comp") -> V:
    g = _guard(ctx, m.key, "stories")
    if g:
        return g
    if m.l_b is None:
        return V(NE, "", why="l_b is None")
    r = f"late@{m.l_b}"
    share, med, _ = ctx.silent(m, "AFTER", r, sub)
    parts = [check(share, ">=", 0.8, f"AFTER {r}: share B among the non-S answers")]
    if m.key == "mistral7":
        parts.append(check(med, ">=", 0.8, f"AFTER {r}: median absolute p(B) of the B answers"))
    return _model_v(parts)


def p5a(ctx, m) -> V:
    g = _guard(ctx, m.key, "edits")
    if g:
        return g
    parts = []
    for f in EDIT_FORMATS:
        for l in EDIT_BLOCKS:
            d = ctx.rec("edits", m.key, f"{f}@{l}:E1-nat:flip_diff", diff(ctx.eflip(m, f, l, "E1"), ctx.eflip(m, f, l, "nat")))
            parts.append(band(d, -0.1, 0.1, f"{f}@{l} flip(E1) - flip(nat)"))
    return _model_v(parts)


def p5b(ctx, m) -> V:
    g = _guard(ctx, m.key, "edits")
    if g:
        return g
    pre = ctx.tau(m, 3, "nat")
    if not pre.ok():
        return V(NE, "", why=f"precondition tau_nat(3) not evaluable: {pre.reason()}")
    if pre.pt < 0.9 - 1e-9:
        return V(NE, f"precondition tau_nat(3) {pre.txt()} < 0.9", why=f"precondition tau_nat(3) = {pre.pt:.3f} < 0.9")
    parts, used = [], []
    for l in EDIT_BLOCKS:
        nf = ctx.eflip(m, "NONE", l, "E4b")
        if nf.ok() and nf.pt >= 0.5 - 1e-9:
            used.append(l)
            parts.append(check(ctx.tau(m, l, "E4b"), "<=", 0.7, f"tau_E4b({l}) (E4b NONE flip {nf.pt:.3f} >= 0.5)"))
    if not parts:
        return V(NE, f"precondition tau_nat(3) {pre.txt()}", why="no block where E4b's NONE flip >= 0.5")
    return _model_v(parts, f"precondition tau_nat(3) {pre.txt()} >= 0.9; blocks {used}")


def p5c(ctx, m):
    """(TRANSFERS / DOES NOT TRANSFER / NOT EVALUABLE, text): CAA's AFTER flip >= 0.5 at block 11 or 15 where the
    natural patch's AFTER flip <= 0.2 (points)."""
    why = ctx.ne(m.key, "edits")
    if why:
        return None, why
    parts, hit = [], False
    for l in (11, 15):
        nat, caa = ctx.eflip(m, "AFTER", l, "nat"), ctx.eflip(m, "AFTER", l, "CAA")
        if not (nat.ok() and caa.ok()):
            parts.append(f"block {l}: not evaluable ({nat.reason() or caa.reason()})")
            continue
        if nat.pt <= 0.2 + 1e-9:
            t = caa.pt >= 0.5 - 1e-9
            hit |= t
            parts.append(f"block {l}: nat AFTER {nat.txt()} <= 0.2, CAA AFTER {caa.txt()} -> {'>= 0.5' if t else '< 0.5'}")
        else:
            parts.append(f"block {l}: nat AFTER {nat.txt()} > 0.2 (the window is open; not read)")
    read = [p for p in parts if "<= 0.2" in p]
    if hit:
        return True, "; ".join(parts)
    if read:
        return False, "; ".join(parts)
    return None, "; ".join(parts) + "; no block with the natural AFTER flip <= 0.2"


def b1(ctx, m) -> V:
    g = _guard(ctx, m.key, "edits")
    if g:
        return g
    return _model_v([check(ctx.eflip(m, "AFTER", 3, "J0.5"), "<=", 0.35, "AFTER@3 J0.5"),
                     check(ctx.eflip(m, "NONE", 3, "J0.5"), ">=", 0.9, "NONE@3 J0.5")])


def b2(ctx, m) -> V:
    g = _guard(ctx, m.key, "edits")
    if g:
        return g
    return _model_v([check(ctx.eflip(m, "NONE", 3, "E1x0.5"), ">=", 0.9, "NONE@3 E1x0.5"),
                     check(ctx.eflip(m, "AFTER", 3, "E1x0.5"), "<=", 0.5, "AFTER@3 E1x0.5")])


def auc_units(ctx, m):
    """The units of B3 (SPEC 9.7): (core, r_K, y) for Z in AUC_ROWS at the dose blocks where the natural AFTER flip
    >= 0.9 (point), t in (S, X), restricted to the units whose NONE answer is t; y = 1[AFTER answer = t]."""
    E = m.ed
    cells, core, r, y = [], [], [], []
    for l in DOSE_BLOCKS:
        nat = ctx.eflip(m, "AFTER", l, "nat")
        if not nat.ok() or nat.pt < 0.9 - 1e-9:
            continue
        cells.append(l)
        for Z in AUC_ROWS:
            if not (E.has_row("AFTER", l, Z) and E.has_row("NONE", l, Z)):
                continue
            for t in ("S", "X"):
                keep = E.unit_hit("NONE", l, Z, t) > 0
                rk = E.rk_unit(l, Z, t)
                yy = E.unit_hit("AFTER", l, Z, t) > 0
                for i in np.flatnonzero(keep & np.isfinite(rk)):
                    core.append(i)
                    r.append(rk[i])
                    y.append(bool(yy[i]))
    return cells, np.array(core, int), np.array(r, float), np.array(y, bool)


def b3(ctx, m):
    """(V, AUC Q, fixed-rule accuracy Q, text)."""
    g = _guard(ctx, m.key, "edits")
    if g:
        return g, None, None, ""
    cells, core, r, y = auc_units(ctx, m)
    if not cells:
        return V(NE, "", why="no dose block where the natural AFTER flip >= 0.9"), None, None, "no transfer cell"
    E = m.ed
    q, npos, nneg = auc(E.B, core, r, y)
    ctx.rec("edits", m.key, "B3:auc", q)
    x = np.zeros(E.n)
    n = np.zeros(E.n)
    np.add.at(x, core, ((r >= RK_RULE - 1e-9) == y).astype(float))
    np.add.at(n, core, 1.0)
    acc = ctx.rec("edits", m.key, "B3:rule_accuracy", ratio(E.B, x, n)) if n.sum() else missing("no units")
    txt = f"cells (blocks) {cells}; {len(r)} units, {npos} positives, {nneg} negatives; rule r_K >= 0.8 accuracy {acc.txt()}"
    return _model_v([check(q, ">=", 0.8, "AUC of r_K")], txt), q, acc, txt


# --------------------------------------------------------------------------- falsifiers
def f1(ctx, m):
    if ctx.ne(m.key, "stories") or m.l_b is None:
        return None, ctx.ne(m.key, "stories") or "l_b is None"
    r = f"early@{m.l_b}"
    qs = [ctx.sflip(m, f, r) for f in ("NONE", "BEFORE")]
    ts = [trig(q, ">=", 0.2) for q in qs]
    t = True if True in ts else (None if None in ts else False)
    return t, "; ".join(f"{f} {r} {q.txt()}" for f, q in zip(("NONE", "BEFORE"), qs))


def f2s(ctx, m):
    if ctx.ne(m.key, "stories"):
        return None, ctx.ne(m.key, "stories")
    q = ctx.rec("stories", m.key, "BEFORE-AFTER:lc_gap_frac", diff(ctx.lc(m, "BEFORE")[0], ctx.lc(m, "AFTER")[0]))
    return trig(q, "<", 0.1), f"[l_c(BEFORE) - l_c(AFTER)]/L {q.txt()}"


def f2p(ctx, m):
    if ctx.ne(m.key, "passages"):
        return None, ctx.ne(m.key, "passages")
    l = m.ps.onsets[1]
    q = ctx.rec("passages", m.key, f"E:OPTB-OPTA:late@{l}:flip_diff",
                diff(ctx.pflip(m, "E", "OPTB", f"late@{l}"), ctx.pflip(m, "E", "OPTA", f"late@{l}")))
    return trig(q, "<", 0.1), f"OPTB - OPTA late@{l} {q.txt()}"


def f3(ctx, m):
    if ctx.ne(m.key, "stories"):
        return None, ctx.ne(m.key, "stories")
    q = ctx.rec("stories", m.key, "NONE-AFTER:lc_gap_frac", diff(ctx.lc(m, "NONE")[0], ctx.lc(m, "AFTER")[0]))
    return trig(q, "<", 0.1), f"[l_c(NONE) - l_c(AFTER)]/L {q.txt()}"


def bf1(ctx, m):
    if ctx.ne(m.key, "edits"):
        return None, ctx.ne(m.key, "edits")
    q = ctx.eflip(m, "AFTER", 3, "J0.5")
    return trig(q, ">=", 0.6), f"AFTER@3 J0.5 flip {q.txt()}"


def bf2(ctx, m):
    if ctx.ne(m.key, "edits"):
        return None, ctx.ne(m.key, "edits")
    q = ctx.rec("dose", m.key, "B-F2:psiV(NONE,3,0.5)-psiK(AFTER,3,0.5)",
                diff(ctx.psi(m, "NONE", 3, "V", 0.5), ctx.psi(m, "AFTER", 3, "K", 0.5)))
    return trig(q, "<=", 0.1), f"psi_V(NONE, 3, 0.5) - psi_K(AFTER, 3, 0.5) {q.txt()}"


# --------------------------------------------------------------------------- running and printing
LINES = (("P1s", p1s, KEYS, "every", "stories"), ("P1p", p1p, KEYS, "every", "passages"),
         ("P2s", p2s, KEYS, "every", "stories"), ("P2p", p2p, KEYS, 2, "passages"), ("P3s", p3s, KEYS, "every", "stories"),
         ("P3p", p3p, KEYS, "every", "passages"), ("P4", p4, KEYS, "every", "stories"), ("P5a", p5a, KEYS, "every", "edits"),
         ("P5b", p5b, E4B_KEYS, "every", "edits"), ("POST", post, KEYS, "every", "stories"))
PAIRS = {"P1": ("P1s", "P1p"), "P2": ("P2s", "P2p"), "P3": ("P3s", "P3p")}
B_LINES = (("B1", b1, KEYS, "every", "edits"), ("B2", b2, KEYS, "every", "edits"))
FALS = (("F1", f1, "stories"), ("F2s", f2s, "stories"), ("F2p", f2p, "passages"), ("F3", f3, "stories"),
        ("B-F1", bf1, "edits"), ("B-F2", bf2, "edits"))


def comb_models(per: dict, rule) -> str:
    lv = [v.level for v in per.values()]
    return combine(lv) if rule == "every" else k_of(lv, rule)


def run_line(ctx, code, f, keys, comb_rule, out, arm="stories", quiet=False, **kw):
    """One line over its models: per-model verdicts (printed with their parts unless quiet) and the combined level."""
    per = {}
    for k in keys:
        m = ctx.models.get(k)
        if m is None:
            per[k] = V(NE, "", why=ctx.ne(k, arm))
            continue
        try:
            per[k] = f(ctx, m, **kw)
        except Exception as ex:  # noqa: BLE001  (a missing record makes this model's line not evaluable)
            per[k] = V(NE, "", why=f"not computable: {type(ex).__name__}: {ex}")
    lv = comb_models(per, comb_rule)
    out(f"  {code} {TITLES.get(code, '')} -> {lv}" + named(per, lv) + ("" if comb_rule == "every" else f" [{comb_rule} of 3 at "
                                                                          "the point]"))
    if quiet:
        out("      " + "; ".join(f"{k}: {v.level}" + (f" ({v.why[:120]})" if v.level == NE and v.why else "") for k, v in per.items()))
        return lv, per
    for k, v in per.items():
        out(f"      {k}: {v.level}" + (f" ({v.why})" if v.level == NE and v.why else "") + (f"  {v.label}" if v.label else ""))
        for p in v.parts:
            out(f"          {p.label}: {p.level}" + (f" ({p.why})" if p.level == NE and p.why else ""))
    return lv, per


def named(per, lv) -> str:
    if lv == NOT:
        return " (NOT SUPPORTED in: " + ", ".join(k for k, v in per.items() if v.level == NOT) + ")"
    if lv == NE:
        ne = [k for k, v in per.items() if v.level == NE]
        return f" ({len(per) - len(ne)} of {len(per)} evaluable; not evaluable: {', '.join(ne)})"
    return ""


P2_PER_MODEL = ("per model: the stories part P2s alone; the passages part P2p is decided over the models (2 of 3 at "
                "the point), not per model")


def predictions(ctx, out):
    for code, f, keys, rule, arm in LINES:
        ctx.res[code] = run_line(ctx, code, f, keys, rule, out, arm=arm)
    for code, (a, b) in PAIRS.items():
        (la, pa), (lb, pb) = ctx.res[a], ctx.res[b]
        # each part combined over the models by its own rule (P2p: 2 of 3), then the parts. Per model (reported): both
        # parts of that model where both are "each model" (P1, P3: the same verdict as combining the per-model
        # verdicts); for P2 the stories part alone, because P2p is a "2 of 3" line whose verdict exists only over the
        # models (a model whose OPTA flip is above 0.4 does not make P2 fail when two others hold it), with that
        # model's P2p level printed beside it
        if LINES_RULE[b] == "every":
            per = {k: V(combine([pa[k].level, pb[k].level])) for k in pa}
        else:
            per = {k: V(pa[k].level, f"P2s {pa[k].level}; P2p here {pb[k].level} (counted 2 of 3 over the models)",
                        why=pa[k].why) for k in pa}
        lv = combine([la, lb])
        ctx.res[code] = (lv, per)
        out(f"  {code} {TITLES[code]} -> {lv} ({a} {la}; {b} {lb}" + ("" if LINES_RULE[b] == "every" else
                                                                       f", {LINES_RULE[b]} of 3 at the point") + ")")
        if LINES_RULE[b] != "every":
            out(f"      {P2_PER_MODEL}: " + "; ".join(f"{k} {v.level} (P2p here {pb[k].level})" for k, v in per.items()))


LINES_RULE = {code: rule for code, _, _, rule, _ in LINES}


def p5c_lines(ctx, out):
    out("  P5c all-position steering (CAA, E1 at every position >= p), reported either way: it transfers if its AFTER "
        "flip >= 0.5 at block 11 or 15 where the natural patch's AFTER flip <= 0.2")
    for k in KEYS:
        m = ctx.models.get(k)
        if m is None:
            ctx.p5c[k] = (None, ctx.ne(k, "edits"))
        else:
            try:
                ctx.p5c[k] = p5c(ctx, m)
            except Exception as ex:  # noqa: BLE001
                ctx.p5c[k] = (None, f"not computable: {type(ex).__name__}: {ex}")
        t, txt = ctx.p5c[k]
        out(f"      {k}: {'TRANSFERS' if t else 'DOES NOT TRANSFER' if t is False else 'NOT EVALUABLE'}  {txt}")


def falsifiers(ctx, out):
    for code, f, arm in FALS:
        per = {}
        for k in KEYS:
            m = ctx.models.get(k)
            try:
                per[k] = (None, ctx.ne(k, arm)) if m is None else f(ctx, m)
            except Exception as ex:  # noqa: BLE001
                per[k] = (None, f"not computable: {type(ex).__name__}: {ex}")
        ts = [t for t, _ in per.values()]
        tot = True if True in ts else (False if all(t is False for t in ts) else None)
        ctx.fal[code] = (tot, per)
        word = {True: "TRIGGERED", False: "not triggered", None: "NOT EVALUABLE"}
        trig_in = [k for k, (t, _) in per.items() if t]
        nev = [k for k, (t, _) in per.items() if t is None]
        out(f"  {code} {FTITLES[code]} -> {word[tot]}" + (f" (in {', '.join(trig_in)})" if trig_in else "")
            + (f" (not evaluable in {', '.join(nev)})" if nev and not trig_in else ""))
        for k, (t, txt) in per.items():
            out(f"      {k}: {word[t]}  {txt}")
    t2 = [ctx.fal["F2s"][0], ctx.fal["F2p"][0]]
    ctx.fal["F2"] = (True if True in t2 else (False if all(x is False for x in t2) else None), {})
    out(f"  F2 (stories or passages) -> {({True: 'TRIGGERED', False: 'not triggered', None: 'NOT EVALUABLE'})[ctx.fal['F2'][0]]}")


def direction_b(ctx, out):
    for code, f, keys, rule, arm in B_LINES:
        ctx.res[code] = run_line(ctx, code, f, keys, rule, out, arm=arm)
    per, extra = {}, {}
    for k in KEYS:
        m = ctx.models.get(k)
        if m is None:
            per[k] = V(NE, "", why=ctx.ne(k, "edits"))
            continue
        try:
            v, q, acc, txt = b3(ctx, m)
        except Exception as ex:  # noqa: BLE001
            v, txt = V(NE, "", why=f"not computable: {type(ex).__name__}: {ex}"), ""
        per[k], extra[k] = v, txt
    lv = combine([v.level for v in per.values()])
    ctx.res["B3"] = (lv, per)
    out(f"  B3 {TITLES['B3']} -> {lv}" + named(per, lv))
    for k, v in per.items():
        out(f"      {k}: {v.level}" + (f" ({v.why})" if v.level == NE and v.why else "") + (f"  {extra.get(k, '')}" if extra.get(k) else ""))
        for p in v.parts:
            out(f"          {p.label}: {p.level}")


# --------------------------------------------------------------------------- the pre-written readings (note section 6)
def readings(ctx) -> list[str]:
    R, Fz = ctx.res, ctx.fal
    lv = lambda c: R.get(c, (NE, {}))[0]  # noqa: E731
    tr = lambda c: Fz.get(c, (None, {}))[0]  # noqa: E731
    pt = lambda c: lv(c) in (SUP, PT)  # noqa: E731
    w = lambda c: {True: "TRIGGERED", False: "not triggered", None: "NOT EVALUABLE"}[tr(c)]  # noqa: E731
    out = []
    fal_ok = all(tr(c) is False for c in ("F1", "F2", "F3"))
    if tr("F2") is True:
        out.append("F2 triggers: report only the evaluation warning (\"intervention verdicts depend on the answer format\"), "
                   "in the appendix, with no mechanism claim.")
    elif pt("P1s") and pt("P2s") and pt("P3s") and not fal_ok and not any(tr(c) is True for c in ("F1", "F2", "F3")):
        # the note's first reading needs F1-F3 "not triggered"; a falsifier that could not be read in some model is not
        # that, so the reading is withheld (never selected by default) and the missing parts are named
        miss = []
        for c in ("F1", "F2s", "F2p", "F3"):
            ks = [k for k, (t, _) in Fz.get(c, (None, {}))[1].items() if t is None]
            if ks:
                miss.append(f"{c} not evaluable in {', '.join(ks)}")
        out.append("P1, P2, P3 (stories) hold at least at the point and no evaluable falsifier triggers, but F1 " + w("F1")
                   + ", F2 " + w("F2") + ", F3 " + w("F3") + " (" + ("; ".join(miss) or "the falsifiers were not computed") + "): the note's reading needs F1-F3 "
                   "not triggered, so it is withheld (not selected); describe the outcome per model and name the falsifier "
                   "parts that could not be read.")
    elif pt("P1s") and pt("P2s") and pt("P3s") and fal_ok:
        nums = []
        for k in KEYS:
            m = ctx.models.get(k)
            if m is None or m.st is None:
                continue
            q, l = ctx.lc(m, "AFTER")
            if l is not None and l < m.st.L:
                st = ctx.sflip(m, "AFTER", f"late@{l}", who="B")
                nums.append(f"{k}: l_c(AFTER)/L {q.pt:.2f}, AFTER answers unchanged (B) at late@{l} {st.pt:.2f}")
        out.append("P1, P2, P3 (stories) at least at the point and F1-F3 not triggered: \"When the candidates are listed "
                   "after the context, a model has finished using a context token by l_c(AFTER)/L of its depth: "
                   "overwriting everything the token holds from there on leaves X-Y % of answers unchanged, with "
                   "confidence, though the same overwrite changes every free-form answer, and overwriting only the layers "
                   "before it does the reverse.\" Numbers: " + ("; ".join(nums) or "none"))
        if pt("P5a") and pt("P5b"):
            out.append("P5a and P5b at least at the point: add the edits clause.")
    elif pt("P1s") and pt("P2s") and (lv("P3s") == NOT or tr("F1") is True):
        out.append("P1, P2 hold, P3 fails or F1 triggers: \"Late edits fail silently with listed candidates; the early "
                   "window also moves free-form answers, so the routes overlap in depth.\"")
    else:
        out.append("No pre-written reading of the window claim applies (P1s " + lv("P1s") + ", P2s " + lv("P2s") + ", P3s "
                   + lv("P3s") + "; F1 " + w("F1") + ", F2 " + w("F2") + ", F3 " + w("F3")
                   + "): describe the outcome per model.")
    if any(t for t, _ in ctx.p5c.values()):
        out.append("P5c transfers: state the claim for single-site edits, patching and KV-cache edits only.")
    if all(pt(c) for c in ("B1", "B2", "B3")) and tr("B-F1") is False and tr("B-F2") is False:
        out.append("B1-B3 hold and B-F1, B-F2 not triggered: \"an edit at a context token reaches a listed answer only if it "
                   "delivers about 80 % or more of the natural key displacement; free-form answers follow partial edits.\"")
    else:
        out.append("Direction B: describe the convexity without a threshold claim (B1 " + lv("B1") + ", B2 " + lv("B2")
                   + ", B3 " + lv("B3") + "; B-F1 " + w("B-F1") + ", B-F2 " + w("B-F2")
                   + "). The failed channel-ratio law stays reported as failed.")
    return out
