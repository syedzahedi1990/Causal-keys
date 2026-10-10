"""Statistics of the stage-8 part-D scorer (P-2026-10-10-J part D): the story bootstrap, estimates as (point, resamples),
interval criteria as one-sided tests of named nulls, the interval components handed to the shared Holm helper
(analysis/stage8_holm.py, decision D2), verdict words and the combination of models.

Bootstrap (entry, Statistics): stories (cores) are resampled with replacement, 10,000 resamples, numpy default_rng(seed
20261010); one fixed index set per population size n, shared by every row, arm and statistic computed on stories of that
size, so contrasts between rows of the same stories are paired. Every ratio of means is recomputed in every resample.
A one-sided test "H0: theta <= t" is rejected when the lower bound of the 95 % percentile interval is > t (a test at
2.5 %). Each test is also recorded as a Holm component {line, name, est, se, bound, direction}: est the point estimate,
se the standard deviation of the defined resamples (ddof 1), direction '>' for H1: theta > t and '<' for H1: theta < t.
"""
from __future__ import annotations

import numpy as np

SEED, NB = 20261010, 10000
NAN = float("nan")
_BOOT: dict = {}


class Boot:
    """Resample weights W [NB, n] (counts of each story in each resample)."""

    def __init__(self, n, nb=NB, seed=SEED):
        self.n, self.nb = n, nb
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, (nb, n)) if n else np.zeros((nb, 0), dtype=int)
        self.W = np.zeros((nb, n))
        if n:
            np.add.at(self.W, (np.arange(nb)[:, None], idx), 1)

    def mean(self, x, w=None):
        """Q of the mean of x [n] (``w``: optional 0/1 mask selecting stories, e.g. a stratum; resamples without any
        selected story give NaN)."""
        x = np.asarray(x, float)
        assert x.shape == (self.n,), (x.shape, self.n)
        W = self.W if w is None else self.W * np.asarray(w, float)[None]
        with np.errstate(all="ignore"):
            tot = W.sum(1)
            bs = np.where(tot > 0, (W @ np.nan_to_num(x)) / np.where(tot > 0, tot, 1), np.nan)
            sel = x if w is None else x[np.asarray(w, bool)]
            pt = float(np.mean(sel)) if sel.size else NAN
        return Q(pt, bs)


def boot(n) -> Boot:
    if n not in _BOOT:
        _BOOT[n] = Boot(n)
    return _BOOT[n]


class Q:
    """A statistic: point estimate and resamples."""

    def __init__(self, pt, bs):
        self.pt = float(pt)
        self.bs = np.asarray(bs, float)

    def pct(self, q):
        b = self.bs[~np.isnan(self.bs)]
        return float(np.percentile(b, q)) if b.size else NAN

    def ci(self, level=0.95):
        a = 100 * (1 - level) / 2
        return self.pct(a), self.pct(100 - a)

    def lo(self, level=0.95):
        return self.ci(level)[0]

    def hi(self, level=0.95):
        return self.ci(level)[1]

    @property
    def drop(self):
        return float(np.isnan(self.bs).mean()) if self.bs.size else 1.0

    @property
    def se(self):
        """Bootstrap standard error: the standard deviation of the defined resamples (ddof 1)."""
        b = self.bs[~np.isnan(self.bs)]
        return float(np.std(b, ddof=1)) if b.size > 1 else NAN

    def p_le(self, t):
        """p of H0: theta <= t (rejected when the lower bound > t)."""
        b = self.bs[~np.isnan(self.bs)]
        return (1 + int((b <= t).sum())) / (b.size + 1)

    def p_ge(self, t):
        """p of H0: theta >= t (rejected when the upper bound < t)."""
        b = self.bs[~np.isnan(self.bs)]
        return (1 + int((b >= t).sum())) / (b.size + 1)

    def txt(self, level=0.95):
        lo, hi = self.ci(level)
        return f"{f3(self.pt)} [{f3(lo)},{f3(hi)}]" + (f" ({100 * self.drop:.1f} % undefined)" if self.drop > 0 else "")

    def __sub__(self, o):
        return Q(self.pt - o.pt, self.bs - o.bs)

    def __add__(self, o):
        return Q(self.pt + o.pt, self.bs + o.bs)

    def __neg__(self):
        return Q(-self.pt, -self.bs)

    def scale(self, c):
        return Q(c * self.pt, c * self.bs)


def f3(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def ratio(a: Q, b: Q) -> Q:
    with np.errstate(all="ignore"):
        bs = np.where(b.bs != 0, a.bs / b.bs, np.nan)
        return Q(a.pt / b.pt if b.pt else NAN, bs)


def one_minus(q: Q) -> Q:
    return Q(1 - q.pt, 1 - q.bs)


def mean_q(qs) -> Q:
    qs = list(qs)
    with np.errstate(all="ignore"):
        return Q(float(np.mean([q.pt for q in qs])), np.mean(np.stack([q.bs for q in qs]), 0))


def V(ok):
    return "NOT EVALUABLE" if ok is None else "MET" if ok else "NOT MET"


def component(code, name, q: Q, bound, direction, own):
    """One interval component for the Holm sensitivity analysis (decision D2; analysis/stage8_holm.py): (line code,
    {line, name, est, se, bound, direction}, rejected by the line's own interval rule). A degenerate bootstrap (se 0) is
    passed as se 1e-12 (p then 0 or 1 by the side of the bound the estimate lies on), as in the other parts."""
    se = q.se
    se = 1e-12 if se == 0 else se
    return (code, {"line": code, "name": name, "est": float(q.pt), "se": float(se), "bound": float(bound),
                   "direction": direction}, bool(own))


class Tests:
    """Collects the interval components of a line: each (label, rejected under the 95 % interval rule), and, into
    ``sink``, the Holm component of every test (``component``). A line evaluates every one of its tests before it
    combines them with its point conditions, so the Holm family does not depend on which point conditions hold."""

    def __init__(self, code=None, sink=None):
        self.code, self.sink, self.items = code, sink, []

    def _rec(self, label, q, t, direction, rej):
        self.items.append((label, rej))
        if self.sink is not None:
            self.sink.append(component(self.code, label, q, t, direction, rej))

    def lower_gt(self, q: Q, t, label):
        """H0: theta <= t; rejected when the lower bound > t."""
        rej = bool(np.isfinite(q.lo()) and q.lo() > t)
        self._rec(f"{label} H0: <= {t:g}", q, t, ">", rej)
        return rej

    def upper_lt(self, q: Q, t, label):
        """H0: theta >= t; rejected when the upper bound < t."""
        rej = bool(np.isfinite(q.hi()) and q.hi() < t)
        self._rec(f"{label} H0: >= {t:g}", q, t, "<", rej)
        return rej

    def inside(self, q: Q, lo, hi, label):
        """Equivalence by two one-sided tests at 2.5 % each: the 95 % interval inside (lo, hi)."""
        a = self.lower_gt(q, lo, label)
        b = self.upper_lt(q, hi, label)
        return a and b


def combine(per_model: dict, need: int):
    """Combination over models: MET if met in every evaluable model and at least ``need`` are evaluable; NOT MET if not
    met in any evaluable model; NOT EVALUABLE otherwise (fewer than ``need`` evaluable, none failing)."""
    ev = {k: v for k, v in per_model.items() if v is not None}
    if any(v is False for v in ev.values()):
        return False
    if len(ev) >= need and all(ev.values()):
        return True
    return None
