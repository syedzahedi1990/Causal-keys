"""Statistics of the stage-9 scorer (docs/stage9/SPEC.md section 10; docs/STAGE9_PREDICTIONS.md sections 2, 3, 5).

Bootstrap: the two-stage cluster bootstrap of analysis/stage8b_parts/stats.Boot (resample the clusters with replacement,
then the items within each drawn cluster with replacement), 10,000 resamples, seed 20261015. One weight matrix W [NB, n]
per population and model over all valid items, shared by every format and row, so contrasts are paired. A rate over a
subset (the competent items, the units of the AUC) is the ratio of weighted sums (W @ (x m)) / (W @ m); a resample whose
subset weight is zero is undefined and dropped; more than 5 % dropped (or an undefined point) makes the statistic NOT
EVALUABLE. Every statistic (rates, l_c, psi, the AUC, the weighted median, differences) is recomputed in every resample.
Intervals: 95 % percentile intervals of the defined resamples (numpy's linear interpolation).

Verdicts per model (SPEC 10): SUPPORTED = the point and the whole interval on the predicted side of the threshold;
SUPPORTED AT THE POINT = the point on the predicted side, the interval crossing; NOT SUPPORTED = the point on the wrong
side; NOT EVALUABLE = missing files, formats or l_b, a failed check, or an undefined statistic. A prediction with
several parts in one model (cells, formats, onsets) combines its parts with the rule that combines models: NOT
SUPPORTED as soon as one part is, else NOT EVALUABLE if one part is, else SUPPORTED if every part is, else SUPPORTED AT
THE POINT. Comparisons with a threshold allow 1e-9 (EPS), so that a rate equal to a threshold as a rational number is
not lost to floating-point rounding (0.3 - 0.2 < 0.1 in binary).
"""
from __future__ import annotations

import math

import numpy as np

from stage8b_parts.stats import Boot

SEED, NB = 20261015, 10000
DROP_MAX = 0.05
EPS = 1e-9
NAN = float("nan")
SUP, PT, NOT, NE = "SUPPORTED", "SUPPORTED AT THE POINT", "NOT SUPPORTED", "NOT EVALUABLE"
_BOOT: dict = {}


def boot_of(clusters, nb: int | None = None) -> Boot:
    """The cached Boot of a population (cluster labels in item order); seed 20261015; nb = NB (10,000)."""
    nb = NB if nb is None else nb
    key = (tuple(clusters), nb)
    if key not in _BOOT:
        _BOOT[key] = Boot(list(clusters), nb=nb, seed=SEED)
    return _BOOT[key]


class Q:
    """A statistic: point estimate ``pt``, resamples ``bs`` (NaN = undefined in that resample), the number of units
    ``n`` and, when it cannot be evaluated for a reason other than undefinedness, ``why``."""

    def __init__(self, pt, bs, n, why=None):
        self.pt = NAN if pt is None else float(pt)
        self.bs = np.asarray(bs, float).reshape(-1)
        self.n = int(n)
        self.why = why

    @property
    def drop(self) -> float:
        return float(np.isnan(self.bs).mean()) if self.bs.size else 1.0

    def _ci(self):
        if not hasattr(self, "_lohi"):
            b = self.bs[~np.isnan(self.bs)]
            self._lohi = tuple(float(x) for x in np.percentile(b, [2.5, 97.5])) if b.size else (NAN, NAN)
        return self._lohi

    @property
    def lo(self) -> float:
        return self._ci()[0]

    @property
    def hi(self) -> float:
        return self._ci()[1]

    def ok(self) -> bool:
        return self.why is None and math.isfinite(self.pt) and self.drop <= DROP_MAX

    def reason(self) -> str:
        if self.why:
            return self.why
        if not math.isfinite(self.pt):
            return "undefined at the point"
        if self.drop > DROP_MAX:
            return f"undefined in {100 * self.drop:.1f} % of resamples (> 5 %)"
        return ""

    def txt(self, d: int = 3) -> str:
        if self.why and not math.isfinite(self.pt):
            return f"NOT EVALUABLE ({self.why})"
        s = f"{self.pt:.{d}f} [{self.lo:.{d}f}, {self.hi:.{d}f}] (n={self.n})"
        return s + ("" if self.ok() else f" (NOT EVALUABLE: {self.reason()})")

    def as_dict(self) -> dict:
        f = lambda x: None if not math.isfinite(x) else round(x, 6)  # noqa: E731
        return {"pt": f(self.pt), "lo": f(self.lo), "hi": f(self.hi), "n": self.n}


def missing(why: str) -> Q:
    return Q(NAN, np.full(NB, NAN), 0, why=why)


def _wsum(W, x):
    return W @ np.asarray(x, float)


def rate(B: Boot, x, m=None) -> Q:
    """The mean of the per-item values x over the subset m (0/1), at the point and in every resample. (Per-item counts
    of units: ``ratio`` of the counts.)"""
    x = np.asarray(x, float)
    m = np.ones_like(x) if m is None else np.asarray(m, float)
    n = int(round(m.sum()))
    if m.sum() <= 0:
        return Q(NAN, np.full(B.nb, NAN), 0, why="no items in the subset")
    with np.errstate(all="ignore"):
        den = _wsum(B.W, m)
        bs = np.where(den > 0, _wsum(B.W, x * m) / np.where(den > 0, den, 1.0), NAN)
    return Q(float((x * m).sum() / m.sum()), bs, n)


def ratio(B: Boot, num, den, m=None) -> Q:
    """sum(num m) / sum(den m) at the point and in every resample (undefined where the denominator is 0)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    m = np.ones_like(num) if m is None else np.asarray(m, float)
    n = int(round(m.sum()))
    if m.sum() <= 0:
        return Q(NAN, np.full(B.nb, NAN), 0, why="no items in the subset")
    with np.errstate(all="ignore"):
        d = _wsum(B.W, den * m)
        bs = np.where(d != 0, _wsum(B.W, num * m) / np.where(d != 0, d, 1.0), NAN)
        D = float((den * m).sum())
        pt = float((num * m).sum()) / D if D != 0 else NAN
    return Q(pt, bs, n)


def fn(f, *qs, n=None) -> Q:
    """f of statistics computed on the same weights (paired), at the point and in every resample."""
    why = next((q.why for q in qs if q.why), None)
    with np.errstate(all="ignore"):
        pt = f(*[q.pt for q in qs])
        bs = f(*[q.bs for q in qs])
    return Q(pt, bs, min(q.n for q in qs) if n is None else n, why=why)


def diff(a: Q, b: Q) -> Q:
    return fn(lambda x, y: x - y, a, b)


def wmedian(B: Boot, v, m) -> Q:
    """The weighted median of v over the subset m (the midpoint convention of numpy.median: where the cumulative weight
    reaches exactly half, the mean of that value and the next one with weight), at the point and in every resample."""
    v, m = np.asarray(v, float), np.asarray(m, bool)
    sel = np.flatnonzero(m)
    if not sel.size:
        return Q(NAN, np.full(B.nb, NAN), 0, why="no items in the subset")
    o = sel[np.argsort(v[sel], kind="stable")]
    vals = v[o]

    def med(Wm):
        c = np.cumsum(Wm, axis=1)
        tot = c[:, -1]
        half = tot / 2.0
        i = np.argmax(c >= half[:, None] - 1e-12, axis=1)
        r = np.arange(Wm.shape[0])
        exact = np.abs(c[r, i] - half) <= 1e-12
        nxt = np.argmax(c > c[r, i][:, None] + 1e-12, axis=1)
        val = np.where(exact, 0.5 * (vals[i] + vals[nxt]), vals[i])
        return np.where(tot > 0, val, NAN)

    pt = float(np.median(vals))
    return Q(pt, med(B.W[:, o]), len(sel))


def auc(B: Boot, core, r, y, min_each: int = 10) -> tuple[Q, int, int]:
    """AUC = P(r of a positive > r of a negative) + 1/2 P(tie) over units (core index, predictor r, label y); in a
    resample every unit carries the weight of its core. (Q, positives, negatives); evaluable with at least ``min_each``
    of each at the point."""
    core, r, y = np.asarray(core, int), np.asarray(r, float), np.asarray(y, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos < min_each or nneg < min_each:
        return (Q(NAN, np.full(B.nb, NAN), len(r), why=f"{npos} positives and {nneg} negatives (at least {min_each} "
                    f"of each needed)"), npos, nneg)
    vals, k = np.unique(r, return_inverse=True)
    n_core = B.W.shape[1]
    Mp = np.zeros((n_core, len(vals)))
    Mn = np.zeros((n_core, len(vals)))
    np.add.at(Mp, (core[y], k[y]), 1.0)
    np.add.at(Mn, (core[~y], k[~y]), 1.0)

    def a(P, N):   # P, N [..., K] weights of the positives / negatives at each value
        above = np.cumsum(P[..., ::-1], axis=-1)[..., ::-1] - P   # weight of positives with a larger value
        num = (N * (above + 0.5 * P)).sum(-1)
        den = P.sum(-1) * N.sum(-1)
        with np.errstate(all="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), NAN)

    ones = np.ones(n_core)
    pt = float(a(ones @ Mp, ones @ Mn))
    return Q(pt, a(B.W @ Mp, B.W @ Mn), len(r)), npos, nneg


def first_below(F, onsets, L, thr=0.5):
    """l_c: the first onset whose flip is < thr (L when none; NaN where a flip is undefined), along the last axis."""
    F = np.asarray(F, float)
    on = np.asarray(onsets, float)
    below = F < thr - EPS
    idx = np.argmax(below, axis=-1)
    val = np.where(below.any(axis=-1), on[idx], float(L))
    return np.where(np.isnan(F).any(axis=-1), NAN, val)


# --------------------------------------------------------------------------- verdicts
class V:
    """A verdict: level (SUP, PT, NOT, NE), a label, the parts it combines and the reason when NOT EVALUABLE."""

    def __init__(self, level, label="", parts=(), why=""):
        self.level, self.label, self.parts, self.why = level, label, list(parts), why

    def __str__(self):
        return f"{self.label}: {self.level}" + (f" ({self.why})" if self.level == NE and self.why else "")


def check(q: Q, op: str, thr: float, label: str) -> V:
    """One part: the statistic against a threshold, op in (">=", "<=") (the predicted side)."""
    if not q.ok():
        return V(NE, f"{label} {q.txt()} {op} {thr:g}", why=q.reason())
    if op == ">=":
        p, i = q.pt >= thr - EPS, q.lo >= thr - EPS
    elif op == "<=":
        p, i = q.pt <= thr + EPS, q.hi <= thr + EPS
    else:
        raise ValueError(op)
    lvl = SUP if p and i else PT if p else NOT
    return V(lvl, f"{label} {q.txt()} {op} {thr:g}")


def band(q: Q, a: float, b: float, label: str) -> V:
    """One part: the statistic inside [a, b] (point; interval inside for SUPPORTED)."""
    if not q.ok():
        return V(NE, f"{label} {q.txt()} in [{a:g}, {b:g}]", why=q.reason())
    p = a - EPS <= q.pt <= b + EPS
    i = q.lo >= a - EPS and q.hi <= b + EPS
    return V(SUP if p and i else PT if p else NOT, f"{label} {q.txt()} in [{a:g}, {b:g}]")


def combine(levels) -> str:
    """Parts within a model, or models (SPEC 10): NOT SUPPORTED as soon as one is; else NOT EVALUABLE if one is (or
    there is none); else SUPPORTED if all are; else SUPPORTED AT THE POINT."""
    levels = list(levels)
    if not levels:
        return NE
    if NOT in levels:
        return NOT
    if NE in levels:
        return NE
    return SUP if all(x == SUP for x in levels) else PT


def k_of(levels, k: int) -> str:
    """A "k of 3" prediction, counted at the point level: SUPPORTED if at least k models are SUPPORTED; SUPPORTED AT
    THE POINT if at least k are at least at the point; NOT SUPPORTED if k can no longer be reached even if every NOT
    EVALUABLE model were at the point; else NOT EVALUABLE."""
    levels = list(levels)
    n_sup = sum(x == SUP for x in levels)
    n_pt = sum(x in (SUP, PT) for x in levels)
    n_ne = sum(x == NE for x in levels)
    if n_pt >= k:
        return SUP if n_sup >= k else PT
    if n_pt + n_ne < k:
        return NOT
    return NE


def trig(q: Q, op: str, thr: float):
    """A falsifier at the point: True (TRIGGERED), False, or None (not evaluable)."""
    if not q.ok():
        return None
    if op == "<":
        return q.pt < thr - EPS
    if op == ">=":
        return q.pt >= thr - EPS
    if op == "<=":
        return q.pt <= thr + EPS
    raise ValueError(op)


def at_least_point(level) -> bool:
    return level in (SUP, PT)
