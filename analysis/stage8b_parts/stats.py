"""Statistics of the stage-8 part-B scorer (P-2026-10-10-J part B): the two-stage cluster bootstrap, estimates with
definedness rules, one-sided interval criteria with their bootstrap p-values (Holm sensitivity), verdict words and the
rules that combine models.

Bootstrap (entry, Statistics): for a population whose items carry cluster labels (lexicon, base, source), resample the
clusters with replacement, then the items within each drawn cluster with replacement; 10,000 resamples, seed 20261013,
one index set per population (shared by every arm, scoring and behavioural measure, so contrasts are paired). A
statistic is a function of item means, recomputed in every resample from the resample's weighted means. The core
bootstrap (resample items directly; the original rules of JB6 (c)) uses the same seed.
Definedness: a statistic returns its value and two masks. "own" fails when the arm under test makes it undefined (a
share whose denominator collapses); "anchor" fails when the LIST-AFTER anchor of a ratio does. At the point or in more
than 5 % of the resamples: own -> the criterion is not met; anchor -> the criterion is not evaluable.
Intervals are percentile intervals at the line's level (95 %: every-model lines and single-model lines; 98.75 %: the
"3 of 4" lines). "lower bound > t" is the one-sided test of H0: theta <= t, with bootstrap p = (1 + #{resamples <= t}) /
(B + 1) over the defined resamples; "upper bound < t" likewise; "inside (a, b)" is two one-sided tests, p the larger.
"""
from __future__ import annotations

import math

import numpy as np

SEED, NB = 20261013, 10000
DROP_MAX = 0.05
NAN = float("nan")
_BOOT: dict = {}


class Boot:
    """Weights W [NB, n] of the two-stage cluster bootstrap (clusters given) or of the item bootstrap (clusters None)."""

    def __init__(self, clusters, n=None, nb=NB, seed=SEED):
        n = len(clusters) if clusters is not None else n
        self.n, self.nb = n, nb
        self.W = np.zeros((nb, n))
        if n == 0:
            self.tot = np.ones(nb)
            return
        rng = np.random.default_rng(seed)
        if clusters is None:
            idx = rng.integers(0, n, (nb, n))
            np.add.at(self.W, (np.arange(nb)[:, None], idx), 1)
        else:
            labs = sorted(set(clusters))
            members = {c: [i for i, x in enumerate(clusters) if x == c] for c in labs}
            draws = rng.integers(0, len(labs), (nb, len(labs)))
            cnt = np.zeros((nb, len(labs)), dtype=np.int64)
            np.add.at(cnt, (np.arange(nb)[:, None], draws), 1)
            for j, c in enumerate(labs):
                ii = members[c]
                self.W[:, ii] = rng.multinomial(cnt[:, j] * len(ii), [1.0 / len(ii)] * len(ii))
        self.tot = self.W.sum(1)

    def means(self, x):
        with np.errstate(all="ignore"):
            return (self.W @ np.asarray(x, float)) / self.tot


def boot_of(clusters, n=None) -> Boot:
    """One Boot per population (the cluster labels in item order; None and n: the item bootstrap)."""
    key = (None, n) if clusters is None else tuple(clusters)
    if key not in _BOOT:
        _BOOT[key] = Boot(list(clusters) if clusters is not None else None, n)
    return _BOOT[key]


class Est:
    """A point estimate and its resamples; ``own`` / ``anchor``: whether the point is defined under each rule, and the
    fraction of resamples where each rule fails."""

    def __init__(self, pt, bs, n, own=True, anchor=True, drop_own=0.0, drop_anchor=0.0):
        self.pt, self.n = float(pt), n
        self.own, self.anchor = bool(own), bool(anchor)
        self.drop_own, self.drop_anchor = float(drop_own), float(drop_anchor)
        bs = np.asarray(bs, float)
        self.bs = bs[~np.isnan(bs)]

    def bound(self, level, side):
        if not self.bs.size:
            return NAN
        q = 100 * (1 - level) / 2
        return float(np.percentile(self.bs, q if side == "lo" else 100 - q))

    def undefined(self):
        """None, ("anchor", why) or ("own", why)."""
        if not self.anchor or self.drop_anchor > DROP_MAX:
            return ("anchor", "the LIST-AFTER anchor of the ratio is undefined" + (
                "" if not self.anchor else f" in {100 * self.drop_anchor:.1f} % of resamples (> 5 %)"))
        if not self.own or self.drop_own > DROP_MAX or math.isnan(self.pt):
            return ("own", "undefined by the arm itself" + ("" if not self.own or math.isnan(self.pt) else
                                                             f" in {100 * self.drop_own:.1f} % of resamples (> 5 %)"))
        return None

    def txt(self, level):
        u = self.undefined()
        s = f"{self.pt:+.3f} [{self.bound(level, 'lo'):+.3f},{self.bound(level, 'hi'):+.3f}]"
        return s + (f" ({u[1]})" if u else "")


def est(boot: Boot, fn, *xs) -> Est:
    """fn(*means) -> value, or (value, own mask, anchor mask); evaluated at the point and in every resample."""
    xs = [np.asarray(x, float) for x in xs]
    n = boot.n
    assert all(len(x) == n for x in xs), "every argument must be over the population"
    if n == 0:
        return Est(NAN, [], 0, own=False)

    def call(ms):
        with np.errstate(all="ignore"):
            r = fn(*ms)
        if isinstance(r, tuple):
            v, o, a = r
        else:
            v, o, a = r, True, True
        v = np.asarray(v, float)
        o = np.broadcast_to(np.asarray(o, bool), v.shape) & ~np.isnan(v)
        a = np.broadcast_to(np.asarray(a, bool), v.shape)
        return v, o, a

    pv, po, pa = call([x.mean() for x in xs])
    bv, bo, ba = call([boot.means(x) for x in xs])
    bs = np.where(bo & ba, bv, NAN)
    return Est(float(pv), bs, n, bool(po), bool(pa), float(1 - bo.mean()), float(1 - ba.mean()))


def mean(x):
    return x


# --------------------------------------------------------------------------- criteria
class Comp:
    """One component of a line in one model: an effect-size condition on the point estimate, or an interval
    criterion (a one-sided test of a named null) with its bootstrap p-value; ``ne`` names why it is not evaluable."""

    def __init__(self, label, passed, p=None, ne=None):
        self.label, self.passed, self.p, self.ne = label, bool(passed), p, ne

    def __str__(self):
        return f"{self.label}: {'NOT EVALUABLE (' + self.ne + ')' if self.ne else 'yes' if self.passed else 'no'}"


def _undef_comp(e, label):
    u = e.undefined()
    if u is None:
        return None
    if u[0] == "anchor":
        return Comp(label, False, ne=u[1])
    return Comp(label + f" -> not met: {u[1]}", False)


def point(label, cond, e=None):
    """An effect-size condition on the point estimate (an undefined estimate: as its definedness rule says)."""
    if e is not None:
        u = _undef_comp(e, label + " (point)")
        if u is not None:
            return u
    return Comp(label + " (point)", bool(cond))


def pval(e, null_side) -> float:
    if not e.bs.size:
        return 1.0
    return float((1 + null_side(e.bs).sum()) / (e.bs.size + 1))


def lower(e, thr, level, what):
    """H0: theta <= thr, rejected when the lower bound > thr."""
    lab = f"H0 {what} <= {thr:g} rejected at {100 * level:g}% (lower bound {e.bound(level, 'lo'):+.3f} > {thr:g})"
    u = _undef_comp(e, lab)
    if u is not None:
        return u
    lo = e.bound(level, "lo")
    return Comp(lab, not math.isnan(lo) and lo > thr, pval(e, lambda b: b <= thr))


def upper(e, thr, level, what):
    """H0: theta >= thr, rejected when the upper bound < thr."""
    lab = f"H0 {what} >= {thr:g} rejected at {100 * level:g}% (upper bound {e.bound(level, 'hi'):+.3f} < {thr:g})"
    u = _undef_comp(e, lab)
    if u is not None:
        return u
    hi = e.bound(level, "hi")
    return Comp(lab, not math.isnan(hi) and hi < thr, pval(e, lambda b: b >= thr))


def inside(e, a, b, level, what):
    """H0: theta <= a or theta >= b, rejected when the interval lies inside (a, b) (two one-sided tests)."""
    lo, hi = e.bound(level, "lo"), e.bound(level, "hi")
    lab = f"H0 {what} outside ({a:g}, {b:g}) rejected at {100 * level:g}% (interval [{lo:+.3f},{hi:+.3f}])"
    u = _undef_comp(e, lab)
    if u is not None:
        return u
    p = max(pval(e, lambda x: x <= a), pval(e, lambda x: x >= b))
    return Comp(lab, not math.isnan(lo) and not math.isnan(hi) and a < lo and hi < b, p)


class Res:
    """A line in one model: ok True / False / None (not evaluable), a text, the components, the reason."""

    def __init__(self, ok, txt, comps=(), why=""):
        self.ok, self.txt, self.comps, self.why = ok, txt, list(comps), why


def decide(comps, txt="") -> Res:
    """Not evaluable if any component is; else met iff every component passes."""
    ne = [c for c in comps if c.ne]
    if ne:
        return Res(None, txt, comps, "; ".join(sorted({c.ne for c in ne})))
    return Res(all(c.passed for c in comps), txt, comps)


def V(ok):
    return "NOT EVALUABLE" if ok is None else "MET" if ok else "NOT MET"


def comb_every(per: dict, min_ev: int = 3):
    """P4f (intersection-union): NOT EVALUABLE if fewer than min_ev models are evaluable; else MET iff every evaluable
    model meets the line."""
    ev = [v for v in per.values() if v is not None]
    if len(ev) < min_ev:
        return None
    return all(ev)


def comb_k_of(per: dict, k: int = 3):
    """N4: NOT EVALUABLE if fewer than k models are evaluable; else MET iff at least k models meet the line (a model not
    evaluable counts as not meeting it)."""
    ev = [v for v in per.values() if v is not None]
    if len(ev) < k:
        return None
    return sum(v is True for v in ev) >= k


def comb_single(per: dict):
    vals = list(per.values())
    return vals[0] if vals else None


def holm(pvals, alpha=0.025):
    """Holm's step-down over one-sided p-values at familywise alpha: a list of booleans (rejected)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    rej = [False] * len(pvals)
    m = len(pvals)
    for k, i in enumerate(order):
        if pvals[i] <= alpha / (m - k):
            rej[i] = True
        else:
            break
    return rej
