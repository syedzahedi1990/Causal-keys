"""Shared statistics of the stage-8 part-A scorer (P-2026-10-10-J part A): the two-stage article cluster bootstrap,
estimates, one-sided interval criteria with their bootstrap p-values (for the Holm sensitivity analysis), verdict words,
the cross-model combination rules, and each line's class and recorded prior.

Bootstrap: for a population of items with article labels, resample the articles with replacement, then the items
within each sampled article with replacement (10,000 resamples, seed 20261010, one fixed index set per population).
A statistic is a function of item means; it is recomputed in every resample from the resample's weighted means.
Intervals are 95 % percentile intervals. A criterion "lower bound > t" is the one-sided test of H0: theta <= t at
2.5 % ("upper bound < t": H0 theta >= t; "inside (a, b)": two one-sided tests). Each such component also carries, for
the Holm sensitivity analysis of entry J (analysis/stage8_holm.py), its point estimate, its bootstrap standard error
(the standard deviation of the resamples), its bound and its direction ('>' for H1 theta > t, '<' for H1 theta < t; an
"inside" criterion gives one of each). A resample in which the statistic is undefined (NaN) is dropped and counted.
"""
import math

import numpy as np

SEED, NB = 20261010, 10000
FRESH = {"llama8", "gemma9", "yi9"}
NAN = float("nan")
_BOOT = {}

# line code -> (class, recorded prior P(met), title); the entry's table. "D" = derived (reported, not counted).
# The class follows the recorded prior (entry J, G4): L = implied by data in hand on the same models and material with
# prior >= 0.9; M = prior >= 0.8; R = prior < 0.8 (tests/test_stage8a_score.py checks the agreement).
LINES = {
    "J-A1": ("M", 0.80, "a key read with later options on natural passages (OPTA)"),
    "J-A1b": ("R", 0.55, "the key read on spans of >= 3 tokens (OPTA)"),
    "J-A1c": ("M", 0.85, "the key read with lettered options (LETA)"),
    "J-A2": ("M", 0.85, "a value copy without a later mention (NOM)"),
    "J-A3": ("D", 0.70, "crossover OPTA - NOM (derived from J-A1 and J-A2; not counted)"),
    "J-A4": ("L", 0.90, "position: options before the passage give no key read (OPTB)"),
    "J-A5": ("R", 0.50, "a natural mention sentence after the passage opens a key read (MENA vs NOM)"),
    "J-A5B": ("M", 0.80, "the same sentence before the passage does not (MENA vs MENB)"),
    "J-A6a": ("R", 0.45, "cue conflict, OPTA: the key's entity is answered"),
    "J-A6b": ("R", 0.60, "cue conflict, LETA: the key's entity's letter is answered"),
    "J-A6c": ("R", 0.45, "cue conflict, NOM: the value's entity is answered"),
    "J-A6d": ("R", 0.45, "flag only (K_S, V_Z): S in OPTA, not in NOM"),
    "J-A6e": ("R", 0.40, "copy fallback (K_Z, V_S) in OPTA: S is answered"),
    "J-A7": ("R", 0.30, "KIVI 2-bit: value quantization hurts free form more than MCQ, beyond keys (DiD)"),
    "J-A7b": ("R", 0.35, "KIVI 2-bit: value quantization hurts free form more than MCQ"),
    "J-A8": ("M", 0.80, "multi-token answers: continuation needs K and V jointly, decision token additive (NOM)"),
    "J-A8d": ("R", 0.60, "hybrid answers under K_Z in NOM, not in OPTA"),
    "J-A-HA1": ("R", 0.75, "sparse natural readers (N*)"),
    "J-A-HA2": ("R", 0.40, "the template readers (T*) transfer to natural text"),
    "J-A-HA3a": ("R", 0.30, "ablating N* at Q+ breaks faithful MCQ answers (prior-free)"),
    "J-A-HA3b": ("R", 0.65, "ablating N* at Q+ leaves free-form answers (prior-free)"),
}
ORDER = list(LINES)


class Boot:
    """The two-stage cluster bootstrap of one population: arts[i] is item i's article."""

    def __init__(self, arts, nb=NB, seed=SEED):
        n = len(arts)
        self.n = n
        self.W = np.zeros((nb, n))
        if n == 0:
            return
        rng = np.random.default_rng(seed)
        labs = sorted(set(arts))
        idx = {a: [i for i, x in enumerate(arts) if x == a] for a in labs}
        draws = rng.integers(0, len(labs), (nb, len(labs)))
        cnt = np.zeros((nb, len(labs)), dtype=np.int64)
        np.add.at(cnt, (np.arange(nb)[:, None], draws), 1)
        for a, lab in enumerate(labs):
            ii = idx[lab]
            na = len(ii)
            self.W[:, ii] = rng.multinomial(cnt[:, a] * na, [1.0 / na] * na)
        self.tot = self.W.sum(1)

    def means(self, x):
        with np.errstate(all="ignore"):
            return (self.W @ np.asarray(x, float)) / self.tot


def boot_of(arts):
    key = tuple(arts)
    if key not in _BOOT:
        _BOOT[key] = Boot(list(arts))
    return _BOOT[key]


class Est:
    def __init__(self, pt, bs, n):
        self.pt, self.n = float(pt), n
        bs = np.asarray(bs, float)
        ok = ~np.isnan(bs)
        self.drop = float(1 - ok.mean()) if bs.size else 1.0
        self.bs = bs[ok]
        self.lo = float(np.percentile(self.bs, 2.5)) if self.bs.size else NAN
        self.hi = float(np.percentile(self.bs, 97.5)) if self.bs.size else NAN
        self.se = float(np.std(self.bs, ddof=1)) if self.bs.size > 1 else NAN

    def __str__(self):
        s = f"{self.pt:+.3f} [{self.lo:+.3f},{self.hi:+.3f}]"
        return s + (f" ({100 * self.drop:.1f} % of resamples dropped)" if self.drop > 0 else "")


def est(arts, fn, *xs):
    """fn(mean x1, mean x2, ...) at the point and in every resample of the population with articles ``arts``."""
    xs = [np.asarray(x, float) for x in xs]
    n = len(arts)
    assert all(len(x) == n for x in xs), "every argument must be over the population"
    if n == 0:
        return Est(NAN, [], 0)
    b = boot_of(arts)
    with np.errstate(all="ignore"):
        pt = fn(*[x.mean() for x in xs])
        bs = fn(*[b.means(x) for x in xs])
    return Est(pt, bs, n)


def mean(x):
    return x


def ratio(a, b):
    return a / b


def sid(k, v):
    return k / (k + v)


def nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


class Comp:
    """One component of a criterion: a point (effect-size) condition (``tests`` empty) or an interval criterion with its
    one-sided tests for the Holm sensitivity analysis: ``tests`` = [(name, passed, {est, se, bound, direction})]."""

    def __init__(self, label, passed, tests=()):
        self.label, self.passed, self.tests = label, bool(passed), list(tests)

    def __str__(self):
        return f"{self.label}: {'yes' if self.passed else 'no'}"


def point(label, cond):
    """An effect-size condition on the point estimate (NaN comparisons are False, so an undefined estimate fails)."""
    return Comp(label + " (point)", bool(cond))


def _test(name, e, thr, direction, passed):
    return (name, bool(passed), {"est": e.pt, "se": e.se, "bound": float(thr), "direction": direction})


def lower(e, thr, what):
    """H0: theta <= thr rejected: lower bound > thr."""
    ok = not nan(e.lo) and e.lo > thr
    return Comp(f"H0 {what} <= {thr:g} rejected (lower bound {e.lo:+.3f} > {thr:g})", ok,
                [_test(f"{what} > {thr:g}", e, thr, ">", ok)])


def upper(e, thr, what):
    """H0: theta >= thr rejected: upper bound < thr."""
    ok = not nan(e.hi) and e.hi < thr
    return Comp(f"H0 {what} >= {thr:g} rejected (upper bound {e.hi:+.3f} < {thr:g})", ok,
                [_test(f"{what} < {thr:g}", e, thr, "<", ok)])


def inside(e, a, b, what):
    """H0: theta <= a or theta >= b rejected: the interval lies inside (a, b) (two one-sided tests)."""
    lo_ok, hi_ok = not nan(e.lo) and a < e.lo, not nan(e.hi) and e.hi < b
    return Comp(f"H0 {what} outside ({a:g}, {b:g}) rejected (interval [{e.lo:+.3f},{e.hi:+.3f}])", lo_ok and hi_ok,
                [_test(f"{what} > {a:g}", e, a, ">", lo_ok), _test(f"{what} < {b:g}", e, b, "<", hi_ok)])


def V(ok):
    return "NOT EVALUABLE" if ok is None else "MET" if ok else "NOT MET"


def comb_models(per, fresh=FRESH, min_models=3):
    """A-lines: NOT MET as soon as one evaluable model does not meet the line (whatever the number of evaluable models);
    else MET with >= 3 evaluable including >= 1 fresh family; else NOT EVALUABLE (an intersection-union test at the 95 %
    intervals)."""
    ev = {k: v for k, v in per.items() if v is not None}
    if any(v is False for v in ev.values()):
        return False
    if len(ev) >= min_models and any(k in fresh for k in ev):
        return True
    return None


def comb_both(per, need=("qwen7", "mistral7")):
    """Head lines: MET if met in both head models, NOT MET if not met in an evaluable one, else NOT EVALUABLE."""
    vals = [per.get(k) for k in need]
    if any(v is False for v in vals):
        return False
    return True if all(v is True for v in vals) else None
