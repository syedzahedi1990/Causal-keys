"""Shared statistics of the stage-7 scorer (P-2026-10-08-I): the fixed core-bootstrap index set, estimates with the
dropped-resample count, the kappa rule, verdict words and the "both formats" combination."""
import math

import numpy as np

SEED, NB = 20261008, 10000
MAXDROP = 0.05
NAN4 = (float("nan"),) * 3 + (1.0,)
NAMES = {"P1": "OPTIONS-AFTER", "LETTER": "LETTERS-AFTER", "POST": "SENTENCE-AFTER", "NONE": "NO-MENTION", "BEFORE": "LIST-BEFORE"}
_W = {}


def W(n):
    """[NB, n] resampling weights (count / n) of one fixed index set per n (seed 20261008): W @ x = the resampled means.
    Every format is resampled with the same W over the same core order, i.e. jointly."""
    if n not in _W:
        idx = np.random.default_rng(SEED).integers(0, n, (NB, n))
        w = np.zeros((NB, n))
        np.add.at(w, (np.arange(NB)[:, None], idx), 1.0 / n)
        _W[n] = w
    return _W[n]


def est(fn, *xs):
    """(point, lo, hi, dropped) of fn(mean x1, mean x2, ...) over cores: point on the full sample, 95 % percentile
    interval over the resamples where fn is defined (not NaN), dropped = the fraction where it is not."""
    xs = [np.asarray(x, float) for x in xs]
    n = len(xs[0])
    assert all(len(x) == n for x in xs), "joint resampling needs the same cores in every argument"
    if not n:
        return NAN4
    with np.errstate(all="ignore"):
        pt = float(fn(*[x.mean(0) for x in xs]))
        bs = np.asarray(fn(*[W(n) @ x for x in xs]), float)
    ok = ~np.isnan(bs)
    drop = 1.0 - ok.mean()
    if not ok.any():
        return pt, float("nan"), float("nan"), drop
    return pt, float(np.percentile(bs[ok], 2.5)), float(np.percentile(bs[ok], 97.5)), float(drop)


def kappa(K, V, D):
    """The kappa rule (H's, adapted): K / (K + V) where psi_K + psi_V >= 0.3, both >= -0.1 on the scale of its own
    condition (psi = . / D) and D >= 3 nats; NaN elsewhere. Works on scalars and on resample arrays."""
    K, V, D = (np.asarray(x, float) for x in (K, V, D))
    with np.errstate(all="ignore"):
        pk, pv = K / D, V / D
        ok = (pk + pv >= 0.3) & (pk >= -0.1) & (pv >= -0.1) & (D >= 3)
        return np.where(ok, K / (K + V), np.nan)


def f3(t):
    s = f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"
    return s + (f" ({100 * t[3]:.1f} % of resamples dropped)" if len(t) > 3 and t[3] > 0 else "")


def V(ok):
    return "NOT EVALUABLE" if ok is None else "MET" if ok else "NOT MET"


def comb(subs):
    """MET when every part is met, NOT MET when an evaluable part is not met, else NOT EVALUABLE (also with no part)."""
    subs = list(subs)
    return False if any(s is not None and not s for s in subs) else None if not subs or any(s is None for s in subs) else True


def nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def lower_ok(t, thr):
    """The lower bound meets thr, counted only when at most 5 % of the resamples were dropped."""
    return not nan(t[1]) and t[1] >= thr and t[3] <= MAXDROP


def upper_ok(t, thr):
    return not nan(t[2]) and t[2] <= thr and t[3] <= MAXDROP


def why_undefined(t, what="kappa"):
    """The reason a bound-carrying criterion is NOT MET although its format is evaluable."""
    if nan(t[0]):
        return f"{what} undefined after blocking (the kappa rule fails at the point estimate: behaviour lost or interaction-carried)"
    if t[3] > MAXDROP:
        return f"{100 * t[3]:.1f} % of resamples dropped by the kappa rule (> 5 %): the bound is not met"
    return ""
