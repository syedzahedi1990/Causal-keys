"""Statistics of the stage-8 part-C scorer (P-2026-10-10-J part C): the hierarchical bootstrap, estimates as (point,
resamples), interval criteria as one-sided tests (each with its bootstrap SE: the input of the shared Holm sensitivity
helper analysis/stage8_holm.py, decision D2), TOST, verdict words.

Bootstrap (entry, Statistics): stories are nested in (base, source) clusters. Two-stage: resample the clusters with
replacement, then the stories within each drawn cluster with replacement; 10,000 resamples, seed 20261012, one index set
per (model, story list), shared by every format, depth, row and scoring of the model, so every contrast is paired. The E4
seeds are a further level: each resample also draws two seeds with replacement from {101, 102}, and an E4 per-story value
is the mean over the drawn seeds (the point estimate weights the two seeds equally). Every ratio, share and log ratio is
recomputed in every resample from the resample's weighted means.
Pair statistics on Prakash et al.'s material (J-C6, J-C-WIN, J-C-SCREEN) use the plain pair bootstrap (no clusters), same
seed and size.
"""
from __future__ import annotations

import math

import numpy as np

SEED, NB = 20261012, 10000
DROP_MAX = 0.05
NAN = float("nan")
LOG125, LOG05, LOG08, LOG2 = math.log(1.25), math.log(0.5), math.log(0.8), math.log(2.0)
PSI_FLOOR = 0.01          # psi values below this are set to it before the log (lambda stays finite; a departure)
_BOOT: dict = {}


class Boot:
    """Weights of the hierarchical bootstrap: W [NB, n] (story counts), the seed weights SW [NB, 2] (counts / 2)."""

    def __init__(self, clusters, n=None, nb=NB, seed=SEED):
        n = len(clusters) if clusters is not None else n
        self.n, self.nb = n, nb
        self.W = np.zeros((nb, n))
        rng = np.random.default_rng(seed)
        if n:
            if clusters is None:
                idx = rng.integers(0, n, (nb, n))
                np.add.at(self.W, (np.arange(nb)[:, None], idx), 1)
            else:
                labs = sorted(set(clusters), key=str)
                members = {c: [i for i, x in enumerate(clusters) if x == c] for c in labs}
                draws = rng.integers(0, len(labs), (nb, len(labs)))
                cnt = np.zeros((nb, len(labs)), dtype=np.int64)
                np.add.at(cnt, (np.arange(nb)[:, None], draws), 1)
                for j, c in enumerate(labs):
                    ii = members[c]
                    self.W[:, ii] = rng.multinomial(cnt[:, j] * len(ii), [1.0 / len(ii)] * len(ii))
        self.tot = np.maximum(self.W.sum(1), 1e-300)
        s = rng.integers(0, 2, (nb, 2))
        self.SW = np.stack([(s == 0).sum(1), (s == 1).sum(1)], 1) / 2.0

    def mean(self, x):
        """(point, resamples [NB]) of the mean of x: [n], or [n, 2] (E4 seeds: the seed level applies)."""
        x = np.asarray(x, float)
        with np.errstate(all="ignore"):
            if x.ndim == 1:
                return float(np.nanmean(x)) if x.size else NAN, (self.W @ np.nan_to_num(x)) / self.tot
            pt = float(np.mean(x))
            m = (self.W @ x) / self.tot[:, None]          # [NB, 2]
            return pt, (m * self.SW).sum(1)


def boot_of(clusters, n=None) -> Boot:
    key = (None, n) if clusters is None else tuple(map(str, clusters))
    if key not in _BOOT:
        _BOOT[key] = Boot(list(clusters) if clusters is not None else None, n)
    return _BOOT[key]


class Q:
    """A statistic: point estimate and resamples (NaN where undefined)."""

    def __init__(self, pt, bs):
        self.pt = float(pt)
        self.bs = np.asarray(bs, float)

    @property
    def drop(self):
        return float(np.isnan(self.bs).mean()) if self.bs.size else 1.0

    def pct(self, q):
        b = self.bs[~np.isnan(self.bs)]
        return float(np.percentile(b, q)) if b.size else NAN

    def ci(self, level=0.95):
        a = 100 * (1 - level) / 2
        return self.pct(a), self.pct(100 - a)

    def lower(self, level=0.95):
        return self.ci(level)[0]

    def upper(self, level=0.95):
        return self.ci(level)[1]

    @property
    def se(self):
        """The bootstrap standard error (standard deviation of the defined resamples): the Holm sensitivity input."""
        b = self.bs[~np.isnan(self.bs)]
        return float(b.std()) if b.size > 1 else NAN

    def txt(self, level=0.95):
        lo, hi = self.ci(level)
        return f"{f3(self.pt)} [{f3(lo)},{f3(hi)}]" + (f" ({100 * self.drop:.1f} % undefined)" if self.drop > 0 else "")


def f3(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def ratio(a: Q, b: Q) -> Q:
    with np.errstate(all="ignore"):
        bs = np.where(b.bs != 0, a.bs / b.bs, np.nan)
        return Q(a.pt / b.pt if b.pt else NAN, bs)


def mean_q(qs) -> Q:
    """The mean of several statistics (point and per resample)."""
    qs = list(qs)
    with np.errstate(all="ignore"):
        return Q(float(np.mean([q.pt for q in qs])), np.mean(np.stack([q.bs for q in qs]), 0))


def log_ratio(psiK: Q, psiV: Q) -> Q:
    """lambda = log(max(psi_K, 0.01)) - log(max(psi_V, 0.01))."""
    f = lambda x: np.log(np.maximum(x, PSI_FLOOR))  # noqa: E731
    with np.errstate(all="ignore"):
        bs = f(np.nan_to_num(psiK.bs, nan=PSI_FLOOR)) - f(np.nan_to_num(psiV.bs, nan=PSI_FLOOR))
        bs = np.where(np.isnan(psiK.bs) | np.isnan(psiV.bs), np.nan, bs)
        return Q(float(f(psiK.pt) - f(psiV.pt)) if np.isfinite(psiK.pt) and np.isfinite(psiV.pt) else NAN, bs)


def tost(q: Q, margin=LOG125, level=0.90):
    """Equivalence by two one-sided tests at 5 % each: H0: lambda <= -margin and H0: lambda >= margin are both rejected
    when the 90 % interval lies inside (-margin, margin); a statistic undefined in > 5 % of resamples is not equivalent."""
    lo, hi = q.ci(level)
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo > -margin and hi < margin and q.drop <= DROP_MAX)


def V(ok):
    return "NOT EVALUABLE" if ok is None else "MET" if ok else "NOT MET"


def component(code, name, q: Q, bound, direction, own):
    """One interval component for the Holm sensitivity analysis of entry J (decision D2; analysis/stage8_holm.py):
    (line code, {line, name, est, se, bound, direction}, rejected by the line's own interval rule). ``direction`` '>' for
    H1: theta > bound, '<' for H1: theta < bound. A degenerate bootstrap (se 0) is passed as se 1e-12 (p then 0 or 1 by
    the side of the bound the estimate lies on)."""
    se = q.se
    se = 1e-12 if se == 0 else se
    return (code, {"line": code, "name": name, "est": float(q.pt), "se": float(se), "bound": float(bound),
                   "direction": direction}, bool(own))
