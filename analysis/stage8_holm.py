"""The Holm sensitivity analysis of preregistration J (P-2026-10-10-J, common part, decision D2), shared by the four
stage-8 scorers so that it is computed identically in every part.

    holm(components, alpha=0.025) -> list of dicts, one per component, in the input order

A component is one bound of one statistic in one model: a dict with the keys
    line       the line code (e.g. "J-B3")
    name       a label unique within the line (e.g. "qwen7: s_ID(POST) lower bound")
    est        the point estimate
    se         its bootstrap standard error (the standard deviation of the resampled statistic)
    bound      the bound of the named null
    direction  ">" (H1: value > bound; H0: value <= bound) or "<" (H1: value < bound)
Any other key is passed through unchanged. The one-sided p-value is the normal approximation from the bootstrap SE,
    p = Phi(-(est - bound) / se)  for ">",      p = Phi((est - bound) / se)  for "<".
A degenerate SE (se == 0) gives the limit of that formula: p = 0 when the estimate lies strictly on the H1 side of the
bound, 1 on the null side and 0.5 on the bound. A component whose est, se or bound is not a finite number (or whose se
is negative) is undefined: it is returned with "undefined": True, p = 1, reject False, and it is left out of the family
(it does not count in m), so a scorer that passes it and a scorer that drops it before the call get the same decisions.

Holm's step-down procedure at familywise one-sided level alpha (0.025) over the m defined components given: the p-values
are sorted increasingly (ties keep the input order); the component of rank i (1-based) has the threshold alpha / (m - i + 1);
the components are rejected in rank order while p <= threshold, and none after the first that is not. Each returned
dict is a copy of the component with these keys added:
    p          the one-sided p-value above
    z          (est - bound) / se, signed so that a large positive z favours H1 ("<": (bound - est) / se); nan if undefined
    rank, m    the rank of p (1 = smallest; None if undefined) and the family size (the defined components)
    threshold  the adjusted threshold alpha / (m - rank + 1) (nan if undefined)
    p_holm     the Holm-adjusted p-value, max over ranks j <= rank of min(1, (m - j + 1) p_(j))
    reject     True when Holm rejects the component's null (equivalently p_holm <= alpha)
    alpha      the familywise level used
    undefined  True when est, se or bound was not usable (see above)
No verdict of entry J uses this; each scorer prints, per R-class account line, whether any component's decision changes
when Holm's decisions replace the interval decisions (tests/test_stage8_holm.py)."""
from __future__ import annotations

import math

ALPHA = 0.025


def phi(x: float) -> float:
    """The standard normal distribution function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x


def one_sided_p(est, se, bound, direction) -> tuple:
    """(p, z, undefined) for one component (see the module docstring)."""
    if direction not in (">", "<"):
        raise ValueError(f"direction must be '>' or '<', got {direction!r}")
    est, se, bound = _num(est), _num(se), _num(bound)
    if not (math.isfinite(est) and math.isfinite(se) and math.isfinite(bound)) or se < 0:
        return 1.0, float("nan"), True
    d = (est - bound) if direction == ">" else (bound - est)   # > 0 favours H1
    if se == 0:
        return (0.0 if d > 0 else 1.0 if d < 0 else 0.5), (math.inf if d > 0 else -math.inf if d < 0 else 0.0), False
    z = d / se
    return phi(-z), z, False


def holm(components, alpha: float = ALPHA) -> list:
    """Holm's step-down at familywise one-sided ``alpha`` over ``components`` (see the module docstring)."""
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    comps = list(components)
    out = []
    for c in comps:
        missing = [k for k in ("line", "name", "est", "se", "bound", "direction") if k not in c]
        if missing:
            raise KeyError(f"component {c.get('line')!r}/{c.get('name')!r} lacks {missing}")
        p, z, undef = one_sided_p(c["est"], c["se"], c["bound"], c["direction"])
        out.append(dict(c, p=p, z=z, undefined=undef, alpha=alpha))
    for r in out:
        if r["undefined"]:
            r.update(rank=None, threshold=float("nan"), p_holm=1.0, reject=False)
    defined = [i for i, r in enumerate(out) if not r["undefined"]]
    m = len(defined)
    for r in out:
        r["m"] = m
    order = sorted(defined, key=lambda i: (out[i]["p"], i))
    running, stopped = 0.0, False
    for rank, i in enumerate(order, start=1):
        r = out[i]
        thr = alpha / (m - rank + 1)
        running = max(running, min(1.0, (m - rank + 1) * r["p"]))
        stopped = stopped or not (r["p"] <= thr)
        r.update(rank=rank, m=m, threshold=thr, p_holm=running, reject=not stopped)
    return out
