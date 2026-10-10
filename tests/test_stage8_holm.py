"""analysis/stage8_holm.py: the Holm sensitivity analysis shared by the four stage-8 scorers (preregistration J, common
part, decision D2). The one-sided normal p-value from the bootstrap SE in both directions (against hand values and the
inverse normal), the degenerate and undefined cases, Holm's step-down at 0.025 (hand-worked families, the stop at the
first non-rejection, ties, input order kept, extra keys passed through), the adjusted p-values against an independent
brute-force computation, and the equivalence with the unadjusted one-sided 2.5% test when the family has one member."""
import itertools
import math
import random
import sys
from pathlib import Path
from statistics import NormalDist

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import stage8_holm as H  # noqa: E402

N = NormalDist()


def comp(p, direction=">", line="J-X1", name=None, bound=0.1, se=0.05):
    """A component whose one-sided p-value is p: est placed |z| SEs from the bound on the side the direction gives."""
    z = -N.inv_cdf(p)                       # p = Phi(-z)
    est = bound + z * se if direction == ">" else bound - z * se
    return {"line": line, "name": name or f"c{p}", "est": est, "se": se, "bound": bound, "direction": direction}


def test_phi_and_p_values():
    assert H.phi(0) == 0.5 and abs(H.phi(1.959963984540054) - 0.975) < 1e-12 and abs(H.phi(-1) - 0.15865525393145707) < 1e-12
    # H1: value > 0.35, est 0.45, se 0.05 -> z = 2, p = Phi(-2)
    p, z, u = H.one_sided_p(0.45, 0.05, 0.35, ">")
    assert abs(z - 2) < 1e-12 and abs(p - 0.022750131948179195) < 1e-12 and not u
    # H1: value < 0.3, est 0.2, se 0.1 -> z = 1, p = Phi(-1); the same estimate tested the other way gives 1 - p
    p, z, _ = H.one_sided_p(0.2, 0.1, 0.3, "<")
    assert abs(z - 1) < 1e-12 and abs(p - 0.15865525393145707) < 1e-12
    q, _, _ = H.one_sided_p(0.2, 0.1, 0.3, ">")
    assert abs(p + q - 1) < 1e-12
    # an estimate on the null side has p > 0.5
    assert H.one_sided_p(-0.1, 0.05, 0.0, ">")[0] > 0.5 and H.one_sided_p(0.1, 0.05, 0.0, "<")[0] > 0.5
    with pytest.raises(ValueError):
        H.one_sided_p(1, 1, 0, ">=")


def test_degenerate_and_undefined():
    assert H.one_sided_p(0.5, 0.0, 0.35, ">")[:1] == (0.0,) and H.one_sided_p(0.2, 0.0, 0.35, ">")[0] == 1.0
    assert H.one_sided_p(0.35, 0.0, 0.35, ">")[0] == 0.5 and H.one_sided_p(0.2, 0.0, 0.35, "<")[0] == 0.0
    for bad in [(float("nan"), 0.1, 0), (0.3, float("nan"), 0), (0.3, 0.1, float("inf")), (0.3, -0.1, 0), (None, 0.1, 0)]:
        p, z, u = H.one_sided_p(*bad, ">")
        assert p == 1.0 and u and math.isnan(z)
    # an undefined component is returned unrejected and left out of the family: passing it or dropping it gives the
    # same decisions for the others (some scorers drop such components before the call, others pass them)
    a, b, c = comp(0.01, name="a"), dict(comp(1e-6, name="b"), est=float("nan")), comp(0.02, name="c")
    r = H.holm([a, b, c])
    assert not r[1]["reject"] and r[1]["undefined"] and r[1]["rank"] is None and math.isnan(r[1]["threshold"])
    assert r[0]["m"] == 2 and r[0]["threshold"] == 0.025 / 2 and r[0]["reject"] and r[2]["reject"]
    r2 = H.holm([a, c])
    assert [(x["reject"], x["threshold"], x["p_holm"]) for x in (r[0], r[2])] == [(x["reject"], x["threshold"], x["p_holm"]) for x in r2]
    assert H.holm([b])[0]["reject"] is False and H.holm([b])[0]["m"] == 0


def test_holm_hand_worked():
    # p = 0.004, 0.012, 0.006, 0.03 at alpha 0.025 (m = 4): sorted 0.004 <= 0.00625 reject; 0.006 <= 0.008333 reject;
    # 0.012 <= 0.0125 reject; 0.03 > 0.025 not rejected
    ps = [0.004, 0.012, 0.006, 0.03]
    r = H.holm([comp(p, name=str(p)) for p in ps])
    assert [x["name"] for x in r] == [str(p) for p in ps]                     # input order kept
    assert [x["rank"] for x in r] == [1, 3, 2, 4] and all(x["m"] == 4 for x in r)
    assert [round(x["threshold"], 10) for x in r] == [round(0.025 / k, 10) for k in (4, 2, 3, 1)]
    assert [x["reject"] for x in r] == [True, True, True, False]
    for x, p in zip(r, ps):
        assert abs(x["p"] - p) < 1e-9
    # the step-down stops at the first non-rejection: 0.007 fails 0.00625 at rank 1, so 0.0071 (rank 2, threshold
    # 0.00833, which it would pass alone) is not rejected either
    r = H.holm([comp(0.0071, name="b"), comp(0.007, name="a"), comp(0.02, name="c"), comp(0.5, name="d")])
    assert not any(x["reject"] for x in r)
    assert 0.0071 <= r[0]["threshold"]      # it would pass its own threshold: only the step-down stops it
    # alpha 0.025, not 0.05: p = 0.03 alone is not rejected, p = 0.02 alone is
    assert not H.holm([comp(0.03)])[0]["reject"] and H.holm([comp(0.02)])[0]["reject"]


def test_directions_mixed_and_passthrough():
    a = comp(0.001, ">", line="J-B3", name="qwen7: s_ID(POST) lower bound > 0.10")
    b = comp(0.001, "<", line="J-B-NULL", name="qwen7: r(POST-NULL) upper bound < 0.10")
    b["extra"] = {"model": "qwen7"}
    r = H.holm([a, b])
    assert all(x["reject"] for x in r) and r[1]["extra"] == {"model": "qwen7"} and r[1]["direction"] == "<"
    assert a.get("p") is None   # the inputs are not modified
    assert H.holm([]) == []
    with pytest.raises(KeyError):
        H.holm([{"line": "J-X", "name": "n", "est": 1, "se": 1, "bound": 0}])


def test_ties_keep_input_order():
    r = H.holm([comp(0.01, name="x"), comp(0.01, name="y"), comp(0.01, name="z")])
    assert [x["rank"] for x in r] == [1, 2, 3]
    assert [x["reject"] for x in r] == [False, False, False]      # 0.01 > 0.025 / 3


def test_adjusted_p_against_brute_force():
    rng = random.Random(7)
    for _ in range(200):
        m = rng.randint(1, 7)
        ps = [10 ** rng.uniform(-5, 0) * 0.999 for _ in range(m)]
        r = H.holm([comp(p, name=str(i)) for i, p in enumerate(ps)])
        got = [x["p"] for x in r]
        # independent Holm: closed testing with Bonferroni local tests rejects H_i iff every intersection containing i
        # has min p <= alpha / |intersection|
        for i in range(m):
            others = [j for j in range(m) if j != i]
            rej = all(min(got[j] for j in (i, *s)) <= 0.025 / (len(s) + 1) + 1e-15
                      for k in range(m) for s in itertools.combinations(others, k))
            assert r[i]["reject"] == rej, (ps, i)
            assert r[i]["reject"] == (r[i]["p_holm"] <= 0.025 + 1e-15)
            assert r[i]["p_holm"] >= r[i]["p"] - 1e-15


def test_single_component_equals_the_unadjusted_test():
    # m = 1: Holm rejects iff the one-sided 2.5% normal test does, i.e. est - 1.95996 se > bound (">")
    for est in (0.30, 0.3985, 0.3995, 0.45):
        r = H.holm([{"line": "J", "name": "n", "est": est, "se": 0.025, "bound": 0.35, "direction": ">"}])[0]
        assert r["reject"] == (est - N.inv_cdf(0.975) * 0.025 > 0.35), est
