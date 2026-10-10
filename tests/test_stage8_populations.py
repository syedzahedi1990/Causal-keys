"""Rule G6 of preregistration J across the parts: every templated story population of stage 8 is disjoint, by the full
core tuple (agent, other, object, distractor, initial, distractor_location, base, source), from U = the union of
make_cores(1000, Random(s)) for s = 0..3 and from every other stage-8 population, except the two labelled re-uses of
stage-1 cores (Part B's S0 = make_cores(150, Random(0)), JB6; Part D's fit set R = make_cores(60, Random(0)) and its
subset R'). Every part computes U identically (3,981 distinct tuples, sha256 abd1f053...). No confirmatory population is
drawn from a seed the pilots used (make_cores seeds 0, 1, 7, 8, 9, 99, 101, 202, 20261011; ckeys.ioi seeds 5, 6), except
the two labelled re-uses (seed 0). Part A uses natural items only (data/stage8a_items.json), no templated cores.

The populations are taken from the functions the GPU steps call:
  Part B  ckeys.fresh.population("F" | "C" | "S0"), ckeys.fresh.used_cores()
  Part C  ckeys.edits.populations() (E, H, TSET, THOLD), ckeys.edits.universe()
  Part D  experiments.stage8_flag.populations() (R, R', E8, BIND; F_IOI_CAND, E_IOI_CAND; XFIT/XEVAL paint, schedule),
          experiments.stage8_flag.u_set()"""
import hashlib
import itertools
import json
import random

import pytest

from ckeys import edits, fresh, ioi
from ckeys.story import make_cores

U_SHA256 = "abd1f0530a3d08a2058743f59102f8dd6dd360af06fc65dd3277c5b5eb176c3d"
S0_SHA256 = "fdd1bf1ba4d8d657f663f786c3ff92d0145e41cf02e123e602d594b183110121"
FIELDS = ("agent", "other", "object", "distractor", "initial", "distractor_location", "base", "source")
PILOT_SEEDS = {0, 1, 7, 8, 9, 99, 101, 202, 20261011}
PILOT_IOI_SEEDS = {5, 6}
REUSE = {"B.S0", "D.R", "D.R'"}       # labelled re-uses of stage-1 cores (seed 0), in U on purpose


def tup(c):
    return tuple(c[f] for f in FIELDS)


@pytest.fixture(scope="module")
def flag():
    import experiments.stage8_flag as fl
    return fl


@pytest.fixture(scope="module")
def U():
    return {tup(c) for s in range(4) for c in make_cores(1000, random.Random(s))}


@pytest.fixture(scope="module")
def pops(flag):
    """{part.population: [core tuples in order]} of every templated story population of stage 8."""
    P = {f"B.{n}": [tup(it.core) for it in fresh.population(n)] for n in ("F", "C", "S0")}
    P.update({f"C.{k}": [tup(c) for c in v] for k, v in edits.populations(check=True).items()})
    d = flag.populations(check=True)["P"]
    P.update({f"D.{k}": [tup(c) for c in v] for k, v in d.items() if "IOI" not in k and not k.startswith("X")})
    return P


def test_every_part_computes_U_identically(U, flag):
    assert len(U) == 3981
    assert hashlib.sha256(json.dumps(sorted(list(t) for t in U)).encode()).hexdigest() == U_SHA256
    assert set(fresh.used_cores()) == U and fresh.u_hash() == U_SHA256 == fresh.U_SHA256
    assert set(edits.universe()) == U and edits.U_SHA256 == U_SHA256
    assert flag.u_set() == U and flag.canon_u(flag.u_set()) == U_SHA256 == flag.U_SHA
    # the field order of every part's tuple is the entry's
    assert fresh.FIELDS == edits.STORY_FIELDS == flag.FIELDS == FIELDS


def test_populations_have_the_expected_sizes(pops):
    want = {"B.F": 150, "B.C": 30, "B.S0": 150, "C.E": 80, "C.H": 200, "C.TSET": 1000, "C.THOLD": 100,
            "D.R": 60, "D.R'": 45, "D.E8": 100, "D.BIND": 100}
    assert {k: len(v) for k, v in pops.items()} == want
    for k, v in pops.items():
        assert len(set(v)) == len(v), f"{k} repeats a core"


def test_fresh_populations_disjoint_from_U_and_from_each_other(pops, U):
    fresh_pops = {k: set(v) for k, v in pops.items() if k not in REUSE}
    assert set(fresh_pops) == {"B.F", "B.C", "C.E", "C.H", "C.TSET", "C.THOLD", "D.E8", "D.BIND"}
    for k, v in fresh_pops.items():
        assert not v & U, f"{k} shares {len(v & U)} cores with U"
    for a, b in itertools.combinations(sorted(fresh_pops), 2):
        assert not fresh_pops[a] & fresh_pops[b], f"{a} and {b} share {len(fresh_pops[a] & fresh_pops[b])} cores"
    # the labelled re-uses touch no fresh population either
    for k in REUSE:
        for a, v in fresh_pops.items():
            assert not set(pops[k]) & v, (k, a)


def test_labelled_reuses(pops, U):
    S0 = make_cores(150, random.Random(0))
    assert pops["B.S0"] == [tup(c) for c in S0]
    assert hashlib.sha256(json.dumps([list(tup(c)) for c in S0]).encode()).hexdigest() == S0_SHA256 == fresh.S0_SHA256
    assert set(pops["B.S0"]) <= U
    assert pops["D.R"] == pops["B.S0"][:60]          # Part D's fit set is the first 60 cores of S0
    assert set(pops["D.R'"]) <= set(pops["D.R"])


def test_no_confirmatory_population_uses_a_pilot_seed(flag):
    seeds = {f"B.{k}": s for k, (_, s) in fresh.POPULATIONS.items()}
    seeds.update({f"C.{k}": s for k, s in edits.SEEDS.items()})
    seeds.update({f"D.{k}": s for k, s in flag.SEEDS.items()})
    bad = {k: s for k, s in seeds.items() if s in PILOT_SEEDS and k not in REUSE}
    assert not bad, bad
    assert seeds["B.S0"] == 0 and seeds["D.R"] == 0
    # the IOI populations of Part D: not the pilots' ckeys.ioi seeds, disjoint from each other and from stage 5's
    assert not {flag.SEEDS["F_IOI"], flag.SEEDS["E_IOI"]} & (PILOT_IOI_SEEDS | {0, 1})
    d = flag.populations(check=True)
    fi = {flag.ituple(c) for c in d["P"]["F_IOI_CAND"]}
    ei = {flag.ituple(c) for c in d["P"]["E_IOI_CAND"]}
    old = {flag.ituple(c) for s in (0, 1, 5, 6) for c in ioi.make_cores(1000, random.Random(s))}
    assert fi and ei and not fi & ei and not (fi | ei) & old
