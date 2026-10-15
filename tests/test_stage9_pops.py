"""Stage 9 populations (docs/stage9/SPEC.md 5.1-5.5, 13.1 items 1 and 2), from the functions the stage-9 steps call
(ckeys/stage9.py): the exclusion set X9 (45,832 tuples: U 3,981, the 15 seed streams 42,296, overlap 445); N9 (150 fresh
story cores of the Random(9101) stream, 161 draws, the attributes of F) and E9 (80 canonical-lexicon cores of the
Random(9102) stream satisfying cond_E, 173 draws) with their pins, determinism, coverage and attributes; C (the stage-8b
calibration cores); N9, E9 and C pairwise disjoint and disjoint from U, X9 and every stage-8 population taken from the
functions the stage-8 steps call (every stage-8 story population is inside X9; the IOI and task lists are other templates,
shown by type); the seeds; the onset grids; and the E9 prompts equal ckeys.encoding.raw_prompt byte for byte."""
import collections
import hashlib
import itertools
import json
import random

import pytest

from ckeys import edits, fresh, stage9
from ckeys.encoding import raw_prompt as enc_raw_prompt
from ckeys.story import LOCATIONS, OBJECTS, PAIR_SWAP, make_cores, pick_x, record

STAGE8_PILOT_SEEDS = {0, 1, 7, 8, 9, 99, 101, 202, 20261011}


def tup(c):
    return fresh.core_tuple(c)


@pytest.fixture(scope="module")
def flag():
    import experiments.stage8_flag as fl
    return fl


@pytest.fixture(scope="module")
def stage8(flag):
    """{name: [story core tuples]} of every templated story population of stage 8, and the other templates' lists."""
    P = {f"B.{n}": [tup(it.core) for it in fresh.population(n)] for n in ("F", "C", "S0")}
    P.update({f"C.{k}": [tup(c) for c in v] for k, v in edits.populations(check=True).items()})
    d = flag.populations(check=True)["P"]
    P.update({f"D.{k}": [tup(c) for c in v] for k, v in d.items() if "IOI" not in k and not k.startswith("X")})
    other = {k: [flag.ituple(c) if "IOI" in k else flag.ttuple(c) for c in v] for k, v in d.items()
             if "IOI" in k or k.startswith("X")}
    return P, other


def test_exclusion_set():
    X = stage9.excluded()
    U = fresh.used_cores()
    streams = {tup(c) for s in stage9.STREAM_SEEDS for c in make_cores(stage9.N_STREAM, random.Random(s))}
    assert len(X) == stage9.X9_SIZE == 45832
    assert len(U) == 3981 and len(streams) == 42296 and len(streams & U) == 445
    assert X == U | streams and stage9.excluded() is X                      # cached
    # make_cores(3000, rng) equals 3000 successive make_cores(1, rng) draws (the stream rule)
    for s in (7, 8104):
        rng = random.Random(s)
        assert [make_cores(1, rng)[0] for _ in range(300)] == make_cores(300, random.Random(s))
    assert stage9.STREAM_SEEDS == (7, 8, 9, 99, 101, 202, 20261011, 20261013, 20261014, 8101, 8102, 8103, 8104, 81, 84)


def test_n9_pins_draws_attributes():
    N = stage9.stories()
    cores = stage9.story_cores()
    assert len(N) == len(cores) == 150 and [it.core for it in N] == cores
    assert fresh.pop_hash(N) == stage9.POP_SHA256["N9"] == "0cf411ab8fe99ec495adfb30944fad69c6b65964fabde849b848cc6358d38da3"
    assert hashlib.sha256(json.dumps([list(tup(c)) for c in cores]).encode()).hexdigest() == stage9.POP_SHA256["N9_tuples"]
    # the rule, drawn independently: the first 150 cores of the Random(9101) stream in neither U nor X9 nor earlier in it
    rng, seen, out, draws = random.Random(9101), set(), [], 0
    while len(out) < 150:
        c = make_cores(1, rng)[0]
        draws += 1
        if tup(c) in stage9.excluded() or tup(c) in seen:
            continue
        seen.add(tup(c))
        out.append(c)
    assert out == cores and draws == stage9.DRAWS["N9"] == 161
    assert stage9.draw(9101, 150, stage9.excluded()) == (cores, 161)
    assert fresh.fresh_cores(150, 9101, stage9.excluded()) == cores
    # determinism: a second, uncached computation gives the same items
    stage9._story_cores.cache_clear(), stage9.stories.cache_clear()
    assert stage9.stories() == N and stage9.story_cores() == cores
    # attributes as F: fresh.attributes, keys N9000..
    assert [(it.lex, it.sent, it.order) for it in N] == fresh.attributes(cores)
    assert [it.key for it in N[:2]] == ["N9000", "N9001"] and all(it.pop == "N9" for it in N)
    assert tup(N[0].core) == ("Alice", "Bob", "ring", "phone", "drawer", "drawer", "basket", "box")
    assert (N[0].lex, N[0].sent, N[0].order) == (1, 7, (5, 0, 4, 1, 3, 2))
    assert tup(N[1].core) == ("Bob", "Alice", "book", "watch", "closet", "closet", "shelf", "box")
    assert (N[1].lex, N[1].sent, N[1].order) == (2, 5, (0, 4, 2, 3, 5, 1))
    # coverage
    assert len({(c["base"], c["source"]) for c in cores}) == 30 and {c["object"] for c in cores} == set(OBJECTS)
    assert len({it.cluster for it in N}) == 53
    assert collections.Counter(it.lex for it in N) == {1: 75, 2: 75}
    assert sorted(collections.Counter(it.sent for it in N).values(), reverse=True) == [19, 19, 19, 19, 19, 19, 18, 18]
    assert [collections.Counter(it.sent for it in N)[k] for k in range(8)] == [19] * 6 + [18] * 2
    assert all(sorted(it.order) == list(range(6)) for it in N)


def test_c_is_the_stage8b_calibration_set():
    C = stage9.calibration()
    assert C == fresh.population("C") and len(C) == 30
    assert fresh.pop_hash(C) == stage9.POP_SHA256["C"] == fresh.POP_SHA256["C"]


def test_e9_pins_draws_condition():
    E = stage9.edit_cores()
    assert len(E) == 80 and edits.pop_hash(E) == stage9.POP_SHA256["E9"]
    assert stage9.POP_SHA256["E9"] == "e4daeebd5227804ed0931302ab93d72b42b2a2d7af6222d922d8e555945c2453"
    excl = set(fresh.used_cores()) | set(stage9.excluded()) | {tup(c) for c in stage9.story_cores()}
    assert edits._take(9102, 80, excl, edits.cond_E) == E
    assert stage9.draw(9102, 80, excl, edits.cond_E) == (E, stage9.DRAWS["E9"]) and stage9.DRAWS["E9"] == 173
    stage9._edit_cores.cache_clear()
    assert stage9.edit_cores() == E                                          # deterministic
    assert all(edits.cond_E(c) for c in E)
    assert all(PAIR_SWAP[c["source"]] != c["base"] and PAIR_SWAP[pick_x(c)] != c["base"] for c in E)
    assert tup(E[0]) == ("Alice", "Bob", "card", "hat", "box", "box", "shelf", "cabinet")
    assert tup(E[1]) == ("Alice", "Bob", "card", "bag", "cabinet", "basket", "box", "shelf")
    admissible = {(b, s) for b in LOCATIONS for s in LOCATIONS if b != s and PAIR_SWAP[s] != b}
    pairs = {(c["base"], c["source"]) for c in E}
    assert len(admissible) == 24 and pairs <= admissible and len(pairs) == 22
    items = stage9.edit_items()
    assert [it.core for it in items] == E and [it.key for it in items[:2]] == ["E9000", "E9001"]
    assert all(it.pop == "E9" and it.lex == 1 and it.order == tuple(range(6)) and it.sent is None for it in items)


def test_disjoint_from_each_other_and_from_stage8(stage8):
    P, other = stage8
    U, X = set(fresh.used_cores()), set(stage9.excluded())
    N9 = {tup(c) for c in stage9.story_cores()}
    E9 = {tup(c) for c in stage9.edit_cores()}
    C = {tup(it.core) for it in stage9.calibration()}
    assert len(N9) == 150 and len(E9) == 80 and len(C) == 30
    for a, b in itertools.combinations((N9, E9, C), 2):
        assert not a & b
    for new in (N9, E9):
        assert not new & U and not new & X
    want = {"B.F", "B.C", "B.S0", "C.E", "C.H", "C.TSET", "C.THOLD", "D.R", "D.R'", "D.E8", "D.BIND"}
    assert set(P) == want
    for k, v in P.items():
        assert set(v) <= X, f"stage-8 population {k} is not inside X9"
        assert not set(v) & N9 and not set(v) & E9, k
    # the IOI candidate lists and the task cores are cores of other templates: no tuple can be a story core
    assert set(other) == {"F_IOI_CAND", "E_IOI_CAND", "XFIT_paint", "XEVAL_paint", "XFIT_schedule", "XEVAL_schedule"}
    for k, v in other.items():
        assert v and not set(v) & (N9 | E9 | C | X), k
        assert all(len(t) != len(fresh.FIELDS) for t in v), k                 # 9 (IOI) or 7 (task) fields, not 8


def test_seeds_are_new(flag):
    new = set(stage9.SEEDS.values())
    assert new == {9101, 9102}
    stage8_pop_seeds = ({s for _, s in fresh.POPULATIONS.values()} | set(fresh.U_SEEDS) | set(edits.SEEDS.values())
                        | {edits.R_SEED} | set(flag.SEEDS.values()) | set(flag.EXCL_SEEDS) | set(flag.IOI_EXCL_SEEDS))
    assert not new & STAGE8_PILOT_SEEDS and not new & stage8_pop_seeds
    # every seed that drew story cores before stage 9 is a stream of X9 (0 and 1 are U seeds)
    story_seeds = {fresh.POPULATIONS["F"][1], fresh.POPULATIONS["C"][1], *edits.SEEDS.values(), edits.R_SEED,
                   flag.SEEDS["E8"], flag.SEEDS["BIND"], *flag.EXCL_SEEDS}
    assert story_seeds | (STAGE8_PILOT_SEEDS - set(fresh.U_SEEDS)) == set(stage9.STREAM_SEEDS)


def test_onset_grids():
    assert stage9.story_onsets(32) == list(range(2, 31, 2)) and len(stage9.story_onsets(32)) == 15
    assert stage9.story_onsets(28) == list(range(2, 27, 2)) and len(stage9.story_onsets(28)) == 13
    assert stage9.story_onsets(24) == list(range(2, 23, 2)) and len(stage9.story_onsets(24)) == 11
    assert stage9.passage_onsets(32) == [5, 10, 14, 19]
    assert stage9.passage_onsets(28) == [4, 8, 13, 17]
    assert stage9.passage_onsets(24) == [4, 7, 11, 14]
    for nL in (24, 28, 32):
        assert set(stage9.KV_ONLY_ONSETS) <= set(stage9.story_onsets(nL))


def test_e9_prompts_equal_encoding_raw_prompt():
    for it in stage9.edit_items():
        c = it.core
        for loc in (c["base"], c["source"], pick_x(c)):
            rec = record(c, "direct", loc)
            for f in stage9.EDIT_FORMATS + ("P1",):
                assert fresh.raw_prompt(it, f, loc) == enc_raw_prompt(f, rec["story"], rec["query"]), (it.key, f, loc)


def test_constants():
    assert stage9.STORY_FORMATS == ("NONE", "BEFORE", "POST-NULL", "POST", "AFTER", "LETTER")
    assert stage9.EDIT_FORMATS == ("NONE", "AFTER", "LETTER", "POST")
    assert stage9.PASSAGE_FORMATS == ("NOM", "OPTB", "OPTA")
    assert (stage9.EDIT_BLOCKS, stage9.DOSE_BLOCKS) == ((3, 7, 11, 15), (3, 7))
    assert stage9.PASSAGE_X == (0.15, 0.3, 0.45, 0.6) and (stage9.BOOT_SEED, stage9.NB) == (20261015, 10000)
    assert all(f in fresh.ARMS_ALL for f in stage9.STORY_FORMATS)


def test_artefact_pins_on_disk():
    missing = [p for p in stage9.ARTEFACT_SHA256 if not (stage9.ROOT / p).exists()]
    if len(missing) == len(stage9.ARTEFACT_SHA256):
        pytest.skip("the stage-8 results are not in this copy")
    for p, h in stage9.ARTEFACT_SHA256.items():
        if (stage9.ROOT / p).exists():
            assert stage9.file_sha(stage9.ROOT / p) == h, p
