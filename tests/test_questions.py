"""Gate J-D-G0 of P-2026-10-10-J, part D (docs/PREREGISTRATION.md), tokenizer part: the prompt layouts of part D on the
first 50 cores of each population, for the Qwen2.5 and the Mistral-7B-v0.3 tokenizers (Qwen2.5-1.5B, 3B and 7B share
the Qwen tokenizer). B, S, X (and the K_N run) differ only at the writing token p; the six option rows lie after p;
the initial-state runs differ only at p_init / p_dloc; the IOI runs differ only at p and the listed names lie after p
(before p under BEFORE). Also the question arms' text, the populations (pinned hashes, disjointness, including the other
stage-8 parts' populations when their modules are importable), the K_N words and the case-marginalised form set.
A tokenizer that is neither cached nor downloadable makes its tests fail (and the scorer counts a skip in any tests/
file of the last pytest run as failing J-D-G0)."""
import json
import random

import pytest

from ckeys import ioi
from ckeys import questions as Q
from ckeys.encoding import ARM_BUILDERS, LISTING, build_prompt, raw_prompt
from ckeys.story import LOCATIONS, PREFIX, record
from experiments import stage8_flag as s8

TOKENIZERS = {"qwen": "Qwen/Qwen2.5-0.5B-Instruct", "mistral": "mistralai/Mistral-7B-Instruct-v0.3"}
MANIFEST_KEY = {"qwen": "qwen0.5", "mistral": "mistral7"}   # Qwen2.5-0.5B/1.5B/3B/7B share tokenizer.json at their pins
N = 50


@pytest.fixture(scope="module")
def pops():
    return s8.populations(check=True)


def _rev(key):
    """The revision pinned in scripts/stage8_models.json (the tokenizer the GPU run loads)."""
    from pathlib import Path
    J = json.loads((Path(__file__).resolve().parents[1] / "scripts" / "stage8_models.json").read_text())
    m = J["models"][MANIFEST_KEY[key]]
    assert m["repo"] == TOKENIZERS[key], (m["repo"], TOKENIZERS[key])
    return m["revision"]


def _tok(name):
    from transformers import AutoTokenizer
    key = next(k for k, v in TOKENIZERS.items() if v == name)
    rev = _rev(key)
    try:
        return AutoTokenizer.from_pretrained(name, revision=rev, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(name, revision=rev)   # the Hub at the pinned revision (not gated)


def test_question_arms_text():
    r = record(s8.make_cores(1, random.Random(5))[0], "direct", "box")
    for arm, q in (("Q_IN", Q.Q_IN), ("Q_OUT", Q.Q_OUT)):
        assert arm in ARM_BUILDERS
        want = PREFIX + r["story"] + "\nQuestion: " + q + "\n" + LISTING + "\nAnswer with exactly one choice.\nAnswer:"
        assert build_prompt(arm, r["story"], r["query"]) == want == raw_prompt(arm, r["story"], r["query"])
        # identical to OPTIONS-AFTER except for the question line
        p1 = raw_prompt("P1", r["story"], r["query"])
        assert p1.replace(r["query"], q) == want
    assert "not mentioned" in Q.Q_OUT and "not" not in Q.Q_IN


def test_inline_chat_text():
    c = ioi.make_cores(3, random.Random(5))[1]
    raw = Q.ioi_raw("INLINE_CHAT", c, c["io_b"], True)
    after = ioi.raw_prompt("AFTER", c, c["io_b"], True)
    assert raw == ioi.HEAD + "\n\nSentence: " + ioi.inline(c, c["io_b"], False) + "\n" + ioi.INSTR + "\nAnswer:"
    # AFTER's wrapper with the in-sentence parenthetical instead of the Choices line
    assert after.replace("\n" + ioi.listing(c), "").replace(ioi.sentence(c, c["io_b"]), ioi.inline(c, c["io_b"], False)) == raw


def test_populations(pops):
    P, sha = pops["P"], pops["sha"]
    assert sha == {k: v for k, v in s8.POP_SHA.items()}
    assert all(v == 0 for v in pops["overlap"].values()), pops["overlap"]
    assert len(P["E8"]) == 100 and len(P["BIND"]) == 100 and len(P["R"]) == 60
    assert all(c["distractor_location"] != c["initial"] for c in P["BIND"] + P["R'"])
    U = s8.u_set()
    assert s8.canon_u(U) == s8.U_SHA and len(U) == 3981
    for name in ("E8", "BIND"):
        assert not {s8.ctuple(c) for c in P[name]} & U
    # the critics' pilots: Random(0) indices 16-46 (inside U), Random(7) and Random(9)
    crit = {s8.ctuple(c) for c in s8.make_cores(47, random.Random(0))[16:]} | \
           {s8.ctuple(c) for s in (7, 9) for c in s8.make_cores(100, random.Random(s))}
    for name in ("E8", "BIND"):
        assert not {s8.ctuple(c) for c in P[name]} & crit
    strata = [s8.order_stratum(c) for c in P["BIND"]]
    assert 30 <= strata.count("object_first") <= 70


def test_disjoint_from_other_parts(pops):
    """Parts B and C: their populations, when their modules are importable (each is also inside EXCL by seed)."""
    mine = {s8.ctuple(c) for k in ("E8", "BIND") for c in pops["P"][k]}
    other = set()
    try:
        from ckeys import fresh
        for name in ("F", "C"):
            other |= {s8.ctuple(it.core) for it in fresh.population(name)}
    except Exception:  # noqa: BLE001  (module or function absent in this checkout)
        pass
    try:
        from ckeys import edits
        Pc = edits.populations(check=False)
        other |= {s8.ctuple(c) for k in ("E", "H", "TSET", "THOLD") for c in Pc.get(k, [])}
    except Exception:  # noqa: BLE001
        pass
    assert not mine & other
    X = s8.excl_set()
    assert not mine & X


@pytest.mark.parametrize("key", list(TOKENIZERS))
def test_belief_layouts(key, pops):
    tok = _tok(TOKENIZERS[key])
    P = pops["P"]
    for c in P["E8"][:N]:
        for arm in ("P1", "POST", "Q_IN", "Q_OUT"):
            d = s8.prep(tok, c, arm, nrun=True)
            assert d is not None, (key, arm, c)
            ib = d["ids"]["B"][0]
            for r in ("S", "X", "N"):
                diff = (ib != d["ids"][r][0]).nonzero().flatten().tolist()
                assert diff == [d["p"]], (key, arm, r)
            assert min(d["G"]) > d["p"] and len(d["G"]) == 6
            assert d["first"] > d["p"] if arm != "POST" else True
    for c in P["BIND"][:N]:
        for view in s8.BIND_VIEWS:
            assert s8.prep(tok, c, "P1", view) is not None
    for c in P["R'"]:
        d = s8.prep(tok, c, "P1", initial_runs=True)
        assert d is not None
        ib = d["ids"]["B"][0]
        for r in ("Iinit", "Idloc"):
            assert (ib != d["ids"][r][0]).nonzero().flatten().tolist() == [d["pos"][r]]
        assert d["Ip"] not in {c["initial"], c["distractor_location"], c["base"], c["source"], d["X"]}


@pytest.mark.parametrize("key", list(TOKENIZERS))
def test_ioi_layouts(key, pops):
    tok = _tok(TOKENIZERS[key])
    arms = ("INLINE", "INLINE_CHAT", "AFTER", "BEFORE")
    ok = 0
    for c in pops["P"]["E_IOI_CAND"][:N]:
        for arm in arms:
            if Q.ioi_valid(tok, c, arm, True) is not None:
                continue
            L = Q.ioi_layout(tok, c, arm, True)
            ib = L["ids"]["B"][0]
            for r in ("S", "X"):
                assert (ib != L["ids"][r][0]).nonzero().flatten().tolist() == [L["p"]]
            opts = L["groups"]["options"]
            assert len(opts) == 4 and sorted(L["named"].values()) == sorted(opts)
            assert all(i < L["p"] for i in opts) if arm == "BEFORE" else all(i > L["p"] for i in opts)
            assert ib[L["named"]["io_b"]].item() == ib[L["p"]].item()      # the listed IO_B is the written name
            ok += 1
    assert ok >= 0.8 * N * len(arms), (key, ok)


@pytest.mark.parametrize("key", list(TOKENIZERS))
def test_words_and_forms(key):
    tok = _tok(TOKENIZERS[key])
    for w in s8.N_WORDS:
        assert len(tok.encode(" " + w, add_special_tokens=False)) == 1 and w not in LOCATIONS
    fs = s8.fs_of(tok)
    for w in LOCATIONS:
        assert fs.lower[w] in fs.sets["E"][w]
        assert {tok.decode(list(s)).strip() for s in fs.sets["E"][w]} <= {w, w.capitalize()}
    for t in ("paint", "schedule"):
        for c in s8.task_cores(t, 3, 85):
            d = s8.prep_task(tok, t, c)
            assert min(d["G"]) > d["p"]


def test_ioi_candidate_lists_fixed():
    a = json.dumps([list(s8.ituple(c)) for c in ioi.make_cores(90, random.Random(82))])
    assert s8.pop_sha(ioi.make_cores(90, random.Random(82)), s8.ituple) == s8.POP_SHA["F_IOI_CAND"] and a
