"""Stage 9, passages arm (docs/stage9/SPEC.md 13.2): FP32 exactness of the window rows of experiments/stage9_passages.py
at Qwen2.5-0.5B-Instruct on the CPU (1e-4 in log-probabilities, 1e-5 in K/V tables), the 48-token generation against the
cache-free stepwise reference, the stored clean B run (clean_B, the reference of the BF16 floor) against a plain forward
of the B prompt, the two answer classes and the containment normaliser (possessives and Unicode punctuation, SPEC 9.3) on
hand cases, the valid E and R sets with the cached study tokenizers, and a TEST_MODE subprocess run of check and eval
(the keys of SPEC 11.3 and 11.6, resume, the deadline).

Items: two E items (IDS) with windowed passages (experiments.natural_factorial.window, as in TEST_MODE) from the
committed SQuAD file; the subprocess run takes the first valid E and R item. Every reference is computed independently
of the batch it checks: a plain forward of the S or B prompt, single-row clamps built from separately captured runs with
the layer sets of SPEC 7.1 written out here (``hand_sets``), nested clamps on disjoint layer ranges, or the cache-free
stepwise greedy reference."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys import stage9 as s9
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.generate import greedy_reference
from ckeys.interventions import blocks
from ckeys.natural_formats import FORMATS, norm
from ckeys.natural_rows import capture, cut, decided_stop, end_ids, generate_rows, prep, score_rows, tables, valid_all
from experiments import stage9_passages as sp
from experiments.natural_factorial import load_items, sha_ids, split

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL, KVTOL = 1e-4, 1e-5
ROOT = Path(__file__).resolve().parents[1]
TOKENIZERS = {"qwen7": "Qwen/Qwen2.5-7B-Instruct", "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
              "llama8": "unsloth/Meta-Llama-3.1-8B-Instruct"}     # llama8: the build's byte-identical tokenizer copy
torch.set_grad_enabled(False)


# --------------------------------------------------------------------------- fixtures and references
@pytest.fixture(scope="module")
def qwen():
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    return model, tok, len(blocks(model))


# Two E items with short windowed prompts (T 79 and 84 in NOM at Qwen2.5-0.5B): the first has a two-token answer span
# and a free-form answer longer than W + 2 words (the B run: "George Low ended up replacing Joseph Francis Shea as ASPO
# Manager."), so the missing stage-8a cut is visible.
IDS = ("5725dd1689a1e219009abfe4", "56e75e4500c9c71400d7702c")


@pytest.fixture(scope="module")
def items(qwen):
    """The two items, passages windowed as in TEST_MODE; both valid in every stage-8a format at frame " "."""
    _, tok, _ = qwen
    its = {i["id"]: i for i in load_items(argparse.Namespace(squad=str(ROOT / sp.SQUAD), items=str(sp.ITEMS), test=True))}
    out = [its[i] for i in IDS]
    assert all(i["split"] == "E" and valid_all(tok, i, FORMATS, " ") is None for i in out)
    return out


def hand_sets(name, nL):
    """(layers whose key comes from S, layers whose value comes from S) of a row, from SPEC 7.1's table."""
    every = set(range(nL))
    if name in ("none", "none2"):
        return set(), set()
    if name == "KV@0":
        return every, every
    if name == "K@0":
        return every, set()
    if name == "V@0":
        return set(), every
    kind, l = name.split("@")
    late, early = set(range(int(l), nL)), set(range(int(l)))
    return {"late": (late, late), "early": (early, early), "Klate": (late, set()), "Vlate": (set(), late)}[kind]


def names_for(nL):
    on = s9.passage_onsets(nL)
    return (["none", "KV@0"] + [f"late@{l}" for l in on] + [f"early@{l}" for l in on] + ["K@0", "V@0"]
            + [f"Klate@{l}" for l in on] + [f"Vlate@{l}" for l in on] + ["none2"])


def lsm(model, ids, start):
    """Plain forward: log-softmax rows from position ``start`` on (FP32)."""
    return model(torch.tensor([ids]), use_cache=False).logits[0, start:].float().log_softmax(-1)


def single_kv(model, ids, pos, nL):
    """K/V at ``pos`` from a single forward of ``ids`` alone (no w)."""
    with capture_kv(model, pos, range(nL)) as C:
        model(torch.tensor([ids]), use_cache=False, logits_to_keep=1)
    return {k: v[0] for k, v in C.items()}


def hand_tabs(own, nL, ks, vs):
    return ({l: own["S" if l in ks else "B"][(l, "k")] for l in range(nL)},
            {l: own["S" if l in vs else "B"][(l, "v")] for l in range(nL)})


def hand_clamps(model, P, nL, kt, vt):
    """Two clamp_kv contexts (keys, values) with [|P|, D] tables per layer: the single-row reference clamp."""
    return (clamp_kv(model, P, {(l, "k"): kt[l] for l in range(nL)}, range(nL), "k"),
            clamp_kv(model, P, {(l, "v"): vt[l] for l in range(nL)}, range(nL), "v"))


def ref_scores(model, d, nL, kt, vt):
    c, j, T = d["c"]["S"], d["j"], d["T"]
    ck, cv = hand_clamps(model, d["P"], nL, kt, vt)
    with ck, cv:
        l0 = lsm(model, d["ids"]["B"] + c, T - 1 + j)[0]
    opts = [d["dec"][Y] for Y in ("B", "S", "X", "D") if Y in d["dec"]]
    return {Y: float(l0[t]) for Y, t in d["dec"].items()}, int(l0.argmax()), float(l0[opts].exp().sum())


def kv_scale(tabs):
    """1e-5 of the tables' scale: FP32 ulps at |K| ~ 100 (the house rule of tests/test_stage8_edits.py)."""
    return KVTOL * max(1.0, max(float(t.abs().max()) for t in tabs.values()))


def close(a, b, tol=TOL):
    return all(abs(a[Y] - b[Y]) < tol for Y in b) and set(a) == set(b)


# --------------------------------------------------------------------------- 13.2.1 rows, tables, identities
def test_rows_layout_and_donors():
    for nL, on in ((24, [4, 7, 11, 14]), (28, [4, 8, 13, 17]), (32, [5, 10, 14, 19])):
        assert s9.passage_onsets(nL) == on
        rows = sp.window_rows(nL)
        assert list(rows) == names_for(nL) and len(rows) == 21
        for n, r in rows.items():
            ks, vs = hand_sets(n, nL)
            for l in range(nL):
                for i in (0, 3):
                    assert r.donor(l, "k", i) == ("S" if l in ks else "B"), (nL, n, l)
                    assert r.donor(l, "v", i) == ("S" if l in vs else "B"), (nL, n, l)
    assert sp.Win("late", 0).donor(0, "k") == "S" and sp.Win("early", 24).donor(23, "v") == "S"
    assert all(sp.Win("early", 0).donor(l, ch) == "B" for l in range(24) for ch in "kv")


@pytest.mark.parametrize("fmt", ["NOM", "OPTB", "OPTA", "LETA"])
def test_window_rows_equal_hand_built_clamps(qwen, items, fmt):
    """Every row of the 21-row scoring batch equals a single-row reference clamp built from separately captured runs
    (decision log-probabilities, option mass and argmax 1e-4); the batch's tables equal the single captures (1e-5 of the
    tables' scale)."""
    model, tok, nL = qwen
    it = items[0]
    d = prep(tok, it, fmt, " ")
    kv, _ = capture(model, d, range(nL), "cpu")
    own = {Y: single_kv(model, d["ids"][Y], d["P"], nL) for Y in "BS"}
    rows = sp.window_rows(nL)
    tabs = tables(kv, list(rows.values()), nL, len(d["P"]))
    tol = kv_scale(tabs)
    for r, n in enumerate(rows):
        ks, vs = hand_sets(n, nL)
        for l in range(nL):
            assert (tabs[(l, "k")][r] - own["S" if l in ks else "B"][(l, "k")]).abs().max() <= tol, (n, l)
            assert (tabs[(l, "v")][r] - own["S" if l in vs else "B"][(l, "v")]).abs().max() <= tol, (n, l)
    sc = score_rows(model, d, kv, rows, nL, "cpu", "S")
    for n in rows:
        lp, am, mass = ref_scores(model, d, nL, *hand_tabs(own, nL, *hand_sets(n, nL)))
        assert close(sc[n]["lp"], lp), (fmt, n)
        assert sc[n]["argmax"] == am and abs(sc[n]["mass"] - mass) < TOL, (fmt, n)
    assert close(sc["none"]["lp"], sc["none2"]["lp"])
    assert sc["KV@0"]["lp"]["S"] - sc["none"]["lp"]["S"] > 0.5, "the S passage must move the S decision token"
    assert set(sc["none"]["lp"]) == ({"B", "S", "X", "D"} if fmt == "LETA" else {"B", "S", "X", "Z", "D"})


@pytest.mark.parametrize("fmt", ["NOM", "OPTA", "LETA"])
def test_full_patch_is_the_s_run_and_identity_is_the_b_run(qwen, items, fmt):
    """The window ends: late@0 = early@L = KV@0 = the S prompt's own forward, and early@0 = late@L = none = the B
    prompt's own (scores 1e-4); the K/V of every position after the span, captured under the clamp of late@0 and of
    early@0, equal the clean S and B runs' (1e-5 of the tables' scale). (SPEC 7.1: early@l takes S on [0, l), so
    early@L is the full patch and early@0 the identity, "KV@0 = late@0 and none = early@0".)"""
    model, tok, nL = qwen
    it = items[1]
    d = prep(tok, it, fmt, " ")
    kv, _ = capture(model, d, range(nL), "cpu")
    ext = {"late@0": sp.Win("late", 0), "early@L": sp.Win("early", nL), "KV@0": sp.Win("KV"),
           "early@0": sp.Win("early", 0), "late@L": sp.Win("late", nL), "none": sp.Win("none")}
    sc = score_rows(model, d, kv, ext, nL, "cpu", "S")
    c, j, T = d["c"]["S"], d["j"], d["T"]
    for row in ext:
        run = "S" if row in ("late@0", "early@L", "KV@0") else "B"
        ref = lsm(model, d["ids"][run] + c, T - 1 + j)[0]
        assert close(sc[row]["lp"], {Y: float(ref[t]) for Y, t in d["dec"].items()}), (fmt, row)
        assert sc[row]["argmax"] == int(ref.argmax()), (fmt, row)
    assert abs(sc["KV@0"]["lp"]["S"] - sc["none"]["lp"]["S"]) > 0.5
    later = list(range(d["P"][-1] + 1, T))
    tabs = tables(kv, [ext["late@0"], ext["early@0"]], nL, len(d["P"]))
    with clamp_kv(model, d["P"], tabs, range(nL)), capture_kv(model, later, range(nL)) as C:
        model(torch.tensor([d["ids"]["B"]] * 2), use_cache=False, logits_to_keep=1)
    for r, run in enumerate("SB"):
        clean = single_kv(model, d["ids"][run], later, nL)
        err = max(float((C[key][r] - clean[key]).abs().max()) for key in clean)
        assert err <= kv_scale(clean), (fmt, run, err)
    assert len(later) > 10


def test_nested_complementarity_at_span_positions(qwen, items):
    """S on [0, l) and S on [l, L) by two nested clamp_kv contexts = KV@0; S on [0, l) nested with B on [l, L) = early@l;
    B on [0, l) nested with S on [l, L) = late@l (decision log-probabilities 1e-4), l in (2, 12, 22)."""
    model, tok, nL = qwen
    it = items[0]
    d = prep(tok, it, "OPTA", " ")
    kv, _ = capture(model, d, range(nL), "cpu")
    own = {Y: single_kv(model, d["ids"][Y], d["P"], nL) for Y in "BS"}
    c, j, T, P = d["c"]["S"], d["j"], d["T"], d["P"]
    ls = (2, 12, 22)
    rows = {"KV@0": sp.Win("KV")} | {f"early@{l}": sp.Win("early", l) for l in ls} | {f"late@{l}": sp.Win("late", l) for l in ls}
    sc = score_rows(model, d, kv, rows, nL, "cpu", "S")

    def part(run, layers):
        return {(l, ch): own[run][(l, ch)] for l in layers for ch in "kv"}

    def nested(lo, hi, l):
        with clamp_kv(model, P, part(lo, range(l)), range(l)), clamp_kv(model, P, part(hi, range(l, nL)), range(l, nL)):
            l0 = lsm(model, d["ids"]["B"] + c, T - 1 + j)[0]
        return {Y: float(l0[t]) for Y, t in d["dec"].items()}

    for l in ls:
        assert close(sc["KV@0"]["lp"], nested("S", "S", l)), l
        assert close(sc[f"early@{l}"]["lp"], nested("S", "B", l)), l
        assert close(sc[f"late@{l}"]["lp"], nested("B", "S", l)), l
    assert abs(sc["early@22"]["lp"]["S"] - sc["late@2"]["lp"]["S"]) > 1e-3, "the windows must differ"


# --------------------------------------------------------------------------- 13.2.2 generation
def unstopped_words(tok, it):
    """W of the stage-8a cut: the most normalised words of a candidate."""
    return max(len(norm(it[k]).split()) for k in ("answer", "S", "X", "Z", "D"))


@pytest.mark.parametrize("fmt", ["NOM", "LETA"])
def test_generation_equals_the_stepwise_reference(qwen, items, fmt):
    """run_item's 48-token generation batch (21 rows and the unclamped S run) equals greedy_reference of each row alone
    under its own hand-built clamp (cache-free forwards at every step), read up to the first special token. NOM has no
    W + 2 word cut; LETA stops at the two-character letter rule."""
    model, tok, nL = qwen
    it = items[0]
    rows = sp.window_rows(nL)
    rec = sp.run_item(model, tok, it, fmt, " ", rows, nL, "cpu", "E")
    d = prep(tok, it, fmt, " ")
    own = {Y: single_kv(model, d["ids"][Y], d["P"], nL) for Y in "BS"}
    stop = decided_stop(tok, it, fmt) if fmt == "LETA" else None
    ends = end_ids(tok)
    xb = torch.tensor([d["ids"]["B"]])
    for n in rows:
        ck, cv = hand_clamps(model, d["P"], nL, *hand_tabs(own, nL, *hand_sets(n, nL)))
        with ck, cv:
            ref = greedy_reference(model, tok, xb, max_new=sp.MAX_NEW, stop=stop)[0]
        assert rec["rows"][n]["ids"] == cut(ref, ends), (fmt, n)
        assert rec["rows"][n]["n_tok"] == len(rec["rows"][n]["ids"])
    ref = greedy_reference(model, tok, torch.tensor([d["ids"]["S"]]), max_new=sp.MAX_NEW, stop=stop)[0]
    assert rec["S_run"]["ids"] == cut(ref, ends)
    assert rec["rows"]["KV@0"]["ids"] == rec["S_run"]["ids"] and rec["rows"]["none"]["ids"] == rec["rows"]["none2"]["ids"]
    # clean_B (the reference of the BF16 floor): the B prompt's own decision log-probabilities from the capture batch,
    # equal to a plain forward of the B prompt and, in FP32, to the none row of the scoring batch (1e-4)
    c, j, T = d["c"]["S"], d["j"], d["T"]
    l0 = lsm(model, d["ids"]["B"] + c, T - 1 + j)[0]
    assert set(rec["clean_B"]) == {"lp"} and set(rec["clean_B"]["lp"]) == set(d["dec"])
    assert close(rec["clean_B"]["lp"], {Y: float(l0[t]) for Y, t in d["dec"].items()}), fmt
    assert close(rec["clean_B"]["lp"], rec["rows"]["none"]["lp"]), fmt
    if fmt == "LETA":
        st = decided_stop(tok, it, fmt)
        fired = [n for n in rows if rec["rows"][n]["ids"] and st(0, rec["rows"][n]["ids"])]
        assert fired, "the letter rule must end some row"
        assert all(n in fired or rec["rows"][n]["n_tok"] < sp.MAX_NEW for n in rows)
        free = generate_rows(model, tok, it, d, capture(model, d, range(nL), "cpu")[0], {n: rows[n] for n in fired}, nL,
                             sp.MAX_NEW, stop=False)
        assert all(free[n]["ids"][:rec["rows"][n]["n_tok"]] == rec["rows"][n]["ids"] for n in fired)
        assert any(len(free[n]["ids"]) > rec["rows"][n]["n_tok"] for n in fired), "the rule must cut some row short"
    else:   # no W + 2 cut: rows run past W + 2 normalised words, where the stage-8a rule (decided_stop) would have ended them
        W, st = unstopped_words(tok, it), decided_stop(tok, it, fmt)
        texts = {n: r["text"] for n, r in rec["rows"].items()} | {"S_run": rec["S_run"]["text"]}
        long_rows = [n for n, t in texts.items() if len(norm(t).split()) > W + 2]
        assert "none" in long_rows and "S_run" in long_rows, long_rows
        g = rec["rows"]["none"]["ids"]
        assert next(i for i in range(1, len(g) + 1) if st(0, g[:i])) < len(g)


# --------------------------------------------------------------------------- 13.2.3 answer classes
IT = {"answer": "Bing Crosby", "S": "Darren Fletcher", "X": "Peyton Manning", "Z": "James Lafayette", "D": "Ed McCaffrey",
      "options": ["Peyton Manning", "Darren Fletcher", "Bing Crosby", "Ed McCaffrey"]}


@pytest.mark.parametrize("text,strict,contain", [
    (" Bing Crosby", "B", "B"),
    (" **Bing Crosby**", "B", "B"),
    (" the Bing Crosby show", "B", "B"),
    (" Darren Fletcher, as the passage says", "S", "S"),
    (" The answer is Darren Fletcher.", "other", "S"),
    (" According to the passage, the network signed Peyton Manning", "other", "X"),
    (" Peyton Manning and then Bing Crosby", "X", "X"),
    (" Not Bing Crosby but James Lafayette", "other", "B"),
    (" It was Ed McCaffrey.\nBing Crosby", "other", "D"),
    (" Ed McCaffreys", "other", "other"),
    (" Crosby", "other", "other"),
    (" Darrenfletcher", "other", "other"),
    ("", "other", "other"),
    (" none of them", "other", "other"),
    # SPEC 9.3's normaliser: a possessive and Unicode punctuation never hide a contained answer (strict: the stage-8a rule)
    (" Bing Crosby's band", "other", "B"),
    (" Bing Crosby\u2019s band", "other", "B"),
    (" Darren Fletcher\u2019s", "other", "S"),
    (" \u201cDarren Fletcher\u201d", "other", "S"),
    (" \u00abPeyton Manning\u00bb", "other", "X"),
    (" Peyton Manning\u2014the quarterback", "other", "X"),
    (" Peyton Manning\u2013Ed McCaffrey", "other", "X"),
    (" Ed McCaffrey/Bing Crosby", "other", "D"),
    (" Bing Crosby\u2026", "other", "B"),
    (" Bing Crosby\u3002", "other", "B"),
    (" (Ed McCaffrey)", "D", "D"),
    (" Crosby's", "other", "other"),
    (" Bing Crosbys", "other", "other"),
])
def test_who_contain_and_who_strict(text, strict, contain):
    assert sp.answers(text, IT, "NOM") == {"who_strict": strict, "who_contain": contain}
    assert sp.who_contain(text, IT, "OPTA") == contain


def test_norm9():
    """The containment normaliser (SPEC 9.3): the stage-8a norm on ASCII text without a possessive, dash or slash; a
    possessive 's / \u2019s dropped; Unicode punctuation removed; dashes and the slash split words; the same rule for
    entities, so an entity with a possessive or punctuation of its own still matches its own spelling."""
    for s in (" The answer is Darren Fletcher.", "**Bing Crosby**", "U.S. Army, 1,000 men", "(an) Ed McCaffrey!", "$5 million",
              "Joneses' house", "C'est la vie", ""):
        assert sp.norm9(s) == norm(s), s
    assert sp.norm9("Bing Crosby's") == sp.norm9("Bing Crosby\u2019s") == sp.norm9("\u201cBing Crosby\u201d") == "bing crosby"
    assert sp.norm9("Crosby\u2014the singer") == "crosby singer" and sp.norm9("Crosby/Hope") == "crosby hope"
    assert sp.norm9("Jay-Z") == "jay z" and sp.norm9("1914\u20131918") == sp.norm9("1914-1918") == "1914 1918"
    assert sp.norm9("Bing Crosbys") == "bing crosbys" and sp.norm9("St. John's Wood") == "st john wood"
    it = dict(IT, answer="St. John's Wood")
    assert sp.who_contain(" It is St John\u2019s Wood.", it, "NOM") == "B" == sp.who_contain(" St. John's Wood", it, "NOM")


def test_who_contain_ties_and_subwords():
    it = {"answer": "John Smith", "S": "John Smith Jr", "X": "Ford", "Z": "Ann Lee", "D": "Lee", "options": []}
    assert sp.who_contain(" John Smith Jr won", it, "NOM") == "S"        # same start: the longer entity
    assert sp.who_contain(" John Smith won", it, "NOM") == "B"
    assert sp.who_contain(" Fordham University", it, "NOM") == "other"   # an entity inside a word is not counted
    assert sp.who_contain(" Henry Ford", it, "NOM") == "X"
    assert sp.who_contain(" Ann Lee", it, "NOM") == "Z"                   # earliest start wins over the shorter D
    assert sp.who_contain(" Lee and Ann Lee", it, "NOM") == "D"


@pytest.mark.parametrize("text,who", [(" B", "S"), (" **C**", "B"), (" (A) Peyton Manning", "X"), (" D.", "D"),
                                      (" Darren Fletcher", "other"), (" The answer is A", "other"), (" Alpha", "other"),
                                      ("", "other")])
def test_leta_letters(text, who):
    assert sp.answers(text, IT, "LETA") == {"who_strict": who, "who_contain": who}


# --------------------------------------------------------------------------- 13.2.4 the valid sets (cached tokenizers)
def _tok(name):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError as ex:
        pytest.skip(f"tokenizer {name} not cached: {type(ex).__name__}")


@pytest.fixture(scope="module")
def full_items():
    return load_items(argparse.Namespace(squad=str(ROOT / sp.SQUAD), items=str(sp.ITEMS), test=False))


@pytest.mark.parametrize("key", list(TOKENIZERS))
def test_valid_sets_cached_tokenizer(key, full_items):
    """At frame " ": the valid E set has 185 items and sha_ids equal to the stage-8a factorial's population_sha256; the
    R set is computed, within split R, and the same on a second computation."""
    tok = _tok(TOKENIZERS[key])
    a = argparse.Namespace(key=key, test=False)
    frame, ffile, fsha = sp.read_frame(a)
    assert frame == " " and fsha == s9.ARTEFACT_SHA256[ffile]
    E, skE = sp.valid_split(tok, full_items, "E", frame)
    assert len(E) == sp.N_E == 185 and not skE
    assert sha_ids([i["id"] for i in E]) == sp.stage8a_population(a) == "57fb5fa1556dc8072950a80b2e992fb374fa96679987c2c1cda76bdda4eeb8ba"
    R, skR = sp.valid_split(tok, full_items, "R", frame)
    R2, skR2 = sp.valid_split(tok, full_items, "R", frame)
    ids = [i["id"] for i in split(full_items, "R")]
    assert [i["id"] for i in R] == [i["id"] for i in R2] and skR == skR2
    assert len(R) + len(skR) == len(ids) == 71 and set(i["id"] for i in R) <= set(ids)
    # the tokenizer check of 2026-10-14: every R item is valid at the three study tokenizers
    assert len(R) == 71 and sha_ids([i["id"] for i in R]) == "ef23b97c9fa58a19693736ec34095befa589339ec6266e9710f767142eabb8df"
    assert len({i["art"] for i in E}) == 34 and len({i["art"] for i in split(full_items, "R")}) == 12


# --------------------------------------------------------------------------- the TEST_MODE subprocess run
ITEM_KEYS = {"id", "art", "title", "sub", "D_in", "stratum", "split", "fmt", "nP", "P", "T", "j", "w", "dec", "rows", "S_run",
             "clean_B"}
ROW_KEYS = {"lp", "mass", "argmax", "text", "ids", "n_tok", "who_strict", "who_contain"}
PROV_KEYS = {"git_commit", "git_dirty", "freeze_commit", "predictions_sha256", "spec_sha256", "started_utc", "finished_utc",
             "stage9", "arm", "stage", "model", "model_key", "revision", "verified", "dtype", "attn_implementation", "device",
             "n_layers", "hidden", "n_heads", "n_kv_heads", "test_mode", "population", "population_sha256", "n_items",
             "skipped_items", "formats", "formats_not_run", "onsets", "kv_only_onsets", "max_new", "reserve_min", "timings",
             "items_sha256", "squad_sha256", "frame", "frames_file", "frames_sha256", "valid_sha256",
             "stage8a_population_sha256", "torch", "transformers", "python", "args", "wrapper"}


def run(out, *args, env=None):
    e = dict(os.environ, TEST_MODE="1", PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1") | (env or {})
    e.pop("STAGE8_DEADLINE", None) if "STAGE8_DEADLINE" not in (env or {}) else None
    p = subprocess.run([sys.executable, "-m", "experiments.stage9_passages", "--key", "qwen7", "--out", str(out), "--n", "1",
                        *args], cwd=ROOT, env=e, capture_output=True, text=True, timeout=3600)
    return p.returncode, p.stdout + p.stderr


def test_test_mode_check_eval_resume_and_deadline(tmp_path):
    rc, log = run(tmp_path, "--stage", "check")
    assert rc == 0, log
    C = json.loads((tmp_path / "check" / "TEST_qwen7_passages.json").read_text())
    assert {"provenance", "pass", "fails", "skipped", "valid", "valid_sha256", "n_valid", "onsets", "rows"} <= set(C)
    assert C["pass"] and not C["fails"] and C["n_valid"] == {"E": 1, "R": 1} and C["onsets"] == [4, 7, 11, 14]
    assert C["provenance"]["arm"] == "passages" and C["provenance"]["stage"] == "check" and C["provenance"]["test_mode"]

    path = tmp_path / "passages" / "TEST_qwen7.json"
    rc, log = run(tmp_path, "--stage", "eval", env={"STAGE8_DEADLINE": str(time.time() - 60)})
    assert rc == 3, log                                           # past the deadline: no piece is started
    J = json.loads(path.read_text())
    assert J["provenance"]["formats_not_run"] == [f"{s}/{f}" for s in ("E", "R") for f in ("NOM", "OPTB", "OPTA")]
    assert J["splits"] == {"E": {}, "R": {}} and not J["provenance"]["complete"]
    assert "weights not loaded" in log and "load" not in J["provenance"]["timings"], log   # the weights never load
    assert PROV_KEYS <= set(J["provenance"]), PROV_KEYS - set(J["provenance"])

    rc, log = run(tmp_path, "--stage", "eval")
    assert rc == 0, log
    J = json.loads(path.read_text())
    P = J["provenance"]
    assert PROV_KEYS <= set(P), PROV_KEYS - set(P)
    assert P["complete"] and not P["formats_not_run"] and P["onsets"] == P["kv_only_onsets"] == [4, 7, 11, 14]
    assert P["dtype"] == "torch.float32" and P["frame"] == " " and P["frames_file"] is None and P["max_new"] == 48
    assert P["valid_sha256"] == C["valid_sha256"] and P["items_sha256"] == sp.ITEMS_SHA256
    assert J["rows"] == names_for(24) and J["frame"] == " "
    for s in ("E", "R"):
        assert list(J["splits"][s]) == ["NOM", "OPTB", "OPTA"]
        for f, recs in J["splits"][s].items():
            assert len(recs) == 1 and recs[0]["id"] == C["valid"][s][0]
            r = recs[0]
            assert set(r) == ITEM_KEYS and r["split"] == s and r["fmt"] == f and list(r["rows"]) == J["rows"]
            assert all(set(x) == ROW_KEYS for x in r["rows"].values())
            assert set(r["S_run"]) == {"text", "ids", "n_tok", "who_strict", "who_contain"}
            assert set(r["dec"]) == {"B", "S", "X", "Z", "D"} and r["nP"] == len(r["P"])
            assert set(r["clean_B"]) == {"lp"} and set(r["clean_B"]["lp"]) == set(r["dec"])
            assert close(r["clean_B"]["lp"], r["rows"]["none"]["lp"])          # FP32: the floor is rounding only

    assert "load" in P["timings"] and P["kept"] == [] and P["carried"] == []
    pieces = [f"{s}/{f}" for s in ("E", "R") for f in ("NOM", "OPTB", "OPTA")]

    rc, log = run(tmp_path, "--stage", "eval")                    # a rerun keeps every piece, in run order, unloaded
    assert rc == 0, log
    J2 = json.loads(path.read_text())
    assert J2["provenance"]["kept"] == pieces and J2["provenance"]["complete"] and "weights not loaded" in log
    assert json.dumps(J2["splits"]) == json.dumps(J["splits"])

    rc, log = run(tmp_path, "--stage", "eval", "--splits", "E", "--formats", "OPTA")   # a narrower rerun drops nothing
    assert rc == 0, log
    J3 = json.loads(path.read_text())
    assert json.dumps(J3["splits"]) == json.dumps(J["splits"]) and "weights not loaded" in log
    assert J3["provenance"]["kept"] == pieces and J3["provenance"]["carried"] == [p for p in pieces if p != "E/OPTA"]
    assert J3["provenance"]["complete"] and J3["provenance"]["formats"] == ["OPTA"] and J3["provenance"]["splits"] == ["E"]

    del J3["splits"]["E"]["OPTB"]                                 # a resume that runs the middle piece again
    path.write_text(json.dumps(J3))
    rc, log = run(tmp_path, "--stage", "eval")
    assert rc == 0, log
    J4 = json.loads(path.read_text())
    assert J4["provenance"]["kept"] == [p for p in pieces if p != "E/OPTB"] and "load" in J4["provenance"]["timings"]
    assert {s: list(fm) for s, fm in J4["splits"].items()} == {s: ["NOM", "OPTB", "OPTA"] for s in ("E", "R")}
    old, new = J["splits"]["E"]["OPTB"][0], J4["splits"]["E"]["OPTB"][0]
    assert {k: v for k, v in new.items() if k != "rows"} == {k: v for k, v in old.items() if k != "rows"}
    for n in J["rows"]:
        assert new["rows"][n]["ids"] == old["rows"][n]["ids"] and close(new["rows"][n]["lp"], old["rows"][n]["lp"]), n
    assert all(J4["splits"][s][f] == J["splits"][s][f] for s in ("E", "R") for f in ("NOM", "OPTA"))

    rc, log = run(tmp_path, "--stage", "eval", "--formats", "LETA", "--splits", "E")
    assert rc == 0, log
    L = json.loads((tmp_path / "passages" / "TEST_qwen7_LETA.json").read_text())
    assert list(L["splits"]) == ["E"] and list(L["splits"]["E"]) == ["LETA"]
    r = L["splits"]["E"]["LETA"][0]
    assert set(r["dec"]) == {"B", "S", "X", "D"} and all(x["who_strict"] == x["who_contain"] for x in r["rows"].values())
    rc, log = run(tmp_path, "--stage", "eval", "--formats", "NOM,LETA")
    assert rc != 0 and "LETA runs alone" in log


class _Loop:
    """stage_eval's piece loop with the item work mocked (in process, TEST_MODE tokenizer and config): run_item returns a
    record named by (split, format, id), load_model a parameter-only model with the tiny model's widths, and the deadline
    passes once ``budget`` pieces have run (None: never), so the gating is exact and needs no timing."""

    def __init__(self, monkeypatch, out):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(NAME)
        self.out, self.loads, self.ran, self.budget = out, 0, [], None

        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))
                self.config = argparse.Namespace(_attn_implementation="sdpa", hidden_size=cfg.hidden_size,
                                                 num_attention_heads=cfg.num_attention_heads,
                                                 num_key_value_heads=cfg.num_key_value_heads)

        def load(a):
            self.loads += 1
            return Fake(), None, "cpu"

        def run_item(model, tok, it, fmt, frame, rows, nL, dev, name, max_new=sp.MAX_NEW):
            if (name, fmt) not in self.ran:
                self.ran.append((name, fmt))
            return {"id": it["id"], "art": it["art"], "split": name, "fmt": fmt, "rec": f"{name}/{fmt}/{it['id']}"}

        def valid_split(tok, items, name, frame, n=0):
            return [{"id": f"{name}{i}", "art": f"a{i % 2}"} for i in range(3 if name == "E" else 2)], []

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.delenv("STAGE8_DEADLINE", raising=False)
        monkeypatch.setattr(sp, "load_model", load)
        monkeypatch.setattr(sp, "blocks", lambda m: [None] * cfg.num_hidden_layers)
        monkeypatch.setattr(sp, "load_items", lambda a: [])
        monkeypatch.setattr(sp, "valid_split", valid_split)
        monkeypatch.setattr(sp, "run_item", run_item)
        monkeypatch.setattr(sp, "deadline_passed",
                            lambda need=0.0, reserve=0.0: self.budget is not None and len(self.ran) >= self.budget)

    def __call__(self, *args, budget=None):
        self.loads, self.ran, self.budget = 0, [], budget
        rc = sp.main(["--stage", "eval", "--key", "qwen7", "--test", "--out", str(self.out), *args])
        return rc, json.loads((self.out / "passages" / "TEST_qwen7.json").read_text())


def test_eval_piece_loop_deadline_resume_and_carry(tmp_path, monkeypatch):
    """The piece loop: past the deadline nothing loads; a deadline after one piece leaves the rest in order (exit 3); the
    resume runs exactly the rest and its file equals a single session's, byte for byte; a narrower or identical rerun
    keeps every piece without loading the weights."""
    pieces = [(s, f) for s in ("E", "R") for f in ("NOM", "OPTB", "OPTA")]
    names = [f"{s}/{f}" for s, f in pieces]
    one = _Loop(monkeypatch, tmp_path / "one")
    rc, single = one()
    assert rc == 0 and one.loads == 1 and one.ran == pieces and single["provenance"]["complete"]
    loop = _Loop(monkeypatch, tmp_path / "parts")
    rc, J = loop(budget=0)
    assert rc == 3 and loop.loads == 0 and loop.ran == [] and J["splits"] == {"E": {}, "R": {}}
    assert J["provenance"]["formats_not_run"] == names and "load" not in J["provenance"]["timings"]
    rc, J = loop(budget=1)
    assert rc == 3 and loop.loads == 1 and loop.ran == pieces[:1] and J["provenance"]["formats_not_run"] == names[1:]
    rc, J = loop()
    assert rc == 0 and loop.loads == 1 and loop.ran == pieces[1:] and J["provenance"]["kept"] == names[:1]
    assert json.dumps(J["splits"]) == json.dumps(single["splits"]) and J["provenance"]["complete"]
    rc, J2 = loop("--splits", "R", "--formats", "OPTB", budget=0)          # all kept: no load, no deadline exit
    assert rc == 0 and loop.loads == 0 and loop.ran == [] and json.dumps(J2["splits"]) == json.dumps(single["splits"])
    assert J2["provenance"]["kept"] == names and J2["provenance"]["carried"] == [n for n in names if n != "R/OPTB"]
    rc, J3 = loop("--splits", "E", "--formats", "NOM,OPTB,OPTA")
    assert rc == 0 and loop.loads == 0 and json.dumps(J3["splits"]) == json.dumps(single["splits"])
    assert J3["provenance"]["carried"] == names[3:]


def _check_args(tmp_path, *extra):
    return ["--stage", "check", "--key", "qwen7", "--test", "--out", str(tmp_path), *extra]


def test_check_records_a_wrong_squad_or_items_file(tmp_path, monkeypatch):
    """A wrong SQuAD file or a wrong items file fails the check with the reason in the check file (exit 1, "pass":
    false), not a traceback; a wrong items file is not read."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    bad = tmp_path / "squad.json"
    bad.write_text(json.dumps({"data": []}))
    assert sp.main(_check_args(tmp_path, "--squad", str(bad))) == 1
    C = json.loads((tmp_path / "check" / "TEST_qwen7_passages.json").read_text())
    assert not C["pass"] and any(f.startswith("items/SQuAD: AssertionError") and "SQuAD v1.1 dev" in f for f in C["fails"])
    assert C["valid"] == {"E": [], "R": []} and C["n_valid"] == {"E": 0, "R": 0}
    items = tmp_path / "items.json"
    items.write_text(sp.ITEMS.read_text() + "\n")
    monkeypatch.setattr(sp, "load_items", lambda a: pytest.fail("a wrong items file must not be read"))
    assert sp.main(_check_args(tmp_path, "--items", str(items))) == 1
    C = json.loads((tmp_path / "check" / "TEST_qwen7_passages.json").read_text())
    assert not C["pass"] and any(f.startswith("items sha256") for f in C["fails"]) and C["valid"] == {"E": [], "R": []}


def test_revision_falls_back_to_the_verified_directory(tmp_path):
    """provenance.revision: --revision, else the verified directory's revision (as the stories arm records it)."""
    (tmp_path / "VERIFIED.json").write_text(json.dumps({"repo": "org/model", "revision": "abc123"}))
    a = argparse.Namespace(model=str(tmp_path), key="qwen7", revision=None, test=True, items=str(sp.ITEMS),
                           max_new=sp.MAX_NEW, reserve_min=0.0)
    p = sp.base_prov(a, "eval")
    assert p["revision"] == "abc123" and p["verified"]["revision"] == "abc123"
    a.revision = "def456"
    assert sp.base_prov(a, "eval")["revision"] == "def456"
    a.model, a.revision = NAME, None
    assert sp.base_prov(a, "eval")["revision"] is None


def test_check_fails_on_a_frames_file_that_differs_from_its_pin(tmp_path, monkeypatch):
    """Outside TEST_MODE the check reads the stage-8a frames file against its pin; a differing sha256 fails the check
    (exit 1, "pass": false, the reason recorded). Tokenizer only (the study tokenizer's family, the first 3 items)."""
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    rel = "results/gpu_stage8a/frames/qwen7.json"
    args = ["--stage", "check", "--key", "qwen7", "--model", NAME, "--out", str(tmp_path), "--n", "3"]
    assert sp.main(args) == 0
    C = json.loads((tmp_path / "check" / "qwen7_passages.json").read_text())
    assert C["pass"] and C["provenance"]["frames_sha256"] == s9.ARTEFACT_SHA256[rel] and C["provenance"]["frames_file"] == rel
    assert C["stage8a_population_sha256"] == "57fb5fa1556dc8072950a80b2e992fb374fa96679987c2c1cda76bdda4eeb8ba"
    monkeypatch.setitem(s9.ARTEFACT_SHA256, rel, "0" * 64)
    assert sp.main(args) == 1
    C = json.loads((tmp_path / "check" / "qwen7_passages.json").read_text())
    assert not C["pass"] and any("frame" in f and "differs from the pin" in f for f in C["fails"])
