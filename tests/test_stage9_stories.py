"""Stage 9, stories arm (docs/stage9/SPEC.md 7, 9.1-9.2, 9.6, 11.3-11.4; 13.1 items 3-7): the window rows of
experiments/stage9_stories.py against independent references in FP32 on the CPU (1e-4 in log-probabilities, 1e-5 in
K/V), the LETTER parser and stop rule, l_b and the file contracts.

Models: a tiny random Qwen2 (24 blocks, hidden 8, the Qwen2.5-0.5B tokenizer) where every row of every case is checked,
and Qwen2.5-0.5B-Instruct for the identities, the complementarity and the generation with real stop rules.
3. run_item9's rows (N9 items of both lexicons; NONE, BEFORE, AFTER, LETTER; every row) equal a batch-1 pass under a
   separately built single-row clamp (tables from a separate capture, the donor rule written out from SPEC 7.1, the
   LETTER letters mapped through the item's listing); one forward over the trie for every row, and
   ckeys.surface.score_reference (one plain pass per form) for two row kinds per case (every kind in some case); the
   stored clean S and clean B runs (clean_B: the reference of the BF16 floor) equal plain passes of the S and B prompts.
4. late@0 = early@L = KV@0 = the clean S run and early@0 = late@L = none = none2 = the clean B run (scores 1e-4; the K/V
   at every position >= p under the clamp 1e-5); S on [0, l) and S on [l, L) by two nested clamp_kv contexts equals KV@0,
   and S on [0, l) nested with B on [l, L) equals early@l, l in (2, 12, 22). (SPEC 7.1: early@l takes S on [0, l), so
   early@L is the full patch and early@0 the identity, SPEC 7.2 and 13.1.4.)
5. The generation batch equals ckeys.generate.greedy_reference of each row alone under its own clamp (16 tokens; the word
   stop under AFTER, stage9.letter_stop under LETTER; the clean S run unclamped); the stored answer fields equal the
   parsers' on the reference text (LETTER: also the word parse "word").
6. stage9.parse_letter and stage9.letter_stop on hand cases; a letter answer contradicted by a word ("A drawer") keeps
   SPEC 9.2's answer and records the word; the LETTER E/L mapping on an item whose order is not the identity.
7. l_b on synthetic flips; the stage-8b pins; the lb-file provenance check of the eval; a TEST_MODE subprocess run (tiny
   model) of check, cal and eval writes files with every key of SPEC 11.3-11.4, resumes by format (records in
   STORY_FORMATS order whatever the run order), exits 3 at the deadline or without the lb file, and exits 1 for a cal
   without AFTER and for an eval whose lb file another commit wrote.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from ckeys import fresh, stage9
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import LETTERS
from ckeys.generate import greedy_reference
from ckeys.interventions import blocks
from ckeys.surface import FormSet, score, score_reference
from experiments import stage9_stories as st
from experiments.fresh_factorial import encode_item

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL, KVTOL = 1e-4, 1e-5
ROOT = Path(__file__).resolve().parents[1]
CASES = [(lex, fmt) for lex in (1, 2) for fmt in ("NONE", "BEFORE", "AFTER", "LETTER")]
KINDS = ["none", "KV@0", "late@12", "early@12", "K@0", "V@0", "Klate@8", "Vlate@16", "none2"]
torch.set_grad_enabled(False)


def tiny_model(nL=24):
    c = Qwen2Config(vocab_size=151936, hidden_size=8, intermediate_size=16, num_hidden_layers=nL, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=4, max_position_embeddings=4096, tie_word_embeddings=False,
                    initializer_range=0.5)
    c._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(c).float().eval()


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(NAME)


@pytest.fixture(scope="module")
def tiny():
    return tiny_model()


@pytest.fixture(scope="module")
def qwen():
    return AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval()


# --------------------------------------------------------------------------- independent references
def item(lex, k=0):
    return [it for it in stage9.stories() if it.lex == lex][k]


def s_layers(row, nL):
    """SPEC 7.1 written out: (layers whose key is S's, layers whose value is S's)."""
    if row in ("none", "none2"):
        return set(), set()
    kind, l0 = row.split("@")
    l0 = int(l0)
    every, hi, lo = set(range(nL)), set(range(l0, nL)), set(range(l0))
    return {"KV": (every, every), "late": (hi, hi), "early": (lo, lo), "K": (every, set()), "V": (set(), every),
            "Klate": (hi, set()), "Vlate": (set(), hi)}[kind]


def kv_at(model, ids, positions, nL):
    with capture_kv(model, positions, range(nL)) as t:
        model(ids, use_cache=False)
    return {k: v[0] for k, v in t.items()}


def single_table(kv, row, nL):
    ks, vs = s_layers(row, nL)
    return {(l, ch): kv["S" if l in (ks if ch == "k" else vs) else "B"][(l, ch)] for l in range(nL) for ch in "kv"}


def canon(res, name, it, fmt, i=0):
    """Six scores in canonical word order; under LETTER the letter the listing gives word j."""
    if fmt == "LETTER":
        letter = dict(zip(it.order, LETTERS))
        return [float(res[name][letter[j]][i]) for j in range(6)]
    return [float(res[name][w][i]) for w in it.words]


def small_fs(tok, it, fmt):
    return st.letter_form_set(tok) if fmt == "LETTER" else FormSet(tok, it.words, {"E": (" ", " the ")}, names=it.names)


def close(a, b, tol=TOL):
    return torch.allclose(torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64), atol=tol, rtol=0)


def same_scores(rec_row, res, it, fmt, what):
    for name in ("E", "L"):
        assert close(rec_row[name], canon(res, name, it, fmt)), (what, name, rec_row[name], canon(res, name, it, fmt))


def listing_letters(it):
    """{letter: word} parsed from the LETTER prompt's listing text."""
    body = fresh.letter_listing(it)[len("Choices: "):]
    return {x.split(") ")[0]: x.split(") ")[1] for x in body.split(", ")}


# --------------------------------------------------------------------------- rows
def test_row_names_and_donors():
    assert len(st.row_names(32)) == 43 and len(st.row_names(28)) == 39 and len(st.row_names(24)) == 35
    r = st.row_names(28)
    on = stage9.story_onsets(28)
    assert r == (["none", "KV@0"] + [f"late@{l}" for l in on] + [f"early@{l}" for l in on] + ["K@0", "V@0"]
                 + [f"Klate@{l}" for l in (4, 8, 12, 16)] + [f"Vlate@{l}" for l in (4, 8, 12, 16)] + ["none2"])
    for nL in (24, 28, 32):
        for row in st.row_names(nL) + ["late@0", "early@0", f"late@{nL}", f"early@{nL}"]:
            ks, vs = s_layers(row, nL)
            for l in range(nL):
                assert st.donor(row, l, "k") == ("S" if l in ks else "B"), (row, l)
                assert st.donor(row, l, "v") == ("S" if l in vs else "B"), (row, l)


@pytest.mark.parametrize("case", range(len(CASES)))
def test_rows_equal_single_row_references(tiny, tok, case):
    lex, fmt = CASES[case]
    model, nL = tiny, len(blocks(tiny))
    it = item(lex)
    fs = small_fs(tok, it, fmt)
    assert len(fs) > 0                                        # a trie with nodes: multi-token forms are scored
    rec = st.run_item9(model, tok, it, fmt, fs, nL, "cpu", gen=False)
    ids, p = encode_item(tok, it, fmt)
    assert rec["p"] == p and rec["T"] == ids["B"].shape[1] and list(rec["rows"]) == st.row_names(nL)
    assert rec["fmt"] == fmt and rec["key"] == it.key and rec["words"] == list(it.words)
    kv = {n: kv_at(model, ids[n], [p], nL) for n in "BS"}
    full = {KINDS[(2 * case) % 9], KINDS[(2 * case + 1) % 9]}
    for row in st.row_names(nL):
        with clamp_kv(model, [p], single_table(kv, row, nL), range(nL)):
            same_scores(rec["rows"][row], score(model, ids["B"], fs), it, fmt, row)
            if row in full:
                same_scores(rec["rows"][row], score_reference(model, ids["B"], fs), it, fmt, row + " (reference)")
    same_scores(rec["clean_S"], score(model, ids["S"], fs), it, fmt, "clean S")
    assert set(rec["clean_B"]) == {"E", "L"}
    same_scores(rec["clean_B"], score(model, ids["B"], fs), it, fmt, "clean B")
    # not vacuous: the S tables move the scores; the two identity rows equal the B run and each other
    d = max(abs(a - b) for a, b in zip(rec["rows"]["KV@0"]["E"], rec["rows"]["none"]["E"]))
    assert d > 1e-2, d
    same_scores(rec["rows"]["none2"], score(model, ids["B"], fs), it, fmt, "none2 = B run")
    assert close(rec["rows"]["none"]["E"], rec["rows"]["none2"]["E"], 1e-5)


# --------------------------------------------------------------------------- identities, complementarity, generation
ID_ROWS = ["none", "KV@0", "late@0", "early@24", "early@0", "late@24", "early@2", "early@12", "early@22", "late@12",
           "K@0", "Vlate@4", "none2"]
GEN_CHECK = ["none", "KV@0", "early@12", "late@12", "K@0"]


@pytest.fixture(scope="module")
def qrecs(qwen, tok):
    out = {}
    for lex, fmt in ((1, "AFTER"), (2, "LETTER")):
        it = item(lex, 1)
        fs = st.form_set9(tok, it, fmt)
        out[fmt] = (it, fs, st.run_item9(qwen, tok, it, fmt, fs, len(blocks(qwen)), "cpu", rows=ID_ROWS, gen=True))
    return out


@pytest.mark.parametrize("fmt", ["AFTER", "LETTER"])
def test_identities_and_complementarity(qwen, tok, qrecs, fmt):
    model, nL = qwen, len(blocks(qwen))
    assert nL == 24
    it, fs, rec = qrecs[fmt]
    ids, p = encode_item(tok, it, fmt)
    R = rec["rows"]
    cS, cB = score(model, ids["S"], fs), score(model, ids["B"], fs)
    for row in ("KV@0", "late@0", "early@24"):
        same_scores(R[row], cS, it, fmt, f"{row} = S run")
    for row in ("none", "early@0", "late@24", "none2"):
        same_scores(R[row], cB, it, fmt, f"{row} = B run")
    same_scores(rec["clean_S"], cS, it, fmt, "clean S")
    same_scores(rec["clean_B"], cB, it, fmt, "clean B")
    assert not close(R["KV@0"]["E"], R["none"]["E"], 1e-2)     # S and B differ
    # the K/V at every position >= p under the clamp are the S run's (KV@0) and the B run's (none)
    T = ids["B"].shape[1]
    kv = {n: kv_at(model, ids[n], [p], nL) for n in "BS"}
    post = {n: kv_at(model, ids[n], range(p, T), nL) for n in "BS"}
    for row, who in (("KV@0", "S"), ("none", "B")):
        with clamp_kv(model, [p], single_table(kv, row, nL), range(nL)):
            got = kv_at(model, ids["B"], range(p, T), nL)
        assert all(close(got[k], post[who][k], KVTOL) for k in got), row
    # complementarity by nested clamps
    tabS = {k: kv["S"][k] for k in kv["S"]}
    tabB = {k: kv["B"][k] for k in kv["B"]}
    for l in (2, 12, 22):
        with clamp_kv(model, [p], tabS, range(l)), clamp_kv(model, [p], tabS, range(l, nL)):
            same_scores(R["KV@0"], score(model, ids["B"], fs), it, fmt, f"S[0,{l}) + S[{l},L)")
            got = kv_at(model, ids["B"], range(p, T), nL)
        assert all(close(got[k], post["S"][k], KVTOL) for k in got), l
        with clamp_kv(model, [p], tabS, range(l)), clamp_kv(model, [p], tabB, range(l, nL)):
            same_scores(R[f"early@{l}"], score(model, ids["B"], fs), it, fmt, f"S[0,{l}) + B[{l},L) = early@{l}")


@pytest.mark.parametrize("fmt", ["AFTER", "LETTER"])
def test_generation_equals_stepwise_reference(qwen, tok, qrecs, fmt):
    model, nL = qwen, len(blocks(qwen))
    it, fs, rec = qrecs[fmt]
    ids, p = encode_item(tok, it, fmt)
    stop = stage9.letter_stop(tok, it) if fmt == "LETTER" else fresh.stop_fn(tok, it.words)
    kv = {n: kv_at(model, ids[n], [p], nL) for n in "BS"}
    for row in GEN_CHECK:
        with clamp_kv(model, [p], single_table(kv, row, nL), range(nL)):
            g = greedy_reference(model, tok, ids["B"], max_new=16, stop=stop)[0]
        text = fresh.gen_text(tok, ids["B"][0].tolist(), g)
        r = rec["rows"][row]
        assert (r["gen"], r["n_tok"]) == (text, len(g)), (row, r["gen"], text)
        w = fresh.parse(text, it.words)
        want = stage9.parse_letter(text, it) if fmt == "LETTER" else (w, "word" if w != "other" else "none")
        assert (r["ans"], r["how"]) == want, (row, text)
        assert r.get("word") == (w if fmt == "LETTER" else None), (row, text)
    g = greedy_reference(model, tok, ids["S"], max_new=16, stop=stop)[0]
    text = fresh.gen_text(tok, ids["S"][0].tolist(), g)
    c = rec["clean_S"]
    assert c["gen"] == text and c["n_tok"] == len(g)
    w = fresh.parse(text, it.words)
    want = stage9.parse_letter(text, it) if fmt == "LETTER" else (w, "word" if w != "other" else "none")
    assert (c["ans"], c["how"]) == want and c.get("word") == (w if fmt == "LETTER" else None)
    assert any(rec["rows"][r]["n_tok"] < 16 for r in GEN_CHECK)        # the stop rule ends answers early


def test_generation_16_tokens_tiny(tiny, tok):
    """The full 16-token path (a random model never stops early) with per-row tables, LETTER and a word format."""
    model, nL = tiny, len(blocks(tiny))
    for lex, fmt in ((2, "LETTER"), (1, "POST-NULL")):
        it = item(lex, 2)
        rows = ["none", "KV@0", "early@6", "Klate@16", "none2"]
        rec = st.run_item9(model, tok, it, fmt, small_fs(tok, it, fmt), nL, "cpu", rows=rows, gen=True)
        ids, p = encode_item(tok, it, fmt)
        stop = st.stop_of(tok, it, fmt)
        kv = {n: kv_at(model, ids[n], [p], nL) for n in "BS"}
        for row in rows:
            with clamp_kv(model, [p], single_table(kv, row, nL), range(nL)):
                g = greedy_reference(model, tok, ids["B"], max_new=16, stop=stop)[0]
            assert rec["rows"][row]["gen"] == fresh.gen_text(tok, ids["B"][0].tolist(), g), (fmt, row)
            assert rec["rows"][row]["n_tok"] == len(g)
        assert any(rec["rows"][r]["n_tok"] == 16 for r in rows)


# --------------------------------------------------------------------------- LETTER parser and stop
class Pieces:
    def decode(self, ids):
        return "".join(ids)


def test_parse_letter_hand_cases():
    for lex in (1, 2):
        it = item(lex)
        assert it.order != tuple(range(6))
        word_of = listing_letters(it)
        for text, L in (("A", "A"), (" B)", "B"), ("**C**", "C"), ("(D) jar", "D"), ("E. box", "E"), ("F", "F"),
                        ("A1", "A"), (" 'B'", "B"), ("`C`", "C"), ("[D]", "D"), ('"E"', "E"), ("_F_", "F")):
            assert stage9.parse_letter(text, it) == (word_of[L], "letter"), (lex, text)
        for text in ("Bob", "b", "", "G", "\ufffd", "Answer: B", "Cat"):
            assert stage9.parse_letter(text, it) == ("other", "none"), (lex, text)
        w = it.words[3]
        assert stage9.parse_letter(f" The {w} is it", it) == (w, "word")
        assert stage9.parse_letter(f"{w.capitalize()}.", it) == (w, "word")
    assert stage9.parse_letter("Box", item(1)) == ("box", "word")
    assert stage9.parse_letter("Box", item(2)) == ("other", "none")
    assert stage9.parse_letter("Bin", item(2)) == ("bin", "word")


def test_letter_word_conflict_recorded():
    """An article and a word ("A drawer") is a letter answer under SPEC 9.2 (option A) and the stop fires after it; the
    record keeps that answer and stores the word parse, so the conflict can be counted."""
    tk = Pieces()
    for lex in (1, 2):
        it = item(lex)
        word_of = listing_letters(it)
        other = next(w for w in it.words if w != word_of["A"])
        text = f" A {other}"
        assert stage9.parse_letter(text, it) == (word_of["A"], "letter")
        assert stage9.letter_stop(tk, it)(0, [" A", f" {other}"])
        assert st.gen_fields(text, it, "LETTER") == {"ans": word_of["A"], "how": "letter", "word": other}
        assert st.gen_fields(f" A) {word_of['A']}", it, "LETTER") == {"ans": word_of["A"], "how": "letter",
                                                                       "word": word_of["A"]}
        assert st.gen_fields(" A", it, "LETTER") == {"ans": word_of["A"], "how": "letter", "word": "other"}
        assert st.gen_fields(f" The {other}.", it, "LETTER") == {"ans": other, "how": "word", "word": other}
        assert st.gen_fields(f" The {other}.", it, "AFTER") == {"ans": other, "how": "word"}
        assert st.gen_fields(" Nowhere", it, "NONE") == {"ans": "other", "how": "none"}


def test_letter_stop_hand_cases():
    tk = Pieces()
    for lex in (1, 2):
        f = stage9.letter_stop(tk, item(lex))
        for ids in (["A", ")"], [" B", "."], ["**", "C", "**"], ["(", "D", ")"], ["E", " "], [" F", ","]):
            assert f(0, ids), (lex, ids)
        for ids in (["A"], [" "], ["A", "\ufffd"], ["\ufffd"], ["\ufffd", "A"], ["Bo"], ["Bob", ","], ["G", ")"], ["b", ")"],
                    ["", ""]):
            assert not f(0, ids), (lex, ids)
    assert stage9.letter_stop(tk, item(1))(0, [" B", "ox"])           # the word stop: a complete candidate word
    assert not stage9.letter_stop(tk, item(2))(0, [" B", "ox"])
    assert stage9.letter_stop(tk, item(2))(0, [" the", " crate", "."])


def test_letter_stop_incomplete_character(tok):
    """A byte-level token that ends inside a multi-byte character decodes to U+FFFD: no stop until the character is
    complete, because the character may be a letter (here U+10348, GOTHIC LETTER HWAIR, after " A": then the answer is
    not the letter A)."""
    it = item(1)
    f = stage9.letter_stop(tok, it)
    ids = tok(" A\U00010348", add_special_tokens=False).input_ids
    cut = next(k for k in range(1, len(ids)) if "\ufffd" in tok.decode(ids[:k]))
    assert stage9.letter_text(tok.decode(ids[:cut])).startswith("A\ufffd") and not f(0, ids[:cut])
    assert "\U00010348".isalpha() and not f(0, ids) and stage9.parse_letter(tok.decode(ids), it) == ("other", "none")
    assert stage9.parse_letter(tok.decode(ids[:1]), it) == (listing_letters(it)["A"], "letter")


def test_letter_mapping_to_canonical_order():
    base = item(2)
    it = fresh.Item("N9", 0, base.core, 2, (3, 5, 0, 1, 4, 2), 0)
    res = {"E": {L: torch.tensor([float(k)]) for k, L in enumerate(LETTERS)},
           "L": {L: torch.tensor([10.0 + k]) for k, L in enumerate(LETTERS)}}
    word_of = listing_letters(it)
    letter_of = {w: L for L, w in word_of.items()}
    want = [float(LETTERS.index(letter_of[w])) for w in it.words]
    assert st.six(res, "E", it, "LETTER", 0) == want == [2.0, 3.0, 5.0, 0.0, 4.0, 1.0]
    assert st.six(res, "L", it, "LETTER", 0) == [10.0 + x for x in want]
    words = {w: torch.tensor([float(i)]) for i, w in enumerate(it.words)}
    assert st.six({"E": words}, "E", it, "AFTER", 0) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


# --------------------------------------------------------------------------- l_b, pins, files
def fake_recs(flips, onsets, n=10, fmt="AFTER"):
    recs = []
    for i in range(n):
        rows = {f"late@{l}": {"ans": "S" if i < round(flips[l] * n) else "B"} for l in onsets}
        recs.append({"fmt": fmt, "track": {"S": "S", "B": "B"}, "rows": rows})
    return recs


def test_lb_rule():
    on = stage9.story_onsets(32)
    fl = {l: max(0.0, 1.0 - 0.1 * k) for k, l in enumerate(on)}             # 1.0, 0.9, ..., 0.2 at the 9th onset
    recs = fake_recs(fl, on) + fake_recs({l: 1.0 for l in on}, on, fmt="NONE")
    got = st.late_flips(recs, on)
    assert all(abs(got[l] - fl[l]) < 1e-12 for l in on)
    assert st.lb_of(got, on) == on[8] == 18                                 # the first onset with flip <= 0.2 (0.2 counts)
    assert st.lb_of({l: 0.25 for l in on}, on) is None
    assert st.lb_of({**{l: 0.9 for l in on}, 30: 0.0, 10: 0.1}, on) == 10   # the first, not the deepest
    assert st.late_flips([], on) == {}


def test_stage8b_pins(tmp_path):
    a = Namespace(test=False, key="qwen7", stage8b=st.STAGE8B)
    if not (ROOT / st.STAGE8B / "frames" / "qwen7.json").exists():
        pytest.skip("the stage-8b results are not in this copy")
    f = st.stage8b_files(a)
    assert f["frames"] == [] and f["scorer"] == "trie" and f["frames_sha256"] == stage9.ARTEFACT_SHA256[f["frames_file"]]
    for kind in ("frames", "g3"):
        (tmp_path / kind).mkdir()
        shutil.copy(ROOT / st.STAGE8B / kind / "qwen7.json", tmp_path / kind / "qwen7.json")
    assert st.stage8b_files(Namespace(test=False, key="qwen7", stage8b=str(tmp_path)))["scorer"] == "trie"
    p = tmp_path / "frames" / "qwen7.json"
    p.write_text(p.read_text().replace('"frames": []', '"frames": [" x "]'))
    with pytest.raises(AssertionError, match="differs from the pin"):
        st.stage8b_files(Namespace(test=False, key="qwen7", stage8b=str(tmp_path)))
    with pytest.raises(AssertionError):
        st.stage8b_files(Namespace(test=False, key="gemma9", stage8b=st.STAGE8B))
    assert st.stage8b_files(Namespace(test=True, key="qwen7", stage8b="nowhere"))["frames_file"] is None


PROV = {"args", "git_commit", "torch", "transformers", "python", "device", "wrapper", "stage9", "arm", "stage", "git_dirty",
        "freeze_commit", "predictions_sha256", "spec_sha256", "started_utc", "finished_utc", "model", "model_key",
        "revision", "verified", "test_mode", "formats", "max_new", "reserve_min", "timings", "kv_only_onsets",
        "frames_file", "frames_sha256", "frames", "g3_file", "g3_sha256", "scorer", "skipped_items"}
PROV_RUN = PROV | {"dtype", "attn_implementation", "n_layers", "hidden", "n_heads", "n_kv_heads", "population",
                   "population_sha256", "n_items", "formats_not_run", "onsets"}
ITEM = {"key", "pop", "idx", "core", "lex", "order", "sent", "words", "track", "cluster", "fmt", "p", "T", "rows",
        "clean_S", "clean_B"}
ROW = {"E", "L", "gen", "ans", "how", "n_tok"}


def test_lb_mismatch(tmp_path):
    a = Namespace(model=str(tmp_path), revision=None, test=True, key="qwen7")
    head = stage9.run_meta(True)["git_commit"]
    lbp = {"git_commit": head, "revision": None, "model_key": "qwen7"}
    assert st.lb_mismatch(lbp, a) == []
    bad = st.lb_mismatch({**lbp, "git_commit": "0" * 40}, a)
    assert len(bad) == 1 and bad[0].startswith("git_commit 0000")
    assert st.lb_mismatch({**lbp, "model_key": "llama8"}, a)[0].startswith("model_key llama8")
    (tmp_path / "VERIFIED.json").write_text(json.dumps({"repo": "x/y", "revision": "abc", "key": "qwen7"}))
    assert st.lb_mismatch(lbp, a)[0] == "revision None != this run's abc"
    assert st.lb_mismatch({**lbp, "revision": "abc"}, a) == []
    a2 = Namespace(**{**vars(a), "revision": "def"})
    assert st.lb_mismatch({**lbp, "revision": "abc"}, a2)[0].startswith("revision abc")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory, tok):
    d = tmp_path_factory.mktemp("tiny") / "model"
    tiny_model().save_pretrained(d)
    tok.save_pretrained(d)
    return d


def cli(tiny_dir, out, *args, deadline=None):
    env = dict(os.environ, PYTHONPATH=str(ROOT), TEST_MODE="1", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    env.pop("STAGE8_DEADLINE", None)
    env.pop("STAGE9_FREEZE", None)
    if deadline is not None:
        env["STAGE8_DEADLINE"] = str(deadline)
    cmd = [sys.executable, "-m", "experiments.stage9_stories", "--key", "qwen7", "--model", str(tiny_dir), "--out", str(out),
           "--n", "1", "--n-cal", "2", *args]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_test_mode_files(tiny_dir, tmp_path):
    out = tmp_path / "out"
    rc, log = cli(tiny_dir, out, "--stage", "check")
    assert rc == 0, log
    chk = json.loads((out / "check" / "TEST_qwen7_stories.json").read_text())
    assert {"provenance", "pass", "fails", "skipped"} <= set(chk) and chk["pass"] and not chk["fails"] and not chk["skipped"]
    assert PROV <= set(chk["provenance"]) and chk["provenance"]["stage"] == "check"
    assert chk["provenance"]["freeze_commit"] is None and chk["provenance"]["test_mode"] is True
    # eval before cal: no lb file -> exit 3, nothing written
    rc, log = cli(tiny_dir, out, "--stage", "eval")
    assert rc == 3 and not (out / "stories").exists(), log
    # cal without AFTER (the l_b rule's format): exit 1 before the model is loaded, nothing written
    rc, log = cli(tiny_dir, out, "--stage", "cal", "--formats", "NONE,LETTER")
    assert rc == 1 and "cal needs AFTER" in log and not (out / "lb").exists(), log
    # cal stopped by the deadline: exit 3, progress kept aside, no lb file
    rc, log = cli(tiny_dir, out, "--stage", "cal", "--formats", "AFTER,LETTER", deadline=time.time() - 60)
    assert rc == 3 and not (out / "lb" / "TEST_qwen7.json").exists(), log
    part = json.loads((out / "lb" / "TEST_qwen7.json.partial").read_text())
    assert part["provenance"]["formats_not_run"] == ["AFTER", "LETTER"] and part["items"] == []
    rc, log = cli(tiny_dir, out, "--stage", "cal", "--formats", "AFTER,LETTER")
    assert rc == 0, log
    assert not (out / "lb" / "TEST_qwen7.json.partial").exists()
    lbf = out / "lb" / "TEST_qwen7.json"
    lb = json.loads(lbf.read_text())
    assert {"provenance", "population", "rows", "l_b", "late_after_flip", "rule", "items"} <= set(lb)
    assert lb["population"] == "C" and lb["rule"] == st.LB_RULE and lb["rows"] == st.row_names(24)
    assert PROV_RUN <= set(lb["provenance"]) and lb["provenance"]["formats_not_run"] == []
    assert lb["provenance"]["population_sha256"] == stage9.POP_SHA256["C"] and lb["provenance"]["n_items"] == 2
    assert len(lb["items"]) == 4 and {r["fmt"] for r in lb["items"]} == {"AFTER", "LETTER"}
    on = stage9.story_onsets(24)
    fl = st.late_flips(lb["items"], on)
    assert lb["l_b"] == st.lb_of(fl, on) and lb["late_after_flip"] == {str(l): round(fl[l], 5) for l in on}
    for r in lb["items"]:
        assert ITEM <= set(r) and set(r["rows"]) == set(lb["rows"]) and ROW <= set(r["clean_S"])
        assert all(ROW <= set(x) and len(x["E"]) == 6 and len(x["L"]) == 6 for x in r["rows"].values())
        assert set(r["clean_B"]) == {"E", "L"} and len(r["clean_B"]["E"]) == 6
        assert all(("word" in x) == (r["fmt"] == "LETTER") for x in list(r["rows"].values()) + [r["clean_S"]])
        assert r["pop"] == "C" and r["key"] == f"C{r['idx']:03d}" and len(r["cluster"]) == 3
    # eval: one format, then resume with two (the first kept byte for byte)
    rc, log = cli(tiny_dir, out, "--stage", "eval", "--formats", "AFTER")
    assert rc == 0, log
    ev1 = json.loads((out / "stories" / "TEST_qwen7.json").read_text())
    rc, log = cli(tiny_dir, out, "--stage", "eval", "--formats", "LETTER,AFTER")   # run order LETTER first
    assert rc == 0 and "formats ['AFTER'] kept" in log, log
    ev = json.loads((out / "stories" / "TEST_qwen7.json").read_text())
    assert {"provenance", "population", "rows", "items"} <= set(ev) and ev["population"] == "N9"
    P = ev["provenance"]
    assert PROV_RUN | {"lb_file", "lb_sha256", "l_b"} <= set(P)
    assert P["lb_sha256"] == stage9.file_sha(lbf) and P["l_b"] == lb["l_b"] and P["formats_not_run"] == []
    assert P["population_sha256"] == stage9.POP_SHA256["N9"] and P["n_items"] == 1 and P["n_layers"] == 24
    assert P["onsets"] == on and P["dtype"] == "torch.float32" and P["attn_implementation"] == "sdpa"
    assert [r["fmt"] for r in ev["items"]] == ["AFTER", "LETTER"] and ev["items"][0] == ev1["items"][0]   # canonical
    assert all(ITEM <= set(r) and r["pop"] == "N9" for r in ev["items"])
    # the deadline before any format: exit 3, every format not run
    out2 = tmp_path / "out2"
    shutil.copytree(out / "lb", out2 / "lb")
    rc, log = cli(tiny_dir, out2, "--stage", "eval", "--formats", "AFTER,LETTER", deadline=time.time() - 60)
    assert rc == 3, log
    ev2 = json.loads((out2 / "stories" / "TEST_qwen7.json").read_text())
    assert ev2["provenance"]["formats_not_run"] == ["AFTER", "LETTER"] and ev2["items"] == []
    # an lb file another commit wrote: exit 1 before the model is loaded, naming the step to redo
    out3 = tmp_path / "out3"
    (out3 / "lb").mkdir(parents=True)
    old = json.loads(lbf.read_text())
    old["provenance"]["git_commit"] = "0" * 40
    (out3 / "lb" / "TEST_qwen7.json").write_text(json.dumps(old))
    rc, log = cli(tiny_dir, out3, "--stage", "eval", "--formats", "AFTER")
    assert rc == 1 and "FORCE_STEPS=cal_qwen7" in log and not (out3 / "stories").exists(), log
    # a new lb file (another l_b fixing) restarts the evaluation instead of resuming it
    lb["late_after_flip"]["2"] = 0.5
    lbf.write_text(json.dumps(lb))
    rc, log = cli(tiny_dir, out, "--stage", "eval", "--formats", "AFTER")
    assert rc == 0 and "rerun from the start" in log, log
