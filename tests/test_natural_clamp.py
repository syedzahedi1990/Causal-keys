"""Gate J-A-G0 of P-2026-10-10-J part A (docs/PREREGISTRATION.md): FP32 exactness of the stage-8 part-A rows and passes
(ckeys/natural_rows.py, the functions experiments/natural_factorial.py calls) at Qwen2.5-0.5B-Instruct on the CPU,
1e-4 in the log-probabilities; plus the tokenizer-only checks of the frame-aware decision ids, the closed-book prompts and
the answer parsing (ckeys/natural_formats.py).

Items: two committed items (a PERSON and a NUMBER item of data/stage8a_items.json) on a short passage written for the
test (the entity once), so the test needs no copy of SQuAD. Every reference is computed independently of the batch it
checks: a plain forward of the S (or B) prompt, single-row clamps from separately captured runs, or the cache-free
stepwise greedy reference."""
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import chat_text
from ckeys.generate import argmax_chain, greedy_reference
from ckeys.interventions import blocks
from ckeys.natural_formats import (FRAMES, answer_of, cb_raw, choose_frame, cont_ids, decision_ids, encode_item,
                                   frame_cont, frame_of, letter_of, passage_range)
from ckeys.natural_rows import (BASE, CUE, LETA_REDUCED, ZROWS, Row, capture, core_rows, decided_stop, explore_rows,
                                cut, end_ids, gen_text, generate_rows, passage_positions, prep, score_rows, tables)

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 1e-4
ROOT = Path(__file__).resolve().parents[1]
ITEMS = {i["id"]: i for i in json.load(open(ROOT / "data" / "stage8a_items.json"))}
CTX = {"57268739708984140094c8f0": "By 1947 the network had signed Bing Crosby, who taped his radio show on Magnetophon recorders.",
       "56f88025aef2371900626121": "For the first choral hymnal Luther wrote 24 hymns, which Johann Walter set for several voices."}
torch.set_grad_enabled(False)


def item(i):
    it = dict(ITEMS[i])
    it["context"] = CTX[i]
    it["start"] = CTX[i].index(it["answer"])
    return it


@pytest.fixture(scope="module")
def qwen():
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    return model, tok, len(blocks(model))


def lsm(model, ids, start):
    """Plain forward: log-softmax rows from position ``start`` on (FP32)."""
    return model(torch.tensor([ids]), use_cache=False).logits[0, start:].float().log_softmax(-1)


def single_kv(model, ids, P, nL):
    """K/V at P from a single forward of ``ids`` alone (no w): the hand-built clamps' donor tables."""
    with capture_kv(model, P, range(nL)) as C:
        model(torch.tensor([ids]), use_cache=False, logits_to_keep=1)
    return {k: v[0] for k, v in C.items()}


@pytest.mark.parametrize("iid", list(CTX))
@pytest.mark.parametrize("fmt", ["NOM", "OPTA", "LETA"])
def test_kv_s_row_is_the_s_run_and_id_is_clean_b(qwen, iid, fmt):
    model, tok, nL = qwen
    it = item(iid)
    d = prep(tok, it, fmt, " ")
    kv, runs = capture(model, d, range(nL), "cpu")
    sc = score_rows(model, d, kv, core_rows(fmt), nL, "cpu", "S")
    c, j, T = d["c"]["S"], d["j"], d["T"]
    for row, run in (("KV_S", "S"), ("ID", "B")):
        ref = lsm(model, d["ids"][run] + c, T - 1 + j)
        for Y, t in d["dec"].items():
            assert abs(sc[row]["lp"][Y] - float(ref[0, t])) < TOL, (row, Y)
            assert abs(runs[run][Y] - float(ref[0, t])) < TOL, (run, Y)    # the capture batch's own decision log-probs
        assert sc[row]["argmax"] == int(ref[0].argmax())
        want = [float(ref[t - j, c[t]]) for t in range(j + 1, len(c))]
        assert len(sc[row]["cont"]) == len(want) and all(abs(x - y) < TOL for x, y in zip(sc[row]["cont"], want)), row
        opts = [d["dec"][Y] for Y in ("B", "S", "X", "D")]
        assert abs(sc[row]["mass"] - float(ref[0, opts].exp().sum())) < TOL
    assert abs(sc["KV_S"]["lp"]["S"] - sc["ID"]["lp"]["S"]) > 0.5, "the S passage must move the S decision token"
    assert fmt == "LETA" or len(sc["KV_S"]["cont"]) == len(c) - j - 1


def test_batched_rows_equal_single_rows(qwen):
    model, tok, nL = qwen
    for iid, fmt in ((list(CTX)[0], "OPTA"), (list(CTX)[1], "NOM")):
        d = prep(tok, item(iid), fmt, " ")
        kv, _ = capture(model, d, range(nL), "cpu")
        rows = core_rows(fmt)
        full = score_rows(model, d, kv, rows, nL, "cpu", "S")
        for n, r in rows.items():
            one = score_rows(model, d, kv, {n: r}, nL, "cpu", "S")[n]
            assert all(abs(one["lp"][Y] - full[n]["lp"][Y]) < TOL for Y in one["lp"]), n
            assert all(abs(x - y) < TOL for x, y in zip(one["cont"], full[n]["cont"])), n


def hand_clamp(model, d, nL, k_tab, v_tab, target="S"):
    """One row on B prompt + c_target with K tables k_tab and V tables v_tab ({l: [|P|, D]}), two clamp_kv contexts."""
    c, j, T = d["c"][target], d["j"], d["T"]
    with clamp_kv(model, d["P"], {(l, "k"): k_tab[l] for l in range(nL)}, range(nL), "k"), \
            clamp_kv(model, d["P"], {(l, "v"): v_tab[l] for l in range(nL)}, range(nL), "v"):
        ref = lsm(model, d["ids"]["B"] + c, T - 1 + j)
    return {Y: float(ref[0, t]) for Y, t in d["dec"].items()}, [float(ref[t - j, c[t]]) for t in range(j + 1, len(c))]


def test_cue_conflict_z_and_explore_rows_equal_hand_built_clamps(qwen):
    model, tok, nL = qwen
    d = prep(tok, item(list(CTX)[0]), "OPTA", " ")
    kv, _ = capture(model, d, range(nL), "cpu")
    own = {Y: single_kv(model, d["ids"][Y], d["P"], nL) for Y in "BSXZ"}
    for Y in "BSXZ":   # the batch capture (prompt + w, four rows) equals single captures of the prompt alone
        for key in own[Y]:
            assert torch.allclose(kv[Y][key], own[Y][key], atol=TOL, rtol=0), (Y, key)
    nP, on = len(d["P"]), round(0.3 * nL)
    assert nP >= 2, "the piece rows need a multi-token span"
    rows = dict(ZROWS) | dict(CUE) | {n: BASE[n] for n in ("K_S", "V_X")} | explore_rows("OPTA", nL)
    sc = score_rows(model, d, kv, rows, nL, "cpu", "S")

    def tab(Y, ch, keep=lambda l, i: True):
        return {l: torch.stack([own[Y if keep(l, i) else "B"][(l, ch)][i] for i in range(nP)]) for l in range(nL)}

    want = {"KS_VX": (tab("S", "k"), tab("X", "v")), "KX_VS": (tab("X", "k"), tab("S", "v")),
            "KS_VZ": (tab("S", "k"), tab("Z", "v")), "KZ_VS": (tab("Z", "k"), tab("S", "v")),
            "K_Z": (tab("Z", "k"), tab("B", "v")), "V_Z": (tab("B", "k"), tab("Z", "v")), "KV_Z": (tab("Z", "k"), tab("Z", "v")),
            "K_S": (tab("S", "k"), tab("B", "v")), "V_X": (tab("B", "k"), tab("X", "v")),
            "K_S@on": (tab("S", "k", lambda l, i: l >= on), tab("B", "v")), "V_S@on": (tab("B", "k"), tab("S", "v", lambda l, i: l >= on)),
            "K_S^1": (tab("S", "k", lambda l, i: i == 0), tab("B", "v")), "K_S^r": (tab("S", "k", lambda l, i: i > 0), tab("B", "v"))}
    moved = 0
    for n, (kt, vt) in want.items():
        lp, cont = hand_clamp(model, d, nL, kt, vt)
        assert all(abs(sc[n]["lp"][Y] - lp[Y]) < TOL for Y in lp), n
        assert all(abs(x - y) < TOL for x, y in zip(sc[n]["cont"], cont)), n
        moved += abs(sc[n]["lp"]["S"] - sc["ID"]["lp"]["S"] if "ID" in sc else 0) > 0.1
    assert moved >= 5
    assert Row("S", "B", piece="first").donor(3, "k", 1) == "B" and Row("S", "B", piece="rest").donor(3, "k", 1) == "S"
    assert Row("S", "B", onset=4).donor(3, "k", 0) == "B" and Row("S", "B", onset=4).donor(4, "k", 0) == "S"
    assert set(core_rows("LETA", True)) == set(LETA_REDUCED) and set(core_rows("OPTB")) == set(BASE)


@pytest.mark.parametrize("fmt", ["NOM", "LETA"])
def test_generation_under_kv_s_is_greedy_on_the_s_prompt(qwen, fmt):
    model, tok, nL = qwen
    it = item(list(CTX)[0])
    d = prep(tok, it, fmt, " ")
    kv, _ = capture(model, d, range(nL), "cpu")
    rows = {n: BASE[n] for n in ("ID", "KV_S", "K_S", "V_S")} | {"KS_VX": CUE["KS_VX"]}
    g = generate_rows(model, tok, it, d, kv, rows, nL, 12, extra={"S_run": "S", "B_run": "B"})
    stop = decided_stop(tok, it, fmt)
    assert g["KV_S"]["ids"] == g["S_run"]["ids"] and g["ID"]["ids"] == g["B_run"]["ids"]
    assert g["S_run"]["ids"] == greedy_reference(model, tok, torch.tensor([d["ids"]["S"]]), max_new=12, stop=stop)[0]
    assert bool(argmax_chain(model, torch.tensor([d["ids"]["S"]]), g["S_run"]["ids"])[0])
    tabs = tables(kv, [BASE["K_S"]], nL, len(d["P"]))       # the cache path under a clamp equals the stepwise reference
    with clamp_kv(model, d["P"], tabs, range(nL)):
        assert g["K_S"]["ids"] == greedy_reference(model, tok, torch.tensor([d["ids"]["B"]]), max_new=12, stop=stop)[0]
    full = generate_rows(model, tok, it, d, kv, rows, nL, 12, extra={"S_run": "S", "B_run": "B"}, stop=False)
    for n in g:   # the early stop never changes the answer class or g1, and only shortens
        assert g[n]["who"] == full[n]["who"] and g[n]["g1"] == full[n]["g1"], n
        assert full[n]["ids"][:len(g[n]["ids"])] == g[n]["ids"], n
    assert sum(len(full[n]["ids"]) - len(g[n]["ids"]) for n in g) > 0 or fmt == "NOM"
    assert g["KV_S"]["text"] == gen_text(tok, d["ids"]["S"], g["S_run"]["ids"])
    assert g["ID"]["who"] in ("B", "other") and g["KV_S"]["who"] in ("S", "other")
    if g["KV_S"]["ids"][:len(d["w"])] == d["w"] and len(g["KV_S"]["ids"]) > len(d["w"]):
        assert g["KV_S"]["g1"] == g["KV_S"]["ids"][len(d["w"])]


def test_answer_ends_at_a_special_token_missing_from_the_eos_list(qwen):
    """Gemma-2-9b-it's generation config lists only <eos>, not <end_of_turn> (Yi-1.5-9B-Chat: not <|im_end|>). The shared
    decoder (ckeys.generate) now stops at the chat end-of-turn tokens too; Part A also reads each answer only up to the
    first special token. Here <|im_end|> is removed from the EOS list: the rows must name the same entities, and an
    answer that runs past the end of the turn is still read correctly once cut, though not without the cut."""
    model, tok, nL = qwen
    it = item(list(CTX)[0])
    d = prep(tok, it, "NOM", " ")
    kv, _ = capture(model, d, range(nL), "cpu")
    rows = {n: BASE[n] for n in ("ID", "KV_S", "KV_X")}
    ref = generate_rows(model, tok, it, d, kv, rows, nL, 12)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos = model.generation_config.eos_token_id
    model.generation_config.eos_token_id = [t for t in eos if t != im_end]
    try:
        got = generate_rows(model, tok, it, d, kv, rows, nL, 12)
        raw = greedy_reference(model, tok, torch.tensor([d["ids"]["B"]]), max_new=12)[0]
    finally:
        model.generation_config.eos_token_id = eos
    assert im_end not in raw                                       # the shared decoder stops at the end of the turn
    past = ref["ID"]["ids"] + [im_end] + tok("\nuser\nmore", add_special_tokens=False).input_ids
    assert answer_of(gen_text(tok, d["ids"]["B"], past), it, "NOM") == "other"     # read with the special token's text
    assert answer_of(gen_text(tok, d["ids"]["B"], cut(past, end_ids(tok))), it, "NOM") == "B"
    assert {n: (g["who"], g["ids"], g["g1"]) for n, g in got.items()} == {n: (g["who"], g["ids"], g["g1"]) for n, g in ref.items()}
    assert [g["who"] for g in got.values()] == ["B", "S", "X"]


def test_passage_positions_and_frames(qwen):
    model, tok, nL = qwen
    it = item(list(CTX)[1])
    d = prep(tok, it, "NOM", " ")
    pos = passage_positions(tok, d, it, "S")
    txt = tok.decode([d["ids"]["S"][p] for p in pos])
    assert txt.strip() == it["context"].replace("24", it["S"]).strip()
    assert set(d["P"]) <= set(pos)
    p0, p1 = passage_range(d["texts"]["S"], it, "NOM", it["S"])
    assert d["texts"]["S"][p0:p1] == it["context"].replace("24", it["S"])
    # frame " **": w ends inside the frame, the decision tokens follow it and stay distinct; frame " " is cont_ids
    it0 = item(list(CTX)[0])
    text, ids, _, _ = encode_item(tok, it0, "OPTA", it0["answer"])
    dd = decision_ids(tok, text, ids, it0, "OPTA", " **")
    assert tok.decode(dd["w"]).strip() == "**" and len(set(dd["dec"].values())) == 5
    for e in (it0["answer"], it0["S"], it["answer"], it["S"]):
        assert frame_cont(tok, text, ids, e, " ") == cont_ids(tok, text, ids, e)


def test_early_stop_never_changes_the_answer_class():
    """decided_stop (tokenizer only, every committed item): along every token prefix of many continuations (each
    candidate, markdown frames, possessives, punctuation, longer and shorter names, letters with markdown), the answer
    class from the prefix where the row stops equals the class of the whole continuation, and the stop fires."""
    tok = AutoTokenizer.from_pretrained(NAME)
    n_stop = 0
    for it in list(ITEMS.values()):
        for fmt in ("NOM", "LETA"):
            text = chat_text(tok, "Read the passage and answer the question.") if fmt == "NOM" else chat_text(tok, "Pick a letter.")
            ids = tok(text, add_special_tokens=False).input_ids
            stop = decided_stop(tok, it, fmt)
            if fmt == "NOM":
                ents = [it[k] for k in ("answer", "S", "X", "Z", "D")]
                gens = [f" {e}{t}" for e in ents for t in ("", ".", "'s house", " Jr. and the rest of it", ", the second one")]
                gens += [f" **{e}**, as the passage says" for e in ents]
                gens += [f" The {ents[0]} and {ents[1]}", f" {ents[0].split()[0]} {ents[2]} is it",
                         f" {ents[1]}-{ents[0]} was named", " none of them, sorry about that"]
            else:
                gens = [f" {L}{t}" for L in "ABCD" for t in ("", ".", ") the answer", "**", "lpha and beta", " is it")]
                gens += [" **B**", " (C) text", " The answer is A", " `D` here"]
            for g in gens:
                full = tok(text + g, add_special_tokens=False).input_ids
                assert full[:len(ids)] == ids
                gen = full[len(ids):]
                k = next((n for n in range(1, len(gen) + 1) if stop(0, gen[:n])), None)
                if k is None:
                    continue
                n_stop += 1
                cut, whole = gen_text(tok, ids, gen[:k]), gen_text(tok, ids, gen)
                assert answer_of(cut, it, fmt) == answer_of(whole, it, fmt), (it["id"], fmt, cut, whole)
    assert n_stop > 1000


def test_frame_cont_is_cont_ids_on_every_item_and_parsing():
    tok = AutoTokenizer.from_pretrained(NAME)
    text = chat_text(tok, "Read the passage and answer the question.")
    ids = tok(text, add_special_tokens=False).input_ids
    n = 0
    for it in ITEMS.values():
        for k in ("answer", "S", "X", "Z", "D"):
            assert frame_cont(tok, text, ids, it[k], " ") == cont_ids(tok, text, ids, it[k]), it[k]
            n += 1
    assert n == 5 * len(ITEMS)
    it = item(list(CTX)[0])
    assert cb_raw("CBOPT", it) == ("Answer the question.\n\nQuestion: " + it["question"].strip() + "\nOptions: "
                                   + "; ".join(it["options"]) + "\nAnswer with exactly one of the options.")
    assert cb_raw("CBLET", it).endswith("D. " + it["options"][3] + "\nAnswer with the letter of the correct option.")
    L = {Y: "ABCD"[it["options"].index(it[k])] for Y, k in (("B", "answer"), ("S", "S"), ("X", "X"), ("D", "D"))}
    assert answer_of(f" {L['S']}. {it['S']}", it, "LETA") == "S" and answer_of(f" **{L['B']}**", it, "LETA") == "B"
    assert answer_of(" Answer: none", it, "LETA") == "other" and letter_of(" A") == "A" and letter_of(" Alpha") is None
    assert answer_of(" **" + it["answer"] + "**", it, "NOM") == "B" and answer_of(" The " + it["Z"] + " show", it, "NOM") == "Z"
    assert answer_of(" " + it["S"].split()[0] + " Crosby", it, "NOM") == "other"
    assert frame_of(" **Bing Crosby**", "Bing Crosby") == " **" and frame_of(" Crosby", "Bing Crosby") is None
    assert choose_frame([" **", " **", " ", None]) == (" **", {f: {" **": 2, " ": 1}.get(f, 0) for f in FRAMES})
    assert choose_frame([None, "x"])[0] == " " and choose_frame([" ", " **"])[0] == " "
