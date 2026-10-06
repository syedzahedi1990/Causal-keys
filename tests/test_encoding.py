"""ckeys.encoding: the arm registry, encode_raw / encode_any, and the standard arms unchanged."""
import random

import pytest
import torch
from transformers import AutoTokenizer

from ckeys.encoding import (ALL_ARMS, ARM_BUILDERS, ARMS, LISTING, PREFIX, ROOM, alphabet, arm_span, build_prompt,
                            candidate_ids, encode, encode_any, encode_raw, raw_prompt, register_arm)
from ckeys.story import LOCATIONS, make_cores, pick_x, record


def test_standard_arms_registered_and_unchanged():
    core = make_cores(1, random.Random(0))[0]
    r = record(core, "direct", core["base"])
    for arm in ARMS:
        assert arm in ARM_BUILDERS and not ARM_BUILDERS[arm].needs_core
        assert build_prompt(arm, r["story"], r["query"], core, pick_x(core)) == raw_prompt(arm, r["story"], r["query"])
    assert raw_prompt("POST", "S.", "Q?") == PREFIX + "S. " + ROOM + "\nQuestion: Q?\nAnswer with one word.\nAnswer:"
    assert raw_prompt("P1", "S.", "Q?") == PREFIX + "S.\nQuestion: Q?\n" + LISTING + "\nAnswer with exactly one choice.\nAnswer:"
    assert arm_span("POST", "S.", "Q?") == ROOM and arm_span("NONE", "S.", "Q?") is None and arm_span("P1", "S.", "Q?") == LISTING
    assert alphabet("LETTER") == ("A", "B", "C", "D", "E", "F") and alphabet("AFTER") == LOCATIONS
    assert ALL_ARMS[:7] == list(ARMS)
    with pytest.raises(ValueError):
        raw_prompt("NO_SUCH_ARM", "S.", "Q?")


def test_register_core_dependent_and_independent_arms():
    sent = "The room has a crate and a hamper."
    register_arm("T_INDEP", lambda story, query: PREFIX + story + " " + sent + "\nQuestion: " + query + "\nAnswer with one word.\nAnswer:",
                 span=lambda story, query: sent)

    def named(core, X):
        return tuple(l for l in LOCATIONS if l in (core["source"], X))

    register_arm("T_DEP", lambda story, query, core, X: PREFIX + story + "\nQuestion: " + query + "\nChoices: " + ", ".join(named(core, X)) + "\nAnswer:",
                 needs_core=True, span=lambda story, query, core, X: "Choices: " + ", ".join(named(core, X)),
                 meta=lambda core, X: {"named": named(core, X)})
    core = make_cores(1, random.Random(0))[0]
    X = pick_x(core)
    r = record(core, "direct", core["base"])
    assert raw_prompt("T_INDEP", r["story"], r["query"]) == build_prompt("T_INDEP", r["story"], r["query"]) and sent in raw_prompt("T_INDEP", r["story"], r["query"])
    with pytest.raises(ValueError):
        raw_prompt("T_DEP", r["story"], r["query"])  # a core-dependent arm needs build_prompt
    p = build_prompt("T_DEP", r["story"], r["query"], core, X)
    assert arm_span("T_DEP", r["story"], r["query"], core, X) in p and ARM_BUILDERS["T_DEP"].meta(core, X)["named"] == named(core, X)
    assert "T_DEP" in ALL_ARMS and "T_INDEP" in ALL_ARMS and alphabet("T_DEP") == LOCATIONS
    register_arm("T_DEP", ARM_BUILDERS["T_DEP"].build, needs_core=True)  # re-registering the same builder is allowed
    with pytest.raises(AssertionError):
        register_arm("T_DEP", lambda *a: "")


def test_encode_raw_and_any():
    tq, tg = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"), AutoTokenizer.from_pretrained("gpt2")
    raw = "When Ruth and Charles got a ticket, Charles gave it to"
    assert encode_raw(tq, raw).tolist() == [tq(raw, add_special_tokens=False).input_ids]           # Qwen: no BOS
    g = encode_raw(tg, raw)
    assert g[0, 0].item() == tg.bos_token_id and g[0, 1:].tolist() == tg(raw, add_special_tokens=False).input_ids
    assert encode_raw(tg, raw, bos=False).shape[1] == g.shape[1] - 1
    assert torch.equal(encode_any(tq, raw, chat=True), encode(tq, raw)) and torch.equal(encode_any(tg, raw, chat=False), g)
    assert len(candidate_ids(tq, "P1")) == 6
