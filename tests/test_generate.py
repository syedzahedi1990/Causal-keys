"""ckeys.generate (preregistration J, gate JB-G0 item 5 and A-G0): greedy decoding with the KV cache under prompt
clamps equals cache-free stepwise argmax decoding with the clamps active at every step (FP32, CPU, Qwen2.5-0.5B and
tiny Gemma-2/Phi-3 models); the teacher-forced argmax chain agrees with greedy output; answer parsing and frames."""
import random

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config, Phi3Config

from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import build_prompt, chat_text
from ckeys.generate import (argmax_chain, candidate_stop, discover_frames, extract_frame, greedy, greedy_reference,
                            parse_answer)
from ckeys.interventions import blocks
from ckeys.story import make_cores, record

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def qwen():
    return (AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval(), AutoTokenizer.from_pretrained(NAME))


def story_ids(tok, arm="NONE"):
    core = make_cores(1, random.Random(7))[0]
    rec = record(core, "direct", core["base"])
    raw = build_prompt(arm, rec["story"], rec["query"], core)
    return tok(chat_text(tok, raw), add_special_tokens=False, return_tensors="pt").input_ids, core


def test_greedy_with_cache_equals_stepwise_reference_under_clamps(qwen):
    model, tok = qwen
    ids, core = story_ids(tok)
    nL = len(blocks(model))
    p = (ids[0] == tok(" " + core["base"], add_special_tokens=False).input_ids[0]).nonzero()[-1].tolist()
    other = ids.clone()
    other[0, p] = tok(" " + core["source"], add_special_tokens=False).input_ids[0]
    with capture_kv(model, p, range(nL)) as kv:
        model(other, use_cache=False)
    tab = {k: v[0] for k, v in kv.items()}
    for ch in ("kv", "k", "v"):
        with clamp_kv(model, p, tab, range(nL), which=ch):
            a = greedy(model, tok, ids.expand(2, -1), max_new=8)
            b = greedy_reference(model, tok, ids, max_new=8)
        assert a[0] == a[1] == b[0], (ch, tok.decode(a[0]), tok.decode(b[0]))
        assert len(a[0]) > 0
        with clamp_kv(model, p, tab, range(nL), which=ch):
            assert bool(argmax_chain(model, ids, a[0])[0])
    with clamp_kv(model, p, tab, range(nL)):
        g = greedy(model, tok, ids, max_new=8)[0]
    with clamp_kv(model, p, tab, range(nL)):
        assert not bool(argmax_chain(model, ids, g[:-1] + [(g[-1] + 1) % 1000])[0])
    stop = candidate_stop(tok)
    s = greedy(model, tok, ids, max_new=12, stop=stop)[0]
    assert parse_answer(tok.decode(s)) != "other" or len(s) == 12


@pytest.mark.parametrize("cfg", [
    lambda: Gemma2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, max_position_embeddings=128, sliding_window=64,
                         query_pre_attn_scalar=16),
    lambda: Phi3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, max_position_embeddings=128, pad_token_id=0)])
def test_greedy_cache_equals_reference_tiny(cfg):
    torch.manual_seed(0)
    c = cfg()
    c._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(c).float().eval()
    model.generation_config.eos_token_id = None

    class T:
        eos_token_id = None

        def decode(self, ids):
            return "".join(chr(65 + i % 26) for i in ids)

    ids = torch.randint(3, 97, (1, 16), generator=torch.Generator().manual_seed(2))
    P = [5, 6]
    other = ids.clone()
    other[0, P] = torch.tensor([7, 8])
    nL = len(blocks(model))
    with capture_kv(model, P, range(nL)) as kv:
        model(other, use_cache=False)
    with clamp_kv(model, P, {k: v[0] for k, v in kv.items()}, range(nL), which="k"):
        assert greedy(model, T(), ids, max_new=6) == greedy_reference(model, T(), ids, max_new=6)


def test_parse_and_frames():
    assert parse_answer(" The shelf") == "shelf"
    assert parse_answer(" Boxes") == "box"
    assert parse_answer("**Shelf**") == "shelf"
    assert parse_answer(" On the shelf.") == "shelf"
    assert parse_answer(" I don't know") == "other"
    names = {"a": "Alice", "b": "Bob", "o": "candle", "d": "map"}
    assert extract_frame(" Alice thinks the candle is in the drawer", names) == " {a} thinks the {o} is in the "
    assert extract_frame(" the box", names) == " the "
    assert extract_frame(" nothing", names) is None
    recs = [("NONE", " {a} thinks it is in the ")] * 3 + [("NONE", " the ")] * 50 + [("NONE", None)] * 47
    assert discover_frames(recs, fixed=(" the ",)) == [" {a} thinks it is in the "]
    assert discover_frames(recs[:2] + [("NONE", None)] * 98, fixed=()) == [" {a} thinks it is in the "]
    assert discover_frames([("NONE", "x ")] + [("NONE", None)] * 99, fixed=()) == []
