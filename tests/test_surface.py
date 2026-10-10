"""Gate JB-G0 (preregistration J), items 1-3: the one-pass trie score of ckeys.surface equals separate plain forward
passes for every form (FP32, CPU, 1e-4 nats) at Qwen2.5-0.5B-Instruct (sdpa and eager; clean and under key-only and
value-only clamps) and on tiny random-weight models of the stage-8 families (Gemma-2 with soft-capping, Phi-3, Llama,
Mistral); FormSet invariants on the cached tokenizers."""
import random

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config, LlamaConfig, MistralConfig, Phi3Config

from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import build_prompt, chat_text
from ckeys.interventions import blocks
from ckeys.story import LOCATIONS, QUERIES, make_cores, record
from ckeys.surface import FRAMES_E_FIXED, FRAMES_SIGMA, FormSet, score, score_reference

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 1e-4
torch.set_grad_enabled(False)


def close(a, b):
    return torch.allclose(a, b, atol=TOL, rtol=0)


def same(r1, r2, fs):
    for name in ("L", *fs.sets):
        for w in fs.words:
            assert close(r1[name][w], r2[name][w]), (name, w, (r1[name][w] - r2[name][w]).abs().max())
    assert close(r1["first"], r2["first"])
    assert close(r1["gap"], r2["gap"])


def prompts(tok, n=2):
    out = []
    for core in make_cores(n, random.Random(5)):
        rec = record(core, "direct", core["base"])
        raw = build_prompt("POST", rec["story"], rec["query"], core)
        out.append(tok(chat_text(tok, raw), add_special_tokens=False, return_tensors="pt").input_ids)
    return out


@pytest.fixture(scope="module", params=["sdpa", "eager"])
def qwen(request):
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation=request.param).eval()
    return model, tok


def test_trie_equals_plain_passes_clean_and_clamped(qwen):
    model, tok = qwen
    fs = FormSet(tok, LOCATIONS[:3], {"sigma": FRAMES_SIGMA, "E": FRAMES_SIGMA + (" On the ", " **", " {a} put it in the ")},
                 names={"a": "Alice"})
    assert len(fs) > 6 and not fs.check_decode()
    ids = prompts(tok, 1)[0]
    same(score(model, ids, fs), score_reference(model, ids, fs), fs)
    plain = torch.log_softmax(model(ids, use_cache=False).logits[:, -1].float(), -1)
    assert close(score(model, ids, fs)["first"], plain)                     # item 3: the answer position
    # under clamps at a prompt position (the decisive pre-RoPE K or V of a token in the story), batched rows
    nL, P = len(blocks(model)), [ids.shape[1] - 30]
    other = ids.clone()
    other[0, P] = tok(" basket", add_special_tokens=False).input_ids[0]
    with capture_kv(model, P, range(nL)) as kv:
        model(other, use_cache=False)
    tab = {k: v[0] for k, v in kv.items()}
    for ch in ("k", "v"):
        with clamp_kv(model, P, tab, range(nL), which=ch):
            r1 = score(model, ids.expand(2, -1), fs)
            r2 = score_reference(model, ids, fs)
        same(row(r1, 0, fs), r2, fs)
        same(row(r1, 1, fs), r2, fs)


def row(r, i, fs):
    out = {name: {w: r[name][w][i:i + 1] for w in fs.words} for name in ("L", *fs.sets)}
    return out | {"first": r["first"][i:i + 1], "gap": r["gap"][i:i + 1]}


SMALL = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=2, max_position_embeddings=256)
TINY = {
    "gemma2": lambda: Gemma2Config(**SMALL, head_dim=16, sliding_window=128, query_pre_attn_scalar=16,
                                   attn_logit_softcapping=50.0, final_logit_softcapping=30.0),
    "phi3": lambda: Phi3Config(**SMALL, pad_token_id=0),
    "llama": lambda: LlamaConfig(**SMALL),
    "mistral": lambda: MistralConfig(**SMALL),
}


class _Tok:
    """A stand-in tokenizer over the tiny vocabulary: one id per character (offset 3)."""

    def __call__(self, s, add_special_tokens=False):
        return type("E", (), {"input_ids": [3 + (ord(c) % 120) for c in s]})()

    def decode(self, ids):
        return "".join(chr(i - 3) for i in ids)


@pytest.mark.parametrize("fam", list(TINY))
def test_trie_equals_plain_passes_tiny_families(fam):
    torch.manual_seed(0)
    cfg = TINY[fam]()
    cfg._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(cfg).float().eval()
    tok = _Tok()
    fs = FormSet(tok, ["box", "shelf", "map"], {"sigma": (" ", " the "), "E": (" ", " the ", " On the ")})
    ids = torch.randint(3, 120, (2, 20), generator=torch.Generator().manual_seed(1))
    same(score(model, ids, fs), score_reference(model, ids, fs), fs)


@pytest.mark.parametrize("name", ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
                                  "unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/gemma-2-9b-it", "microsoft/phi-4",
                                  "tiiuae/Falcon3-7B-Instruct", "allenai/OLMo-2-1124-7B-Instruct"])
def test_formset_invariants_on_study_tokenizers(name):
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except OSError as ex:
        pytest.skip(f"tokenizer not cached: {ex}")
    fs = FormSet(tok, LOCATIONS)
    assert not fs.check_decode()
    for w in LOCATIONS:
        assert len(fs.sets["E"][w]) >= len(fs.sets["sigma"][w]) >= 4
