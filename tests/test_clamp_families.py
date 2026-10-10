"""ckeys.clamp on the stage-8 families (preregistration J): K/V sites of Phi-3/Phi-4 (fused qkv_proj), Gemma-2 and
Llama/Falcon3 (k_proj/v_proj), and the span form of ckeys.headsplice.HeadSplice. Tiny random-weight models built from
each family's config class (FP32, CPU, eager attention): clamping every layer's K and V at the positions where two
inputs differ, from a capture of the second input, reproduces the second input's logits at every later position; K
alone and V alone each move the logits; a clamp from the run's own capture is the identity."""
import pytest
import torch
from transformers import Gemma2Config, LlamaConfig, Phi3Config, AutoModelForCausalLM

from ckeys.clamp import capture_kv, clamp_kv, kv_sites
from ckeys.headsplice import HeadSplice, head_masks
from ckeys.interventions import blocks

TOL = 1e-4
torch.set_grad_enabled(False)

SMALL = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
             num_key_value_heads=2, max_position_embeddings=128)
CONFIGS = {
    "phi3": lambda: Phi3Config(**SMALL, pad_token_id=0),
    "gemma2": lambda: Gemma2Config(**SMALL, head_dim=16, sliding_window=64, query_pre_attn_scalar=16),
    "llama": lambda: LlamaConfig(**SMALL),
}


def tiny(name):
    torch.manual_seed(0)
    cfg = CONFIGS[name]()
    cfg._attn_implementation = "eager"
    return AutoModelForCausalLM.from_config(cfg).float().eval()


def inputs():
    g = torch.Generator().manual_seed(1)
    b = torch.randint(3, 97, (1, 24), generator=g)
    s = b.clone()
    P = [9, 10, 11]
    s[0, P] = torch.randint(3, 97, (len(P),), generator=g)
    assert (s != b).any()
    return b, s, P


@pytest.mark.parametrize("name", list(CONFIGS))
def test_kv_clamp_from_layer0_reproduces_the_donor(name):
    model = tiny(name)
    nL, (b, s, P) = len(blocks(model)), inputs()
    if name == "phi3":
        assert kv_sites(model, 0)[0][0] is blocks(model)[0].self_attn.qkv_proj
    lg = lambda x: model(x, use_cache=False).logits[0]  # noqa: E731
    with capture_kv(model, P, range(nL)) as kvS:
        ref = lg(s)
    with capture_kv(model, P, range(nL)) as kvB:
        clean = lg(b)
    tab = {k: v[0] for k, v in kvS.items()}
    with clamp_kv(model, P, tab, range(nL)):
        out = lg(b)
    t = max(P) + 1
    assert torch.allclose(out[t:], ref[t:], atol=TOL, rtol=0), (out[t:] - ref[t:]).abs().max()
    assert (clean[t:] - ref[t:]).abs().max() > 1e-2
    for ch in "kv":
        with clamp_kv(model, P, tab, range(nL), which=ch):
            assert (lg(b)[t:] - clean[t:]).abs().max() > 1e-3, ch
    with clamp_kv(model, P, {k: v[0] for k, v in kvB.items()}, range(nL)):
        assert torch.allclose(lg(b), clean, atol=TOL, rtol=0)


@pytest.mark.parametrize("B", [1, 3])
def test_headsplice_span_table_all_heads_equals_key_clamp(B):
    """A [|P|, D] span table with |P| = 3: all heads in all rows equal the K-only clamp; B = 3 is the case the old
    broadcast mis-assigned (|P| equal to the batch size)."""
    model = tiny("llama")
    nL, (b, s, P) = len(blocks(model)), inputs()
    H, T = model.config.num_attention_heads, b.shape[1]
    lg = lambda x: model(x, use_cache=False).logits  # noqa: E731
    with capture_kv(model, P, range(nL), which="k") as kS:
        lg(s)
    tab = {k: v[0] for k, v in kS.items()}
    with clamp_kv(model, P, tab, range(nL), which="k"):
        ref = lg(b)
    clean = lg(b)
    hs = HeadSplice(model)
    hs.ks, hs.pos, hs.mode, hs.mu = {l: tab[(l, "k")] for l in range(nL)}, list(P), "splice", None
    hs.masks, hs.active = head_masks(torch.ones(1, nL, H, dtype=torch.bool), slice(None), T), True
    try:
        out = lg(b.expand(B, -1))
    finally:
        hs.active, hs.masks = False, None
    assert torch.allclose(out, ref.expand(B, -1, -1), atol=TOL, rtol=0), (out - ref).abs().max()
    assert (ref - clean).abs().max() > 1e-3
    hs.masks, hs.active = {}, True
    try:
        assert torch.allclose(lg(b), clean, atol=TOL, rtol=0)
    finally:
        hs.active, hs.masks = False, None
