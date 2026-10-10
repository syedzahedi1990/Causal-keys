"""ckeys.kvquant (preregistration J, Gate J-A-G0, the J-A7 hooks): KIVI-style fake quantization of the passage keys
and values. FP32, CPU, Qwen2.5-0.5B-Instruct, 1e-4 in the logits.

Checks: the quantizer equals an independent per-group reference (keys per channel over groups of 32 positions, values
per token over groups of 32 channels, last group shorter); 16 bits is the identity to 1e-3 relative; the hooked run
equals a layer-by-layer reference built from captured keys/values, the reference quantizer and ckeys.clamp; a mixed
batch equals its rows run singly; greedy decoding with the cache equals the cache-free stepwise reference under the
hooks; the recorded relative errors are those of the reference."""
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import chat_text
from ckeys.generate import greedy, greedy_reference
from ckeys.interventions import blocks
from ckeys.kvquant import fake_quant, kivi, quantize_kv, rel_error

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 1e-4
torch.set_grad_enabled(False)
PASSAGE = ("The first choral hymnal of the Reformation appeared in Wittenberg in 1524 with a preface by Martin Luther, "
           "who wrote twenty-four of its hymns; Johann Walter set them for four or five voices.")


def ref_quant(x, bits, axis, group):
    """Independent reference: numpy, one group at a time, every other index looped."""
    a = np.moveaxis(x.double().numpy(), axis, -1)
    out = np.empty_like(a)
    top = 2 ** bits - 1
    for idx in np.ndindex(*a.shape[:-1]):
        v = a[idx]
        for s in range(0, len(v), group):
            g = v[s:s + group]
            mn, mx = g.min(), g.max()
            if mx == mn:
                out[idx + (slice(s, s + group),)] = g
                continue
            sc = (mx - mn) / top
            out[idx + (slice(s, s + group),)] = np.clip(np.round((g - mn) / sc), 0, top) * sc + mn
    return torch.from_numpy(np.moveaxis(out, -1, axis))


@pytest.fixture(scope="module")
def qwen():
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    text = chat_text(tok, "Read the passage and answer the question.\n\nPassage: " + PASSAGE +
                     "\nQuestion: How many hymns did Luther write?\nAnswer with the exact words from the passage.")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    c0 = text.index(PASSAGE)
    pos = [i for i, (s, e) in enumerate(enc.offset_mapping) if e > c0 and s < c0 + len(PASSAGE)]
    assert 32 < len(pos) < 64, len(pos)   # two key groups, the second partial
    return model, tok, torch.tensor([enc.input_ids]), pos


def test_quantizer_equals_reference():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 45, 96, generator=g) * torch.linspace(0.1, 3.0, 96)
    x[0, :, 5] = 1.25   # a constant channel: returned unchanged
    for bits in (2, 3):
        k, v = kivi(x, "k", bits), kivi(x, "v", bits)
        assert torch.allclose(k.double(), ref_quant(x, bits, 1, 32), atol=1e-5)
        assert torch.allclose(v.double(), ref_quant(x, bits, 2, 32), atol=1e-5)
        assert torch.equal(k[0, :, 5], x[0, :, 5])
        # every key group of every channel takes at most 2^bits distinct values
        assert max(len(torch.unique(k[1, s:s + 32, c])) for s in (0, 32) for c in range(96)) <= 2 ** bits
        assert max(len(torch.unique(v[1, t, s:s + 32])) for t in range(45) for s in (0, 32, 64)) <= 2 ** bits
    e2 = rel_error(x, kivi(x, "v", 2)).mean()
    e16 = rel_error(x, kivi(x, "v", 16)).max()
    assert e16 < 1e-3 * e2 and e16 < 1e-4
    assert torch.allclose(fake_quant(x, 16, 1, 32), x, atol=float(x.abs().max()) / (2 ** 16 - 1), rtol=0)


def layerwise_reference(model, ids, pos, bits, chs):
    """Keys/values of every layer quantized in turn: layer l's inputs are captured with layers < l clamped to their
    reference-quantized tables, then quantized with ref_quant; the final pass clamps every layer."""
    nL = len(blocks(model))
    tabs, errs = {}, {}
    for l in range(nL):
        with clamp_kv(model, pos, tabs, list({k[0] for k in tabs}), which=chs) if tabs else _null():
            with capture_kv(model, pos, [l], chs) as C:
                model(ids, use_cache=False, logits_to_keep=1)
        for ch in chs:
            x = C[(l, ch)][0]
            xq = ref_quant(x, bits, 0 if ch == "k" else 1, 32).float()
            tabs[(l, ch)] = xq
            errs[(l, ch)] = float(rel_error(x, xq))
    with clamp_kv(model, pos, tabs, range(nL), which=chs):
        return model(ids, use_cache=False, logits_to_keep=1).logits[:, -1], errs


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.mark.parametrize("chs", ["k", "v"])
def test_hooked_run_equals_layerwise_reference(qwen, chs):
    model, tok, ids, pos = qwen
    nL = len(blocks(model))
    clean = model(ids, use_cache=False, logits_to_keep=1).logits[:, -1]
    ref, errs = layerwise_reference(model, ids, pos, 2, chs)
    stats = {}
    with quantize_kv(model, pos, 2, {chs: [True]}, stats=stats):
        got = model(ids, use_cache=False, logits_to_keep=1).logits[:, -1]
    assert (ref - clean).abs().max() > 1e-2, "2-bit quantization must change the logits"
    assert torch.allclose(got, ref, atol=TOL, rtol=0), float((got - ref).abs().max())
    assert sorted(stats) == [(l, chs) for l in range(nL)]
    for k, e in errs.items():
        assert abs(float(stats[k][0]) - e) < 1e-4, (k, float(stats[k][0]), e)
    with quantize_kv(model, pos, 16, {chs: [True]}):
        id16 = model(ids, use_cache=False, logits_to_keep=1).logits[:, -1]
    assert torch.allclose(id16, clean, atol=1e-2, rtol=0), float((id16 - clean).abs().max())


def test_mixed_batch_equals_single_rows_and_decoding(qwen):
    model, tok, ids, pos = qwen
    x = ids.expand(3, -1)
    with quantize_kv(model, pos, 2, {"k": [False, True, False], "v": [False, False, True]}):
        mixed = model(x, use_cache=False, logits_to_keep=1).logits[:, -1]
    single = [model(ids, use_cache=False, logits_to_keep=1).logits[:, -1]]
    for ch in "kv":
        with quantize_kv(model, pos, 2, {ch: [True]}):
            single.append(model(ids, use_cache=False, logits_to_keep=1).logits[:, -1])
    assert torch.allclose(mixed, torch.cat(single), atol=TOL, rtol=0)
    with quantize_kv(model, pos, 2, {"v": [True, False], "k": [False, True]}):
        a = greedy(model, tok, ids.expand(2, -1), max_new=6)
    for r, ch in enumerate("vk"):
        with quantize_kv(model, pos, 2, {ch: [True]}):
            assert a[r] == greedy_reference(model, tok, ids, max_new=6)[0], ch
