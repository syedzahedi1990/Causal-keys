"""KIVI-style fake quantization of cached keys and values (stage 8 part A, preregistration J, line J-A7).

KIVI (Liu et al. 2024) stores the key cache quantized per channel and the value cache per token. Here the quantization
is simulated ("fake": quantize, then dequantize at once) on the outputs of the key and value projections (the sites of
ckeys.clamp: pre-RoPE k_proj / v_proj outputs, or the key / value slices of a fused projection) at a chosen set of
positions (the passage tokens), in every layer:
  * keys, per channel: the positions are cut into consecutive groups of ``key_group`` (32) in the order given; each
    channel of each group gets its own scale and zero point (min and max over the group's positions);
  * values, per token: the channels of each position are cut into consecutive groups of ``value_group`` (32); each
    group gets its own scale and zero point (min and max over its channels).
Quantization is asymmetric uniform with round-to-nearest: q = round((x - min) / s), s = (max - min) / (2^bits - 1),
x' = q * s + min, computed in FP32 and cast back. A last group shorter than the group size is quantized as it is
(KIVI keeps a full-precision residual of recent tokens; here every passage token is quantized). A group whose entries
are all equal is returned unchanged.

``quantize_kv`` acts only in full-sequence passes (a forward whose length does not reach the positions passes through),
so in greedy decoding with the KV cache the prompt pass writes the quantized keys and values into the cache and every
decode step reads them, as a quantized cache would. It records the relative reconstruction error ||x' - x|| / ||x||
(Frobenius, over the quantized block of each selected batch row) per layer and channel.
"""
from __future__ import annotations

import contextlib
from typing import Iterable, Sequence

import torch

from .clamp import site
from .interventions import _out_tensor, _with_tensor, blocks, hooks


def fake_quant(x: torch.Tensor, bits: int, axis: int, group: int) -> torch.Tensor:
    """Fake-quantize ``x`` with one scale and zero point per group of ``group`` consecutive entries along ``axis`` (every
    other index has its own groups). Returns a tensor of x's shape and dtype."""
    assert bits >= 1 and group >= 1
    xf = x.float().movedim(axis, -1)
    n, top = xf.shape[-1], float(2 ** bits - 1)
    out = torch.empty_like(xf)
    for s in range(0, n, group):
        g = xf[..., s:s + group]
        mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        scale = (mx - mn) / top
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.round((g - mn) / safe).clamp(0, top)
        out[..., s:s + group] = torch.where(scale > 0, q * scale + mn, g)
    return out.movedim(-1, axis).to(x.dtype)


def kivi(x: torch.Tensor, ch: str, bits: int, key_group: int = 32, value_group: int = 32) -> torch.Tensor:
    """``x`` [..., positions, D]: keys per channel (groups of positions), values per token (groups of channels)."""
    return fake_quant(x, bits, -2, key_group) if ch == "k" else fake_quant(x, bits, -1, value_group)


def rel_error(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """||xq - x|| / ||x|| over the last two axes (one value per leading index), in FP32."""
    x, xq = x.float(), xq.float()
    return (xq - x).flatten(-2).norm(dim=-1) / x.flatten(-2).norm(dim=-1).clamp_min(1e-30)


@contextlib.contextmanager
def quantize_kv(model, positions: Sequence[int], bits: int, rows: dict, layers: Iterable[int] | None = None,
                key_group: int = 32, value_group: int = 32, stats: dict | None = None):
    """Fake-quantize the keys and/or values at ``positions`` in ``layers`` (default every layer).

    ``rows``: {"k": bool [R] or None, "v": bool [R] or None}: the batch rows whose keys / values are quantized (None or a
    missing channel: that channel is not quantized in any row; a channel whose mask has no True entry is skipped).
    ``stats``: optional dict filled with {(layer, ch): FP32 tensor [n selected rows]} of relative reconstruction errors
    from the last full-sequence pass.
    """
    pos = list(positions)
    hs = []
    layers = range(len(blocks(model))) if layers is None else layers
    for l in layers:
        for ch in "kv":
            sel = rows.get(ch)
            if sel is None:
                continue
            sel = torch.as_tensor(sel, dtype=torch.bool)
            if not bool(sel.any()):
                continue
            mod, sl = site(model, l, ch)

            def fn(_m, _i, out, key=(l, ch), sl=sl, sel=sel):
                h = _out_tensor(out)
                if h.shape[1] <= max(pos):
                    return out
                assert sel.shape[0] == h.shape[0], f"rows mask has {sel.shape[0]} rows, the batch {h.shape[0]}"
                h = h.clone()
                idx = sel.nonzero().flatten().to(h.device)
                x = h[idx][:, pos, sl]
                xq = kivi(x, key[1], bits, key_group, value_group)
                blk = h[idx]
                blk[:, pos, sl] = xq
                h[idx] = blk
                if stats is not None:
                    stats[key] = rel_error(x, xq).cpu()
                return _with_tensor(out, h)

            hs.append(mod.register_forward_hook(fn))
    with hooks(hs):
        yield
