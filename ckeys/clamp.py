"""Position-set key/value clamps: the format_factorial hooks at one position p, generalised to any position set
and to per-batch-row tables, for Qwen2/Qwen3/Mistral/Llama/OLMo-2/Gemma-2/Falcon3 (k_proj / v_proj), GPT-2 (fused
c_attn) and Phi-3/Phi-4 (fused qkv_proj).

Sites are the pre-RoPE projection outputs: later positions see a clamped position only through its per-layer
K and V, so clamping every layer from a captured run reproduces that run's logits exactly (FP32).
Tables are keyed by ``(layer, ch)`` with ``ch`` in ``"kv"``; a table is ``[P, D]`` (shared by the batch) or
``[R, P, D]`` (one row per batch row), ``P = len(positions)``.
"""
from __future__ import annotations

import contextlib
from typing import Iterable, Sequence

import torch

from .interventions import _out_tensor, _with_tensor, blocks, hooks


def kv_sites(model, layer: int) -> tuple[tuple[torch.nn.Module, slice], tuple[torch.nn.Module, slice]]:
    """((module, slice) of the key, (module, slice) of the value): ``module``'s output at ``[:, p, slice]``."""
    blk = blocks(model)[layer]
    if hasattr(blk, "attn") and hasattr(blk.attn, "c_attn"):  # GPT-2: c_attn -> (q, k, v) split at D
        D = blk.attn.split_size
        return (blk.attn.c_attn, slice(D, 2 * D)), (blk.attn.c_attn, slice(2 * D, 3 * D))
    at = blk.self_attn
    if hasattr(at, "qkv_proj"):  # Phi-3/Phi-4: qkv_proj -> (q [H*hd], k [KVH*hd], v [KVH*hd])
        cfg = model.config
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        q, kv = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
        return (at.qkv_proj, slice(q, q + kv)), (at.qkv_proj, slice(q + kv, q + 2 * kv))
    return (at.k_proj, slice(None)), (at.v_proj, slice(None))


def site(model, layer: int, ch: str) -> tuple[torch.nn.Module, slice]:
    return kv_sites(model, layer)["kv".index(ch)]


@contextlib.contextmanager
def capture_kv(model, positions: Sequence[int], layers: Iterable[int], which: str = "kv"):
    """Yields ``{(layer, ch): tensor [B, P, D]}`` of the keys/values at ``positions`` (full-sequence forwards only)."""
    store, hs, pos = {}, [], list(positions)
    for l in layers:
        for ch in which:
            mod, sl = site(model, l, ch)

            def fn(_m, _i, out, key=(l, ch), sl=sl):
                h = _out_tensor(out)
                if h.shape[1] > max(pos):
                    store[key] = h[:, pos, sl].detach().clone()

            hs.append(mod.register_forward_hook(fn))
    with hooks(hs):
        yield store


@contextlib.contextmanager
def clamp_kv(model, positions: Sequence[int], tables: dict, layers: Iterable[int], which: str = "kv",
             per_row: torch.Tensor | None = None):
    """Overwrite the keys/values at ``positions`` in ``layers`` by ``tables[(layer, ch)]`` during the block.

    ``per_row``: optional boolean ``[R]`` or ``[R, P]`` selecting which batch rows (and positions) are written;
    unselected entries keep the run's own key/value. Rows of a ``[R, P, D]`` table index the batch.
    """
    hs, pos = [], list(positions)
    if per_row is not None:
        per_row = per_row.bool()
        per_row = per_row[:, None].expand(-1, len(pos)) if per_row.dim() == 1 else per_row
    for l in layers:
        for ch in which:
            mod, sl = site(model, l, ch)

            def fn(_m, _i, out, t=tables[(l, ch)], sl=sl):
                h = _out_tensor(out)
                if h.shape[1] <= max(pos):
                    return out
                h = h.clone()
                new = t.to(h.device, h.dtype)
                if per_row is None:
                    h[:, pos, sl] = new
                else:
                    new = (new if new.dim() == 3 else new[None]).expand(h.shape[0], -1, -1)
                    h[:, pos, sl] = torch.where(per_row.to(h.device)[..., None], new, h[:, pos, sl])
                return _with_tensor(out, h)

            hs.append(mod.register_forward_hook(fn))
    with hooks(hs):
        yield


def stack_rows(kv: dict, rows: Sequence, layers: Iterable[int], which: str = "kv") -> dict:
    """Per-row tables from captured runs: ``kv[name][(l, ch)]`` is ``[P, D]``; ``rows[r](l, ch)`` names the donor run
    of batch row ``r`` at layer ``l`` and channel ``ch``. Returns ``{(l, ch): [R, P, D]}``."""
    return {(l, ch): torch.stack([kv[who(l, ch)][(l, ch)] for who in rows]) for l in layers for ch in which}
