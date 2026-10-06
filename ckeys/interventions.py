"""Intervention primitives for HF decoder-only LMs (Qwen2/Llama/Mistral/Gemma layouts; GPT-2 for ``blocks``).

Sites:
  * residual stream output of decoder block ``layer`` (``resid``),
  * post-projection, pre-RoPE attention keys / values (``k_proj`` / ``v_proj`` outputs).

All position arguments index the token axis of a single (unpadded) sequence batch.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Iterable, Sequence

import torch
from torch import nn


def blocks(model) -> nn.ModuleList:
    return model.transformer.h if hasattr(model, "transformer") else model.model.layers


def _out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _with_tensor(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


@contextlib.contextmanager
def hooks(handles: list):
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- capture
@contextlib.contextmanager
def capture(model, layers: Iterable[int], site: str = "resid"):
    """Record activations at ``site`` for each layer. Yields dict layer -> tensor [B, T, D]."""
    store: dict[int, torch.Tensor] = {}
    handles = []
    for l in layers:
        mod = _site_module(model, l, site)

        def fn(_m, _i, out, l=l):
            store[l] = _out_tensor(out).detach()

        handles.append(mod.register_forward_hook(fn))
    with hooks(handles):
        yield store


def _site_module(model, layer: int, site: str) -> nn.Module:
    blk = blocks(model)[layer]
    if site == "resid":
        return blk
    if site == "k":
        return blk.self_attn.k_proj
    if site == "v":
        return blk.self_attn.v_proj
    if site == "q":
        return blk.self_attn.q_proj
    raise ValueError(site)


# --------------------------------------------------------------------------- edit
@contextlib.contextmanager
def edit(model, layer: int, site: str, positions: Sequence[int] | torch.Tensor,
         fn: Callable[[torch.Tensor], torch.Tensor]):
    """Replace activations at ``positions`` of ``site`` in ``layer`` by ``fn(acts[:, positions])``.

    ``positions`` may be a 1-D index list shared by the batch or a [B, P] tensor of per-example
    positions. The hook only fires on full-sequence forwards (not on single-token decode steps).
    """
    mod = _site_module(model, layer, site)

    def hook(_m, _i, out):
        h = _out_tensor(out)
        if isinstance(positions, torch.Tensor) and positions.dim() == 2:
            idx = positions.to(h.device)
            gather = idx.unsqueeze(-1).expand(-1, -1, h.shape[-1])
            sel = torch.gather(h, 1, gather)
            new = fn(sel)
            h = h.scatter(1, gather, new.to(h.dtype))
        else:
            pos = list(positions)
            if max(pos) >= h.shape[1]:
                return out
            h = h.clone()
            h[:, pos] = fn(h[:, pos]).to(h.dtype)
        return _with_tensor(out, h)

    handle = mod.register_forward_hook(hook)
    with hooks([handle]):
        yield


@contextlib.contextmanager
def edits(model, specs: list[tuple[int, str, Sequence[int] | torch.Tensor, Callable]]):
    with contextlib.ExitStack() as stack:
        for layer, site, pos, fn in specs:
            stack.enter_context(edit(model, layer, site, pos, fn))
        yield


# --------------------------------------------------------------------------- subspaces
class RotatedSubspace(nn.Module):
    """Rank-r orthonormal subspace U (rows orthonormal), interchange I(h_b, h_s) = h_b + U^T U (h_s - h_b)."""

    def __init__(self, dim: int, rank: int, init: torch.Tensor | None = None):
        super().__init__()
        lin = nn.Linear(dim, rank, bias=False)
        if init is not None:
            with torch.no_grad():
                lin.weight.copy_(init[:rank])
        self.proj = nn.utils.parametrizations.orthogonal(lin)

    @property
    def U(self) -> torch.Tensor:  # [r, d]
        return self.proj.weight

    def interchange(self, h_base: torch.Tensor, h_src: torch.Tensor) -> torch.Tensor:
        U = self.U.to(h_base.dtype)
        return h_base + (h_src - h_base) @ U.T @ U


def pca_basis(diffs: torch.Tensor, rank: int, center: bool = True) -> torch.Tensor:
    """Top-``rank`` right singular vectors of source-minus-base differences [N, d] -> [rank, d]."""
    X = diffs.float()
    if center:
        X = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[:rank]


def fixed_subspace_interchange(U: torch.Tensor):
    def f(h_base, h_src):
        Uc = U.to(h_base.dtype)
        return h_base + (h_src - h_base) @ Uc.T @ Uc
    return f


# --------------------------------------------------------------------------- key exchange
def signed_permutation_null(recipient: torch.Tensor, donor: torch.Tensor, n_kv: int,
                            generator: torch.Generator) -> torch.Tensor:
    """Matched null: apply a fixed signed coordinate permutation per KV group to (donor - recipient).

    Preserves the norm of the displacement within each KV group but scrambles its direction.
    ``recipient``/``donor``: [..., n_kv * head_dim].
    """
    hd = recipient.shape[-1] // n_kv
    disp = (donor - recipient).reshape(*recipient.shape[:-1], n_kv, hd)
    out = torch.empty_like(disp)
    for g in range(n_kv):
        perm = torch.randperm(hd, generator=generator)
        sign = (torch.randint(0, 2, (hd,), generator=generator) * 2 - 1).to(disp.dtype)
        out[..., g, :] = disp[..., g, perm] * sign
    return recipient + out.reshape(recipient.shape)
