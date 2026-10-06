"""Per-head and per-row attention splices (stage 6, P-2026-10-05-H part (a)) for the q/k/v/o_proj layout of
Qwen2/Mistral/Llama (eager or sdpa attention; every pass with use_cache=False).

HeadSplice wraps every ``self_attn.forward``. Mode "splice": a layer whose mask is non-empty runs twice on the same
hidden states, once with the run's own key at ``pos`` and once with ``ks[l]`` there (k_proj hook, all KV groups); the
o_proj input (head h = columns h*hd:(h+1)*hd) takes head h's slice in row t of batch row b from the second pass where
``masks[l][b, t, h]`` and from the first elsewhere, and o_proj runs once on the spliced tensor. Head h's slice depends
on the key at ``pos`` only through head h's own attention, so this is "the swapped key is visible to query head h only,
in the masked rows". Mode "ablate": one pass; the masked slices are replaced by ``mu[l][b, t, h]`` (``mu`` None: 0).

HopSplice overwrites the K/V (k_proj/v_proj outputs) of the rows ``rows`` per batch row, from ``tabs["A"]`` in pass A
and ``tabs["B"]`` in pass B (``which[pass][ch]``: bool per batch row, K and V separate; unselected rows keep their own
values); output row t of batch row b comes from pass B where ``mask[b, t]``, else from pass A (one pass when no row
of the mask is set). Tables are keyed ``(layer, "k"|"v")`` as in ckeys.clamp, ``[1 or B, len(rows), D]``.

Masks, means and tables with batch dimension 1 broadcast over the batch; any other mismatch raises. The wrappers pass
through while inactive, so they nest with each other and with row_restricted_keys.RowSplice; one instance per model.
"""
from __future__ import annotations

import torch

from .interventions import blocks


def _bcast(x, B):
    assert x.shape[0] in (1, B), f"batch dimension {x.shape[0]} does not match the batch ({B}); only 1 broadcasts"
    return x.expand(B, *x.shape[1:]) if x.shape[0] != B else x


def _once(model, name, obj):
    assert not getattr(model, name, None), f"{type(obj).__name__} is already installed on this model (one per model)"
    setattr(model, name, True)


class HeadSplice:
    def __init__(self, model):
        _once(model, "_ckeys_headsplice", self)
        cfg = model.config
        self.model, self.H, self.nL = model, cfg.num_attention_heads, len(blocks(model))
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.active, self.mode, self.pos, self.ks, self.masks, self.mu = False, "splice", None, None, None, None
        self._src, self._oin, self.n_double = False, {}, 0
        for l, blk in enumerate(blocks(model)):
            at = blk.self_attn
            at.k_proj.register_forward_hook(self._khook(l))
            at.o_proj.register_forward_pre_hook(self._ohook(l))
            at.forward = self._wrap(at.forward, l, at)

    def _khook(self, l):
        def hk(_m, _i, out):
            if not (self.active and self._src):
                return out
            out, k = out.clone(), self.ks[l].to(out.device, out.dtype)
            out[:, self.pos] = k if k.dim() == 1 else _bcast(k, out.shape[0])
            return out
        return hk

    def _ohook(self, l):
        def hk(_m, args):
            if self.active and self.masks and l in self.masks:
                self._oin[l] = args[0]
        return hk

    def _wrap(self, f, l, at):
        def fwd(*a, **kw):
            M = self.masks.get(l) if self.active and self.masks else None
            if M is None or not bool(M.any()):
                return f(*a, **kw)
            base = f(*a, **kw)
            ob = self._oin[l]
            B, T = ob.shape[:2]
            assert M.shape[1:] == (T, self.H), (M.shape, T, self.H)
            if self.mode == "splice":
                self._src = True
                try:
                    f(*a, **kw)
                finally:
                    self._src = False
                alt = self._oin[l]
                self.n_double += 1
            elif self.mode == "ablate":
                alt = torch.zeros_like(ob) if self.mu is None else _bcast(self.mu[l], B).reshape(B, T, -1).to(ob.device, ob.dtype)
            else:
                raise ValueError(self.mode)
            m = _bcast(M.to(ob.device), B)[..., None].expand(B, T, self.H, self.hd).reshape(B, T, -1)
            self._oin.pop(l, None)
            return (at.o_proj(torch.where(m, alt, ob)),) + tuple(base[1:])
        return fwd


def head_masks(dense, rows, T: int) -> dict:
    """``dense``: bool [B, nL, H] (head (l, h) of batch row b is masked); returns {layer: bool [B, T, H]} with the mask
    set in ``rows`` (a list, or slice(None) for every row), for the layers where any entry is set."""
    dense = torch.as_tensor(dense, dtype=torch.bool)
    B, _, H = dense.shape
    out = {}
    for l in dense.any(2).any(0).nonzero().flatten().tolist():
        m = torch.zeros(B, T, H, dtype=torch.bool)
        m[:, rows] = dense[:, l, None, :]
        out[l] = m
    return out


def cells_dense(sets, nL: int, H: int) -> torch.Tensor:
    """bool [len(sets), nL, H] from one list of (layer, head) per batch row."""
    d = torch.zeros(len(sets), nL, H, dtype=torch.bool)
    for b, cells in enumerate(sets):
        if len(cells):
            idx = torch.as_tensor(list(cells))
            d[b, idx[:, 0], idx[:, 1]] = True
    return d


def mean_table(mu_rows, rows, T: int) -> torch.Tensor:
    """[1, T, H, hd] with the per-row means ``mu_rows`` [len(rows), H, hd] at ``rows`` (zeros elsewhere, never used)."""
    t = torch.zeros(1, T, *mu_rows.shape[1:], dtype=mu_rows.dtype)
    t[0, rows] = mu_rows
    return t


class HopSplice:
    def __init__(self, model):
        _once(model, "_ckeys_hopsplice", self)
        self.model, self.active, self.rows, self.tabs, self.which, self.mask = model, False, None, {}, {}, None
        self._pass = "A"
        for l, blk in enumerate(blocks(model)):
            at = blk.self_attn
            at.k_proj.register_forward_hook(self._hook(l, "k"))
            at.v_proj.register_forward_hook(self._hook(l, "v"))
            at.forward = self._wrap(at.forward)

    def _hook(self, l, ch):
        def hk(_m, _i, out):
            if not self.active:
                return out
            t, use = self.tabs.get(self._pass, {}).get((l, ch)), self.which.get(self._pass, {}).get(ch)
            if t is None or use is None:
                return out
            B = out.shape[0]
            use = _bcast(torch.as_tensor(use, dtype=torch.bool).view(-1), B).to(out.device)
            if not bool(use.any()):
                return out
            out = out.clone()
            out[:, self.rows] = torch.where(use[:, None, None], _bcast(t, B).to(out.device, out.dtype), out[:, self.rows])
            return out
        return hk

    def _wrap(self, f):
        def fwd(*a, **kw):
            self._pass = "A"
            if not self.active or self.mask is None or not bool(self.mask.any()):
                return f(*a, **kw)
            ra = f(*a, **kw)
            self._pass = "B"
            try:
                rb = f(*a, **kw)
            finally:
                self._pass = "A"
            B, T = ra[0].shape[:2]
            m = _bcast(self.mask.to(ra[0].device), B)
            assert m.shape[1] == T, (m.shape, T)
            return (torch.where(m[..., None], rb[0], ra[0]),) + tuple(ra[1:])
        return fwd
