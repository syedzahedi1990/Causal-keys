"""Stage 7 (P-2026-10-08-I) hooks for the q/k/v/o_proj layout of Qwen2/Mistral/Llama (eager or sdpa; use_cache=False):
the head-restricted reader knockout, the per-row key/value pin used for the remap exchanges, and mask helpers.

ReaderKO: in layer l and batch row b, the query rows t with ``masks[l][b, t, h]`` of head h get attention weight exactly 0
on the column ``cols[b]`` (the writing token p), the row renormalised over its remaining columns; every other head, row
and edge is unchanged. Implemented like ckeys.headsplice.HeadSplice: ``self_attn.forward`` is wrapped; a layer whose mask
is non-empty runs twice on the same hidden states, the second time with ``attention_mask`` replaced by the knockout mask
(the pairs rows(b) x {cols[b]} forbidden per batch row, rows(b) = the rows with any head set); the o_proj input slice of
head h in row t of batch row b comes from the second pass where the mask is set, and o_proj runs once on the spliced
tensor. Head h's slice depends on column p only through head h's own attention, so this is "head h cannot attend from
the masked rows to p". The incoming mask must be an explicit 4D ``[B, 1, T, T]`` plain causal mask (boolean for sdpa,
additive for eager; ckeys.knockout.mask_for_model(model, T, [], B)): the wrapper asserts it, which guards padding and a
forgotten explicit mask (transformers passes a user 4D mask through unchanged). ``null=True``: the second pass uses the
incoming mask unchanged (the knockout floor). Forbidden rows must lie after the column (positions <= p then compute
identically), except with ``allow_before=True``, where every forbidden edge must already be causally masked (the
LIST-BEFORE structural check: the call then leaves the mask unchanged). Column 0 and the diagonal are never forbidden.

KeyPin: a forward hook on k_proj (``ch="k"``) or v_proj (``ch="v"``) of each layer in ``layers``, registered with
prepend=True so that it runs before every hook registered earlier on that module (HeadSplice's and RowSplice's own k-hooks
among them, which therefore override it in their second pass). While active it writes ``tab[l][b]`` (kv_dim) at position
``pos[b]`` of every batch row with ``use[b]``; other rows keep their own key/value. The stage-7 code uses one instance per
channel both for the key/value-only exchanges (P + K_M, M + V_P, ...) and for the B_x key pin (the host's own captured key
at p in every pass, so that heads outside the HeadSplice mask see exactly K_P).

The wrappers pass through while inactive, so they nest with HeadSplice, HopSplice and RowSplice; one instance per model
(per channel for KeyPin).
"""
from __future__ import annotations

import inspect

import torch

from .interventions import blocks


def _once(model, name, obj):
    assert not getattr(model, name, None), f"{type(obj).__name__} is already installed on this model (one per model)"
    setattr(model, name, True)


def _bcast(x, B):
    assert x.shape[0] in (1, B), f"batch dimension {x.shape[0]} does not match the batch ({B}); only 1 broadcasts"
    return x.expand(B, *x.shape[1:]) if x.shape[0] != B else x


def plain_causal(am: torch.Tensor) -> torch.Tensor:
    """bool [B, T, T]: True where ``am`` (4D, boolean or additive) lets a row attend."""
    return am[:, 0] if am.dtype == torch.bool else am[:, 0] == 0


class ReaderKO:
    def __init__(self, model):
        _once(model, "_ckeys_readerko", self)
        cfg = model.config
        self.model, self.H, self.nL = model, cfg.num_attention_heads, len(blocks(model))
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.active, self.masks, self.cols, self.null, self.allow_before = False, None, None, False, False
        self._cap, self._oin, self.n_double = False, {}, 0
        for l, blk in enumerate(blocks(model)):
            at = blk.self_attn
            at.o_proj.register_forward_pre_hook(self._ohook(l))
            at.forward = self._wrap(at.forward, l, at)

    def _ohook(self, l):
        def hk(_m, args):
            if self._cap:
                self._oin[l] = args[0]
        return hk

    def _kmask(self, am, M):
        """The knockout mask of this call: ``am`` with rows(b) x {cols[b]} forbidden per batch row."""
        B, T = M.shape[:2]
        cols = _bcast(torch.as_tensor(self.cols).view(-1), B).tolist()
        rows = M.any(-1).cpu()                         # [B, T]
        ko = am.clone()
        off = False if am.dtype == torch.bool else torch.finfo(am.dtype).min
        allowed = plain_causal(am)
        for b in range(B):
            rb = rows[b].nonzero().flatten().tolist()
            if not rb:
                continue
            c = int(cols[b])
            assert c != 0 and c not in rb, ("column 0 and the diagonal are never forbidden", c, rb)
            if self.allow_before:
                assert not bool(allowed[b, rb, c].any()), "allow_before: a forbidden edge is not causally masked already"
            else:
                assert min(rb) > c, ("knocked rows must lie after the column", min(rb), c)
            ko[b, 0, rb, c] = off
        return ko

    def _wrap(self, f, l, at):
        sig = inspect.signature(type(at).forward)   # the attention class's own signature (f may be another wrapper)

        def fwd(*a, **kw):
            M = self.masks.get(l) if self.active and self.masks else None
            if M is None or not bool(M.any()):
                return f(*a, **kw)
            ba = sig.bind(at, *a, **kw)
            am, hsx = ba.arguments.get("attention_mask"), ba.arguments.get("hidden_states")
            assert am is not None and am.dim() == 4, "ReaderKO needs an explicit 4D attention mask (mask_for_model(model, T, [], B))"
            B, T = hsx.shape[:2]
            assert am.shape == (B, 1, T, T), (tuple(am.shape), B, T)
            tril = torch.ones(T, T, dtype=torch.bool, device=am.device).tril()
            assert bool((plain_causal(am) == tril).all()), "the incoming attention mask is not the plain causal mask on every row"
            M = _bcast(M.to(hsx.device), B)
            assert M.shape[1:] == (T, self.H), (M.shape, T, self.H)
            self._cap = True
            try:
                base = f(*a, **kw)
                ob = self._oin.pop(l)
                ba.arguments["attention_mask"] = am if self.null else self._kmask(am, M)
                f(*ba.args[1:], **ba.kwargs)
                alt = self._oin.pop(l)
            finally:
                self._cap = False
            self.n_double += 1
            m = M[..., None].expand(B, T, self.H, self.hd).reshape(B, T, -1)
            return (at.o_proj(torch.where(m, alt, ob)),) + tuple(base[1:])
        return fwd


class KeyPin:
    def __init__(self, model, layers, ch="k"):
        assert ch in ("k", "v"), ch
        _once(model, f"_ckeys_keypin_{ch}", self)
        self.model, self.ch, self.layers = model, ch, list(layers)
        self.active, self.tab, self.use, self.pos, self.n_writes = False, None, None, None, 0
        for l in self.layers:
            mod = getattr(blocks(model)[l].self_attn, f"{ch}_proj")
            mod.register_forward_hook(self._hook(l), prepend=True)

    def _hook(self, l):
        def hk(_m, _i, out):
            if not self.active or self.tab is None or l not in self.tab:
                return out
            B = out.shape[0]
            use = _bcast(torch.as_tensor(self.use, dtype=torch.bool).view(-1), B).to(out.device)
            if not bool(use.any()):
                return out
            pos = _bcast(torch.as_tensor(self.pos, dtype=torch.long).view(-1), B).to(out.device)
            ar = torch.arange(B, device=out.device)
            out = out.clone()
            t = _bcast(self.tab[l], B).to(out.device, out.dtype)
            out[ar, pos] = torch.where(use[:, None], t, out[ar, pos])
            self.n_writes += 1
            return out
        return hk


def layer_cells(l: int, H: int) -> list:
    """The H (layer, head) cells of layer l."""
    return [(l, h) for h in range(H)]


def blind_masks(specs, G, T: int, nL: int, H: int, onset: int) -> dict:
    """HeadSplice masks for the B_x batches: ``specs`` = one (cells, outside) per batch row, ``cells`` the (layer, head)
    cells set in the rows G and ``outside`` True when every head is set in every row outside G; only layers >= onset
    carry a mask. Returns {layer: bool [B, T, H]} for the layers where any entry is set."""
    B = len(specs)
    inG = torch.zeros(B, nL, H, dtype=torch.bool)
    out = torch.zeros(B, dtype=torch.bool)
    for b, (cells, outside) in enumerate(specs):
        if len(cells):
            idx = torch.as_tensor(list(cells))
            inG[b, idx[:, 0], idx[:, 1]] = True
        out[b] = bool(outside)
    notG = torch.ones(T, dtype=torch.bool)
    notG[list(G)] = False
    res = {}
    for l in range(onset, nL):
        if not (inG[:, l].any() or out.any()):
            continue
        m = torch.zeros(B, T, H, dtype=torch.bool)
        m[:, list(G)] = inG[:, l, None, :]
        m[out[:, None] & notG[None, :]] = True
        res[l] = m
    return res


def cell_masks(sets, G, T: int, nL: int, H: int) -> dict:
    """{layer: bool [len(sets), T, H]} with the cells of ``sets[b]`` set in the rows G of batch row b (A, A+ and N)."""
    d = torch.zeros(len(sets), nL, H, dtype=torch.bool)
    for b, cells in enumerate(sets):
        if len(cells):
            idx = torch.as_tensor(list(cells))
            d[b, idx[:, 0], idx[:, 1]] = True
    res = {}
    for l in d.any(2).any(0).nonzero().flatten().tolist():
        m = torch.zeros(len(sets), T, H, dtype=torch.bool)
        m[:, list(G)] = d[:, l, None, :]
        res[l] = m
    return res
