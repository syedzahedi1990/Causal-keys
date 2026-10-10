"""Pilot helpers for part D (identity flag): per-head o_proj-input capture at chosen rows, additive injection at the
o_proj output, directional mean-ablation, encodings of the B/S/X runs with the option rows. CPU FP32 pilots only."""
from __future__ import annotations

import math
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/Causal-keys")
from ckeys.clamp import clamp_kv  # noqa: E402
from ckeys.encoding import LISTING, ROOM, candidate_ids, raw_prompt  # noqa: E402
from ckeys.interventions import blocks, capture  # noqa: E402
from ckeys.story import LOCATIONS, make_cores, pick_x, record, PREFIX  # noqa: E402
from experiments.row_restricted_keys import encode_with_offsets, rows_in  # noqa: E402

SPAN = {"P1": LISTING, "POST": ROOM, "IN": LISTING, "OUT": LISTING}
Q_IN = "Which of the choices is mentioned in the story?"
Q_OUT = "Which of the choices is not mentioned in the story?"


def raw(arm, story, query):
    if arm in ("P1", "POST", "NONE", "BEFORE", "LETTER", "AFTER"):
        return raw_prompt(arm, story, query)
    q = Q_IN if arm == "IN" else Q_OUT
    return PREFIX + story + "\nQuestion: " + q + "\n" + LISTING + "\nAnswer with exactly one choice.\nAnswer:"


def prep(tok, core, arm):
    X = pick_x(core)
    enc = {}
    for k, loc in (("B", core["base"]), ("S", core["source"]), ("X", X)):
        r = record(core, "direct", loc)
        enc[k] = encode_with_offsets(tok, raw(arm, r["story"], r["query"]))
    text, ib, off = enc["B"]
    assert ib.shape == enc["S"][1].shape == enc["X"][1].shape
    diff = (ib[0] != enc["S"][1][0]).nonzero().flatten().tolist()
    assert len(diff) == 1
    p = diff[0]
    cid = candidate_ids(tok, "P1")
    seg = SPAN[arm]
    c0 = text.find(seg)
    G = [i for i in rows_in(off, c0, c0 + len(seg)) if ib[0, i].item() in set(cid)]
    assert len(G) == 6 and all(g > p for g in G), (G, p)
    first = rows_in(off, c0, c0 + len(seg))[0]
    qrow = rows_in(off, text.find("Question"), text.find("Question") + 8)[0]
    iB, iS, iX = LOCATIONS.index(core["base"]), LOCATIONS.index(core["source"]), LOCATIONS.index(X)
    return dict(core=core, X=X, ids={k: v[1] for k, v in enc.items()}, p=p, G=G, T=ib.shape[1], cid=cid,
                iB=iB, iS=iS, iX=iX, iI=LOCATIONS.index(core["initial"]), iD=LOCATIONS.index(core["distractor_location"]),
                rowB=G[iB], rowS=G[iS], rowX=G[iX], first=first, qrow=qrow)


class OCap:
    """Capture o_proj inputs at given rows (per batch row) for all layers."""

    def __init__(self, model):
        self.model, self.active, self.rows, self.store = model, False, None, {}
        for l, blk in enumerate(blocks(model)):
            blk.self_attn.o_proj.register_forward_pre_hook(self._h(l))

    def _h(self, l):
        def hk(_m, args):
            if self.active:
                self.store[l] = args[0][:, self.rows].detach().float().clone()
        return hk


class Inject:
    """Add add[l] ([B, T, D]) to the o_proj output of layer l; or directional mean-ablation at rows."""

    def __init__(self, model):
        self.model, self.active, self.add, self.proj = model, False, {}, {}
        for l, blk in enumerate(blocks(model)):
            blk.self_attn.o_proj.register_forward_hook(self._h(l))

    def _h(self, l):
        def hk(_m, _i, out):
            if not self.active:
                return out
            if l in self.add:
                out = out + self.add[l].to(out.dtype)
            if l in self.proj:   # (u [D] unit, mu scalar, rowmask [B, T] bool)
                u, mu, rm = self.proj[l]
                u = u.to(out.dtype)
                c = out @ u
                out = out - (rm.to(out.dtype) * (c - mu))[..., None] * u
            return out
        return hk


def head_out(model, l, z, heads, hd):
    """Summed residual contribution of ``heads`` (list of h) at layer l from o_proj inputs z [..., H*hd]."""
    W = blocks(model)[l].self_attn.o_proj.weight.float()
    tot = 0
    for h in heads:
        tot = tot + z[..., h * hd:(h + 1) * hd] @ W[:, h * hd:(h + 1) * hd].T
    return tot


def ks_of(model, ids, p):
    nL = len(blocks(model))
    with capture(model, range(nL), "k") as K:
        model(ids, use_cache=False, logits_to_keep=1)
    return {l: K[l][0, p].clone() for l in range(nL)}


def lp_last(model, ids):
    out = model(ids, use_cache=False, logits_to_keep=1)
    return torch.log_softmax(out.logits[:, -1].float(), -1)
