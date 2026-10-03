"""Distributed alignment search (DAS) over a candidate-restricted answer, plus evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .interventions import RotatedSubspace, capture, edit


@dataclass
class Pair:
    base_ids: torch.Tensor      # [1, T]
    src_ids: torch.Tensor       # [1, T'] (positions aligned via base_pos/src_pos)
    base_pos: list[int]         # positions patched in the base run
    src_pos: list[int]          # matching positions in the source run
    target: int                 # index into the candidate list


def source_acts(model, pairs: Sequence[Pair], layer: int) -> list[torch.Tensor]:
    out = []
    with torch.no_grad():
        for p in pairs:
            with capture(model, [layer], "resid") as st:
                model(p.src_ids)
            out.append(st[layer][:, p.src_pos].clone())
    return out


def base_acts(model, pairs: Sequence[Pair], layer: int) -> list[torch.Tensor]:
    out = []
    with torch.no_grad():
        for p in pairs:
            with capture(model, [layer], "resid") as st:
                model(p.base_ids)
            out.append(st[layer][:, p.base_pos].clone())
    return out


def patched_logits(model, pair: Pair, layer: int, h_src: torch.Tensor,
                   interchange: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                   cand_ids: torch.Tensor, ids: torch.Tensor | None = None,
                   pos: list[int] | None = None) -> torch.Tensor:
    """Candidate logits at the last token of ``ids`` (default: the pair's base prompt) with the patch applied.

    Passing a different ``ids``/``pos`` reuses the same materialised source activations on a
    follow-up query that shares the patched event span (held-fixed patch evaluation).
    """
    ids = pair.base_ids if ids is None else ids
    pos = pair.base_pos if pos is None else pos
    with edit(model, layer, "resid", pos, lambda h: interchange(h, h_src)):
        lg = model(ids).logits[0, -1]
    return lg[cand_ids]


def train_das(model, pairs: Sequence[Pair], layer: int, rank: int, cand_ids: torch.Tensor,
              steps: int = 300, lr: float = 1e-3, init: torch.Tensor | None = None,
              seed: int = 0, log_every: int = 50, batch: int = 8) -> RotatedSubspace:
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    sub = RotatedSubspace(model.config.hidden_size, rank, init=init)
    opt = torch.optim.AdamW(sub.parameters(), lr=lr, weight_decay=0.0)
    H = source_acts(model, pairs, layer)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, len(pairs), (batch,), generator=g).tolist()
        loss = 0.0
        for i in idx:
            lg = patched_logits(model, pairs[i], layer, H[i], sub.interchange, cand_ids)
            loss = loss + F.cross_entropy(lg[None], torch.tensor([pairs[i].target]))
        loss = loss / batch
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sub.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:4d} loss {loss.item():.4f}", flush=True)
    return sub


@torch.no_grad()
def accuracy(model, pairs: Sequence[Pair], layer: int, interchange, cand_ids: torch.Tensor,
             targets: Sequence[int] | None = None) -> float:
    H = source_acts(model, pairs, layer)
    hits = 0
    for i, p in enumerate(pairs):
        lg = patched_logits(model, p, layer, H[i], interchange, cand_ids)
        t = p.target if targets is None else targets[i]
        hits += int(lg.argmax().item() == t)
    return hits / len(pairs)
