"""E4 of stage 8, part C (preregistration J): a rank-16 DAS remap at the writing token p, trained with the recipe released
by Anonymous (2026) at a new position (the single token p, not the event span), at new depths and in new models.

Recipe: the pair-swap objective (base B', source S', target pi(S') = story.PAIR_SWAP[S']); a RotatedSubspace U of rank 16
(ckeys.interventions, orthonormal rows) at the output of 0-based block l at p, interchange h <- h_B' + U^T U (h_S' - h_B');
the loss is the six-way cross-entropy over the candidate scores, a candidate's score being the logsumexp of the logits of
its single-token forms " x" and " X" (a form that is not one token is left out; the softmax normaliser cancels in the
six-way cross-entropy); AdamW (lr 1e-3, weight decay 0, gradient-norm clip 1.0 as ckeys.das.train_das), batch 1, one
epoch over the 1,000 training pairs in the order random.Random(seed).shuffle gives. Format NO-MENTION with the "Answer:"
prefill (the released recipe used the reply start; a disclosed deviation). Initialisation: seed 101 the PCA basis of
h_S',l(p) - h_B',l(p) over the training pairs (ckeys.interventions.pca_basis, centred), seed 102 a random orthonormal
basis (QR of a Gaussian [D, 16], Generator seed 102). The training items are ckeys.das.Pair records.
"""
from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from .das import Pair
from .interventions import RotatedSubspace, pca_basis
from .edits import write_resid
from .story import LOCATIONS

RANK, LR, CLIP = 16, 1e-3, 1.0
SEED_INIT = {101: "pca", 102: "random"}


def form_ids(tok) -> list[list[int]]:
    """Per candidate, the ids of its single-token forms " x" and " X" (at least " x", asserted)."""
    out = []
    for w in LOCATIONS:
        ids = []
        for f in (" " + w, " " + w.capitalize()):
            t = tok(f, add_special_tokens=False).input_ids
            if len(t) == 1 and t[0] not in ids:
                ids.append(t[0])
        assert ids and tok(" " + w, add_special_tokens=False).input_ids[0] == ids[0], w
        out.append(ids)
    return out


def cand_scores(logits: torch.Tensor, forms) -> torch.Tensor:
    """logits [..., V] -> [..., 6] logsumexp over each candidate's forms."""
    return torch.stack([torch.logsumexp(logits[..., ids].float(), -1) for ids in forms], -1)


def init_basis(kind: str, diffs: torch.Tensor | None, D: int, seed: int) -> torch.Tensor:
    if kind == "pca":
        return pca_basis(diffs, RANK).float()
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(D, RANK, generator=g))[0].T.contiguous()


def patched_scores(model, pair: Pair, layer: int, h_base, h_src, U, forms):
    """Six candidate scores at the last position with h_base + U^T U (h_src - h_base) written at (p, layer)."""
    dev = next(model.parameters()).device
    h = h_base.float() + (h_src.float() - h_base.float()) @ U.T @ U
    with write_resid(model, layer, pair.base_pos[0], h[None].to(dev)):
        lg = model(pair.base_ids.to(dev), use_cache=False, logits_to_keep=1).logits[0, -1]
    return cand_scores(lg, forms)


def train_remap_at(model, pairs, h_base, h_src, layer: int, forms, seed: int, init: str | None = None, log=print,
                   log_every: int = 100):
    """Fit U on ``pairs`` (Pair: base_ids [1, T] NO-MENTION prompt of B', base_pos [p], target = index of pi(S')) with
    their captured residuals h_base[i], h_src[i] ([D], block-l output at p). Returns (U [16, D] float32 on the CPU with
    U U^T = I asserted, losses per step)."""
    init = init or SEED_INIT[seed]
    D = model.config.hidden_size
    dev = next(model.parameters()).device
    for prm in model.parameters():
        prm.requires_grad_(False)
    assert init != "pca" or len(pairs) >= RANK, f"a PCA initialisation needs at least {RANK} training pairs"
    diffs = torch.stack([(s - b).float() for b, s in zip(h_base, h_src)]) if init == "pca" else None
    U0 = init_basis(init, diffs, D, seed)
    torch.manual_seed(seed)
    sub = RotatedSubspace(D, RANK, init=U0).to(dev)
    opt = torch.optim.AdamW(sub.parameters(), lr=LR, weight_decay=0.0)
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    losses, t0 = [], time.time()
    with torch.enable_grad():
        for step, i in enumerate(order):
            pr = pairs[i]
            sc = patched_scores(model, pr, layer, h_base[i].to(dev), h_src[i].to(dev), sub.U.float(), forms)
            loss = F.cross_entropy(sc[None], torch.tensor([pr.target], device=dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sub.parameters(), CLIP)
            opt.step()
            losses.append(float(loss.detach()))
            if log_every and (step % log_every == 0 or step == len(order) - 1):
                log(f"    das l={layer} seed={seed} step {step:4d} loss {losses[-1]:.4f} ({time.time() - t0:.0f}s)")
    U = sub.U.detach().float().cpu()
    assert torch.allclose(U @ U.T, torch.eye(RANK), atol=1e-4), "the fitted basis is not orthonormal"
    for prm in sub.parameters():
        prm.requires_grad_(False)
    return U, losses


@torch.no_grad()
def flip_rate(model, pairs, h_base, h_src, layer: int, U, forms) -> float:
    """The NO-MENTION flip rate on held-out pairs: the share whose six-way argmax under the remap is pi(S')."""
    dev = next(model.parameters()).device
    hits = 0
    for i, pr in enumerate(pairs):
        sc = patched_scores(model, pr, layer, h_base[i].to(dev), h_src[i].to(dev), U.to(dev), forms)
        hits += int(sc.argmax()) == pr.target
    return hits / max(1, len(pairs))
