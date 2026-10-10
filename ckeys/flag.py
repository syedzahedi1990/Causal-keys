"""Stage 8, part D of P-2026-10-10-J (docs/PREREGISTRATION.md): the identity flag that the reader heads write.

Hooks (q/k/v/o_proj layout of Qwen2/Mistral/Llama; o_proj has no bias there; eager or sdpa; use_cache=False):
  OCap    o_proj forward-pre-hook: while active, records the o_proj input (the concatenated head outputs z, [B, |rows|,
          H*hd], FP32 copy) at ``rows`` in each layer of ``layers``.
  Inject  o_proj forward hook, two modes that may be combined:
            add[l]  = (b_idx [K], t_idx [K], V [K, D]): y_l[b, t] += V (additive injection of vectors at chosen batch
                      rows and token rows; repeated (b, t) accumulate);
            proj[l] = (u [D] unit, mu: float, [B] or [1 or B, T], rowmask bool [1 or B, T]): directional mean-ablation
                      y_l[b, t] <- y_l[b, t] - (<y_l[b, t], u> - mu) u at the masked rows.
          Both are computed in FP32 on a copy of the o_proj output and cast back to the model dtype (rows not touched
          round-trip exactly). Sequences longer than the maps (the trie nodes of ckeys.surface.score) are untouched past
          the mask's T. Passes through while inactive; one instance per model.
Composition. HeadSplice ("splice" and "ablate") and HopSplice call o_proj inside their wrapped attention forward; the
injection therefore applies to every o_proj output those wrappers return, exactly once per returned row: HeadSplice
discards the outputs of its inner calls and returns o_proj(spliced input) + add; HopSplice takes each output row from one
of its two passes, both with the add. tests/test_flag.py checks this with nonzero adds (Gate J-D-G0).

Flag algebra. For a story s and a layer l of L* (the layers of the reader set H*), with z^B and z^K the o_proj inputs of
the clean base run and of the run with the key at p from the source run (K_S, every layer), and W_O^{l,h} the o_proj
columns of head h:
  delta_l(s) = 1/2 sum_{h in H*_l} W_O^{l,h} ([z^K_h(r_S) - z^B_h(r_S)] + [z^B_h(r_B) - z^K_h(r_B)])
(the readers' matched-minus-unmatched write at the option rows of S and B), Delta_l = mean_s delta_l(s) (``flag_of``).
The same formula with other runs or rows gives the other flags of the part (natural S run: Delta^KV; POST rows; the
initial-state sentences; IOI). Controls: isotropic and head-span random vectors, a layer derangement, the story's own
write orthogonal to the flag, a direction rescaled to the flag's per-layer norm (``norm_match``).
"""
from __future__ import annotations

import hashlib
import random as _random

import numpy as np
import torch

from .interventions import blocks


def _once(model, name, obj):
    assert not getattr(model, name, None), f"{type(obj).__name__} is already installed on this model (one per model)"
    setattr(model, name, True)


def dims(model):
    cfg = model.config
    H = cfg.num_attention_heads
    return len(blocks(model)), H, getattr(cfg, "head_dim", None) or cfg.hidden_size // H, cfg.hidden_size


class OCap:
    def __init__(self, model):
        _once(model, "_ckeys_ocap", self)
        self.model, self.active, self.rows, self.layers, self.store = model, False, None, None, {}
        for l, blk in enumerate(blocks(model)):
            blk.self_attn.o_proj.register_forward_pre_hook(self._h(l))

    def _h(self, l):
        def hk(_m, args):
            if self.active and (self.layers is None or l in self.layers):
                self.store[l] = args[0][:, self.rows].detach().float().clone()
        return hk

    def run(self, fn, rows, layers=None):
        """Call ``fn()`` with the capture on; returns ({l: [B, |rows|, H*hd] float32 (device of the model)}, fn's value)."""
        self.store, self.rows, self.layers, self.active = {}, list(rows), None if layers is None else set(layers), True
        try:
            r = fn()
        finally:
            self.active = False
        out, self.store = self.store, {}
        return out, r


class Inject:
    def __init__(self, model):
        _once(model, "_ckeys_inject", self)
        self.model, self.active, self.add, self.proj, self.n_calls = model, False, {}, {}, 0
        for l, blk in enumerate(blocks(model)):
            blk.self_attn.o_proj.register_forward_hook(self._h(l))

    def _h(self, l):
        def hk(_m, _i, out):
            if not self.active or (l not in self.add and l not in self.proj):
                return out
            self.n_calls += 1
            y = out.float()          # a copy (out is bf16 on the GPU); FP32 already: clone below
            if y.data_ptr() == out.data_ptr():
                y = y.clone()
            if l in self.add:
                b, t, V = self.add[l]
                y.index_put_((b.to(y.device), t.to(y.device)), V.to(y.device, torch.float32), accumulate=True)
            if l in self.proj:
                u, mu, rm = self.proj[l]
                u = u.to(y.device, torch.float32)
                B, T = y.shape[:2]
                rm = rm.to(y.device).bool()
                rm = rm.expand(B, -1) if rm.shape[0] == 1 else rm
                assert rm.shape[0] == B and rm.shape[1] <= T, (tuple(rm.shape), B, T)
                m = torch.zeros(B, T, dtype=torch.bool, device=y.device)
                m[:, :rm.shape[1]] = rm
                c = y @ u                                          # [B, T]
                mu_t = torch.as_tensor(mu, dtype=torch.float32, device=y.device)
                if mu_t.dim() == 1:                                # one value per batch row
                    mu_t = mu_t.view(-1, 1)
                elif mu_t.dim() == 2 and mu_t.shape[1] < T:         # [1 or B, T'] per token row, padded past T'
                    mu_t = torch.cat([mu_t, mu_t.new_zeros(mu_t.shape[0], T - mu_t.shape[1])], 1)
                y = torch.where(m[..., None], y - (c - mu_t)[..., None] * u, y)
            return y.to(out.dtype)
        return hk

    def clear(self):
        self.active, self.add, self.proj = False, {}, {}


def add_map(spec, layers, scale=1.0) -> dict:
    """``spec``: one list per batch row of (token row, {l: vector [D]}, coefficient); returns Inject.add for ``layers``
    (layers a vector dict lacks are skipped for that entry)."""
    out = {}
    for l in layers:
        b, t, V = [], [], []
        for bi, items in enumerate(spec):
            for row, vec, c in items:
                if vec is None or l not in vec:
                    continue
                b.append(bi)
                t.append(row)
                V.append(c * scale * vec[l].float())
        if b:
            out[l] = (torch.tensor(b), torch.tensor(t), torch.stack(V))
    return out


# --------------------------------------------------------------------------- head outputs and flags
def wo(model, l):
    """o_proj weight of layer l, FP32 [D, H*hd] (no bias in the supported layouts)."""
    o = blocks(model)[l].self_attn.o_proj
    assert getattr(o, "bias", None) is None, "o_proj has a bias: the head decomposition below would omit it"
    return o.weight.float()


def head_out(W, z, heads, hd):
    """Summed residual write of ``heads`` from o_proj inputs z [..., H*hd] with o_proj weight W [D, H*hd] (FP32)."""
    z = z.float()
    W = W.to(z.device)
    idx = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in heads]).to(z.device)
    return z[..., idx] @ W[:, idx].T


def by_layer(cells) -> dict:
    out = {}
    for l, h in cells:
        out.setdefault(int(l), []).append(int(h))
    return {l: sorted(v) for l, v in sorted(out.items())}


def flag_delta(Ws, zk, zb, iS, iB, byL, hd):
    """delta_l for one story: ``zk``/``zb`` {l: [rows, H*hd]} (the matched and the clean run), ``iS``/``iB`` the index of
    the S and B rows in those captures."""
    return {l: 0.5 * (head_out(Ws[l], zk[l][iS] - zb[l][iS], hh, hd) + head_out(Ws[l], zb[l][iB] - zk[l][iB], hh, hd))
            for l, hh in byL.items()}


def flag_of(deltas, keep=None) -> dict:
    """Delta_l = mean over the stories (``keep``: a bool list selecting them) of delta_l(s)."""
    ds = [d for i, d in enumerate(deltas) if keep is None or keep[i]]
    assert ds, "no story to fit the flag on"
    return {l: torch.stack([d[l] for d in ds]).mean(0) for l in ds[0]}


def geometry(deltas, Delta) -> dict:
    """Consistency c_l = mean_s cos(delta_l(s), Delta_l), energy share phi_l = |Delta_l|^2 / mean_s |delta_l(s)|^2, norms
    and the cross-layer cosine matrix of Delta (E1, exploratory)."""
    Ls = sorted(Delta)
    out = {"layers": Ls, "c": {}, "phi": {}, "norm": {}, "mean_story_norm": {}}
    for l in Ls:
        F = torch.stack([d[l] for d in deltas])
        out["c"][l] = float(torch.nn.functional.cosine_similarity(F, Delta[l][None], dim=-1).mean())
        out["phi"][l] = float(Delta[l].norm() ** 2 / (F.norm(dim=-1) ** 2).mean())
        out["norm"][l] = float(Delta[l].norm())
        out["mean_story_norm"][l] = float(F.norm(dim=-1).mean())
    U = torch.stack([Delta[l] / Delta[l].norm() for l in Ls])
    out["cross_cos"] = (U @ U.T).tolist()
    return out


def wcos(A, B) -> float:
    """Norm-weighted mean over the shared layers of cos(A_l, B_l), weights |A_l|."""
    Ls = sorted(set(A) & set(B))
    w = np.array([float(A[l].norm()) for l in Ls])
    c = np.array([float(torch.nn.functional.cosine_similarity(A[l], B[l], dim=0)) for l in Ls])
    return float((w * c).sum() / w.sum())


def identity_share(deltas, labels) -> dict:
    """Per layer, the share of the variance of delta_l(s) around its mean explained by the story's B identity (between-
    group sum of squares / total), against the null (groups - 1) / (n - 1)."""
    lab = np.asarray(labels)
    out = {}
    for l in deltas[0]:
        X = torch.stack([d[l] for d in deltas]).double().numpy()
        tot = ((X - X.mean(0)) ** 2).sum()
        between = sum(((X[lab == g].mean(0) - X.mean(0)) ** 2).sum() * (lab == g).sum() for g in np.unique(lab))
        out[l] = float(between / tot) if tot > 0 else float("nan")
    return {"share": out, "null": (len(np.unique(lab)) - 1) / max(len(lab) - 1, 1)}


def logit_lens(model, Delta, loc_ids, n_rand=500, seed=3) -> dict:
    """max |cos(Delta_l, W_U[location])| over the six location rows against the mean |cos| with random token rows."""
    WU = model.get_output_embeddings().weight
    g = torch.Generator().manual_seed(seed)
    rid = torch.randint(0, WU.shape[0], (n_rand,), generator=g)
    out = {}
    for l, v in Delta.items():
        u = (v / v.norm()).to(WU.device, torch.float32)
        cl = torch.nn.functional.cosine_similarity(WU[loc_ids].float(), u[None], dim=-1).abs().max()
        cr = torch.nn.functional.cosine_similarity(WU[rid.to(WU.device)].float(), u[None], dim=-1).abs().mean()
        out[l] = (float(cl), float(cr))
    return out


# --------------------------------------------------------------------------- controls
def unit(v):
    return v / v.norm()


def norm_match(vecs, ref):
    """Each vecs[l] rescaled to |ref[l]| (layers of ref)."""
    return {l: unit(vecs[l].float()) * ref[l].norm() for l in ref}


def total_match(vecs, ref):
    """vecs rescaled by one factor so that sum_l |vecs_l|^2 = sum_l |ref_l|^2 (vectors on other layers than ref)."""
    a = sum(float(v.norm() ** 2) for v in vecs.values())
    b = sum(float(v.norm() ** 2) for v in ref.values())
    return {l: v.float() * (b / a) ** 0.5 for l, v in vecs.items()}


def iso_random(ref, n, seed):
    """n draws of isotropic Gaussian directions, norm-matched per layer (torch.Generator seed ``seed``)."""
    g = torch.Generator().manual_seed(seed)
    D = next(iter(ref.values())).shape[0]
    return [{l: unit(torch.randn(D, generator=g)) * ref[l].norm() for l in sorted(ref)} for _ in range(n)]


def headspan_random(Ws, byL, hd, ref, n, seed):
    """n draws of sum_{h in H*_l} W_O^{l,h} z_h with z ~ N(0, I), norm-matched per layer: directions the readers could
    write."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        d = {}
        for l in sorted(ref):
            z = torch.randn(len(byL[l]) * hd, generator=g)
            W = Ws[l].cpu()
            idx = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in byL[l]])
            d[l] = unit(W[:, idx] @ z) * ref[l].norm()
        out.append(d)
    return out


def derangement(Ls, seed):
    """A fixed derangement of the layer list (no layer maps to itself), from random.Random(seed)."""
    rng = _random.Random(seed)
    Ls = list(Ls)
    assert len(Ls) >= 2, "a derangement needs two layers"
    while True:
        p = Ls[:]
        rng.shuffle(p)
        if all(a != b for a, b in zip(Ls, p)):
            return dict(zip(Ls, p))


def layer_perm(Delta, seed):
    """Delta of layer pi(l) placed at layer l, rescaled to |Delta_l| (pi a fixed derangement)."""
    pi = derangement(sorted(Delta), seed)
    return {l: unit(Delta[pi[l]].float()) * Delta[l].norm() for l in Delta}, {int(k): int(v) for k, v in pi.items()}


def orth_to(own, Delta, match=True):
    """The story's own write with its component along Delta_l removed; ``match``: rescaled to |Delta_l|."""
    out = {}
    for l in Delta:
        u = unit(Delta[l].float())
        v = own[l].float() - (own[l].float() @ u) * u
        out[l] = unit(v) * Delta[l].norm() if match else v
    return out


def top_pc(Y):
    """Top principal direction (unit) of the centred rows of Y [n, D]."""
    X = Y.double() - Y.double().mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[0].float()


def sha_tensors(d) -> str:
    """sha256 over a nested dict of tensors / lists / numbers in sorted key order (the flag file's content hash)."""
    h = hashlib.sha256()

    def walk(x, pre=""):
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                walk(x[k], f"{pre}/{k}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f"{pre}[{i}]")
        elif isinstance(x, torch.Tensor):
            h.update(pre.encode())
            h.update(x.detach().cpu().contiguous().float().numpy().tobytes())
        else:
            h.update(f"{pre}={x!r}".encode())
    walk(d)
    return h.hexdigest()


# --------------------------------------------------------------------------- head masks with per-row rows
def row_masks(specs, T: int, nL: int, H: int) -> dict:
    """HeadSplice masks from one (cells, rows) per batch row: ``cells`` a list of (l, h) or "all"; ``rows`` a list of
    token rows. Returns {l: bool [B, T, H]} for the layers where any entry is set."""
    B = len(specs)
    out = {}
    for b, (cells, rows) in enumerate(specs):
        if not rows:
            continue
        if cells == "all":
            cl = {l: list(range(H)) for l in range(nL)}
        else:
            cl = by_layer(cells)
        for l, hh in cl.items():
            m = out.setdefault(l, torch.zeros(B, T, H, dtype=torch.bool))
            for r in rows:
                m[b, r, hh] = True
    return out
