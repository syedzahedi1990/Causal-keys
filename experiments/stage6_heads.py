"""Stage 6, part (a) of P-2026-10-05-H (docs/PREREGISTRATION.md): which heads read the writing token's key, are they
canonical duplicate-token heads, what does ablating them do to ID_K / ID_V, and does the answer position read the
re-mentioned words directly (second hop).

Per model and arm (P1 = OPTIONS-AFTER, confirmatory; POST = SENTENCE-AFTER, reported); eager attention, use_cache=False
in every pass, m = log p(S) - log p(B) at the last position, G = the six re-mention rows (order of LOCATIONS), p = the
writing token:
  phase 1, ranking set R = make_cores(n_rank, Random(0)): source run (K at p), clean base run with attentions and the
    o_proj inputs at G, full K_S clamp with attentions -> a3, T_dup, T_ctrl, mu, activity norms; the single-head add
    grid (row h: only (l, h) sees K_S in G; in-batch none row) and the leave-one-out grid (row h: all heads but (l, h);
    in-batch all_G row). Then, before any evaluation pass: the rankings a3 / f+ (mean d+) / d- (mean d-),
    k* = ceil(kstar_frac x n_heads), KS_eff = sorted(set(KS) | {k*}), n_rand random permutations (numpy seed 2; their
    first k heads are the random sets), the active-at-G control (top-k* by mean o_proj-input norm at G outside the top-2k*
    by a3), the next-k* a3 set, mu[l][j, h] = mean over R of head h's o_proj input at the row of word j.
  phase 2, evaluation set E = make_cores(n_eval, Random(1)): the three passes; per ranking and random permutation a
    sufficiency batch (top-k sees K_S in G for k in KS_eff, none, all_G, all_T = every head in every row) and a knockout
    batch (all heads but the top-k see K_S in G, none, all_G); the layer profile; the 13-row ID_K/ID_V batch of
    format_factorial.run_item with its three clean passes under each ablation condition (HeadSplice "ablate" at G in
    every row); the second hop: an in-batch capture pass, the 8-row HopSplice batch (HOP_ROWS) and exploratory batches.
  duplicate (D), induction (I), previous-token (P) scores on n_seq random sequences of `half` ids repeated once.
Output <out>/<model>.json (atomic write; rewritten after each arm):
  provenance: commit, args, model, revision, dtype, attn_implementation, device, versions, n_layers, n_heads,
    n_kv_heads, head_dim, kstar, KS_eff, n_rand, hop_rows, ablation conditions, dup token pool, test_mode, timings;
  dup: {D, I, P: [nL][H] means, D_seq, I_seq: [n_seq][nL][H]};
  arms[arm]: rank: [per R story: core, p, G, T, mB, mF, a3, tdup, n_dup, tctrl, n_ctrl, act, dplus, dminus, none_grid,
    allG_loo]; rankings {a3, fplus, dminus: [[l, h], ...] all heads}; sets {rand: [perm...], active_kstar, next_kstar};
    eval: [per E story: core, p, G, T, mB, mF, a3, tdup, n_dup, tctrl, n_ctrl, curves {set: {suff: [m per KS_eff], none, allG, allT, ko: [...],
    ko_none, ko_allG}}, layer {m, none, allG}, ablation {cond: {idK, idV, dK, dV, dKV, mass, argmax_cand, base_ok,
    m_id, m_cleanB}}, hop {m: [8], explo: {name: {rows, m}}}].
TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B-Instruct FP32 on the CPU with n_rank = n_eval = 2, grid layers 4,8,12,
KS 1,2,4,8, k* fraction 0.012 (k* = 5, KS_eff = 1,2,4,5,8), one random set, 5 sequences.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import experiments.format_factorial as ff
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import LISTING, ROOM, candidate_ids, raw_prompt
from ckeys.headsplice import HeadSplice, HopSplice, cells_dense, head_masks, mean_table
from ckeys.interventions import blocks, capture, edits
from ckeys.story import LOCATIONS, make_cores, record
from experiments.ioi_factorial import device_name
from experiments.row_restricted_keys import encode_with_offsets, rows_in

KS_DEFAULT = "1,2,3,5,8,12,16,20,24,32,40,48,64,96,128"
ARM_SPAN = {"P1": LISTING, "POST": ROOM}
# second-hop rows: (K_S at p, pass-A table and channels, pass-B table and channels, rows reading pass B)
HOP_ROWS = {"ID": (False, None, None, None), "K_S": (True, None, None, None),
            "ans_K": (True, None, ("base", "k"), "ans"), "ans_V": (True, None, ("base", "v"), "ans"),
            "ans_KV": (True, None, ("base", "kv"), "ans"), "all_KV": (True, None, ("base", "kv"), "all"),
            "other_KV": (True, ("ksrun", "kv"), ("base", "kv"), "others"), "exact": (True, None, ("ksrun", "kv"), "ans")}
HOP_EXPLO = {"other_K": (True, ("ksrun", "k"), ("base", "k"), "others"), "other_V": (True, ("ksrun", "v"), ("base", "v"), "others"),
             "all_K": (True, None, ("base", "k"), "all"), "all_V": (True, None, ("base", "v"), "all")}


def configure_hop(hop, names, kv, rows, T, spec=None):
    """Point HopSplice at the batch whose rows are ``names`` (keys of ``spec``, default HOP_ROWS) over the K/V rows
    ``rows``; ``kv[src]`` = {(l, ch): [len(rows), D]} for src in base / ksrun. Returns the per-row K_S-at-p flags."""
    spec = spec or HOP_ROWS
    S = [spec[n] for n in names]
    B, any_t = len(S), next(iter(kv["base"].values()))
    hop.rows, hop.tabs, hop.which = list(rows), {}, {}
    for i, P in ((1, "A"), (2, "B")):
        hop.which[P] = {ch: torch.tensor([s[i] is not None and ch in s[i][1] for s in S]) for ch in "kv"}
        hop.tabs[P] = {key: torch.stack([kv[s[i][0] if s[i] else "base"][key] for s in S]) for key in kv["base"]}
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b, s in enumerate(S):
        if s[3] == "ans":
            mask[b, T - 1] = True
        elif s[3] == "all":
            mask[b] = True
        elif s[3] == "others":
            mask[b, :T - 1] = True
    hop.mask = mask
    assert any_t.shape[0] == len(rows)
    return torch.tensor([s[0] for s in S])


class Stage6:
    def __init__(self, model, tok, arm, hs, hop):
        self.model, self.tok, self.arm, self.hs, self.hop = model, tok, arm, hs, hop
        self.dev = next(model.parameters()).device
        self.nL, self.H, self.hd = hs.nL, hs.H, hs.hd
        self.cid = candidate_ids(tok, arm)
        self.allcells = [(l, h) for l in range(self.nL) for h in range(self.H)]

    def lp(self, ids, **kw):
        out = self.model(ids, use_cache=False, logits_to_keep=1, **kw)
        return torch.log_softmax(out.logits[:, -1].float(), -1), out

    def m(self, lp, d):
        return (lp[:, d["iS"]] - lp[:, d["iB"]]).cpu().double().numpy()

    def mb(self, d, n):
        return self.m(self.lp(d["ib"].expand(n, -1))[0], d)

    def prep(self, core):
        rb, rs = record(core, "direct", core["base"]), record(core, "direct", core["source"])
        tb, ib, off = encode_with_offsets(self.tok, raw_prompt(self.arm, rb["story"], rb["query"]))
        _, is_, _ = encode_with_offsets(self.tok, raw_prompt(self.arm, rs["story"], rs["query"]))
        assert ib.shape == is_.shape, ("B and S encodings differ in length", core)
        diff = (ib[0] != is_[0]).nonzero().flatten().tolist()
        assert len(diff) == 1, (diff, core)
        p, seg = diff[0], ARM_SPAN[self.arm]
        assert tb.count(seg) == 1
        c0 = tb.find(seg)
        G = [i for i in rows_in(off, c0, c0 + len(seg)) if ib[0, i].item() in set(self.cid)]
        assert len(G) == 6 and all(g > p for g in G) and ib[0, G].tolist() == self.cid, (G, p)
        ids = ib[0].tolist()
        assert ids[p] == self.cid[LOCATIONS.index(core["base"])]
        sep = [i for i in range(G[0] + 1, G[-1]) if i not in G]
        return dict(core=core, ib=ib.to(self.dev), is_=is_.to(self.dev), ids=ids, p=p, G=G, sep=sep, T=ib.shape[1],
                    iS=self.cid[LOCATIONS.index(core["source"])], iB=self.cid[LOCATIONS.index(core["base"])],
                    rowS=G[LOCATIONS.index(core["source"])], rowB=G[LOCATIONS.index(core["base"])])

    def ks_of(self, d):
        self.hs.active = self.hop.active = False
        with capture(self.model, range(self.nL), "k") as K:
            self.lp(d["is_"])
        return {l: K[l][0, d["p"]].clone() for l in range(self.nL)}

    def base_runs(self, d, ks, phase1=False):
        """Clean base run and full K_S clamp, both with attentions: m_B, m_full, a3 [nL, H], T_dup / T_ctrl sums [nL, H]
        with their word counts; in phase 1 also the o_proj inputs at G [nL, 6, H, hd] and their norms."""
        p, G, ids, nL = d["p"], d["G"], d["ids"], self.nL
        store, hk = {}, []
        if phase1:
            hk = [blocks(self.model)[l].self_attn.o_proj.register_forward_pre_hook(
                lambda _m, a, l=l: store.__setitem__(l, a[0][0, G].float().cpu())) for l in range(nL)]
        try:
            lpb, ob = self.lp(d["ib"], output_attentions=True)
        finally:
            for h in hk:
                h.remove()
        with edits(self.model, [(l, "k", [p], (lambda h, l=l: ks[l].expand_as(h))) for l in range(nL)]):
            lpf, of = self.lp(d["ib"], output_attentions=True)
        assert ob.attentions is not None and ob.attentions[0] is not None, "attentions missing: load with attn_implementation='eager'"
        Ab = torch.stack([a[0].float() for a in ob.attentions]).cpu()        # [nL, H, T, T]
        Af = torch.stack([a[0].float() for a in of.attentions]).cpu()
        rS, rB = d["rowS"], d["rowB"]
        a3 = 0.5 * ((Af[:, :, rS, p] - Ab[:, :, rS, p]) + (Ab[:, :, rB, p] - Af[:, :, rB, p]))
        out = dict(mB=float(self.m(lpb, d)[0]), mF=float(self.m(lpf, d)[0]), a3=a3.numpy())
        core, cset = d["core"], set(self.cid)   # task-side duplicate scores, in both phases (H4 (ii) is scored on E)
        dup_words = sorted({core["initial"], core["distractor_location"]} - {core["base"]})
        ctrl_words = [w for w in LOCATIONS if w not in {core["initial"], core["distractor_location"], core["base"]}]
        locs = [t for t in range(G[0]) if t != p and ids[t] in cset]
        tdup, nd, tctrl, nc = torch.zeros(nL, self.H), 0, torch.zeros(nL, self.H), 0
        for w in dup_words:
            j = LOCATIONS.index(w)
            occ = [t for t in range(G[0]) if t != p and ids[t] == self.cid[j]]
            assert occ, (w, core)
            tdup += Ab[:, :, G[j], occ].sum(-1)
            nd += 1
        for w in ctrl_words:
            tctrl += Ab[:, :, G[LOCATIONS.index(w)], locs].sum(-1)
            nc += 1
        out |= dict(tdup=tdup.numpy(), n_dup=nd, tctrl=tctrl.numpy(), n_ctrl=nc)
        if phase1:
            oin = torch.stack([store[l].view(6, self.H, self.hd) for l in range(nL)])
            out |= dict(oin=oin, act=oin.norm(dim=-1).mean(1).numpy())
        return out

    def splice(self, d, ks, dense, extra_all=()):
        """m per batch row with the heads of ``dense`` [B, nL, H] seeing K_S in rows G; rows in ``extra_all``: every head
        in every row."""
        hs = self.hs
        masks = head_masks(dense, d["G"], d["T"])
        for b in extra_all:
            for l in range(self.nL):
                masks.setdefault(l, torch.zeros(dense.shape[0], d["T"], self.H, dtype=torch.bool))[b] = True
        hs.ks, hs.pos, hs.mode, hs.masks, hs.active = ks, d["p"], "splice", masks, True
        try:
            return self.mb(d, dense.shape[0])
        finally:
            hs.active, hs.masks = False, None

    def grids(self, d, ks, layers, loo):
        nL, H = self.nL, self.H
        dplus, dminus, none, allg = np.full((nL, H), np.nan), np.full((nL, H), np.nan), [], []
        ar = torch.arange(H)
        for l in layers:
            dense = torch.zeros(H + 1, nL, H, dtype=torch.bool)
            dense[ar, l, ar] = True
            mm = self.splice(d, ks, dense)
            dplus[l] = mm[:H] - mm[H]
            none.append(float(mm[H]))
            if loo:
                dense = torch.ones(H + 1, nL, H, dtype=torch.bool)
                dense[ar, l, ar] = False
                mm = self.splice(d, ks, dense)
                dminus[l] = mm[H] - mm[:H]
                allg.append(float(mm[H]))
        return dplus, dminus, none, allg

    def curves(self, d, ks, sets, KS):
        out, nL, H = {}, self.nL, self.H
        for name, rk in sets.items():
            n = len(KS)
            dense = cells_dense([rk[:k] for k in KS] + [[], self.allcells, []], nL, H)
            ms = self.splice(d, ks, dense, extra_all=(n + 2,))
            dense = ~cells_dense([rk[:k] for k in KS] + [self.allcells, []], nL, H)
            mk = self.splice(d, ks, dense)
            out[name] = dict(suff=ms[:n].tolist(), none=float(ms[n]), allG=float(ms[n + 1]), allT=float(ms[n + 2]),
                             ko=mk[:n].tolist(), ko_none=float(mk[n]), ko_allG=float(mk[n + 1]))
        dense = torch.zeros(nL + 2, nL, H, dtype=torch.bool)
        dense[torch.arange(nL), torch.arange(nL)] = True
        dense[nL + 1] = True
        ml = self.splice(d, ks, dense)
        return out, dict(m=ml[:nL].tolist(), none=float(ml[nL]), allG=float(ml[nL + 1]))

    def ablation(self, d, conds, MU):
        """conds: name -> (cells, "mean" | "zero"); run_item's 13-row batch and its three clean passes, each with the
        listed heads' o_proj-input slices at rows G replaced by mu (or 0); masks and mu have batch dimension 1."""
        hs, res = self.hs, {}
        mut = {l: mean_table(MU[l], d["G"], d["T"]) for l in range(self.nL)}
        for name, (cells, how) in conds.items():
            if cells:
                hs.mode, hs.masks = "ablate", head_masks(cells_dense([cells], self.nL, self.H), d["G"], d["T"])
                hs.mu, hs.active = (mut if how == "mean" else None), True
            try:
                r = ff.run_item(self.model, self.tok, d["core"], self.arm, "direct", self.dev)
            finally:
                hs.active, hs.masks, hs.mu = False, None, None
            assert r is not None and r["pos"] == d["p"] and r["len"] == d["T"], name
            m, idr = r["m"], r["m"]["ID@0"]
            dl = {k: {t: m[k]["lp"][t] - idr["lp"][t] for t in ("S", "B", "X")} for k in m}
            cB = r["clean"]["B"]
            res[name] = dict(idK=0.5 * ((dl["K_S@0"]["S"] - dl["K_X@0"]["S"]) + (dl["K_X@0"]["X"] - dl["K_S@0"]["X"])),
                             idV=0.5 * ((dl["V_S@0"]["S"] - dl["V_X@0"]["S"]) + (dl["V_X@0"]["X"] - dl["V_S@0"]["X"])),
                             dK=m["K_S@0"]["m"] - idr["m"], dV=m["V_S@0"]["m"] - idr["m"], dKV=m["KV_S@0"]["m"] - idr["m"],
                             mass=cB["mass"], argmax_cand=cB["argmax_cand"], base_ok=cB["argmax_cand"] == d["core"]["base"],
                             m_id=idr["m"], m_cleanB=cB["m"])
        return res

    def hop_batch(self, d, ks, names, kv, rows, spec=None):
        kp = configure_hop(self.hop, names, kv, rows, d["T"], spec)
        tab = {(l, "k"): ks[l][None] for l in range(self.nL)}
        self.hop.active = True
        try:
            with clamp_kv(self.model, [d["p"]], tab, range(self.nL), "k", per_row=kp):
                return self.mb(d, len(names))
        finally:
            self.hop.active = False

    def hop_cache(self, d, ks, rows, B=len(HOP_ROWS)):
        """K/V at ``rows`` captured in-batch (B rows, K_S at p in rows 1..B-1): {"base": row 0, "ksrun": row 1}."""
        self.hs.active = self.hop.active = False
        kp = torch.tensor([False] + [True] * (B - 1))
        with clamp_kv(self.model, [d["p"]], {(l, "k"): ks[l][None] for l in range(self.nL)}, range(self.nL), "k", per_row=kp):
            with capture_kv(self.model, rows, range(self.nL)) as C:
                self.mb(d, B)
        return {"base": {k: v[0] for k, v in C.items()}, "ksrun": {k: v[1] for k, v in C.items()}}

    def second_hop(self, d, ks):
        names = list(HOP_ROWS)
        kv = self.hop_cache(d, ks, d["G"])
        out = dict(m=self.hop_batch(d, ks, names, kv, d["G"]).tolist(), explo={})
        en = ["ID", "K_S"] + list(HOP_EXPLO)
        out["explo"]["G"] = dict(rows=d["G"], names=en, m=self.hop_batch(d, ks, en, kv, d["G"], HOP_ROWS | HOP_EXPLO).tolist())
        for r, rows in (("rowS", [d["rowS"]]), ("rowB", [d["rowB"]]), ("sep", d["sep"])):
            kr = self.hop_cache(d, ks, rows)
            rn = ["ID", "K_S", "ans_KV", "all_KV", "other_KV"]
            out["explo"][r] = dict(rows=rows, names=rn, m=self.hop_batch(d, ks, rn, kr, rows).tolist())
        return out


def token_pool(tok, model_type):
    """Random-sequence ids: the base vocabulary without special/added/control ids (Mistral v0.3: ids >= 1027, past the
    [control_N] block, 10-770, and the byte block <0x00>-<0xFF>, 771-1026)."""
    bad = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}) or {})
    lo = 1027 if model_type == "mistral" else 0
    return np.array([i for i in range(lo, tok.vocab_size) if i not in bad])


def duplicate_scores(model, tok, n_seq, half, seed=0):
    nL, H = len(blocks(model)), model.config.num_attention_heads
    pool = token_pool(tok, model.config.model_type)
    rng, dev = np.random.default_rng(seed), next(model.parameters()).device
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    b, Ds, Is, Ps = len(bos), [], [], []
    for _ in range(n_seq):
        x = rng.choice(pool, half, replace=False)
        o = model(torch.tensor(bos + list(x) + list(x))[None].to(dev), use_cache=False, logits_to_keep=1, output_attentions=True)
        A = torch.stack([a[0].float() for a in o.attentions]).cpu()        # [nL, H, T, T]
        q, k = b + half + torch.arange(half), b + torch.arange(half)
        Ds.append(A[:, :, q, k].mean(-1).numpy())
        Is.append(A[:, :, q[:-1], k[1:]].mean(-1).numpy())
        j = torch.arange(b + 1, b + 2 * half)
        Ps.append(A[:, :, j, j - 1].mean(-1).numpy())
    prov = dict(pool_size=int(len(pool)), pool_min=int(pool.min()), pool_max=int(pool.max()), bos=bool(bos), half=half, n_seq=n_seq, seed=seed)
    D, I, P = (np.mean(x, 0) for x in (Ds, Is, Ps))
    return dict(D=D.tolist(), I=I.tolist(), P=P.tolist(), D_seq=np.array(Ds).tolist(), I_seq=np.array(Is).tolist()), prov


def rank_of(score):
    s = np.nan_to_num(np.asarray(score, float), nan=-np.inf)
    return [list(map(int, np.unravel_index(i, s.shape))) for i in np.argsort(-s.ravel(), kind="stable")]


def run_arm(S, a, cores_rank, cores_eval, layers, KS, kstar, log):
    nL, H, t0 = S.nL, S.H, time.time()
    R = {"rank": [], "eval": []}
    oins = []
    for core in cores_rank:
        d = S.prep(core)
        ks = S.ks_of(d)
        b = S.base_runs(d, ks, phase1=True)
        dplus, dminus, none, allg = S.grids(d, ks, layers, not a.no_loo)
        oins.append(b.pop("oin"))
        R["rank"].append(dict(core=core, p=d["p"], G=d["G"], T=d["T"], **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items()},
                              dplus=dplus.tolist(), dminus=dminus.tolist(), none_grid=none, allG_loo=allg))
    log(f"[{S.arm}] phase 1: {len(cores_rank)} ranking stories ({time.time() - t0:.0f}s)")

    def mean(k):
        x = np.array([r[k] for r in R["rank"]], float)
        ok = ~np.isnan(x).all(0)
        return np.where(ok, np.nanmean(np.where(ok, x, 0.0), 0), np.nan)

    A3, ACT = mean("a3"), mean("act")
    rk = {"a3": rank_of(A3), "fplus": rank_of(mean("dplus"))}
    if not a.no_loo:
        rk["dminus"] = rank_of(mean("dminus"))
    rng = np.random.default_rng(2)
    rand = [[S.allcells[i] for i in rng.permutation(len(S.allcells))] for _ in range(a.n_rand)]
    top2 = {tuple(c) for c in rk["a3"][:2 * kstar]}
    active = [c for c in rank_of(ACT) if tuple(c) not in top2][:kstar]
    MU = {l: torch.stack([o[l] for o in oins]).mean(0) for l in range(nL)}
    R["rankings"], R["sets"] = rk, dict(rand=[[list(c) for c in r] for r in rand], active_kstar=active, next_kstar=rk["a3"][kstar:2 * kstar])
    sets = dict(rk) | {f"rand{i}": r for i, r in enumerate(rand)}
    conds = {"none": ([], "mean"), "top_kstar_mean": (rk["a3"][:kstar], "mean"), "top_10_mean": (rk["a3"][:10], "mean"),
             "top_20_mean": (rk["a3"][:20], "mean"), "top_kstar_zero": (rk["a3"][:kstar], "zero")}
    conds |= {f"rand{i}_kstar_mean": (r[:kstar], "mean") for i, r in enumerate(rand)}
    conds |= {"active_kstar_mean": (active, "mean"), "next_kstar_mean": (rk["a3"][kstar:2 * kstar], "mean")}
    assert kstar in KS
    for core in cores_eval:
        d = S.prep(core)
        ks = S.ks_of(d)
        b = S.base_runs(d, ks)
        cv, layer = S.curves(d, ks, sets, KS)
        R["eval"].append(dict(core=core, p=d["p"], G=d["G"], T=d["T"], mB=b["mB"], mF=b["mF"], a3=b["a3"].tolist(), curves=cv,
                              tdup=b["tdup"].tolist(), n_dup=b["n_dup"], tctrl=b["tctrl"].tolist(), n_ctrl=b["n_ctrl"],
                              layer=layer, ablation=S.ablation(d, conds, MU), hop=S.second_hop(d, ks)))
    log(f"[{S.arm}] phase 2: {len(cores_eval)} evaluation stories ({time.time() - t0:.0f}s)")
    return R, list(conds)


def write_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--arms", default="P1,POST")
    ap.add_argument("--n-rank", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--grid-layers", default="all", help="'all' or a comma list")
    ap.add_argument("--no-loo", action="store_true", help="skip the leave-one-out grid (no d- ranking)")
    ap.add_argument("--ks", default=KS_DEFAULT)
    ap.add_argument("--kstar-frac", type=float, default=0.05)
    ap.add_argument("--n-rand", type=int, default=3)
    ap.add_argument("--n-seq", type=int, default=100)
    ap.add_argument("--half", type=int, default=30)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/gpu_stage6/heads")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B FP32 on the CPU, reduced sizes")
    a = ap.parse_args(argv)
    test = a.test or bool(os.environ.get("TEST_MODE"))
    if test:
        a.model, a.dtype, a.n_rank, a.n_eval, a.grid_layers = "Qwen/Qwen2.5-0.5B-Instruct", "float32", 2, 2, "4,8,12"
        a.ks, a.kstar_frac, a.n_rand, a.n_seq = "1,2,4,8", 0.012, 1, 5
    a.test = test
    dev = "cuda" if torch.cuda.is_available() and not test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), revision=a.revision,
                                                 attn_implementation="eager").eval().to(dev)
    assert model.config._attn_implementation == "eager"
    hs, hop = HeadSplice(model), HopSplice(model)
    nL, H = hs.nL, hs.H
    n_heads = nL * H
    layers = list(range(nL)) if a.grid_layers == "all" else [int(x) for x in a.grid_layers.split(",")]
    kstar = math.ceil(a.kstar_frac * n_heads)
    KS_eff = sorted({int(k) for k in a.ks.split(",") if int(k) <= n_heads} | {kstar})
    assert kstar in KS_eff
    cores_rank, cores_eval = make_cores(a.n_rank, random.Random(0)), make_cores(a.n_eval, random.Random(1))
    log = lambda s: print(s, flush=True)  # noqa: E731
    t0 = time.time()
    dup, dprov = duplicate_scores(model, tok, a.n_seq, a.half)
    log(f"duplicate scores: {a.n_seq} sequences ({time.time() - t0:.0f}s)")
    prov = ff.provenance(a) | {"model": a.model, "revision": a.revision, "dtype": str(next(model.parameters()).dtype),
                               "attn_implementation": model.config._attn_implementation, "device": device_name(dev), "test_mode": test,
                               "n_layers": nL, "n_heads": n_heads, "heads_per_layer": H, "n_kv_heads": model.config.num_key_value_heads,
                               "head_dim": hs.hd, "kstar": kstar, "KS_eff": KS_eff, "n_rand": a.n_rand, "grid_layers": layers,
                               "loo": not a.no_loo, "hop_rows": list(HOP_ROWS), "dup": dprov, "timings": {}}
    out = {"provenance": prov, "dup": dup, "arms": {}}
    path = Path(a.out) / f"{a.model.split('/')[-1]}.json"
    for arm in a.arms.split(","):
        R, conds = run_arm(Stage6(model, tok, arm, hs, hop), a, cores_rank, cores_eval, layers, KS_eff, kstar, log)
        out["arms"][arm] = R
        prov["conditions"], prov["timings"][arm] = conds, round(time.time() - t0, 1)
        write_atomic(out, path)
    log(f"wrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
