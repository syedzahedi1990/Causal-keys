"""Stage 7 of P-2026-10-08-I (docs/PREREGISTRATION.md): find the reader heads at Mistral-Small-24B and block them at the
option words while applying the released remap of our predecessor, Anonymous (2026).

Material: E = the native cores of the predecessor's release (gpu/component_data/native_story_120.json) with B, S, T
distinct (96, in file order); R = make_cores(60, Random(0)) (the ranking stories of stage 6). M = the released
original_1000 m3 basis, P = the pca basis of the same seed (101-103); M and P patch the output of 0-based layer 3
(FIT_LAYER0) over the event span, exactly as experiments/paper1_frames.run_family; the writing token p is the critical
location word; key- or value-only exchanges write the other patched run's K or V at p from 0-based layer 5
(FIRST_EXCHANGE0 = ONSET). Prompts: chat_text(..., prefill="Answer:"). Natural readout m_nat = lp[S] - lp[B]; remap readout
m = lp[T] - lp[S] (letters under LETTER), over the six candidates of the arm, full-vocabulary log-probs at the last
position. G = the six location-word rows of the arm's re-mention span (canonical order): P1 "Choices:" list, LETTER the
location words of the lettered list, POST the room sentence, BEFORE the list before p (structural check), NONE empty.

Stages (each a separate process; BF16, use_cache=False; the pipeline is scripts/gpu_stage7.sh):
  preflight  tokenizer only: the predecessor's release (refit_remap.tolerant_verify, which pins RELEASE.json), the nine
             Mistral bases (BASES_SHA) and the stories file (NATIVE_SHA); the frames tokenizer check (fix_mistral_regex)
             on every E and R prompt; prep() on every R x {P1, LETTER, POST} and E x five arms; R and E disjoint.
             -> OUT/preflight.json
  rank       step 1, phase 1, eager, on R per arm in ARMS_LINK: a3 (H's formula) with the natural K_S clamp from ONSET,
             the layer-0 a3 (exploratory), o_proj-input norms at G; H*_f = top-k* eligible heads (layers >= ONSET), three
             random sets (numpy default_rng(2), shared), the active-at-G set, the next-k* set, the means MU[arm][l] of the
             o_proj inputs at G (l >= L4; OUT/heads/mu.pt, sha256 recorded); duplicate scores (100 sequences).
             -> OUT/heads/rank.json (sets_sha256 = the hash every later file records)
  gate       step 1, phase 2, sdpa, on E: HeadSplice sufficiency / knockout batches (stage 6 layout, masks at layers >=
             ONSET) for H*, the random and active sets; the clean pass and the full K_S clamp from ONSET; the NO-MENTION
             clamp. -> OUT/heads/gate.json
  link       step 2, sdpa, on E, all five arms: natural passes and patches as run_core/run_family; the 6-row capture of
             the exchange tables; the B_x batch (HeadSplice with the key pin), the curve batch, the LIST-BEFORE batch; the
             21-row family batches under the conditions ∅ ("0"), A (HeadSplice "ablate" at G with MU), A+ (A of S plus
             layer L4), N (ReaderKO at G -> p). -> OUT/link/<tag>.json
  remaprank  exploratory, eager, last: a3 for the remap (P_s vs P_s + K_M) on E, its top-k* set H_rem, overlap with H*,
             KO_x(H_rem) with in-batch P, x_all, x_notG rows. -> OUT/heads/remaprank.json
The family run itself is experiments/paper1_frames.py, unchanged (the stage-3b reproduction, Gate I-G1 (a)).

Batch layouts (per arm and core; seeds 101-103):
  B_x (36 rows): per seed [P, M, x_all, x_H, x_rand0, x_rand1, x_rand2, x_active, x_notG, s_H, s_G, rem_H]: hosts P_s
    (M_s for M and rem_H); HeadSplice ks = K_M on a P_s host, K_P on an M_s host; masks at layers >= ONSET: x_S = every
    head in every row except the cells of S in the rows G (x_all: S empty; x_notG: S = all heads), s_S = the cells of S
    in the rows G only; the KeyPin writes the host's own captured key at p from ONSET in every pass.
  curve (18; LETTER 21): per seed [P, x_all, x_top-k for k in KS without k*] (+ x_{H*_P1} under LETTER).
  before (9, BEFORE): per seed [P, x_all, x_{H*_P1}] with G = the list rows before p (I-G1 (d)).
  family (21): per seed [P, M, PK (P + K_M), PV (P + V_M), MK (M + K_P), MV (M + V_P)], then the natural B, S, T; every
    call with an explicit plain causal 4D mask; conditions per arm in the order "0", A:H, A:rand0-2, A:active, A+:H,
    N:H, N:null, then exploratory A:L4, N:rand0, N:allG (every layer and head, the stage-5 M1), A:H_P1 (LETTER only).
    NONE: "0" only; BEFORE: "0" and N:H_P1 at the list rows (allow_before).
Stored per row: candidate log-probs ("cand", six) and the global argmax, keyed f"{cond}/{name}_{seed}" or
f"{cond}/B|S|T" (cond in nat, cap, x, curve, before, 0, A:<set>, A+:<set>, N:<set>).
TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B-Instruct FP32 on the CPU, random bases (width 896, generator seed 0, as
paper1_frames --bases-override), n_rank = n_eval = 2, KS = (1, 2, 4, 5, 8), k* fraction 0.012 (k* = 5 of 336 heads),
5 duplicate sequences; L4 = the 14 heads of layer 4. Outputs carry test_mode and the tag TEST_Qwen2.5-0.5B-Instruct.
A deadline (env STAGE7_DEADLINE, epoch seconds, set by the pipeline) skips the exploratory conditions and remaprank once
passed; what was skipped is recorded in the provenance.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.encoding import LETTER_LISTING, LISTING, ROOM, candidate_ids, chat_text, raw_prompt
from ckeys.headsplice import HeadSplice, mean_table
from ckeys.interventions import blocks, capture, edits
from ckeys.knockout import mask_for_model
from ckeys.readerblind import KeyPin, ReaderKO, blind_masks, cell_masks, layer_cells
from ckeys.story import LOCATIONS, PAIR_SWAP, make_cores, record
from experiments.format_factorial import provenance as ff_provenance
from experiments.ioi_factorial import device_name
from experiments.paper1_frames import (FIRST_EXCHANGE0, FIT_LAYER0, OBJECTIVES, PROFILES, SEEDS, encode_with_span,
                                       fixed_tokenizer, load_bases, lp_rows, sha, tokenizer_check)
from experiments.refit_remap import PINNED, tolerant_verify
from experiments.row_restricted_keys import encode_with_offsets, rows_in
from experiments.stage6_heads import duplicate_scores, rank_of, write_atomic

MODEL = "mistral"
REPO, REV = PROFILES[MODEL]
TINY = "Qwen/Qwen2.5-0.5B-Instruct"
ONSET = FIRST_EXCHANGE0                # 5: exchange onset, natural clamp, eligible heads, B_x and N layers
L4 = FIRST_EXCHANGE0 - 1               # 4: the layer A+ adds (the first layer after the block-4 patch)
assert L4 == FIT_LAYER0 + 1
ARMS_LINK = ("P1", "LETTER", "POST")
ARMS_REF = ("NONE", "BEFORE")
ARMS_ALL = ARMS_LINK + ARMS_REF
SPAN = {"P1": LISTING, "LETTER": LETTER_LISTING, "POST": ROOM, "BEFORE": LISTING, "NONE": None}
KSTAR_FRAC, KS, N_RAND, RAND_SEED = 0.05, (8, 16, 32, 64, 128), 3, 2
TEST_KSTAR_FRAC, TEST_KS = 0.012, (1, 2, 4, 5, 8)
NATIVE = "gpu/component_data/native_story_120.json"
BASES_SHA = {  # pinned to the provenance of results/gpu_stage4/frames/mistral.json
    "m3_101": "6444f90e195a30799b4009f31ccf231e38199e8d45d93986e93cf61ae1944d3b",
    "m3_102": "6f100591fef7ffa10dfb937789c02ca504344913e4fb3bd4314bd1ab1329789b",
    "m3_103": "14625533dfb68aabed9cf8fb5b69217eedafea5127f1ceeda81aac7d25554791",
    "pca_101": "de63064e0cf61b6591f4342942af41f391bc41749c85c2d4067a51611d3fd692",
    "pca_102": "43079cf1e10d6a8d81a10cdc3bd758562752a028f216e0c5ac98fe542fee28d8",
    "pca_103": "16d56f309a4499879742bd663fa02202114b8f9df719fb5d24d2861be64b8242",
    "f_star_101": "b9eb0a4be36e6ac087def95ee5418248a5f5a4a1cfde51c88b06c5d97722ec67",
    "f_star_102": "8481bea608b25e0d5380a3d8dc85c2ff7bff0d2e2044ca827314b09c44794f7b",
    "f_star_103": "a22dec297e3ed77b4d10076a2b212857529cb557502873d1110ee2b52d77bd9a"}
NATIVE_SHA = "22af9f5bcd15e89fa3eb804b246507124a6c12990f0356f3527d6e900f25d931"
# NATIVE_SHA is the sha256 that the release manifest (RELEASE.json) lists for the stories file; RELEASE.json itself is
# pinned by refit_remap.PINNED["RELEASE.json"], which tolerant_verify checks. tests/test_stage7_link.py checks both.
STORY_FIELDS = ("agent", "other", "object", "distractor", "initial", "distractor_location", "base", "source")
BX_NAMES = ("P", "M", "x_all", "x_H", "x_rand0", "x_rand1", "x_rand2", "x_active", "x_notG", "s_H", "s_G", "rem_H")
FAM_NAMES = ("P", "M", "PK", "PV", "MK", "MV")


def log(s):
    print(s, flush=True)


def canon(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def deadline_passed():
    d = os.environ.get("STAGE7_DEADLINE")
    return bool(d) and time.time() > float(d)


# --------------------------------------------------------------------------- material
def native_cores(p1_root):
    cores = json.load(open(Path(p1_root) / NATIVE))["stories"]
    for c in cores:
        assert c["target"] == PAIR_SWAP[c["source"]]
    return [c for c in cores if len({c["base"], c["source"], c["target"]}) == 3]


def random_bases(d):
    """paper1_frames --bases-override: the same generator, seed and order."""
    g = torch.Generator().manual_seed(0)
    return {(o, s): torch.linalg.qr(torch.randn(d, 16, generator=g))[0].T.contiguous() for o in OBJECTIVES for s in SEEDS}


def release_check(p1_root, test):
    """The predecessor's release: manifest and RELEASE.json pin, the nine bases, the stories file."""
    integ = tolerant_verify(p1_root)
    b = {f"{o}_{s}": sha(Path(p1_root) / "gpu/component_data/bases" / f"{MODEL}_original_1000_{o}_{s}.npz") for o in OBJECTIVES for s in SEEDS}
    bad = sorted(k for k in BASES_SHA if b[k] != BASES_SHA[k])
    assert not bad, f"bases differ from the stage-4 provenance: {bad}"
    nat = sha(Path(p1_root) / NATIVE)
    assert nat == NATIVE_SHA, f"stories file sha256 {nat} != {NATIVE_SHA}"
    return {"release_sha256": integ["release_sha256"], "release_pinned": PINNED["RELEASE.json"], "status": integ["status"],
            "bases_sha256": b, "native_sha256": nat, "bases_used": "random (TEST_MODE)" if test else "released original_1000"}


# --------------------------------------------------------------------------- encoding
def encode(tok, arm, core, loc):
    rec = record(core, "direct", loc)
    raw = raw_prompt(arm, rec["story"], rec["query"])
    text = chat_text(tok, raw, prefill="Answer:")
    e = tok(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    return text, raw, e.input_ids, e.offset_mapping[0].tolist()


def prep(tok, core, arm, native):
    """Encodings of B, S (and T for native cores), p, the rows G (canonical order), readout indices. None when the
    encodings differ in length (skipped, as run_core)."""
    locs = {"B": core["base"], "S": core["source"]} | ({"T": core["target"]} if native else {})
    enc = {k: encode(tok, arm, core, v) for k, v in locs.items()}
    if len({e[2].shape for e in enc.values()}) > 1:
        return None
    text, raw, ib, off = enc["B"]
    diff = (ib[0] != enc["S"][2][0]).nonzero().flatten().tolist()
    assert len(diff) == 1, (arm, core, diff)
    p = diff[0]
    d = dict(core=core, arm=arm, p=p, T=ib.shape[1], ids={k: e[2] for k, e in enc.items()})
    if native:
        assert (ib[0] != enc["T"][2][0]).nonzero().flatten().tolist() == [p]
        for k, v in locs.items():
            ids, span, vpos = encode_with_span(tok, arm, core, v)
            assert torch.equal(ids, enc[k][2]) and vpos == p, (arm, k, vpos, p)
        d["span"] = span
    else:
        for k in locs:
            assert torch.equal(encode_with_offsets(tok, enc[k][1])[1], enc[k][2]), (arm, k)
    loc_ids = candidate_ids(tok, "P1")
    seg = SPAN[arm]
    G, letters = [], []
    if seg is not None:
        assert text.count(seg) == 1, (arm, seg)
        c0 = text.index(seg)
        rows = rows_in(off, c0, c0 + len(seg))
        G = [i for i in rows if ib[0, i].item() in set(loc_ids)]
        assert len(G) == 6 and ib[0, G].tolist() == loc_ids, (arm, "location rows of the span", G)
        assert all(g < p for g in G) if arm == "BEFORE" else all(g > p for g in G), (arm, G, p)
        if arm == "LETTER":
            let = candidate_ids(tok, "LETTER")
            letters = [i for i in rows if ib[0, i].item() in set(let)]
            assert len(letters) == 6 and ib[0, letters].tolist() == let, (arm, letters)
    iB, iS = LOCATIONS.index(core["base"]), LOCATIONS.index(core["source"])
    d |= dict(G=G, letters=letters, cid=candidate_ids(tok, arm), iB=iB, iS=iS, init=LOCATIONS.index(core["initial"]),
              rowB=G[iB] if G else None, rowS=G[iS] if G else None)
    if native:
        d["iT"] = LOCATIONS.index(core["target"])
        d["rowT"] = G[d["iT"]] if G else None
    return d


# --------------------------------------------------------------------------- hooks
class RowPatch:
    """Forward hook on decoder block FIT_LAYER0: writes ``tab[b]`` ([span, d]) over the event span [lo, hi) of every
    batch row with ``use[b]`` (natural rows are left alone), as run_family's patch hook does for every row."""

    def __init__(self, model):
        assert not getattr(model, "_ckeys_rowpatch", None), "RowPatch is already installed on this model"
        model._ckeys_rowpatch = True
        self.active, self.tab, self.use, self.lo, self.hi = False, None, None, None, None
        blocks(model)[FIT_LAYER0].register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if not self.active:
            return out
        o = out[0] if isinstance(out, tuple) else out
        o = o.clone()
        use = self.use.to(o.device)
        o[:, self.lo:self.hi] = torch.where(use[:, None, None], self.tab.to(o.device, o.dtype), o[:, self.lo:self.hi])
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o


class Link:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.hs, self.ko = HeadSplice(model), ReaderKO(model)
        self.nL, self.H, self.hd = self.hs.nL, self.hs.H, self.hs.hd
        self.kpin, self.vpin = KeyPin(model, range(ONSET, self.nL), "k"), KeyPin(model, range(ONSET, self.nL), "v")
        self.patch = RowPatch(model)
        self.dev = next(model.parameters()).device
        self.elig = [(l, h) for l in range(ONSET, self.nL) for h in range(self.H)]
        self.allcells = [(l, h) for l in range(self.nL) for h in range(self.H)]

    # ---- one forward with every hook configured, reset afterwards
    @contextlib.contextmanager
    def cfg(self, patch=None, k=None, v=None, hs=None, ko=None):
        try:
            if patch is not None:
                self.patch.tab, self.patch.use, self.patch.lo, self.patch.hi = patch
                self.patch.active = True
            for pin, t in ((self.kpin, k), (self.vpin, v)):
                if t is not None:
                    pin.tab, pin.use, pin.pos = t
                    pin.active = True
            if hs is not None:
                for kk, vv in hs.items():
                    setattr(self.hs, kk, vv)
                self.hs.active = True
            if ko is not None:
                for kk, vv in ko.items():
                    setattr(self.ko, kk, vv)
                self.ko.active = True
            yield
        finally:
            self.patch.active = self.kpin.active = self.vpin.active = self.hs.active = self.ko.active = False
            self.patch.tab = self.kpin.tab = self.vpin.tab = None
            self.hs.masks, self.hs.mu, self.hs.mode = None, None, "splice"
            self.ko.masks, self.ko.null, self.ko.allow_before = None, False, False

    def lp(self, ids, cid, explicit=True, am=None, **kw):
        """Candidate log-probs, global argmax and the model output of one batch; ``explicit``: pass the plain causal 4D
        mask (``am`` overrides it; tests). The last-position logits stay in ``self.last`` (tests)."""
        B, T = ids.shape
        if am is None and explicit:
            am = mask_for_model(self.model, T, [], B)
        out = self.model(ids.to(self.dev), attention_mask=None if am is None else am.to(self.dev), use_cache=False,
                         logits_to_keep=1, **kw)
        self.last = out.logits[:, -1].float()
        lp = torch.log_softmax(self.last, -1)
        return lp[:, cid].cpu(), lp.argmax(-1).cpu(), out

    def m_nat(self, ids, d, **kw):
        c, _, out = self.lp(ids, d["cid"], **kw)
        return (c[:, d["iS"]] - c[:, d["iB"]]).double().numpy(), out

    def ks_of(self, d):
        with capture(self.model, range(self.nL), "k") as K:
            self.lp(d["ids"]["S"], d["cid"], explicit=False)
        return {l: K[l][0, d["p"]].clone() for l in range(self.nL)}

    def clamp_specs(self, d, ks, onset):
        return [(l, "k", [d["p"]], (lambda h, l=l: ks[l].expand_as(h))) for l in range(onset, self.nL)]

    # ---- step 1, phase 1 (eager)
    def rank_core(self, d):
        """One ranking story: the clean base run with attentions and the o_proj inputs at G (layers >= L4), the K_S clamp
        from ONSET and (exploratory) from layer 0 with attentions; a3 as in H (rowS, rowB = the G rows of S and B)."""
        model, nL, H, hd = self.model, self.nL, self.H, self.hd
        p, G = d["p"], d["G"]
        ks = self.ks_of(d)
        store = {}
        hk = [blocks(model)[l].self_attn.o_proj.register_forward_pre_hook(
            lambda _m, x, l=l: store.__setitem__(l, x[0][0, G].float().cpu())) for l in range(L4, nL)]
        try:
            mB, ob = self.m_nat(d["ids"]["B"], d, explicit=False, output_attentions=True)
        finally:
            for h in hk:
                h.remove()
        assert ob.attentions is not None and ob.attentions[0] is not None, "attentions missing: load with attn_implementation='eager'"
        with edits(model, self.clamp_specs(d, ks, ONSET)):
            mF, of = self.m_nat(d["ids"]["B"], d, explicit=False, output_attentions=True)
        with edits(model, self.clamp_specs(d, ks, 0)):
            _, o0 = self.m_nat(d["ids"]["B"], d, explicit=False, output_attentions=True)
        Ab = torch.stack([x[0].float() for x in ob.attentions]).cpu()        # [nL, H, T, T]
        rS, rB = d["rowS"], d["rowB"]

        def a3_of(o):
            Af = torch.stack([x[0].float() for x in o.attentions]).cpu()
            return 0.5 * ((Af[:, :, rS, p] - Ab[:, :, rS, p]) + (Ab[:, :, rB, p] - Af[:, :, rB, p]))
        a3, a30 = a3_of(of), a3_of(o0)
        oin = torch.stack([store[l].view(6, H, hd) for l in range(L4, nL)])      # [nL - L4, 6, H, hd]
        act = np.zeros((nL, H))
        act[L4:] = oin.norm(dim=-1).mean(1).numpy()
        r = dict(core=d["core"], p=p, G=G, T=d["T"], mB=float(mB[0]), mF=float(mF[0]), a3=a3.numpy().tolist(),
                 a3_l0=a30.numpy().tolist(), act=act.tolist(), a3_below_onset_max=float(a3[:ONSET].abs().max()))
        return r, oin

    # ---- material of the remap (link, remaprank)
    def natural(self, d, bases, res=None):
        """run_core's natural passes (B, S, T; one at a time, no explicit mask) and run_family's patches for M_s, P_s."""
        span, h = d["span"], {}
        for name in ("B", "S", "T"):
            with capture(self.model, [FIT_LAYER0], "resid") as st:
                c, g = lp_rows(self.model, d["ids"][name].to(self.dev), d["cid"])
            h[name] = st[FIT_LAYER0][0, span[0]:span[-1] + 1]
            if res is not None:
                res[f"nat/{name}"] = {"cand": c[0].tolist(), "argmax": int(g[0])}
        hb, hs = h["B"], h["S"]
        P = {}
        for obj, tag in (("m3", "M"), ("pca", "P")):
            for s in SEEDS:
                U = bases[(obj, s)].to(hb.device, dtype=hb.dtype)
                P[(tag, s)] = hb + ((hs - hb) @ U.T) @ U
        return P

    def tables(self, d, P, res=None):
        """One capture pass of the six patched runs (M_s, P_s; no explicit mask, as run_family): K and V at p for
        layers ONSET..nL-1. Returns {(tag, s): {"k": {l: [D]}, "v": {l: [D]}}}."""
        keys = [("M", s) for s in SEEDS] + [("P", s) for s in SEEDS]
        span, p = d["span"], d["p"]
        tab = torch.stack([P[k] for k in keys])
        lay = range(ONSET, self.nL)
        with self.cfg(patch=(tab, torch.ones(len(keys), dtype=torch.bool), span[0], span[-1] + 1)), \
                capture(self.model, lay, "k") as K, capture(self.model, lay, "v") as V:
            c, g = lp_rows(self.model, d["ids"]["B"].to(self.dev).expand(len(keys), -1), d["cid"])
        out = {}
        for i, k in enumerate(keys):
            out[k] = {"k": {l: K[l][i, p].clone() for l in lay}, "v": {l: V[l][i, p].clone() for l in lay}}
            if res is not None:
                res[f"cap/{'m3' if k[0] == 'M' else 'pca'}_{k[1]}"] = {"cand": c[i].tolist(), "argmax": int(g[i])}
        return out

    def bx(self, d, P, X, rows, G=None, explicit=True, pin=True, **kw):
        """B_x-type batch: rows = [(name, seed, host 'P'|'M', cells in G, outside)]; HeadSplice ks = the other run's
        key (K_M on a P host, K_P on an M host); the KeyPin writes the host's own key at p from ONSET."""
        G = d["G"] if G is None else G
        span, p, T, B = d["span"], d["p"], d["T"], len(rows)
        other = {"P": "M", "M": "P"}
        lay = range(ONSET, self.nL)
        patch = torch.stack([P[(h, s)] for _, s, h, _, _ in rows])
        own = {l: torch.stack([X[(h, s)]["k"][l] for _, s, h, _, _ in rows]) for l in lay}
        ks = {l: torch.stack([X[(other[h], s)]["k"][l] for _, s, h, _, _ in rows]) for l in lay}
        masks = blind_masks([(c, o) for _, _, _, c, o in rows], G, T, self.nL, self.H, ONSET)
        one = torch.ones(B, dtype=torch.bool)
        with self.cfg(patch=(patch, one, span[0], span[-1] + 1), k=(own, one, torch.full((B,), p)) if pin else None,
                      hs=dict(mode="splice", ks=ks, pos=p, masks=masks, mu=None)):
            c, g, out = self.lp(d["ids"]["B"].expand(B, -1), d["cid"], explicit=explicit, **kw)
        return c, g, out

    def pk_rows(self, d, P, X):
        """Per seed P_s and P_s + K_M (the key-only exchange from ONSET, no splice), with attentions (remaprank)."""
        span, p, lay = d["span"], d["p"], range(ONSET, self.nL)
        rows = [(h, s) for s in SEEDS for h in ("P", "PK")]
        B = len(rows)
        patch = torch.stack([P[("P", s)] for _, s in rows])
        use = torch.tensor([h == "PK" for h, _ in rows])
        tab = {l: torch.stack([X[("M", s)]["k"][l] for _, s in rows]) for l in lay}
        with self.cfg(patch=(patch, torch.ones(B, dtype=torch.bool), span[0], span[-1] + 1), k=(tab, use, torch.full((B,), p))):
            return self.lp(d["ids"]["B"].expand(B, -1), d["cid"], output_attentions=True)[2]

    def family(self, d, P, X, cond=None, sel=None, am=None):
        """The 21-row family batch under one condition: cond None (∅), ("A", cells, mu tables {l: [1 or B, T, H, hd]}),
        ("N", cells, null, allow_before). Rows: per seed P, M, PK, PV, MK, MV; then B, S, T (natural, own ids).
        ``sel`` (tests): run only these row keys, in this order; ``am`` (tests): this 4D mask instead of the plain one."""
        span, p, T = d["span"], d["p"], d["T"]
        lay = range(ONSET, self.nL)
        host = {"P": "P", "M": "M", "PK": "P", "PV": "P", "MK": "M", "MV": "M"}
        src = {"PK": "M", "PV": "M", "MK": "P", "MV": "P"}
        keys = [f"{n}_{s}" for s in SEEDS for n in FAM_NAMES] + ["B", "S", "T"]
        sel = keys if sel is None else list(sel)
        B = len(sel)
        spec = [(k.split("_")[0], int(k.split("_")[1])) if "_" in k else (k, None) for k in sel]
        ref = P[("P", SEEDS[0])]
        patch = torch.zeros(B, *ref.shape, dtype=ref.dtype, device=ref.device)
        for i, (n, s) in enumerate(spec):
            if s is not None:
                patch[i] = P[(host[n], s)]
        hosted = torch.tensor([s is not None for _, s in spec])

        def tab(ch, which):
            use = torch.tensor([n in which for n, _ in spec])
            z = torch.zeros_like(X[("P", SEEDS[0])][ch][ONSET])
            t = {l: torch.stack([X[(src[n], s)][ch][l] if n in which else z for n, s in spec]) for l in lay}
            return (t, use, torch.full((B,), p))
        ids = torch.cat([d["ids"][n if s is None else "B"] for n, s in spec])
        hs = ko = None
        if cond is not None and cond[0] == "A":
            _, cells_, mut = cond
            masks = cell_masks([cells_], d["G"], T, self.nL, self.H)
            hs = dict(mode="ablate", masks=masks, mu={l: mut[l] for l in masks}, ks=None, pos=p)
        elif cond is not None and cond[0] == "N":
            _, cells_, null, before = cond
            ko = dict(masks=cell_masks([cells_], d["G"], T, self.nL, self.H), cols=torch.tensor([p]), null=null, allow_before=before)
        with self.cfg(patch=(patch, hosted, span[0], span[-1] + 1), k=tab("k", ("PK", "MK")), v=tab("v", ("PV", "MV")), hs=hs, ko=ko):
            c, g, _ = self.lp(ids, d["cid"], am=am)
        return {k: {"cand": c[i].tolist(), "argmax": int(g[i])} for i, k in enumerate(sel)}


# --------------------------------------------------------------------------- model
def load_model(a, attn):
    dev = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.repo, revision=a.revision)
    kw = dict(dtype=getattr(torch, a.dtype), revision=a.revision, attn_implementation=attn)
    if dev == "cuda":
        kw["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.repo, **kw).eval()
    if dev == "cpu":
        model = model.to(dev)
    assert model.config._attn_implementation == attn
    return model, tok, dev


def prov(a, model, dev, extra=None):
    cfg = model.config
    H = cfg.num_attention_heads
    nL = cfg.num_hidden_layers
    p = ff_provenance(a) | {"model": a.repo, "revision": a.revision, "dtype": str(next(model.parameters()).dtype),
                            "attn_implementation": cfg._attn_implementation, "device": device_name(dev), "test_mode": a.test,
                            "n_layers": nL, "heads_per_layer": H, "n_heads": nL * H, "n_kv_heads": cfg.num_key_value_heads,
                            "head_dim": getattr(cfg, "head_dim", None) or cfg.hidden_size // H, "onset": ONSET, "L4": L4,
                            "kstar": kstar_of(a, nL * H), "KS": list(a.KS), "n_eligible": (nL - ONSET) * H, "timings": {}}
    return p | (extra or {})


def kstar_of(a, n_heads):
    k = math.ceil(a.kstar_frac * n_heads)
    assert k in a.KS, (k, a.KS)
    return k


def cores_of(a):
    R = make_cores(a.n_rank, random.Random(0))
    E = native_cores(a.p1_root)
    if a.n_eval:
        E = E[: a.n_eval]
    return R, E


def read_rank(out):
    J = json.load(open(Path(out) / "heads" / "rank.json"))
    S = J["sets"]
    assert canon(S) == J["sets_sha256"], "rank.json: sets changed after hashing"
    mu_f = Path(out) / "heads" / J["mu_file"]
    assert sha(mu_f) == J["mu_sha256"], "mu.pt differs from the one rank.json recorded"
    return J, S, torch.load(mu_f)


def cells(x):
    return [tuple(c) for c in x]


# --------------------------------------------------------------------------- stage preflight
def stage_preflight(a):
    t0 = time.time()
    rel = release_check(a.p1_root, a.test)
    log(f"the predecessor's release: {rel['status']}, RELEASE.json {rel['release_sha256'][:12]}, bases and stories file pinned")
    tok = AutoTokenizer.from_pretrained(a.repo, revision=a.revision)
    R, E = cores_of(a)
    strict = not a.test
    tc = tokenizer_check(a.repo, a.revision, tok, list(ARMS_ALL), E, ["Answer:"], strict=strict)
    tok_fix, _ = fixed_tokenizer(a.repo, a.revision, strict)
    n = 0
    for arm in ARMS_LINK:
        for core in R:
            for loc in (core["base"], core["source"]):
                text = encode(tok, arm, core, loc)[0]
                assert tok(text, add_special_tokens=False).input_ids == tok_fix(text, add_special_tokens=False).input_ids, \
                    f"fix_mistral_regex changes the ids (ranking story, {arm}, {loc})"
                n += 1
    tc["ranking_prompts"] = n
    skipped = {"R": 0, "E": 0}
    for arm in ARMS_LINK:
        for core in R:
            skipped["R"] += prep(tok, core, arm, False) is None
    for arm in ARMS_ALL:
        for core in E:
            skipped["E"] += prep(tok, core, arm, True) is None
    key = lambda c: tuple(c[f] for f in STORY_FIELDS)  # noqa: E731
    overlap = len({key(c) for c in R} & {key(c) for c in E})
    assert overlap == 0, f"{overlap} ranking stories equal an evaluation core"
    out = {"provenance": ff_provenance(a) | {"model": a.repo, "revision": a.revision, "test_mode": a.test},
           "release": rel, "tokenizer_check": tc, "n_rank": len(R), "n_eval": len(E), "skipped_items": skipped,
           "overlap_R_E": overlap, "seconds": round(time.time() - t0, 1)}
    write_atomic(out, Path(a.out) / "preflight.json")
    log(f"preflight OK: {len(R)} ranking stories, {len(E)} evaluation cores, skipped {skipped}, {tc['prompts'] + n} prompts checked")


# --------------------------------------------------------------------------- stage rank (eager)
@torch.no_grad()
def stage_rank(a):
    model, tok, dev = load_model(a, "eager")
    L = Link(model, tok)
    nL, H, hd = L.nL, L.H, L.hd
    kstar = kstar_of(a, nL * H)
    R, _ = cores_of(a)
    P = prov(a, model, dev)
    t0 = time.time()
    out = {"provenance": P, "arms": {}}
    MU = {}
    for arm in ARMS_LINK:
        rows, oin_sum = [], None
        for core in R:
            d = prep(tok, core, arm, False)
            assert d is not None, ("ranking story with B/S of different lengths", core)
            r, oin = L.rank_core(d)
            oin_sum = oin if oin_sum is None else oin_sum + oin
            rows.append(r)
        MU[arm] = {l: (oin_sum[l - L4] / len(R)).contiguous() for l in range(L4, nL)}
        A3 = np.mean([r["a3"] for r in rows], 0)
        A30 = np.mean([r["a3_l0"] for r in rows], 0)
        ACT = np.mean([r["act"] for r in rows], 0)       # zero below L4 (not eligible)
        elig = np.full((nL, H), False)
        elig[ONSET:] = True
        rk = [tuple(c) for c in rank_of(np.where(elig, A3, np.nan)) if elig[c[0], c[1]]]
        top2 = set(rk[:2 * kstar])
        active = [tuple(c) for c in rank_of(np.where(elig, ACT, np.nan)) if elig[c[0], c[1]] and tuple(c) not in top2][:kstar]
        out["arms"][arm] = {"rank": rows, "a3_mean": A3.tolist(), "a3_l0_mean": A30.tolist(), "act_mean": ACT.tolist(),
                            "ranking": [list(c) for c in rk], "ranking_l0": rank_of(A30)}
        out["arms"][arm]["_sets"] = dict(H=[list(c) for c in rk[:kstar]], active=[list(c) for c in active],
                                         next=[list(c) for c in rk[kstar:2 * kstar]])
        P["timings"][f"rank_{arm}"] = round(time.time() - t0, 1)
        log(f"[{arm}] ranked on {len(R)} stories ({time.time() - t0:.0f}s); top-5 {rk[:5]}")
    eligible = [(l, h) for l in range(ONSET, nL) for h in range(H)]
    rng = np.random.default_rng(RAND_SEED)
    rand = [[list(eligible[i]) for i in rng.permutation(len(eligible))[:kstar]] for _ in range(N_RAND)]
    sets = {"arms": {arm: out["arms"][arm].pop("_sets") for arm in ARMS_LINK}, "rand": rand,
            "L4": [list(c) for c in layer_cells(L4, H)], "kstar": kstar}
    out["sets"], out["sets_sha256"] = sets, canon(sets)
    mu_f = Path(a.out) / "heads" / "mu.pt"
    mu_f.parent.mkdir(parents=True, exist_ok=True)
    torch.save(MU, mu_f)
    out["mu_file"], out["mu_sha256"] = mu_f.name, sha(mu_f)
    dup, dprov = duplicate_scores(model, tok, a.n_seq, 30)
    out["dup"], P["dup"] = dup, dprov
    P["timings"]["total"] = round(time.time() - t0, 1)
    write_atomic(out, Path(a.out) / "heads" / "rank.json")
    log(f"wrote rank.json, sets {out['sets_sha256'][:12]}, mu {out['mu_sha256'][:12]} ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage gate (sdpa)
@torch.no_grad()
def stage_gate(a):
    J, S, _ = read_rank(a.out)
    model, tok, dev = load_model(a, "sdpa")
    L = Link(model, tok)
    nL, H = L.nL, L.H
    kstar = kstar_of(a, nL * H)
    assert kstar == S["kstar"]
    _, E = cores_of(a)
    P = prov(a, model, dev, {"sets_sha256": J["sets_sha256"], "mu_sha256": J["mu_sha256"]})
    out, t0 = {"provenance": P, "arms": {}}, time.time()
    KS = list(a.KS)
    ik = KS.index(kstar)

    def splice(d, ks, rows):
        masks = blind_masks(rows, d["G"], d["T"], nL, H, ONSET)
        with L.cfg(hs=dict(mode="splice", ks=ks, pos=d["p"], masks=masks, mu=None)):
            return L.m_nat(d["ids"]["B"].expand(len(rows), -1), d, explicit=False)[0]

    for arm in ARMS_LINK:
        Hs = cells(S["arms"][arm]["H"])
        ctrl = {f"rand{i}": cells(r) for i, r in enumerate(S["rand"])} | {"active": cells(S["arms"][arm]["active"])}
        rows_out, skipped = [], 0
        for core in E:
            d = prep(tok, core, arm, True)
            if d is None:
                skipped += 1
                continue
            ks = L.ks_of(d)
            mB = float(L.m_nat(d["ids"]["B"], d, explicit=False)[0][0])
            with edits(model, L.clamp_specs(d, ks, ONSET)):
                mF = float(L.m_nat(d["ids"]["B"], d, explicit=False)[0][0])
            el = set(L.elig)
            ms = splice(d, ks, [(Hs[:k], False) for k in KS] + [([], False), (L.elig, False), (L.elig, True)])
            mk = splice(d, ks, [(sorted(el - set(Hs[:k])), False) for k in KS] + [([], False), (L.elig, False)])
            n = len(KS)
            cur = {"H": dict(suff=ms[:n].tolist(), none=float(ms[n]), allG=float(ms[n + 1]), allT=float(ms[n + 2]),
                             ko=mk[:n].tolist(), ko_none=float(mk[n]), ko_allG=float(mk[n + 1]))}
            for name, C in ctrl.items():
                m6 = splice(d, ks, [(C, False), ([], False), (L.elig, False), (sorted(el - set(C)), False), ([], False), (L.elig, False)])
                cur[name] = dict(suff=[float(m6[0])], none=float(m6[1]), allG=float(m6[2]), ko=[float(m6[3])],
                                 ko_none=float(m6[4]), ko_allG=float(m6[5]))
            rows_out.append(dict(core=core, p=d["p"], G=d["G"], T=d["T"], mB=mB, mF=mF, curves=cur))
        out["arms"][arm] = {"eval": rows_out, "skipped_items": skipped, "k_index": ik}
        P["timings"][arm] = round(time.time() - t0, 1)
        write_atomic(out, Path(a.out) / "heads" / "gate.json")
        log(f"[{arm}] gate on {len(rows_out)} cores ({time.time() - t0:.0f}s)")
    rows_out, skipped = [], 0
    for core in E:   # the natural clamp under NO-MENTION (d_full(NONE))
        d = prep(tok, core, "NONE", True)
        if d is None:
            skipped += 1
            continue
        ks = L.ks_of(d)
        mB = float(L.m_nat(d["ids"]["B"], d, explicit=False)[0][0])
        with edits(model, L.clamp_specs(d, ks, ONSET)):
            mF = float(L.m_nat(d["ids"]["B"], d, explicit=False)[0][0])
        rows_out.append(dict(core=core, p=d["p"], T=d["T"], mB=mB, mF=mF))
    out["none_clamp"] = {"eval": rows_out, "skipped_items": skipped}
    P["timings"]["NONE"] = round(time.time() - t0, 1)
    write_atomic(out, Path(a.out) / "heads" / "gate.json")
    log(f"wrote gate.json ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage link (sdpa)
def bases_of(a, model):
    if a.test:
        return random_bases(model.config.hidden_size), "random (TEST_MODE)"
    return load_bases(a.p1_root, MODEL), "released original_1000"


@torch.no_grad()
def stage_link(a):
    J, S, MU = read_rank(a.out)
    model, tok, dev = load_model(a, "sdpa")
    L = Link(model, tok)
    nL, H = L.nL, L.H
    kstar = kstar_of(a, nL * H)
    assert kstar == S["kstar"]
    rel = release_check(a.p1_root, a.test)
    bases, bsrc = bases_of(a, model)
    _, E = cores_of(a)
    P = prov(a, model, dev, {"sets_sha256": J["sets_sha256"], "mu_sha256": J["mu_sha256"], "release": rel, "bases_used": bsrc,
                             "bases_tensor_sha256": {f"{o}_{s}": hashlib.sha256(bases[(o, s)].numpy().tobytes()).hexdigest() for (o, s) in bases},
                             "explo_skipped": [], "skipped_items": {}})
    path = Path(a.out) / "link" / f"{a.tag}.json"
    out, t0 = {"provenance": P, "arms": {}}, time.time()
    KS = list(a.KS)
    H1 = cells(S["arms"]["P1"]["H"])
    L4c = cells(S["L4"])
    rand = [cells(r) for r in S["rand"]]
    for arm in a.arms.split(","):
        rows_out, skipped = [], 0
        if arm in ARMS_LINK:
            Hs = cells(S["arms"][arm]["H"])
            act = cells(S["arms"][arm]["active"])
            el = set(L.elig)
            comp = lambda C: sorted(el - set(C))  # noqa: E731
        for ci, core in enumerate(E):
            d = prep(tok, core, arm, True)
            if d is None:
                skipped += 1
                continue
            res = {}
            Pt = L.natural(d, bases, res)
            X = L.tables(d, Pt, res)
            rec = dict(core=core, core_index=ci, p=d["p"], G=d["G"], letters=d["letters"], T=d["T"], span=[d["span"][0], d["span"][-1] + 1],
                       iB=d["iB"], iS=d["iS"], iT=d["iT"], init=d["init"], runs=res)
            if arm in ARMS_LINK:
                spec = {"P": ([], False), "M": ([], False), "x_all": (L.elig, True), "x_H": (comp(Hs), True),
                        "x_rand0": (comp(rand[0]), True), "x_rand1": (comp(rand[1]), True), "x_rand2": (comp(rand[2]), True),
                        "x_active": (comp(act), True), "x_notG": ([], True), "s_H": (Hs, False), "s_G": (L.elig, False),
                        "rem_H": (Hs, False)}
                rows = [(n, s, "M" if n in ("M", "rem_H") else "P", *spec[n]) for s in SEEDS for n in BX_NAMES]
                c, g, _ = L.bx(d, Pt, X, rows)
                res |= {f"x/{n}_{s}": {"cand": c[i].tolist(), "argmax": int(g[i])} for i, (n, s, *_) in enumerate(rows)}
                mut = {l: mean_table(MU[arm][l], d["G"], d["T"]).to(dev, next(model.parameters()).dtype) for l in MU[arm]}
                conds = [("0", None), ("A:H", ("A", Hs, mut))] + [(f"A:rand{i}", ("A", r, mut)) for i, r in enumerate(rand)] + \
                        [("A:active", ("A", act, mut)), ("A+:H", ("A", sorted(set(Hs) | set(L4c)), mut)),
                         ("N:H", ("N", Hs, False, False)), ("N:null", ("N", Hs, True, False))]
                explo = [("A:L4", ("A", L4c, mut)), ("N:rand0", ("N", rand[0], False, False)), ("N:allG", ("N", L.allcells, False, False))]
                if arm == "LETTER":
                    explo.append(("A:H_P1", ("A", H1, mut)))
            elif arm == "NONE":
                conds, explo = [("0", None)], []
            else:   # BEFORE: the structural check at the list rows before p
                conds, explo = [("0", None), ("N:H_P1", ("N", H1, False, True))], []
                rows = [(n, s, "P", C, o) for s in SEEDS for n, C, o in (("P", [], False), ("x_all", L.elig, True), ("x_HP1", comp_all(L, H1), True))]
                c, g, _ = L.bx(d, Pt, X, rows)
                res |= {f"before/{n}_{s}": {"cand": c[i].tolist(), "argmax": int(g[i])} for i, (n, s, *_) in enumerate(rows)}
            for name, cond in conds:
                res |= {f"{name}/{k}": v for k, v in L.family(d, Pt, X, cond).items()}
            if arm in ARMS_LINK and not deadline_passed():
                cur = [(n, s, "P", C, o) for s in SEEDS for n, C, o in
                       [("P", [], False), ("x_all", L.elig, True)] + [(f"x_k{k}", comp(Hs[:k]), True) for k in KS if k != kstar]
                       + ([("x_HP1", comp(H1), True)] if arm == "LETTER" else [])]
                c, g, _ = L.bx(d, Pt, X, cur)
                res |= {f"curve/{n}_{s}": {"cand": c[i].tolist(), "argmax": int(g[i])} for i, (n, s, *_) in enumerate(cur)}
            for name, cond in explo:
                if deadline_passed():
                    P["explo_skipped"].append(f"{arm}/{ci}/{name}")
                    continue
                res |= {f"{name}/{k}": v for k, v in L.family(d, Pt, X, cond).items()}
            rows_out.append(rec)
        out["arms"][arm] = rows_out
        P["skipped_items"][arm] = skipped
        P["timings"][arm] = round(time.time() - t0, 1)
        P["n_double_passes"] = {"headsplice": L.hs.n_double, "readerko": L.ko.n_double}
        write_atomic(out, path)
        log(f"[{arm}] link on {len(rows_out)} cores, skipped {skipped} ({time.time() - t0:.0f}s)")
    log(f"wrote {path} ({time.time() - t0:.0f}s)")


def comp_all(L, C):
    return sorted(set(L.elig) - set(C))


# --------------------------------------------------------------------------- stage remaprank (eager, exploratory)
@torch.no_grad()
def stage_remaprank(a):
    J, S, _ = read_rank(a.out)
    path = Path(a.out) / "heads" / "remaprank.json"
    if deadline_passed():
        write_atomic({"provenance": {"skipped": "deadline passed", "sets_sha256": J["sets_sha256"]}, "arms": {}}, path)
        log("remaprank skipped: the pipeline deadline has passed")
        return
    model, tok, dev = load_model(a, "eager")
    L = Link(model, tok)
    nL, H = L.nL, L.H
    kstar = kstar_of(a, nL * H)
    bases, bsrc = bases_of(a, model)
    _, E = cores_of(a)
    P = prov(a, model, dev, {"sets_sha256": J["sets_sha256"], "bases_used": bsrc})
    out, t0 = {"provenance": P, "arms": {}}, time.time()
    elig = np.full((nL, H), False)
    elig[ONSET:] = True
    for arm in ARMS_LINK:
        cache, A = [], []
        for core in E:
            d = prep(tok, core, arm, True)
            if d is None:
                continue
            Pt = L.natural(d, bases)
            X = L.tables(d, Pt)
            o = L.pk_rows(d, Pt, X)                                             # per seed P_s, P_s + K_M (exchange)
            At = torch.stack([x.float() for x in o.attentions], 1).cpu()      # [B, nL, H, T, T]
            p, rT, rS = d["p"], d["rowT"], d["rowS"]
            a3 = [0.5 * ((At[2 * j + 1, :, :, rT, p] - At[2 * j, :, :, rT, p]) + (At[2 * j, :, :, rS, p] - At[2 * j + 1, :, :, rS, p]))
                  for j in range(len(SEEDS))]
            A.append(torch.stack(a3).mean(0).numpy())
            cache.append((d, {k: v.cpu() for k, v in Pt.items()}, {k: {ch: {l: t.cpu() for l, t in X[k][ch].items()} for ch in "kv"} for k in X}))
        A3 = np.mean(A, 0)
        Hrem = [tuple(c) for c in rank_of(np.where(elig, A3, np.nan)) if elig[c[0], c[1]]][:kstar]
        Hs = cells(S["arms"][arm]["H"])
        comp = sorted(set(L.elig) - set(Hrem))
        ko = []
        for d, Pt, X in cache:
            Pt = {k: v.to(dev) for k, v in Pt.items()}
            X = {k: {ch: {l: t.to(dev) for l, t in X[k][ch].items()} for ch in "kv"} for k in X}
            rows = [(n, s, "P", C, o) for s in SEEDS for n, C, o in (("P", [], False), ("x_all", L.elig, True), ("x_notG", [], True), ("x_Hrem", comp, True))]
            c, g, _ = L.bx(d, Pt, X, rows)
            ko.append({"core": d["core"], "iS": d["iS"], "iT": d["iT"],
                       "runs": {f"{n}_{s}": {"cand": c[i].tolist(), "argmax": int(g[i])} for i, (n, s, *_) in enumerate(rows)}})
        ov = len(set(Hrem) & set(Hs))
        out["arms"][arm] = {"a3rem_mean": A3.tolist(), "H_rem": [list(c) for c in Hrem], "overlap_H": ov, "n_eval": len(cache), "ko": ko}
        P["timings"][arm] = round(time.time() - t0, 1)
        write_atomic(out, path)
        log(f"[{arm}] remap ranking on {len(cache)} cores, |H_rem & H*| = {ov} ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "rank", "gate", "link", "remaprank"])
    ap.add_argument("--p1-root", required=True, help="the predecessor's unpacked reviewer repository")
    ap.add_argument("--out", default="results/gpu_stage7")
    ap.add_argument("--revision", default=REV)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-rank", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=0, help="0 = all 96 evaluation cores")
    ap.add_argument("--n-seq", type=int, default=100)
    ap.add_argument("--arms", default=",".join(ARMS_ALL), help="link: the arms to run")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B FP32 on the CPU, random bases, reduced sizes")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    a.repo, a.KS, a.kstar_frac = REPO, KS, KSTAR_FRAC
    if a.test:
        a.repo, a.revision, a.dtype, a.n_rank, a.n_eval, a.n_seq = TINY, None, "float32", 2, 2, 5
        a.KS, a.kstar_frac = TEST_KS, TEST_KSTAR_FRAC
    a.tag = "TEST_" + a.repo.split("/")[-1] if a.test else MODEL
    torch.manual_seed(0)
    {"preflight": stage_preflight, "rank": stage_rank, "gate": stage_gate, "link": stage_link, "remaprank": stage_remaprank}[a.stage](a)


if __name__ == "__main__":
    main()
