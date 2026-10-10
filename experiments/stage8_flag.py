"""Stage 8, part D of P-2026-10-10-J (docs/PREREGISTRATION.md): what the reader heads write (the identity flag), the sign
of the key read, and the 1.5B / 3B route. One process per stage and model (scripts/gpu_stage8d.sh); BF16, sdpa except the
eager ranking passes of ``sets``; use_cache=False in every pass; every scored quantity is a difference against a row of
the same batch. Hooks: ckeys/flag.py (OCap, Inject), ckeys/headsplice.py (HeadSplice, HopSplice), ckeys/clamp.py.

Populations (compared by full core tuple; U = make_cores(1000, Random(s)), s = 0..3; a stream is make_cores(1, rng)
drawn repeatedly from one Random(seed), which gives the cores of make_cores(n, rng) in order):
  R      make_cores(60, Random(0)): the stage-6 ranking set; flags are fit here (in sample by design; R is in U).
  R'     the cores of R with distractor_location != initial (the initial-state flags of J-D5).
  E8     the first 100 cores of the Random(81) stream not in U nor in EXCL (the supersets of the pilots' and the other
         parts' populations below); A = its first 60 (ablation), its first 60 at Qwen2.5-3B (dissociation).
  BIND   the first 100 cores of the Random(84) stream with distractor_location != initial, not in U, EXCL or E8.
  F_ioi  the first 60 cores of ckeys.ioi.make_cores(90, Random(82)) valid in INLINE for the model's tokenizer.
  E_ioi  the first 100 cores of ckeys.ioi.make_cores(150, Random(83)) valid in every IOI arm the model runs
         (INLINE, INLINE_CHAT; Qwen2.5-7B also AFTER and BEFORE); the IOI candidate lists are disjoint from each other
         and from ckeys.ioi.make_cores(1000, Random(s)), s in {0, 1, 5, 6} (stage 5 and the pilots).
  XFIT / XEVAL  ckeys.tasks paint and schedule cores (exploratory): the first 60 / 40 cores of the Task.make_cores(1, rng)
         streams of Random(85) / Random(86) not in Task.make_cores(1000, Random(s)), s = 0..3, nor in XFIT.
EXCL = make_cores(1000, Random(s)) for the pilot seeds 7, 8, 9, 99, 101, 202, 20261011 and Part B's 20261013, 20261014,
and make_cores(3000, Random(s)) for Part C's 8101-8104 (each a superset of the population drawn from that seed).

Stages (outputs <out>/<stage>/<tag>.json, tag = the model key, TEST_<key> in TEST_MODE; atomic writes; provenance):
  preflight  tokenizer only: populations and hashes, disjointness, every prompt layout of the part (B/S/X/N differ only
             at p; six option rows after p; the initial-state runs differ only at p_init / p_dloc), the IOI validity per
             arm, the K_N words single tokens, the case-marginalised FormSet, the task values.
  sets       eager. qwen7 / mistral7: H* = arms.P1.rankings.a3[:k*] of results/gpu_stage6/heads/<model>.json (file sha256
             and the canonical sets hash asserted), the three random sets (first k* of the stored permutations), the
             active-at-G set; the hop-2 ranking on R under P1 (two eager passes per story: hop(l, h) = 1/2[(A^{K_S} -
             A^B)[END, r_S] + (A^B - A^{K_S})[END, r_B]], top 10) and an in-run a3 (reported overlap with H*).
             qwen1.5 / qwen3b (and every key in TEST_MODE): a3 ranked on R with Stage6.base_runs (phase 1), k* =
             ceil(0.05 x n_heads) (TEST 0.012), random sets from numpy default_rng(2) as stage 6, the active-at-G set.
  fit        on R (P1): per story the batch [B, B + K_S, S] with the o_proj inputs at the six option rows -> delta^P1,
             delta^KV (the natural S run), delta^act (active set), the clean H* output and attention output at G; the
             leave-one-word-out flags Delta^{-w} (stories with B != w and S != w); POST (Delta^POST); on R' the
             initial-state flags Delta^init / Delta^dloc (key at p_init / p_dloc from the run whose word there is I', the
             location absent from {init, dloc, B, S, X}; rows r_I' and r_init / r_dloc); on F_ioi Delta^IOI (INLINE, rows
             of the listed IO_S and IO_B). Controls: isotropic and head-span random draws, a layer derangement, the mean
             H* output direction and the top principal direction of the clean attention output at G, the ablation means
             mu (absent-word option rows). -> <tag>.pt (its sha256 in <tag>.json; every later file records it)
  inject     E8, P1 (J-D1, J-D2, J-D4, J-D-ADDR, J-D-KN; secondary rows). Main batch (INJ_ROWS) and the hop-2 route
             batch [none, move, move+ans_K, move+ans_V, move+ans_KV, move+other_KV] (HopSplice over the option rows).
  ablate     A (first 60 of E8), P1: format_factorial.run_item under the directional ablation A(u, mu) at the option
             rows and layers L* for u in {flag, pc1, meanH, iso0-2}; exploratory: the flag at r_init / r_dloc only and at
             r_B only (deadline-gated).
  bind       BIND, P1, queries direct / other_agent / irrelevant_object: [none, +ev, +init, +dloc, iso at r_X, K_X]
             with the three flags norm-matched per layer (J-D5 and the role-confounded event contrast).
  sign       E8 in Q_IN and Q_OUT: format_factorial.run_item (ID_K, ID_V), the 24-row HeadSplice transfer batch, the
             six-candidate batch [none, none2, K_S, K_X, V_S, V_X, +Delta^P1 at r_X, iso0-2], the per-head hop-2 batch;
             P1: the hop-2 batch; E_ioi in INLINE: ioi_factorial.run_item, transfer, the four-name batch, the IOI route
             batch (K_S and the injected IOI flag) and the hop-2 batch; INLINE_CHAT (and AFTER at qwen7): run_item and
             transfer (reported cells).
  diss       qwen1.5 / qwen3b / qwen7 (D6), E8 (n = 100, 60 at qwen3b), P1 and POST, case-marginalised scoring
             (ckeys.surface FormSet, frames " " and ""): delta^f per story (omega), the injection batch [none, none2,
             +Delta^P1 at r_X, +Delta^POST at r_X, iso, K_X, K_S], competence (clean B and S argmax), the POST route
             batch (HopSplice, answer row and its trie nodes), and at qwen1.5 / qwen3b the R(k*) gate (HeadSplice).
  before     exploratory, qwen7: IOI BEFORE on the first 60 cores of E_ioi: ID_K with and without p's attention to the
             listed names (ckeys.knockout, in the base and in the source runs), then the RowSplice localisation
             (experiments/row_restricted_keys.run with ckeys.ioi.RowTask).
  xtask      exploratory: the belief flag against each task's own flag at the X option row (paint, schedule; P1).
TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B-Instruct FP32 on the CPU, every population cut to n = 2, k* fraction
0.012; the stage-6 head file is still hashed, the sets are ranked in-run.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import experiments.format_factorial as ff
from ckeys import ioi
from ckeys import questions as Q
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import WRAPPER_USED, arm_span, build_prompt, candidate_ids
from ckeys.flag import (Inject, OCap, add_map, by_layer, dims, flag_delta, flag_of, geometry, head_out, headspan_random,
                        identity_share, iso_random, layer_perm, logit_lens, norm_match, orth_to, row_masks, sha_tensors,
                        top_pc, total_match, unit, wcos, wo)
from ckeys.headsplice import HeadSplice, HopSplice
from ckeys.interventions import blocks
from ckeys.knockout import mask_for_model
from ckeys.story import LOCATIONS, make_cores, pick_x, record
from ckeys.surface import FormSet
from ckeys.surface import score as fs_score
from ckeys.tasks import TASKS
from experiments.ioi_factorial import device_name
from experiments.row_restricted_keys import encode_with_offsets, rows_in
from experiments.stage6_heads import Stage6, configure_hop, rank_of, write_atomic

TINY = "Qwen/Qwen2.5-0.5B-Instruct"
KEYS = ("qwen7", "mistral7", "qwen1.5", "qwen3b")
BIG, SMALL, DISS = ("qwen7", "mistral7"), ("qwen1.5", "qwen3b"), ("qwen1.5", "qwen3b", "qwen7")
STAGE6 = {  # model key -> (file under --heads-dir, its sha256, the revision stage 6 ran, k*)
    "qwen7": ("Qwen2.5-7B-Instruct.json", "ed828a9b701d60f552eb8dc10247d85de44364b75f6086e5abe4eedb637353be", "a09a35458c702b33eeacc393d103063234e8bc28", 40),
    "mistral7": ("Mistral-7B-Instruct-v0.3.json", "88ababd91bdb3155ef8e4aa691eff30d9edf88a13260148c145e654ed37e69af", "c170c708c41dac9275d15a8fff4eca08d52bab71", 52)}
SETS_SHA = {"qwen7": "dea841d0d49899bac7ad35c71a851ca7ec5d212d8ecd5310407de54d5c77e72a",
            "mistral7": "55a1b17b1ba16b6afc4e0ae94b15818b8b5c11f02d985deea214aa0d49152220"}
KSTAR_FRAC, TEST_KSTAR_FRAC, N_RAND, RAND_SEED = 0.05, 0.012, 3, 2
N_HOP = 10
SEEDS = {"R": 0, "E8": 81, "F_IOI": 82, "E_IOI": 83, "BIND": 84, "XFIT": 85, "XEVAL": 86}
SIZES = {"R": 60, "E8": 100, "A": 60, "BIND": 100, "F_IOI": 60, "E_IOI": 100, "F_IOI_CAND": 90, "E_IOI_CAND": 150,
         "BEFORE": 60, "XFIT": 60, "XEVAL": 40}
N_DISS = {"qwen1.5": 100, "qwen3b": 60, "qwen7": 100}
ISO_SEED, HSPAN_SEED, PERM_SEED, ABL_ISO_SEED, HOP_RAND_SEED = 11, 12, 13, 14, 15
N_WORDS = ("garage", "kitchen", "pocket", "desk", "hallway", "porch")   # K_N event words: one token, not a candidate
FS_FRAMES = (" ", "")
ALPHAS = (0.5, 1.0, 2.0)
FIELDS = ("agent", "other", "object", "distractor", "initial", "distractor_location", "base", "source")
IOI_FIELDS = ("template", "pattern", "place", "object", "io_b", "io_s", "io_x", "subj", "order")
TFIELDS = ("agent", "object", "distractor", "initial", "distractor_location", "base", "source")
U_SHA = "abd1f0530a3d08a2058743f59102f8dd6dd360af06fc65dd3277c5b5eb176c3d"
EXCL_SEEDS = {7: 1000, 8: 1000, 9: 1000, 99: 1000, 101: 1000, 202: 1000, 20261011: 1000, 20261013: 1000, 20261014: 1000,
              8101: 3000, 8102: 3000, 8103: 3000, 8104: 3000}
IOI_EXCL_SEEDS = (0, 1, 5, 6)
POP_SHA = {  # sha256(json.dumps([[*tuple] for each core in order])) of the full populations (pinned at the build)
    "R": "9036af1a838a12d58a7a7eb40f70dd800dd659bfd6e95ed1ab5a2ce10862f936",
    "R'": "fe348ba40a19182a67cee382a3533c1af009273acb740c01d418662a570122a2",
    "E8": "2c198705c595d8068467a188d46e5c22793e079f837ba39bc1e0b43fb14b0f13",
    "BIND": "495c14620cd22e38af0ac67c376441442c40264996db98872d2bae4009a7ee92",
    "F_IOI_CAND": "d41e791a42cc3a833b32220c54f6aa5b129bfda79e904fb8bf5c382ebf0f8b9a",
    "E_IOI_CAND": "2b165365fc60e69af9bbe0d943bb9ea7fc30ec65a2747cd6f04630e58ddd0334",
    "XFIT_paint": "9b9a320f4a58309b388517d63b9d31b6ba58d29546f13e73bdc8258c01cf2189",
    "XEVAL_paint": "be234aaf7083406fa87ee16d896feae675ff473b461694673a9027cffb36970c",
    "XFIT_schedule": "4a56ae029499e3d93b7adcb375febca35394c7f234fc459d32af7336041433c4",
    "XEVAL_schedule": "8b4c7c05fdbe82e9cf5b947024e8dc4a904d07b0f6d955e0e5bf155752396edc"}
INJ_ROWS = ("none", "none2", "K_S", "K_X", "K_N", "add0.5", "add1", "add2", "move", "moveS", "subB", "looS", "looX",
            "kvS", "kvX", "iso0", "iso1", "iso2", "hspan0", "hspan1", "hspan2", "perm", "orth", "meanH", "active", "own",
            "choices", "question", "postflag")
ROUTE_ROWS = ("none", "move", "ans_K", "ans_V", "ans_KV", "other_KV")
ROUTE_SPEC = {"none": (False, None, None, None), "move": (False, None, None, None),
              "ans_K": (False, None, ("base", "k"), "ans"), "ans_V": (False, None, ("base", "v"), "ans"),
              "ans_KV": (False, None, ("base", "kv"), "ans"), "other_KV": (False, ("ksrun", "kv"), ("base", "kv"), "others")}
IOI_ROUTE = ("none", "K_S", "K_S+ans_K", "K_S+ans_V", "K_S+ans_KV", "K_S+other_KV", "inj", "inj+ans_K", "inj+ans_V",
             "inj+ans_KV", "inj+other_KV")
IOI_ROUTE_SPEC = {"none": (False, None, None, None), "K_S": (True, None, None, None),
                  "K_S+ans_K": (True, None, ("base", "k"), "ans"), "K_S+ans_V": (True, None, ("base", "v"), "ans"),
                  "K_S+ans_KV": (True, None, ("base", "kv"), "ans"), "K_S+other_KV": (True, ("ksrun", "kv"), ("base", "kv"), "others"),
                  "inj": (False, None, None, None), "inj+ans_K": (False, None, ("base", "k"), "ans"),
                  "inj+ans_V": (False, None, ("base", "v"), "ans"), "inj+ans_KV": (False, None, ("base", "kv"), "ans"),
                  "inj+other_KV": (False, ("injrun", "kv"), ("base", "kv"), "others")}
TRANSFER_SETS = ("none", "H", "rand0", "rand1", "rand2", "allG", "koH", "korand0", "korand1", "korand2", "allGc", "allT")
HOP2_ROWS = ("none", "inj", "top10", "rand10", "all")
DISS_ROWS = ("none", "none2", "injP1", "injPOST", "iso", "K_X", "K_S")
DISS_ROUTE = ("none", "inj", "ans_K", "ans_V", "ans_KV")
ABL_CONDS = ("none", "flag", "pc1", "meanH", "iso0", "iso1", "iso2")
ABL_EXPLO = ("flag@init_dloc", "flag@B")
BIND_VIEWS = ("direct", "other_agent", "irrelevant_object")
BIND_ROWS = ("none", "ev", "init", "dloc", "iso", "K_X")


def log(s):
    print(s, flush=True)


def canon(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def sha_file(f) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r5(x):
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [r5(v) for v in x]
    return round(float(x), 5)


def deadline_passed() -> bool:
    d = os.environ.get("STAGE8_DEADLINE")
    return bool(d) and time.time() > float(d)


# --------------------------------------------------------------------------- populations
def ctuple(c):
    return tuple(c[f] for f in FIELDS)


def ituple(c):
    return tuple(tuple(c[f]) if f == "order" else c[f] for f in IOI_FIELDS)


def ttuple(c):
    return tuple(c[f] for f in TFIELDS)


def pop_sha(cores, fn=ctuple) -> str:
    return hashlib.sha256(json.dumps([list(fn(c)) for c in cores]).encode()).hexdigest()


def u_set():
    return {ctuple(c) for s in range(4) for c in make_cores(1000, random.Random(s))}


def excl_set():
    return {ctuple(c) for s, n in EXCL_SEEDS.items() for c in make_cores(n, random.Random(s))}


def stream(seed, n, excl, cond=lambda c: True):
    """The first n cores of the make_cores(1, rng) stream of Random(seed) that are not in ``excl``, not repeated and
    satisfy ``cond``; also the number of draws."""
    rng, out, seen, draws = random.Random(seed), [], set(), 0
    while len(out) < n:
        c = make_cores(1, rng)[0]
        draws += 1
        t = ctuple(c)
        if t in excl or t in seen or not cond(c):
            continue
        seen.add(t)
        out.append(c)
    return out, draws


def task_cores(task, n, seed, excl=frozenset()):
    """The first n cores of the Task.make_cores(1, rng) stream of Random(seed) not in ``excl`` and not repeated."""
    rng, out, seen = random.Random(seed), [], set()
    while len(out) < n:
        c = TASKS[task].make_cores(1, rng)[0]
        t = ttuple(c)
        if t in excl or t in seen:
            continue
        seen.add(t)
        out.append(c)
    return out


def populations(check=True) -> dict:
    """Every population of the part (full sizes) with hashes and the disjointness checks (asserted when ``check``)."""
    U = u_set()
    assert canon_u(U) == U_SHA, "U changed"
    X = excl_set()
    R = make_cores(SIZES["R"], random.Random(SEEDS["R"]))
    Rp = [c for c in R if c["distractor_location"] != c["initial"]]
    E8, dE = stream(SEEDS["E8"], SIZES["E8"], U | X)
    BIND, dB = stream(SEEDS["BIND"], SIZES["BIND"], U | X | {ctuple(c) for c in E8},
                      lambda c: c["distractor_location"] != c["initial"])
    FI = ioi.make_cores(SIZES["F_IOI_CAND"], random.Random(SEEDS["F_IOI"]))
    EI = ioi.make_cores(SIZES["E_IOI_CAND"], random.Random(SEEDS["E_IOI"]))
    IX = {ituple(c) for s in IOI_EXCL_SEEDS for c in ioi.make_cores(1000, random.Random(s))}
    P = {"R": R, "R'": Rp, "E8": E8, "BIND": BIND, "F_IOI_CAND": FI, "E_IOI_CAND": EI}
    TX = {}
    for t in ("paint", "schedule"):   # stage 3 drew task cores from seeds 0-3 at most: excluded, as U
        TX[t] = {ttuple(c) for s in range(4) for c in TASKS[t].make_cores(1000, random.Random(s))}
        P[f"XFIT_{t}"] = task_cores(t, SIZES["XFIT"], SEEDS["XFIT"], TX[t])
        P[f"XEVAL_{t}"] = task_cores(t, SIZES["XEVAL"], SEEDS["XEVAL"], TX[t] | {ttuple(c) for c in P[f"XFIT_{t}"]})
    ov = {"E8&U": len({ctuple(c) for c in E8} & U), "BIND&U": len({ctuple(c) for c in BIND} & U),
          "E8&EXCL": len({ctuple(c) for c in E8} & X), "BIND&EXCL": len({ctuple(c) for c in BIND} & X),
          "E8&BIND": len({ctuple(c) for c in E8} & {ctuple(c) for c in BIND}),
          "R&E8": len({ctuple(c) for c in R} & {ctuple(c) for c in E8}),
          "F_IOI&E_IOI": len({ituple(c) for c in FI} & {ituple(c) for c in EI}),
          "IOI&stage5_pilots": len(({ituple(c) for c in FI} | {ituple(c) for c in EI}) & IX)}
    for t in ("paint", "schedule"):
        f, e = {ttuple(c) for c in P[f"XFIT_{t}"]}, {ttuple(c) for c in P[f"XEVAL_{t}"]}
        ov[f"X_{t}_fit&eval"] = len(f & e)
        ov[f"X_{t}&seeds0-3"] = len((f | e) & TX[t])
    sha = {k: pop_sha(v, ituple if "IOI" in k else ttuple if k.startswith("X") else ctuple) for k, v in P.items()}
    if check:
        bad = {k: v for k, v in ov.items() if v}
        assert not bad, f"populations overlap: {bad}"
        pinned = {k: v for k, v in POP_SHA.items() if v is not None}
        diff = {k: sha[k] for k in pinned if sha[k] != pinned[k]}
        assert not diff, f"population hashes differ from the pins: {diff}"
    return {"P": P, "sha": sha, "overlap": ov, "draws": {"E8": dE, "BIND": dB}, "U_sha256": U_SHA, "n_R'": len(Rp)}


def canon_u(U) -> str:
    return hashlib.sha256(json.dumps(sorted(U)).encode()).hexdigest()


def cut(a, cores, name):
    n = 2 if a.test else SIZES[name]
    return cores[:n]


# --------------------------------------------------------------------------- model
def load_model(a, attn=None):
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    attn = attn or ("eager" if cfg.model_type.startswith("gemma2") else "sdpa")
    dev = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    kw = dict(dtype=getattr(torch, a.dtype), revision=a.revision, attn_implementation=attn)
    if dev == "cuda":
        kw["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if dev == "cpu":
        model = model.to(dev)
    assert model.config._attn_implementation == attn
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model, tok, dev


def verified(model_dir):
    f = Path(model_dir) / "VERIFIED.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    return {"sha256": sha_file(f), "repo": j.get("repo"), "revision": j.get("revision"), "key": j.get("key"),
            "attn": j.get("attn"), "token_used": j.get("token_used")}


def prov(a, model=None, dev=None, extra=None):
    p = ff.provenance(a)
    p["args"] = {k: v for k, v in vars(a).items() if k != "pops"}          # the populations are recorded by hash
    p |= {"model": a.model, "model_key": a.key, "revision": a.revision, "verified": verified(a.model), "test_mode": a.test,
          "stage": a.stage, "populations": dict(a.pops["sha"]) if a.pops else None, "U_sha256": U_SHA, "timings": {},
          "skipped_items": [], "deadline_skipped": []}
    if model is not None:
        cfg = model.config
        p |= {"dtype": str(next(model.parameters()).dtype), "attn_implementation": cfg._attn_implementation,
              "device": device_name(dev), "n_layers": len(blocks(model)), "n_heads": cfg.num_attention_heads,
              "n_kv_heads": cfg.num_key_value_heads, "hidden": cfg.hidden_size}
    p["wrapper"] = dict(WRAPPER_USED)
    return p | (extra or {})


class M:
    """A loaded model with the part's hooks installed once."""

    def __init__(self, a, attn=None):
        self.model, self.tok, self.dev = load_model(a, attn)
        self.nL, self.H, self.hd, self.D = dims(self.model)
        self.hs, self.hop = HeadSplice(self.model), HopSplice(self.model)
        self.inj, self.oc = Inject(self.model), OCap(self.model)
        self.cid = candidate_ids(self.tok, "P1")
        self.allcells = [(l, h) for l in range(self.nL) for h in range(self.H)]
        self._W = {}

    def W(self, l):
        if l not in self._W:
            self._W[l] = wo(self.model, l)
        return self._W[l]

    def Ws(self, layers):
        return {l: self.W(l) for l in layers}


def reset(m: M):
    m.inj.clear()
    m.hs.active, m.hs.masks, m.hs.mu, m.hs.mode, m.hs.ks, m.hs.pos = False, None, None, "splice", None, None
    m.hop.active, m.hop.mask = False, None
    m.oc.active = False


@torch.no_grad()
def fwd(m: M, ids, n=None, k=None, v=None, add=None, proj=None, hs=None, hop=False, fs=None, kv=None, z=None):
    """One forward of ``ids`` ([1, T], expanded to ``n`` rows, or [n, T]) with these hooks, all reset afterwards:
    k / v: (positions, {(l, ch): [n, P, D]}, use [n] or [n, P]) clamps from layer 0 (ckeys.clamp.clamp_kv);
    add / proj: the Inject maps; hs: HeadSplice attributes (mode, pos, ks, masks, mu); hop: True when HopSplice was
    configured by stage6_heads.configure_hop (its mask is extended over the trie nodes with the answer row's value);
    fs: a FormSet (case-marginalised scoring, ckeys.surface.score; else the full log-softmax at the last position);
    kv: (positions, layers, which) captured (ckeys.clamp.capture_kv); z: (rows, layers) o_proj inputs captured (OCap).
    Returns (out, kv captures or None, z captures or None): out = {"lp": [n, V]} or {"E": [n, 6], "L": [n, 6]} (CPU)."""
    ids = (ids.expand(n, -1) if n is not None else ids).to(m.dev)
    n, T = ids.shape
    cap = zst = None
    with contextlib.ExitStack() as st:
        if k is not None:
            st.enter_context(clamp_kv(m.model, k[0], k[1], range(m.nL), "k", per_row=k[2]))
        if v is not None:
            st.enter_context(clamp_kv(m.model, v[0], v[1], range(m.nL), "v", per_row=v[2]))
        if kv is not None:
            cap = st.enter_context(capture_kv(m.model, kv[0], kv[1], kv[2]))
        try:
            if add or proj:
                m.inj.add, m.inj.proj, m.inj.active = add or {}, proj or {}, True
            if hs:
                for key, val in hs.items():
                    setattr(m.hs, key, val)
                m.hs.active = True
            if hop:
                if fs is not None and len(fs):
                    mk = m.hop.mask
                    m.hop.mask = torch.cat([mk, mk[:, -1:].expand(-1, len(fs))], 1)
                m.hop.active = True
            if z is not None:
                m.oc.store, m.oc.rows, m.oc.layers, m.oc.active = {}, list(z[0]), set(z[1]), True
            if fs is not None:
                res = fs_score(m.model, ids, fs)
                out = {"E": torch.stack([res["E"][w] for w in LOCATIONS], 1).float().cpu(),
                       "L": torch.stack([res["L"][w] for w in LOCATIONS], 1).float().cpu()}
            else:
                o = m.model(ids, use_cache=False, logits_to_keep=1)
                out = {"lp": torch.log_softmax(o.logits[:, -1].float(), -1).cpu()}
            if z is not None:
                zst = {l: t.cpu() for l, t in m.oc.store.items()}
        finally:
            reset(m)
    capd = {kk: vv.clone() for kk, vv in cap.items()} if cap is not None else None
    return out, capd, zst


def keys_at(m: M, ids_list, pos, which="k"):
    """One batched pass over the runs ``ids_list`` (equal length); returns per run {l: [D]} at ``pos[i]`` (one position per
    run) for each channel of ``which`` ({ch: [per run dict]})."""
    ids = torch.cat(ids_list)
    P = sorted(set(pos))
    _, cap, _ = fwd(m, ids, kv=(P, range(m.nL), which))
    out = {}
    for ch in which:
        out[ch] = [{l: cap[(l, ch)][i, P.index(pos[i])].clone() for l in range(m.nL)} for i in range(len(ids_list))]
    return out


def runs_kv(m: M, ids, names, p):
    """K and V at p of the named runs (``ids[name]``, equal length), one batched pass: {name: {"k": {l}, "v": {l}}}."""
    K = keys_at(m, [ids[n] for n in names], [p] * len(names), "kv")
    return {n: {"k": K["k"][i], "v": K["v"][i]} for i, n in enumerate(names)}


def kv_clamp(p, srcs, caps, nL):
    """Clamp arguments for position p in every layer, as format_factorial.run_item: ``srcs`` one entry per batch row,
    None (the row's own run) or (key source, value source), names of ``caps`` (runs_kv). A key-only row holds the value
    at the base run's (and vice versa), so that p's own later-layer value cannot drift. Returns (k, v) for fwd."""
    if all(s is None for s in srcs):
        return None, None
    use = torch.tensor([s is not None for s in srcs])
    out = []
    for j, ch in enumerate("kv"):
        any_t = next(iter(caps.values()))[ch]
        zero = torch.zeros_like(any_t[0])
        tabs = {(l, ch): torch.stack([caps[s[j]][ch][l] if s is not None else zero for s in srcs])[:, None] for l in range(nL)}
        out.append(([p], tabs, use))
    return tuple(out)


ROWSRC = {"K_S": ("S", "B"), "K_X": ("X", "B"), "K_N": ("N", "B"), "V_S": ("B", "S"), "V_X": ("B", "X")}


def six(lp, cid):
    return lp[:, cid]


def rec_rows(names, lp6):
    return {n: r5(lp6[i]) for i, n in enumerate(names)}


# --------------------------------------------------------------------------- belief prompts
def enc(tok, arm, core, view, loc, X, mod=None):
    c = core | (mod or {})
    r = record(c, view, loc)
    return encode_with_offsets(tok, build_prompt(arm, r["story"], r["query"], core, X))


def n_word(core) -> str:
    return N_WORDS[int(hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8], 16) % len(N_WORDS)]


def i_prime(core, X):
    """The location absent from {init, dloc, B, S, X} (exists when dloc != init)."""
    rest = [w for w in LOCATIONS if w not in {core["initial"], core["distractor_location"], core["base"], core["source"], X}]
    assert len(rest) == 1, (core, X, rest)
    return rest[0]


def prep(tok, core, arm="P1", view="direct", nrun=False, initial_runs=False):
    """Encodings and rows of one belief prompt: runs B, S, X (and N with the K_N event word; Iinit / Idloc with I' as the
    initial or the distractor's location), p, the six option rows G (LOCATIONS order), the first row of the span and
    of the question line. None if a run differs from B in length or at more than one position."""
    X = pick_x(core)
    cid = candidate_ids(tok, "P1")
    runs = {"B": (core["base"], None), "S": (core["source"], None), "X": (X, None)}
    if nrun:
        runs["N"] = (n_word(core), None)
    if initial_runs:
        Ip = i_prime(core, X)
        runs["Iinit"] = (core["base"], {"initial": Ip})
        runs["Idloc"] = (core["base"], {"distractor_location": Ip})
    E = {k: enc(tok, arm, core, view, loc, X, mod) for k, (loc, mod) in runs.items()}
    text, ib, off = E["B"]
    if any(e[1].shape != ib.shape for e in E.values()):
        return None
    pos = {}
    for k, e in E.items():
        if k == "B":
            continue
        diff = (ib[0] != e[1][0]).nonzero().flatten().tolist()
        if len(diff) != 1:
            return None
        pos[k] = diff[0]
    p = pos["S"]
    assert pos["X"] == p and pos.get("N", p) == p, (core, pos)
    r = record(core, view, core["base"])
    seg = arm_span(arm, r["story"], r["query"], core, X)
    assert seg is not None and text.count(seg) == 1, (arm, seg)
    c0 = text.index(seg)
    span = rows_in(off, c0, c0 + len(seg))
    G = [i for i in span if ib[0, i].item() in set(cid)]
    assert len(G) == 6 and ib[0, G].tolist() == cid and min(G) > p, (arm, core, G, p)
    assert ib[0, p].item() == cid[LOCATIONS.index(core["base"])]
    q0 = text.index("\nQuestion: ") + 1
    qrow = rows_in(off, q0, q0 + len("Question"))[0]
    ix = {k: LOCATIONS.index(w) for k, w in (("B", core["base"]), ("S", core["source"]), ("X", X), ("I", core["initial"]),
                                              ("D", core["distractor_location"]))}
    d = dict(core=core, arm=arm, view=view, X=X, ids={k: e[1] for k, e in E.items()}, p=p, pos=pos, G=G, T=ib.shape[1],
             first=span[0], qrow=qrow, ix=ix, row={k: G[i] for k, i in ix.items()}, cid=cid)
    if initial_runs:
        d["Ip"] = Ip
        d["ix"]["Ip"] = LOCATIONS.index(Ip)
        d["row"]["Ip"] = G[LOCATIONS.index(Ip)]
        assert ib[0, pos["Iinit"]].item() == cid[d["ix"]["I"]] and ib[0, pos["Idloc"]].item() == cid[d["ix"]["D"]]
        assert pos["Iinit"] < p and pos["Idloc"] < p and pos["Iinit"] != pos["Idloc"]
    return d


def order_stratum(core) -> str:
    """ckeys.story sorts the two initial-state sentences by object name: the queried object's sentence comes first when
    its name sorts first."""
    return "object_first" if core["object"] < core["distractor"] else "distractor_first"


def mv(vec, r_to, r_from, c=1.0):
    return [(r_to, vec, c), (r_from, vec, -c)]


def fs_of(tok):
    return FormSet(tok, LOCATIONS, frames=FS_FRAMES)


# --------------------------------------------------------------------------- sets and flags on disk
def stage6_sets(a, key):
    """H*, the random and active sets of stage 6 for a 7B key (file sha256, revision, k* and the canonical hash asserted)."""
    fn, sha, rev, kstar = STAGE6[key]
    f = Path(a.heads_dir) / fn
    got = sha_file(f)
    assert got == sha, f"{f}: sha256 {got} != {sha}"
    J = json.load(open(f))
    assert J["provenance"]["revision"] == rev and J["provenance"]["kstar"] == kstar, "stage-6 file: revision or k* differs"
    A = J["arms"]["P1"]
    S = {"H": A["rankings"]["a3"][:kstar], "rand": [r[:kstar] for r in A["sets"]["rand"]], "active": A["sets"]["active_kstar"],
         "kstar": kstar}
    assert canon(S) == SETS_SHA[key], "stage-6 sets hash differs"
    return S, {"file": str(f), "file_sha256": got, "sets_sha256": canon(S), "revision": rev}


def read_sets(a):
    f = Path(a.out) / "sets" / f"{a.tag}.json"
    J = json.load(open(f))
    S = J["sets"]
    assert canon(S) == J["sets_sha256"], "sets file changed after hashing"
    return J, S


def read_flags(a):
    J = json.load(open(Path(a.out) / "fit" / f"{a.tag}.json"))
    f = Path(a.out) / "fit" / f"{a.tag}.pt"
    assert sha_file(f) == J["pt_sha256"], "flags file differs from the one the fit recorded"
    return J, torch.load(f)


def cells(x):
    return [tuple(c) for c in x]


# --------------------------------------------------------------------------- stage preflight (tokenizer only)
def ioi_arms(key):
    return ("INLINE", "INLINE_CHAT", "AFTER", "BEFORE") if key == "qwen7" else ("INLINE", "INLINE_CHAT")


def ioi_pops(tok, a):
    """F_ioi and E_ioi for this tokenizer: (cores, indices, skipped with reasons)."""
    P = a.pops["P"]
    F, Fi, skF = [], [], []
    for i, c in enumerate(P["F_IOI_CAND"]):
        why = Q.ioi_valid(tok, c, "INLINE", True)
        if why:
            skF.append({"index": i, "arm": "INLINE", "reason": why})
        elif len(F) < SIZES["F_IOI"]:
            F.append(c)
            Fi.append(i)
    E, Ei, skE = [], [], []
    for i, c in enumerate(P["E_IOI_CAND"]):
        whys = {arm: Q.ioi_valid(tok, c, arm, True) for arm in ioi_arms(a.key)}
        bad = {k: v for k, v in whys.items() if v}
        if bad:
            skE.append({"index": i, "reasons": bad})
        elif len(E) < SIZES["E_IOI"]:
            E.append(c)
            Ei.append(i)
    if not a.test:
        assert len(F) == SIZES["F_IOI"] and len(E) == SIZES["E_IOI"], (len(F), len(E))
    return {"F": cut(a, F, "F_IOI"), "F_idx": Fi, "E": cut(a, E, "E_IOI"), "E_idx": Ei, "skipped": skF + skE}


def stage_preflight(a):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    P = a.pops["P"]
    n, bad = 0, []
    for name, arms in (("R", ("P1", "POST")), ("E8", ("P1", "POST", "Q_IN", "Q_OUT")), ("BIND", ("P1",))):
        for c in P[name]:
            for arm in arms:
                views = BIND_VIEWS if name == "BIND" else ("direct",)
                for view in views:
                    d = prep(tok, c, arm, view, nrun=(name == "E8" and arm == "P1"), initial_runs=False)
                    n += 1
                    if d is None:
                        bad.append((name, arm, view, c))
    for c in P["R'"]:
        n += 1
        if prep(tok, c, "P1", "direct", initial_runs=True) is None:
            bad.append(("R'", "P1", "direct", c))
    ioi_p = ioi_pops(tok, a)
    for c in ioi_p["F"] + ioi_p["E"]:
        for arm in ioi_arms(a.key):
            if c in ioi_p["E"] or arm == "INLINE":
                Q.ioi_layout(tok, c, arm, True)
                n += 1
    nw = {w: len(tok.encode(" " + w, add_special_tokens=False)) for w in N_WORDS}
    assert all(v == 1 for v in nw.values()), f"K_N words not single tokens: {nw}"
    assert not set(N_WORDS) & set(LOCATIONS)
    fs = fs_of(tok)
    forms = {w: sorted(tok.decode(list(s)) for s in fs.sets["E"][w]) for w in LOCATIONS}
    tasks_ok = {}
    for t in ("paint", "schedule"):
        vals = TASKS[t].values
        single = all(len(tok.encode(" " + v, add_special_tokens=False)) == 1 for v in vals)
        tasks_ok[t] = single
        if single:
            for c in P[f"XFIT_{t}"][:5] + P[f"XEVAL_{t}"][:5]:
                prep_task(tok, t, c)
    assert not bad, f"{len(bad)} prompts with runs of different lengths or positions, first {bad[0]}"
    out = {"provenance": prov(a) | {"skipped_items": ioi_p["skipped"]}, "sha": a.pops["sha"], "overlap": a.pops["overlap"],
           "draws": a.pops["draws"], "n_R'": a.pops["n_R'"], "prompts_checked": n, "k_n_words": nw, "case_forms": forms,
           "trie_nodes": len(fs), "ioi": {"F_idx": ioi_p["F_idx"], "E_idx": ioi_p["E_idx"], "n_skipped": len(ioi_p["skipped"])},
           "tasks_single_token": tasks_ok, "seconds": round(time.time() - t0, 1)}
    write_atomic(out, Path(a.out) / "preflight" / f"{a.tag}.json")
    log(f"preflight OK: {n} prompts, IOI F {len(ioi_p['F_idx'])} E {len(ioi_p['E_idx'])} (skipped {len(ioi_p['skipped'])}), "
        f"trie nodes {len(fs)}, tasks {tasks_ok}")


# --------------------------------------------------------------------------- stage sets (eager)
@torch.no_grad()
def hop_rank(m: M, R):
    """hop(l, h) = 1/2[(A^{K_S} - A^B)[END, r_S] + (A^B - A^{K_S})[END, r_B]] and a3 (stage 6's formula), means over R."""
    H2, A3 = [], []
    for core in R:
        d = prep(m.tok, core)
        ks = keys_at(m, [d["ids"]["S"]], [d["p"]])["k"][0]
        ids = d["ids"]["B"].to(m.dev)
        ob = m.model(ids, use_cache=False, logits_to_keep=1, output_attentions=True)
        with clamp_kv(m.model, [d["p"]], {(l, "k"): ks[l][None, None] for l in range(m.nL)}, range(m.nL), "k"):
            of = m.model(ids, use_cache=False, logits_to_keep=1, output_attentions=True)
        assert ob.attentions[0] is not None, "attentions missing: eager attention needed"
        Ab = torch.stack([x[0].float() for x in ob.attentions]).cpu()
        Af = torch.stack([x[0].float() for x in of.attentions]).cpu()
        rS, rB, p, T = d["row"]["S"], d["row"]["B"], d["p"], d["T"]
        H2.append((0.5 * ((Af[:, :, T - 1, rS] - Ab[:, :, T - 1, rS]) + (Ab[:, :, T - 1, rB] - Af[:, :, T - 1, rB]))).numpy())
        A3.append((0.5 * ((Af[:, :, rS, p] - Ab[:, :, rS, p]) + (Ab[:, :, rB, p] - Af[:, :, rB, p]))).numpy())
    return np.mean(H2, 0), np.mean(A3, 0)


@torch.no_grad()
def stage_sets(a):
    m = M(a, "eager")
    nL, H = m.nL, m.H
    R = cut(a, a.pops["P"]["R"], "R")
    t0 = time.time()
    src, s6 = None, None
    if a.key in STAGE6:
        S6, s6 = stage6_sets(a, a.key)       # hashed in every mode (TEST_MODE: recorded, not used at 0.5B)
    if a.key in STAGE6 and not a.test:
        S, src = dict(S6), "stage6"
    else:
        S6r = Stage6(m.model, m.tok, "P1", m.hs, m.hop)
        kstar = math.ceil((TEST_KSTAR_FRAC if a.test else KSTAR_FRAC) * nL * H)
        rows, ACT = [], []
        for core in R:
            d = S6r.prep(core)
            ks = S6r.ks_of(d)
            b = S6r.base_runs(d, ks, phase1=True)
            b.pop("oin")
            rows.append(b["a3"])
            ACT.append(b["act"])
        A3 = np.mean(rows, 0)
        rk = rank_of(A3)
        rng = np.random.default_rng(RAND_SEED)
        rand = [[list(m.allcells[i]) for i in rng.permutation(len(m.allcells))][:kstar] for _ in range(N_RAND)]
        top2 = {tuple(c) for c in rk[:2 * kstar]}
        active = [c for c in rank_of(np.mean(ACT, 0)) if tuple(c) not in top2][:kstar]
        S = {"H": rk[:kstar], "rand": rand, "active": active, "kstar": kstar}
        src = "ranked in-run (Stage6.base_runs, phase 1)"
    hop, a3 = hop_rank(m, R)
    hk = rank_of(hop)
    rng = np.random.default_rng(HOP_RAND_SEED)
    S["hop_top"] = hk[:N_HOP]
    S["hop_rand"] = [list(m.allcells[i]) for i in rng.permutation(len(m.allcells))[:N_HOP]]
    kst = S["kstar"]
    ov = len({tuple(c) for c in rank_of(a3)[:kst]} & {tuple(c) for c in S["H"]})
    out = {"provenance": prov(a, m.model, m.dev, {"source": src, "stage6": s6, "n_rank": len(R)}), "sets": S,
           "sets_sha256": canon(S), "layers": sorted({l for l, _ in S["H"]}), "hop_mean": hop.tolist(),
           "a3_inrun_mean": a3.tolist(), "a3_inrun_overlap_H": ov}
    out["provenance"]["timings"]["total"] = round(time.time() - t0, 1)
    write_atomic(out, Path(a.out) / "sets" / f"{a.tag}.json")
    log(f"sets: {src}; k* {kst}, L* {out['layers']}; hop-2 top {S['hop_top'][:5]}; in-run a3 overlap with H* {ov}/{kst} "
        f"({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage fit
@torch.no_grad()
def write_pass(m: M, d, runs, rows, layers):
    """o_proj inputs at ``rows`` (layers ``layers``) for one batch of ``runs``: each run is (run name in d["ids"], key
    table {l: [D]} or None, clamp position). A key table is the stage-6 natural clamp: the key at that position in every
    layer, nothing else. Returns one {l: [|rows|, H*hd]} per run."""
    ids = torch.cat([d["ids"][r[0]] for r in runs])
    k = None
    P = sorted({r[2] for r in runs if r[1] is not None})
    if P:
        ref = next(r[1] for r in runs if r[1] is not None)[0]
        tabs = {(l, "k"): torch.zeros(len(runs), len(P), ref.shape[0], dtype=ref.dtype, device=ref.device) for l in range(m.nL)}
        use = torch.zeros(len(runs), len(P), dtype=torch.bool)
        for i, (_, s, pp) in enumerate(runs):
            if s is not None:
                for l in range(m.nL):
                    tabs[(l, "k")][i, P.index(pp)] = s[l]
                use[i, P.index(pp)] = True
        k = (P, tabs, use)
    _, _, z = fwd(m, ids, k=k, z=(rows, layers))
    return [{l: z[l][i] for l in z} for i in range(len(runs))]


@torch.no_grad()
def stage_fit(a):
    J, S = read_sets(a)
    m = M(a)
    t0 = time.time()
    P = a.pops["P"]
    R = cut(a, P["R"], "R")
    Hs = cells(S["H"])
    byH, byA = by_layer(Hs), by_layer(cells(S["active"]))
    Ls = sorted(byH)
    Lcap = sorted(set(Ls) | set(byA))
    Ws = m.Ws(Lcap)
    hd = m.hd
    pr = prov(a, m.model, m.dev, {"sets_sha256": J["sets_sha256"]})
    small = a.key in SMALL
    # ---- P1 on R: delta^P1 (K_S), delta^KV (natural S run), delta^act, clean outputs at G
    dP1, dKV, dAct, yG, hG, Bs = [], [], [], [], [], []
    for core in R:
        d = prep(m.tok, core)
        ks = keys_at(m, [d["ids"]["S"]], [d["p"]])["k"][0]
        zb, zk, zs = write_pass(m, d, [("B", None, None), ("B", ks, d["p"]), ("S", None, None)], d["G"], Lcap)
        iS, iB = d["ix"]["S"], d["ix"]["B"]
        dP1.append({l: v.cpu() for l, v in flag_delta(Ws, zk, zb, iS, iB, byH, hd).items()})
        dKV.append({l: v.cpu() for l, v in flag_delta(Ws, zs, zb, iS, iB, byH, hd).items()})
        dAct.append({l: v.cpu() for l, v in flag_delta(Ws, zk, zb, iS, iB, byA, hd).items()})
        yG.append({l: (zb[l].float() @ Ws[l].T.to(zb[l].device)).cpu() for l in Ls})              # [6, D] attention output
        hG.append({l: head_out(Ws[l], zb[l], byH[l], hd).cpu() for l in Ls})                     # [6, D] H* output
        absent = [i for i, w in enumerate(LOCATIONS) if w not in (core["base"], core["initial"], core["distractor_location"])]
        Bs.append((d["ix"]["B"], d["ix"]["S"], absent))
    pr["timings"]["P1"] = round(time.time() - t0, 1)
    D = flag_of(dP1)
    F = {"P1": D, "KV": flag_of(dKV), "act": total_match(flag_of(dAct), D)}
    loo, loo_n, loo_fb = {}, {}, []
    for w in range(6):
        keep = [b != w and s != w for b, s, _ in Bs]
        loo_n[LOCATIONS[w]] = int(sum(keep))
        if sum(keep) == 0:      # TEST_MODE only (two fit stories): the full flag stands in, recorded
            assert a.test, f"no fit story for the leave-one-word-out flag of {LOCATIONS[w]}"
            loo[w] = dict(D)
            loo_fb.append(LOCATIONS[w])
        else:
            assert a.test or sum(keep) >= 10, (LOCATIONS[w], sum(keep))
            loo[w] = flag_of(dP1, keep)
    F["loo"] = loo
    F["iso"] = iso_random(D, 3, ISO_SEED)
    F["hspan"] = headspan_random(Ws, byH, hd, D, 3, HSPAN_SEED)
    F["perm"], perm_map = layer_perm(D, PERM_SEED) if len(D) >= 2 else (dict(D), {})
    # directions for the ablation (unit) and for the injection controls (rescaled to |Delta_l|)
    meanH = {l: unit(torch.cat([h[l] for h in hG]).mean(0)) for l in Ls}
    pc1 = {l: top_pc(torch.cat([y[l] for y in yG])) for l in Ls}
    g = torch.Generator().manual_seed(ABL_ISO_SEED)
    abl_iso = [{l: unit(torch.randn(m.D, generator=g)) for l in Ls} for _ in range(3)]
    dirs = {"flag": {l: unit(D[l]) for l in Ls}, "pc1": pc1, "meanH": meanH} | {f"iso{i}": u for i, u in enumerate(abl_iso)}
    mu = {}
    for name, U in dirs.items():
        mu[name] = {l: float(np.mean([float(y[l][j] @ U[l]) for y, (_, _, ab) in zip(yG, Bs) for j in ab])) for l in Ls}
    F["meanH"] = norm_match(meanH, D)
    F["dirs"], F["mu"] = dirs, mu
    # ---- POST on R
    dPO = []
    for core in R:
        d = prep(m.tok, core, "POST")
        ks = keys_at(m, [d["ids"]["S"]], [d["p"]])["k"][0]
        zb, zk = write_pass(m, d, [("B", None, None), ("B", ks, d["p"])], d["G"], Ls)
        dPO.append({l: v.cpu() for l, v in flag_delta(Ws, zk, zb, d["ix"]["S"], d["ix"]["B"], byH, hd).items()})
    F["POST"] = flag_of(dPO)
    pr["timings"]["POST"] = round(time.time() - t0, 1)
    extra = {}
    if not small:
        # ---- initial-state flags on R'
        Rp = cut(a, P["R'"], "R")
        dI, dD = [], []
        for core in Rp:
            d = prep(m.tok, core, initial_runs=True)
            kk = keys_at(m, [d["ids"]["Iinit"], d["ids"]["Idloc"]], [d["pos"]["Iinit"], d["pos"]["Idloc"]])["k"]
            zb, zi, zd = write_pass(m, d, [("B", None, None), ("B", kk[0], d["pos"]["Iinit"]), ("B", kk[1], d["pos"]["Idloc"])],
                                    d["G"], Ls)
            iP = d["ix"]["Ip"]
            dI.append({l: v.cpu() for l, v in flag_delta(Ws, zi, zb, iP, d["ix"]["I"], byH, hd).items()})
            dD.append({l: v.cpu() for l, v in flag_delta(Ws, zd, zb, iP, d["ix"]["D"], byH, hd).items()})
        F["init"], F["dloc"] = flag_of(dI), flag_of(dD)
        extra["n_R'"] = len(Rp)
        pr["timings"]["initial"] = round(time.time() - t0, 1)
        # ---- IOI INLINE flag on F_ioi
        ip = ioi_pops(m.tok, a)
        dI2 = []
        for core in ip["F"]:
            L = Q.ioi_layout(m.tok, core, "INLINE", True)
            ks = keys_at(m, [L["ids"]["S"]], [L["p"]])["k"][0]
            rows = [L["named"]["io_s"], L["named"]["io_b"]]
            dd = {"ids": L["ids"]}
            zb, zk = write_pass(m, dd, [("B", None, None), ("B", ks, L["p"])], rows, Ls)
            dI2.append({l: v.cpu() for l, v in flag_delta(Ws, zk, zb, 0, 1, byH, hd).items()})
        F["IOI"] = flag_of(dI2)
        extra |= {"n_F_ioi": len(ip["F"]), "F_ioi_idx": ip["F_idx"]}
        pr["skipped_items"] += ip["skipped"]
        pr["timings"]["IOI"] = round(time.time() - t0, 1)
    # ---- descriptive statistics
    geo = geometry(dP1, D)
    named = [k for k in ("P1", "KV", "POST", "init", "dloc", "IOI") if k in F]
    cosm = {f"{x}|{y}": {l: float(torch.nn.functional.cosine_similarity(F[x][l], F[y][l], dim=0)) for l in Ls}
            for i, x in enumerate(named) for y in named[i + 1:]}
    story_cos_kv = {l: float(np.mean([float(torch.nn.functional.cosine_similarity(k_[l], v_[l], dim=0)) for k_, v_ in zip(dP1, dKV)]))
                    for l in Ls}
    summ = {"layers": Ls, "layers_active": sorted(byA), "norms": {k: {l: float(F[k][l].norm()) for l in F[k]} for k in named + ["act"]},
            "geometry": geo, "cos": cosm, "wcos_K_KV": wcos(D, F["KV"]), "story_cos_K_KV": story_cos_kv,
            "kappa_POST_P1": wcos(D, F["POST"]), "identity_share": identity_share(dP1, [b for b, _, _ in Bs]),
            "logit_lens": logit_lens(m.model, D, m.cid), "loo_n": loo_n, "loo_fallback": loo_fb, "perm": perm_map,
            "n_R": len(R)} | extra
    f = Path(a.out) / "fit" / f"{a.tag}.pt"
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in F.items()}, f)
    out = {"provenance": pr | {"flags_content_sha256": sha_tensors(F)}, "summary": summ, "pt_file": f.name, "pt_sha256": sha_file(f)}
    write_atomic(out, Path(a.out) / "fit" / f"{a.tag}.json")
    log(f"fit: L* {Ls}; |Delta| {[round(summ['norms']['P1'][l], 3) for l in Ls]}; cos(K, KV) {summ['wcos_K_KV']:.3f}; "
        f"kappa(POST, P1) {summ['kappa_POST_P1']:.3f} ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage inject (E8, P1)
def common(a, stage, need_flags=True):
    J, S = read_sets(a)
    FJ, F = read_flags(a) if need_flags else (None, None)
    return J, S, FJ, F


@torch.no_grad()
def stage_inject(a):
    J, S, FJ, F = common(a, "inject")
    m = M(a)
    t0 = time.time()
    Hs = cells(S["H"])
    byH = by_layer(Hs)
    Ls = sorted(byH)
    Ws = m.Ws(Ls)
    D, loo = F["P1"], F["loo"]
    allL = sorted({l for v in [D, F["act"]] for l in v})
    pr = prov(a, m.model, m.dev, {"sets_sha256": J["sets_sha256"], "flags_sha256": FJ["pt_sha256"], "rows": list(INJ_ROWS),
                                  "route_rows": list(ROUTE_ROWS)})
    stories = []
    for si, core in enumerate(cut(a, a.pops["P"]["E8"], "E8")):
        d = prep(m.tok, core, nrun=True)
        p, rX, rB, rS = d["p"], d["row"]["X"], d["row"]["B"], d["row"]["S"]
        caps = runs_kv(m, d["ids"], ["B", "S", "X", "N"], p)
        zb, zk = write_pass(m, d, [("B", None, None), ("B", caps["S"]["k"], p)], [rS, rB], Ls)
        own = flag_delta(Ws, zk, zb, 0, 1, byH, m.hd)
        own = {l: v.cpu() for l, v in own.items()}
        iS, iX = d["ix"]["S"], d["ix"]["X"]
        spec = {"none": [], "none2": [], "K_S": [], "K_X": [], "K_N": [],
                "add0.5": [(rX, D, 0.5)], "add1": [(rX, D, 1.0)], "add2": [(rX, D, 2.0)],
                "move": mv(D, rX, rB), "moveS": mv(D, rS, rB), "subB": [(rB, D, -1.0)],
                "looS": mv(loo[iS], rS, rB), "looX": mv(loo[iX], rX, rB), "kvS": mv(F["KV"], rS, rB), "kvX": mv(F["KV"], rX, rB),
                "iso0": mv(F["iso"][0], rX, rB), "iso1": mv(F["iso"][1], rX, rB), "iso2": mv(F["iso"][2], rX, rB),
                "hspan0": mv(F["hspan"][0], rX, rB), "hspan1": mv(F["hspan"][1], rX, rB), "hspan2": mv(F["hspan"][2], rX, rB),
                "perm": mv(F["perm"], rX, rB), "orth": mv(orth_to(own, D), rX, rB), "meanH": mv(F["meanH"], rX, rB),
                "active": mv(F["act"], rX, rB), "own": mv(own, rX, rB),
                "choices": [(d["first"], D, 1.0)], "question": [(d["qrow"], D, 1.0)], "postflag": [(rX, F["POST"], 1.0)]}
        names = list(INJ_ROWS)
        kk, vv = kv_clamp(p, [ROWSRC.get(nm) for nm in names], caps, m.nL)
        add = add_map([spec[nm] for nm in names], allL)
        out, _, _ = fwd(m, d["ids"]["B"], len(names), k=kk, v=vv, add=add)
        rows = rec_rows(names, six(out["lp"], d["cid"]))
        # hop-2 route of the injected flag (HopSplice over G; the move rows carry the injection)
        rn = list(ROUTE_ROWS)
        add2 = add_map([[] if nm == "none" else mv(D, rX, rB) for nm in ["none", "move"]], allL)
        _, cap, _ = fwd(m, d["ids"]["B"], 2, add=add2, kv=(d["G"], range(m.nL), "kv"))
        kvt = {"base": {kk: vv[0] for kk, vv in cap.items()}, "ksrun": {kk: vv[1] for kk, vv in cap.items()}}
        configure_hop(m.hop, rn, kvt, d["G"], d["T"], ROUTE_SPEC)
        addr = add_map([[] if nm == "none" else mv(D, rX, rB) for nm in rn], allL)
        outr, _, _ = fwd(m, d["ids"]["B"], len(rn), add=addr, hop=True)
        stories.append(dict(index=si, core=core, X=d["X"], p=p, T=d["T"], G=d["G"], ix=d["ix"], n_word=n_word(core),
                            rows=rows, route=rec_rows(rn, six(outr["lp"], d["cid"])),
                            own_norm={l: float(own[l].norm()) for l in Ls}))
        if si % 10 == 0:
            log(f"inject story {si} ({time.time() - t0:.0f}s)")
    pr["timings"]["total"] = round(time.time() - t0, 1)
    pr["inject_calls"] = m.inj.n_calls
    write_atomic({"provenance": pr, "stories": stories}, Path(a.out) / "inject" / f"{a.tag}.json")
    log(f"inject: {len(stories)} stories ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage ablate (A, P1)
@torch.no_grad()
def stage_ablate(a):
    J, S, FJ, F = common(a, "ablate")
    m = M(a)
    t0 = time.time()
    Ls = sorted(by_layer(cells(S["H"])))
    explo = not deadline_passed()
    conds = list(ABL_CONDS) + (list(ABL_EXPLO) if explo else [])
    pr = prov(a, m.model, m.dev, {"sets_sha256": J["sets_sha256"], "flags_sha256": FJ["pt_sha256"], "conditions": conds})
    if not explo:
        pr["deadline_skipped"] = list(ABL_EXPLO)
    stories = []
    for si, core in enumerate(cut(a, a.pops["P"]["E8"], "A")):
        d = prep(m.tok, core)
        res = {}
        for c in conds:
            if c == "none":
                proj = None
            else:
                name, rows = (c.split("@")[0], d["G"]) if "@" not in c else ("flag", [d["row"]["I"], d["row"]["D"]] if c.endswith("init_dloc") else [d["row"]["B"]])
                rm = torch.zeros(1, d["T"], dtype=torch.bool)
                rm[0, rows] = True
                proj = {l: (F["dirs"][name][l], F["mu"][name][l], rm) for l in Ls}
            try:
                if proj:
                    m.inj.proj, m.inj.active = proj, True
                r = ff.run_item(m.model, m.tok, core, "P1", "direct", m.dev)
            finally:
                reset(m)
            assert r is not None and r["pos"] == d["p"] and r["len"] == d["T"], c
            mm, idr = r["m"], r["m"]["ID@0"]
            dl = {k: {t: mm[k]["lp"][t] - idr["lp"][t] for t in ("S", "B", "X")} for k in mm}
            cB = r["clean"]["B"]
            res[c] = dict(idK=0.5 * ((dl["K_S@0"]["S"] - dl["K_X@0"]["S"]) + (dl["K_X@0"]["X"] - dl["K_S@0"]["X"])),
                          idV=0.5 * ((dl["V_S@0"]["S"] - dl["V_X@0"]["S"]) + (dl["V_X@0"]["X"] - dl["V_S@0"]["X"])),
                          dK=mm["K_S@0"]["m"] - idr["m"], dV=mm["V_S@0"]["m"] - idr["m"], mass=cB["mass"],
                          argmax=cB["argmax_cand"], base_ok=cB["argmax_cand"] == core["base"], init_ans=cB["argmax_cand"] == core["initial"])
        stories.append(dict(index=si, core=core, res=res))
        if si % 10 == 0:
            log(f"ablate story {si} ({time.time() - t0:.0f}s)")
    pr["timings"]["total"] = round(time.time() - t0, 1)
    write_atomic({"provenance": pr, "stories": stories}, Path(a.out) / "ablate" / f"{a.tag}.json")
    log(f"ablate: {len(stories)} stories x {len(conds)} conditions ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage bind (BIND, P1, three queries)
@torch.no_grad()
def stage_bind(a):
    J, S, FJ, F = common(a, "bind")
    m = M(a)
    t0 = time.time()
    Ls = sorted(by_layer(cells(S["H"])))
    # the event, initial and distractor flags at one per-layer norm: the mean of their three norms
    nrm = {l: float(np.mean([float(F[k][l].norm()) for k in ("P1", "init", "dloc")])) for l in Ls}
    V = {k: {l: unit(F[k2][l]) * nrm[l] for l in Ls} for k, k2 in (("ev", "P1"), ("init", "init"), ("dloc", "dloc"))}
    V["iso"] = {l: unit(F["iso"][0][l]) * nrm[l] for l in Ls}
    pr = prov(a, m.model, m.dev, {"sets_sha256": J["sets_sha256"], "flags_sha256": FJ["pt_sha256"], "rows": list(BIND_ROWS),
                                  "views": list(BIND_VIEWS), "norm": nrm})
    stories = []
    for si, core in enumerate(cut(a, a.pops["P"]["BIND"], "BIND")):
        rec = dict(index=si, core=core, stratum=order_stratum(core), views={})
        for view in BIND_VIEWS:
            d = prep(m.tok, core, "P1", view)
            p, rX = d["p"], d["row"]["X"]
            caps = runs_kv(m, d["ids"], ["B", "X"], p)
            names = list(BIND_ROWS)
            spec = [[] if nm in ("none", "K_X") else [(rX, V[nm], 1.0)] for nm in names]
            kk, vv = kv_clamp(p, [ROWSRC.get(nm) for nm in names], caps, m.nL)
            out, _, _ = fwd(m, d["ids"]["B"], len(names), k=kk, v=vv, add=add_map(spec, Ls))
            rec["views"][view] = rec_rows(names, six(out["lp"], d["cid"]))
            rec["ix"] = d["ix"]
        stories.append(rec)
        if si % 10 == 0:
            log(f"bind story {si} ({time.time() - t0:.0f}s)")
    pr["timings"]["total"] = round(time.time() - t0, 1)
    write_atomic({"provenance": pr, "stories": stories}, Path(a.out) / "bind" / f"{a.tag}.json")
    log(f"bind: {len(stories)} stories ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage sign
def transfer_specs(S, G, Gc, T, allcells):
    """The 24 HeadSplice rows (12 sets x {K_S, K_X}): (cells, rows) per batch row."""
    Hs, rand = cells(S["H"]), [cells(r) for r in S["rand"]]
    allset = set(allcells)
    sets = {"none": ([], []), "H": (Hs, G), "rand0": (rand[0], G), "rand1": (rand[1], G), "rand2": (rand[2], G),
            "allG": ("all", G), "koH": (sorted(allset - set(Hs)), G), "korand0": (sorted(allset - set(rand[0])), G),
            "korand1": (sorted(allset - set(rand[1])), G), "korand2": (sorted(allset - set(rand[2])), G),
            "allGc": ("all", Gc), "allT": ("all", list(range(T)))}
    return [sets[n] for n in TRANSFER_SETS] * 2


def transfer(m, ids, p, keys, G, Gc, T, S, tid):
    """c = lp(S) - lp(X) per row of the 24-row batch (rows 0-11 with K_S, 12-23 with K_X)."""
    specs = transfer_specs(S, G, Gc, T, m.allcells)
    ks = {l: torch.stack([keys["S"][l]] * 12 + [keys["X"][l]] * 12) for l in range(m.nL)}
    masks = row_masks(specs, T, m.nL, m.H)
    out, _, _ = fwd(m, ids, len(specs), hs=dict(mode="splice", pos=p, ks=ks, masks=masks, mu=None))
    lp = out["lp"]
    c = (lp[:, tid["S"]] - lp[:, tid["X"]]).tolist()
    return {"S": dict(zip(TRANSFER_SETS, r5(c[:12]))), "X": dict(zip(TRANSFER_SETS, r5(c[12:])))}


def hop2(m, ids, T, G, inj_spec, S, layers, cand):
    """[none, inj, inj + the top-10 hop-2 heads at the answer row reading the clean keys at G, the same for 10 random
    heads, for every head]; returns the candidate log-probs per row."""
    _, cap, _ = fwd(m, ids, 1, kv=(G, range(m.nL), "k"))
    ks = {l: cap[(l, "k")][0] for l in range(m.nL)}
    specs = [([], []), ([], []), (cells(S["hop_top"]), [T - 1]), (cells(S["hop_rand"]), [T - 1]), ("all", [T - 1])]
    masks = row_masks(specs, T, m.nL, m.H)
    add = add_map([[] if i == 0 else inj_spec for i in range(len(specs))], layers)
    out, _, _ = fwd(m, ids, len(specs), add=add, hs=dict(mode="splice", pos=list(G), ks=ks, masks=masks, mu=None))
    return rec_rows(list(HOP2_ROWS), out["lp"][:, cand])


@torch.no_grad()
def stage_sign(a):
    J, S, FJ, F = common(a, "sign")
    m = M(a)
    t0 = time.time()
    Ls = sorted(by_layer(cells(S["H"])))
    D = F["P1"]
    pr = prov(a, m.model, m.dev, {"sets_sha256": J["sets_sha256"], "flags_sha256": FJ["pt_sha256"], "transfer_sets": list(TRANSFER_SETS),
                                  "hop2_rows": list(HOP2_ROWS), "ioi_route_rows": list(IOI_ROUTE)})
    res = {"Q_IN": [], "Q_OUT": [], "P1": [], "INLINE": [], "INLINE_CHAT": [], "AFTER": []}
    path = Path(a.out) / "sign" / f"{a.tag}.json"
    E8 = cut(a, a.pops["P"]["E8"], "E8")
    for arm in ("Q_IN", "Q_OUT", "P1"):
        for si, core in enumerate(E8):
            d = prep(m.tok, core, arm)
            p, rX, G = d["p"], d["row"]["X"], d["G"]
            rec = dict(index=si, core=core, ix=d["ix"])
            if arm != "P1":
                it = ff.run_item(m.model, m.tok, core, arm, "direct", m.dev)
                assert it is not None and it["pos"] == p
                mm, idr = it["m"], it["m"]["ID@0"]
                dl = {k: {t: mm[k]["lp"][t] - idr["lp"][t] for t in ("S", "B", "X")} for k in mm}
                rec["item"] = dict(idK=0.5 * ((dl["K_S@0"]["S"] - dl["K_X@0"]["S"]) + (dl["K_X@0"]["X"] - dl["K_S@0"]["X"])),
                                   idV=0.5 * ((dl["V_S@0"]["S"] - dl["V_X@0"]["S"]) + (dl["V_X@0"]["X"] - dl["V_S@0"]["X"])),
                                   mass=it["clean"]["B"]["mass"], argmax=it["clean"]["B"]["argmax_cand"])
                caps = runs_kv(m, d["ids"], ["B", "S", "X"], p)
                keys = {"S": caps["S"]["k"], "X": caps["X"]["k"]}
                tid = {"S": d["cid"][d["ix"]["S"]], "X": d["cid"][d["ix"]["X"]]}
                Gc = [d["row"][k] for k in ("B", "S", "X")]
                rec["transfer"] = transfer(m, d["ids"]["B"], p, keys, G, Gc, d["T"], S, tid)
                names = ["none", "none2", "K_S", "K_X", "V_S", "V_X", "inj", "iso0", "iso1", "iso2"]
                spec = [[(rX, D, 1.0)] if nm == "inj" else [(rX, F["iso"][int(nm[3])], 1.0)] if nm.startswith("iso") else [] for nm in names]
                kk, vv = kv_clamp(p, [ROWSRC.get(nm) for nm in names], caps, m.nL)
                out, _, _ = fwd(m, d["ids"]["B"], len(names), k=kk, v=vv, add=add_map(spec, Ls))
                rec["rows"] = rec_rows(names, six(out["lp"], d["cid"]))
            rec["hop2"] = hop2(m, d["ids"]["B"], d["T"], G, [(rX, D, 1.0)], S, Ls, d["cid"])
            res[arm].append(rec)
        log(f"sign {arm}: {len(E8)} stories ({time.time() - t0:.0f}s)")
        pr["timings"][arm] = round(time.time() - t0, 1)
        write_atomic({"provenance": pr, "results": res}, path)
    ip = ioi_pops(m.tok, a)
    pr["ioi"] = {"E_idx": ip["E_idx"]}
    pr["skipped_items"] += ip["skipped"]
    arms = ["INLINE", "INLINE_CHAT"] + (["AFTER"] if a.key == "qwen7" else [])
    for arm in arms:
        for si, core in enumerate(ip["E"]):
            L = Q.ioi_layout(m.tok, core, arm, True)
            it = Q.ioi_run_item(m.model, m.tok, core, arm, True, None, m.dev)
            assert it is not None and it["pos"] == L["p"]
            meas = ioi.identity_measures(it)
            rec = dict(index=ip["E_idx"][si], core=core, item={k: meas[k] for k in ("idK", "idV", "idKV", "id4K", "id4V", "two_B", "four_B", "LD_B", "mass_B")},
                       floor_B=it["floor_B"], floor_S=it["floor_S"])
            p, T, nm_ = L["p"], L["T"], L["named"]
            caps = runs_kv(m, L["ids"], ["B", "S", "X"], p)
            keys = {"S": caps["S"]["k"], "X": caps["X"]["k"]}
            tid = {"S": L["tid"]["S"], "X": L["tid"]["X"]}
            G = L["groups"]["options"]
            Gc = [nm_["io_b"], nm_["io_s"], nm_["io_x"]]
            rec["transfer"] = transfer(m, L["ids"]["B"], p, keys, G, Gc, T, S, tid)
            if arm == "INLINE":
                four = [L["tid"][k] for k in ("B", "S", "X", "Subj")]
                DI = F["IOI"]
                names = ["none", "none2", "K_S", "K_X", "injIOI", "moveIOI", "injP1", "iso"]
                spec = {"none": [], "none2": [], "K_S": [], "K_X": [], "injIOI": [(nm_["io_x"], DI, 1.0)],
                        "moveIOI": mv(DI, nm_["io_x"], nm_["io_b"]), "injP1": [(nm_["io_x"], D, 1.0)],
                        "iso": [(nm_["io_x"], norm_match(F["iso"][0], DI), 1.0)]}
                kk, vv = kv_clamp(p, [ROWSRC.get(x) for x in names], caps, m.nL)
                out, _, _ = fwd(m, L["ids"]["B"], len(names), k=kk, v=vv, add=add_map([spec[x] for x in names], Ls))
                rec["rows"] = rec_rows(names, out["lp"][:, four])
                # the route: K_S, and the IOI flag injected as K_S would write it (+ at IO_S, - at IO_B)
                inj = mv(DI, nm_["io_s"], nm_["io_b"])
                kk, vv = kv_clamp(p, [None, ROWSRC["K_S"], None], caps, m.nL)
                _, cap, _ = fwd(m, L["ids"]["B"], 3, k=kk, v=vv, add=add_map([[], [], inj], Ls), kv=(G, range(m.nL), "kv"))
                kvt = {nm: {key: val[i] for key, val in cap.items()} for i, nm in enumerate(("base", "ksrun", "injrun"))}
                rn = list(IOI_ROUTE)
                kp = configure_hop(m.hop, rn, kvt, G, T, IOI_ROUTE_SPEC)
                kk, vv = kv_clamp(p, [ROWSRC["K_S"] if b else None for b in kp.tolist()], caps, m.nL)
                out, _, _ = fwd(m, L["ids"]["B"], len(rn), k=kk, v=vv,
                                add=add_map([inj if x.startswith("inj") else [] for x in rn], Ls), hop=True)
                rec["route"] = rec_rows(rn, out["lp"][:, four])
                rec["hop2"] = hop2(m, L["ids"]["B"], T, G, [(nm_["io_x"], DI, 1.0)], S, Ls, four)
            res[arm].append(rec)
        log(f"sign {arm}: {len(ip['E'])} cores ({time.time() - t0:.0f}s)")
        pr["timings"][arm] = round(time.time() - t0, 1)
        write_atomic({"provenance": pr, "results": res}, path)
    pr["n_double_passes"] = {"headsplice": m.hs.n_double}
    write_atomic({"provenance": pr, "results": res}, path)


# --------------------------------------------------------------------------- stage diss (D6)
@torch.no_grad()
def stage_diss(a):
    J, S, FJ, F = common(a, "diss")
    m = M(a)
    t0 = time.time()
    Hs = cells(S["H"])
    byH = by_layer(Hs)
    Ls = sorted(byH)
    Ws = m.Ws(Ls)
    fs = fs_of(m.tok)
    n = 2 if a.test else N_DISS[a.key]
    E = a.pops["P"]["E8"][:n]
    D = F["P1"]
    pr = prov(a, m.model, m.dev, {"sets_sha256": J["sets_sha256"], "flags_sha256": FJ["pt_sha256"], "rows": list(DISS_ROWS),
                                  "route_rows": list(DISS_ROUTE), "trie_nodes": len(fs), "n": n})
    stories = []
    for si, core in enumerate(E):
        rec = dict(index=si, core=core, fmt={})
        for f in ("P1", "POST"):
            d = prep(m.tok, core, f)
            p, rX, rB, rS = d["p"], d["row"]["X"], d["row"]["B"], d["row"]["S"]
            run3 = ["B", "S", "X"]
            o1, cap, _ = fwd(m, torch.cat([d["ids"][r] for r in run3]), fs=fs, kv=([p], range(m.nL), "kv"))
            caps = {r: {ch: {l: cap[(l, ch)][i, 0] for l in range(m.nL)} for ch in "kv"} for i, r in enumerate(run3)}
            keys = {"S": caps["S"]["k"], "X": caps["X"]["k"]}
            zb, zk = write_pass(m, d, [("B", None, None), ("B", keys["S"], p)], [rS, rB], Ls)
            delta = flag_delta(Ws, zk, zb, 0, 1, byH, m.hd)
            W = {l: float(delta[l].cpu() @ unit(D[l])) for l in Ls}
            names = list(DISS_ROWS)
            spec = {"none": [], "none2": [], "injP1": [(rX, D, 1.0)], "injPOST": [(rX, F["POST"], 1.0)],
                    "iso": [(rX, F["iso"][0], 1.0)], "K_X": [], "K_S": []}
            kk, vv = kv_clamp(p, [ROWSRC.get(x) for x in names], caps, m.nL)
            out, _, _ = fwd(m, d["ids"]["B"], len(names), k=kk, v=vv, add=add_map([spec[x] for x in names], Ls), fs=fs)
            r = dict(rows=rec_rows(names, out["E"]), rowsL=rec_rows(names, out["L"]), cleanB=r5(o1["E"][0]), cleanS=r5(o1["E"][1]), W=W)
            if f == "POST":
                rn = list(DISS_ROUTE)
                _, cap2, _ = fwd(m, d["ids"]["B"], 2, add=add_map([[], [(rX, D, 1.0)]], Ls), kv=(d["G"], range(m.nL), "kv"))
                kvt = {"base": {kk: vv[0] for kk, vv in cap2.items()}, "ksrun": {kk: vv[1] for kk, vv in cap2.items()}}
                configure_hop(m.hop, rn, kvt, d["G"], d["T"], ROUTE_SPEC | {"inj": ROUTE_SPEC["move"]})
                outr, _, _ = fwd(m, d["ids"]["B"], len(rn), add=add_map([[] if x == "none" else [(rX, D, 1.0)] for x in rn], Ls),
                                 hop=True, fs=fs)
                r["route"] = rec_rows(rn, outr["E"])
            if f == "P1" and a.key in SMALL:      # Gate J-D-G6 (iii): the in-run H* carries the P1 read (R(k*))
                specs = [([], []), (Hs, d["G"]), ("all", d["G"])] * 2
                ks = {l: torch.stack([keys["S"][l]] * 3 + [keys["X"][l]] * 3) for l in range(m.nL)}
                og, _, _ = fwd(m, d["ids"]["B"], 6, hs=dict(mode="splice", pos=p, ks=ks, masks=row_masks(specs, d["T"], m.nL, m.H), mu=None))
                c = (og["lp"][:, d["cid"][d["ix"]["S"]]] - og["lp"][:, d["cid"][d["ix"]["X"]]]).tolist()
                r["gate"] = {"S": dict(zip(("none", "H", "allG"), r5(c[:3]))), "X": dict(zip(("none", "H", "allG"), r5(c[3:])))}
            rec["fmt"][f] = r
        rec["ix"] = d["ix"]
        stories.append(rec)
        if si % 10 == 0:
            log(f"diss story {si} ({time.time() - t0:.0f}s)")
    pr["timings"]["total"] = round(time.time() - t0, 1)
    write_atomic({"provenance": pr, "stories": stories, "kappa_POST_P1": wcos(D, F["POST"]),
                  "norms_P1": {l: float(D[l].norm()) for l in Ls}}, Path(a.out) / "diss" / f"{a.tag}.json")
    log(f"diss: {len(stories)} stories ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage before (exploratory, qwen7)
@torch.no_grad()
def stage_before(a):
    path = Path(a.out) / "before" / f"{a.tag}.json"
    if deadline_passed():
        write_atomic({"provenance": {"skipped": "deadline passed"}}, path)
        log("before skipped: the deadline has passed")
        return
    model, tok, dev = load_model(a)
    t0 = time.time()
    nL = len(blocks(model))
    ip = ioi_pops(tok, a)
    cores = ip["E"][:2 if a.test else SIZES["BEFORE"]]
    pr = prov(a, model, dev, {"n": len(cores), "E_idx": ip["E_idx"][:len(cores)]})
    items = []
    for core in cores:
        L = Q.ioi_layout(tok, core, "BEFORE", True)
        p, T, tid = L["p"], L["T"], L["tid"]
        rec = {"core": core, "p": p, "options": L["groups"]["options"], "choices": L["groups"]["choices"], "ko": {}}
        for name, rows in (("none", []), ("names", L["groups"]["options"]), ("list", L["groups"]["choices"])):
            pairs = [(p, c) for c in rows if c != 0]
            mk = lambda B: mask_for_model(model, T, pairs, B).to(dev)  # noqa: E731
            K = {}                   # K and V at p of the S, X and B runs, each under the same knockout
            for r in ("S", "X", "B"):
                with capture_kv(model, [p], range(nL), "kv") as C:
                    model(L["ids"][r].to(dev), attention_mask=mk(1), use_cache=False, logits_to_keep=1)
                K[r] = {key: C[key][0, 0].clone() for key in C}
            use = torch.tensor([False, True, True])
            tk = {(l, "k"): torch.stack([K["B"][(l, "k")], K["S"][(l, "k")], K["X"][(l, "k")]])[:, None] for l in range(nL)}
            tv = {(l, "v"): torch.stack([K["B"][(l, "v")]] * 3)[:, None] for l in range(nL)}   # the value held at B's
            with clamp_kv(model, [p], tk, range(nL), "k", per_row=use), clamp_kv(model, [p], tv, range(nL), "v", per_row=use):
                lp = torch.log_softmax(model(L["ids"]["B"].to(dev).expand(3, -1), attention_mask=mk(3), use_cache=False,
                                             logits_to_keep=1).logits[:, -1].float(), -1).cpu()
            lpk = {k: lp[:, tid[k]] for k in ("S", "X", "B", "Subj")}
            idK = 0.5 * ((lpk["S"][1] - lpk["S"][2]) + (lpk["X"][2] - lpk["X"][1]))
            rec["ko"][name] = {"idK": float(idK), "lp": {k: r5(v) for k, v in lpk.items()}}
        items.append(rec)
    pr["timings"]["knockout"] = round(time.time() - t0, 1)
    out = {"provenance": pr, "knockout": items}
    write_atomic(out, path)
    from experiments.row_restricted_keys import run as rs_run
    task = ioi.RowTask(chat=True)
    task.cores = lambda n, seed=None: cores[:n]          # the same E_ioi cores
    res = rs_run(model, tok, task, ["BEFORE"], len(cores), SEEDS["E_IOI"], 0, log)
    out["rowsplice"] = res
    pr["timings"]["rowsplice"] = round(time.time() - t0, 1)
    write_atomic(out, path)
    log(f"before: {len(items)} cores ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage xtask (exploratory)
def prep_task(tok, task, core):
    T_ = TASKS[task]
    vals = T_.values
    cid = [tok.encode(" " + v, add_special_tokens=False)[0] for v in vals]
    used = {core["base"], core["source"], core["initial"], core["distractor_location"]}
    pool = [v for v in vals if v not in used]
    X = random.Random(int(hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8], 16)).choice(pool)
    E = {k: encode_with_offsets(tok, T_.raw_prompt("P1", T_.story(core, v), T_.question(core))) for k, v in
         (("B", core["base"]), ("S", core["source"]), ("X", X))}
    text, ib, off = E["B"]
    assert all(e[1].shape == ib.shape for e in E.values())
    diff = (ib[0] != E["S"][1][0]).nonzero().flatten().tolist()
    assert len(diff) == 1 and (ib[0] != E["X"][1][0]).nonzero().flatten().tolist() == diff
    p = diff[0]
    seg = "Choices: " + ", ".join(vals)
    c0 = text.index(seg)
    G = [i for i in rows_in(off, c0, c0 + len(seg)) if ib[0, i].item() in set(cid)]
    assert len(G) == len(vals) and ib[0, G].tolist() == cid and min(G) > p, (task, G)
    ix = {"B": vals.index(core["base"]), "S": vals.index(core["source"]), "X": vals.index(X)}
    return dict(ids={k: e[1] for k, e in E.items()}, p=p, G=G, T=ib.shape[1], cid=cid, ix=ix, X=X,
                row={k: G[i] for k, i in ix.items()})


@torch.no_grad()
def stage_xtask(a):
    path = Path(a.out) / "xtask" / f"{a.tag}.json"
    if deadline_passed():
        write_atomic({"provenance": {"skipped": "deadline passed"}}, path)
        log("xtask skipped: the deadline has passed")
        return
    J, S, FJ, F = common(a, "xtask")
    m = M(a)
    t0 = time.time()
    byH = by_layer(cells(S["H"]))
    Ls = sorted(byH)
    Ws = m.Ws(Ls)
    pr = prov(a, m.model, m.dev, {"sets_sha256": J["sets_sha256"], "flags_sha256": FJ["pt_sha256"]})
    out = {"provenance": pr, "tasks": {}}
    for t in ("paint", "schedule"):
        if any(len(m.tok.encode(" " + v, add_special_tokens=False)) != 1 for v in TASKS[t].values):
            out["tasks"][t] = {"skipped": "values are not single tokens"}
            continue
        deltas = []
        for core in cut(a, a.pops["P"][f"XFIT_{t}"], "XFIT"):
            d = prep_task(m.tok, t, core)
            ks = keys_at(m, [d["ids"]["S"]], [d["p"]])["k"][0]
            zb, zk = write_pass(m, d, [("B", None, None), ("B", ks, d["p"])], [d["row"]["S"], d["row"]["B"]], Ls)
            deltas.append({l: v.cpu() for l, v in flag_delta(Ws, zk, zb, 0, 1, byH, m.hd).items()})
        DT = flag_of(deltas)
        rows = []
        for core in cut(a, a.pops["P"][f"XEVAL_{t}"], "XEVAL"):
            d = prep_task(m.tok, t, core)
            rX = d["row"]["X"]
            caps = runs_kv(m, d["ids"], ["B", "X"], d["p"])
            names = ["none", "own", "belief", "iso", "K_X"]
            spec = {"none": [], "own": [(rX, DT, 1.0)], "belief": [(rX, F["P1"], 1.0)], "iso": [(rX, norm_match(F["iso"][0], DT), 1.0)], "K_X": []}
            kk, vv = kv_clamp(d["p"], [ROWSRC.get(x) for x in names], caps, m.nL)
            o, _, _ = fwd(m, d["ids"]["B"], len(names), k=kk, v=vv, add=add_map([spec[x] for x in names], Ls))
            rows.append(dict(core=core, ix=d["ix"], rows=rec_rows(names, o["lp"][:, d["cid"]])))
        out["tasks"][t] = {"stories": rows, "wcos_task_belief": wcos(DT, F["P1"]),
                           "norms": {l: float(DT[l].norm()) for l in Ls}}
        pr["timings"][t] = round(time.time() - t0, 1)
        write_atomic(out, path)
    write_atomic(out, path)
    log(f"xtask done ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- main
STAGES = {"preflight": stage_preflight, "sets": stage_sets, "fit": stage_fit, "inject": stage_inject, "ablate": stage_ablate,
          "bind": stage_bind, "sign": stage_sign, "diss": stage_diss, "before": stage_before, "xtask": stage_xtask}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--model", default=None, help="a local verified directory (s8_fetch) or a Hub id")
    ap.add_argument("--key", required=True, choices=KEYS, help="the model key of scripts/stage8_models.json")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--heads-dir", default="results/gpu_stage6/heads")
    ap.add_argument("--out", default="results/gpu_stage8d")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B FP32 on the CPU, n = 2")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    if a.test:
        a.model, a.revision, a.dtype = TINY, None, "float32"
    assert a.model, "--model is required"
    a.tag = ("TEST_" if a.test else "") + a.key
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    a.pops = populations(check=True)
    rc = STAGES[a.stage](a)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
