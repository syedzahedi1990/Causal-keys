"""Stage 8, part C of P-2026-10-10-J (docs/PREREGISTRATION.md): independently obtained identity edits at the writing token
and the channel-ratio law. Material and families are defined in ckeys/edits.py (populations E, H, TSET, THOLD; families
T, R, E1-E5; the PAR / PERP and LEX / NONLEX components), ckeys/sae.py (E3), ckeys/das_at.py (E4), ckeys/neutral.py (the 24
sentences). Prompts: ckeys.encoding.raw_prompt in LETTER, P1, POST, NONE with chat_text(..., prefill="Answer:").

Stages (each a separate process; BF16, use_cache=False in every scoring pass; scripts/gpu_stage8c.sh):
  preflight  tokenizer only: the populations and their pinned hashes, disjointness from U and from each other; prep() of
             every E core (one length, one differing position p, the same p and the same prefix through p in all four
             formats, also for pi(S) and pi(X)) and of every H core; the neutral sentences (sha256, every family clean in
             >= 20 sentences, English in all 24); the E form sets with the fixed frames; the SAE pins against the Hub
             (qwen7; recorded, a network failure is not fatal: the download asserts the hashes) -> preflight/<tag>.json
  calib      on H only: frame discovery (greedy generations of the clean B, S, X runs of H_cal in P1, POST and NONE;
             ckeys.generate.discover_frames; a frame is admitted only if every H_cal and E form set still builds); E1
             means over H_fit; E2 and E5 means over the sentences; the lexical spans; the random directions; at qwen7 the
             SAE (download, sha256, J-C-G3 FVE at blocks l-1, l, l+1 over every prompt position >= 1 of the clean B NONE
             prompts of H, FVE at p, selection on H_fit, the k_F / beta rule on H_cal); the E5 alpha rule on H_cal under
             NONE. calib/<tag>.pt is written first and its sha256 recorded in calib/<tag>.json.
  dasfit     qwen7, mistral7: E4 fits at l in DEPTHS, seeds 101 (PCA init) and 102 (random init), on TSET (NONE); the
             NONE flip rate on THOLD -> das/<tag>.pt (bases) and das/<tag>.json (losses, flip rates, the .pt sha256)
  eval       every E story: prefix captures of the natural runs and of every edit vector at every depth, then per format
             and depth one clamp-row batch (chunks of --chunk rows, each chunk led by the self row), scored with
             ckeys.surface.score (E: the 32 fixed frames plus the discovered ones; letters under LETTER) and L -> eval/<tag>.json
             (resumable story by story; exit 3 when the deadline minus --reserve-min is reached, so the next session goes on)
  readers    qwen7, mistral7: J-C-READ at P1, l = 7: HeadSplice x_S rows (the stage-7 construction with ks = the donor's
             key from block 8 and the KeyPin / value pin holding B's K and V at p) for the natural donor and E1, E2, E3,
             E4 (seed 101), targets S and X, blinded sets H*_{>7} (stage 6's top-k* a3 heads under OPTIONS-AFTER in blocks
             > 7) and three size-matched random sets of blocks > 7 (numpy default_rng(2)); scored with L -> readers/<tag>.json
  explore    exploratory: greedy-generation flip rates of the KV rows at l = 7 under NONE and P1 (first 40 stories)
TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B-Instruct FP32 on the CPU; n = 2 E stories, H_fit 3, H_cal 2, TSET 20,
THOLD 4; a random SAE (d x 1024, seed 0) stands in for the dictionary; outputs tagged TEST_<key>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ckeys import das_at, edits, neutral, sae
from ckeys.clamp import clamp_kv
from ckeys.encoding import LISTING, WRAPPER_USED, candidate_ids, chat_text, raw_prompt
from ckeys.generate import discover_frames, extract_frame, greedy, candidate_stop, parse_answer
from ckeys.headsplice import HeadSplice
from ckeys.interventions import blocks, fixed_subspace_interchange
from ckeys.readerblind import KeyPin, blind_masks
from ckeys.story import LOCATIONS, PAIR_SWAP, pick_x, record
from ckeys.surface import FRAMES_E_FIXED, FormSet
from experiments.format_factorial import provenance as ff_provenance
from experiments.ioi_factorial import device_name
from experiments.row_restricted_keys import rows_in
from experiments.stage6_heads import write_atomic

TINY = "Qwen/Qwen2.5-0.5B-Instruct"
DEPTHS = (3, 7, 11, 15)
TEST_DEPTHS = (7,)          # TEST_MODE: one depth (the reader depth), so the CPU plumbing run stays within the hour
TEST_SENTENCES = 4          # TEST_MODE: the first four neutral sentences
FAMILIES = {"qwen7": ("E1", "E2", "E3", "E4", "E5"), "mistral7": ("E1", "E2", "E4", "E5"), "llama8": ("E1", "E2", "E5"),
            "yi9": ("E1", "E2", "E5")}    # yi9: the fallback of the new family (rule G2), in llama8's place
E5_SUB = ("FR", "DE", "SYN", "NL")
SYNTH = (("SK", 0.5), ("SK", 0.8), ("SV", 0.5), ("SV", 0.8))
DAS_SEEDS = (101, 102)
READ_L = 7
HEADS_FILES = {"qwen7": "Qwen2.5-7B-Instruct.json", "mistral7": "Mistral-7B-Instruct-v0.3.json"}
CAL_FORMATS = ("P1", "POST", "NONE")
EXPLORE_N, EXPLORE_FORMATS = 40, ("NONE", "P1")
KF_MIN_FLIP, E5_MIN_PHI = 0.7, 0.3
MAX_NEW = 16


def log(s):
    print(s, flush=True)


def sha_file(path) -> str:
    return sae.sha_file(path)


def canon(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def deadline_left(a) -> float:
    """Seconds left before STAGE8_DEADLINE minus the reserve the later steps need (inf without a deadline)."""
    d = os.environ.get("STAGE8_DEADLINE")
    return float("inf") if not d else float(d) - time.time() - 60.0 * a.reserve_min


def r5(x):
    return [round(float(v), 5) for v in x] if isinstance(x, (list, tuple)) else round(float(x), 5)


# --------------------------------------------------------------------------- model and provenance
def load_model(a, need_model=True):
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    attn = a.attn if a.attn != "auto" else ("eager" if cfg.model_type.startswith("gemma2") else "sdpa")
    dev = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if not need_model:
        return None, tok, dev
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
    p = ff_provenance(a) | {"model": a.model, "model_key": a.key, "revision": a.revision, "verified": verified(a.model),
                            "test_mode": a.test, "stage": a.stage, "families": list(FAMILIES[a.key]), "depths": list(a.depths),
                            "populations": edits.POP_SHA256, "U_sha256": edits.U_SHA256,
                            "neutral_sha256": neutral.SENTENCES_SHA256, "timings": {}, "skipped_items": []}
    if model is not None:
        cfg = model.config
        p |= {"dtype": str(next(model.parameters()).dtype), "attn_implementation": cfg._attn_implementation,
              "device": device_name(dev), "n_layers": len(blocks(model)), "n_heads": cfg.num_attention_heads,
              "n_kv_heads": cfg.num_key_value_heads, "hidden": cfg.hidden_size}
    p["wrapper"] = dict(WRAPPER_USED)
    return p | (extra or {})


def pops(a):
    P = edits.populations(check=True)
    if a.test:
        P = {"E": P["E"][:a.n or 2], "H": P["H"][:3] + P["H"][-2:], "TSET": P["TSET"][:20], "THOLD": P["THOLD"][:4]}
    elif a.n:
        P["E"] = P["E"][:a.n]
    return P


def hfit_hcal(a, H):
    return (H[:3], H[3:]) if a.test else (H[:edits.N_HFIT], H[edits.N_HFIT:])


def out_path(a, sub, ext="json"):
    return Path(a.out) / sub / f"{a.tag}.{ext}"


# --------------------------------------------------------------------------- stage preflight (tokenizer only)
def stage_preflight(a):
    t0 = time.time()
    _, tok, _ = load_model(a, need_model=False)
    P = pops(a)
    U = edits.universe()
    keys = {k: {edits.key(c) for c in v} for k, v in P.items()}
    overlap = {"U": {k: len(v & U) for k, v in keys.items()},
               "pairwise": {f"{x}-{y}": len(keys[x] & keys[y]) for i, x in enumerate(keys) for y in list(keys)[i + 1:]}}
    bad = [k for k, v in overlap["U"].items() if v] + [k for k, v in overlap["pairwise"].items() if v]
    assert not bad, f"populations not disjoint: {overlap}"
    skipped, ps = [], {}
    for name in ("E", "H"):
        for i, c in enumerate(P[name]):
            try:
                d = edits.prep(tok, c) if name == "E" else edits.prep(tok, c, {"B": c["base"]} | {f"v{j}": x for j, x in enumerate(LOCATIONS) if x != c["base"]})
                ps.setdefault(name, []).append(d["p"])
            except AssertionError as ex:
                skipped.append({"population": name, "index": i, "reason": repr(ex)[:300]})
    assert not skipped, f"cores that fail the single-position / shared-prefix rule: {skipped[:3]}"
    nrep = neutral.check(tok)
    assert neutral.sentences_sha256() == neutral.SENTENCES_SHA256, "the neutral sentences differ from their pinned hash"
    assert nrep["EN"]["n"] == len(neutral.SENTENCES), "an English location word is not clean in every neutral sentence"
    unusable = [f for f, r in nrep.items() if not r["usable"]]
    fs_fail = []
    for c in P["E"]:
        for f in edits.FORMATS:
            try:
                fs = edits.form_set(tok, f, c)
                bad_dec = fs.check_decode()
                if bad_dec:
                    fs_fail.append((edits.key(c), f, bad_dec[:2]))
            except ValueError as ex:
                fs_fail.append((edits.key(c), f, str(ex)))
    assert not fs_fail, f"form sets that do not build or decode: {fs_fail[:3]}"
    sae_check = None
    if "E3" in FAMILIES[a.key]:
        sae_check = hub_sae_check()
    out = {"provenance": prov(a), "sizes": {k: len(v) for k, v in P.items()}, "hashes": {k: edits.pop_hash(v) for k, v in P.items()},
           "overlap": overlap, "p_positions": {k: sorted(set(v)) for k, v in ps.items()}, "neutral": nrep,
           "e5_unusable": unusable, "sae_hub": sae_check, "skipped_items": skipped, "seconds": round(time.time() - t0, 1)}
    write_atomic(out, out_path(a, "preflight"))
    log(f"preflight OK: E {len(P['E'])}, H {len(P['H'])}, TSET {len(P['TSET'])}, THOLD {len(P['THOLD'])}; neutral families "
        f"{ {f: r['n'] for f, r in nrep.items()} }; E5 unusable {unusable}; SAE hub {sae_check and sae_check.get('status')}")


def hub_sae_check():
    """The LFS sha256 of each pinned ae.pt on the Hub at the pinned revision (best effort: a network failure is recorded)."""
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_tree(sae.REPO, revision=sae.REVISION, recursive=True, expand=True)
        meta = {f.path: f for f in files if hasattr(f, "size")}
        got = {l: getattr(meta.get(f"resid_post_layer_{l}/{sae.TRAINER}/ae.pt"), "lfs", None) for l in sae.LAYERS}
        got = {l: (v.sha256 if v is not None else None) for l, v in got.items()}
        bad = [l for l in sae.LAYERS if got[l] != sae.PINS[l]["ae.pt"]]
        assert not bad, f"the Hub's ae.pt sha256 at {sae.REVISION} differs from the pins at layers {bad}"
        return {"status": "OK", "sha256": got}
    except AssertionError:
        raise
    except Exception as ex:  # noqa: BLE001
        return {"status": f"not checked: {type(ex).__name__}: {str(ex)[:200]}"}


# --------------------------------------------------------------------------- shared: H runs, rows, ID on calibration rows
def id_contrast(rows, kind, iS, iX, field="dE"):
    """ID of one instance from score_rows output: 1/2[(d_S(Z_S) - d_S(Z_X)) + (d_X(Z_X) - d_X(Z_S))]."""
    zS, zX = rows[f"{kind}|S"][field], rows[f"{kind}|X"][field]
    return 0.5 * ((zS[iS] - zX[iS]) + (zX[iX] - zS[iX]))


def kv_rows(name, tabZ, tabB, layers):
    """The (name, ktab, vtab) of the KV row of an instance (both channels from its tables)."""
    return (name, {q: tabZ[(q, "k")] for q in layers}, {q: tabZ[(q, "v")] for q in layers})


def split_tables(kv, i, layers):
    return {(q, ch): kv[(q, ch)][i] for q in layers for ch in "kv"}


class Calib:
    """The calibration file's tensors, loaded with its sha256 checked against calib/<tag>.json."""

    def __init__(self, a):
        J = json.loads(out_path(a, "calib").read_text())
        f = out_path(a, "calib", "pt")
        h = sha_file(f)
        assert h == J["calib_pt_sha256"], f"{f}: sha256 {h} != the one calib.json recorded"
        self.J, self.sha, self.T = J, h, torch.load(f, weights_only=False)
        self.frames = J["frames"]


def e3_vectors_for(sae_l: sae.BatchTopK, hB, t, b, cal_l):
    F = [torch.tensor(x, dtype=torch.long) for x in cal_l["F"]]
    abar = cal_l["abar"].float()
    dev = sae_l.W_dec.device
    v = sae.edit(sae_l, hB.float().to(dev), t, b, [x.to(dev) for x in F], abar.to(dev), cal_l["beta"])
    return v.float().cpu()


def load_sae(a, l, dev):
    """The dictionary of block l: the pinned file (downloaded and hashed) or, in TEST_MODE, a random one."""
    if a.test:
        return sae.BatchTopK.random(a.hidden, 1024, seed=l, device=dev), {"test_random": True}
    rec = sae.fetch(l, Path(a.sae_dir))
    return sae.BatchTopK.load(rec["ae.pt"]["path"], device=dev), rec


# --------------------------------------------------------------------------- stage calib
@torch.no_grad()
def stage_calib(a):
    model, tok, dev = load_model(a)
    a.hidden = model.config.hidden_size
    P = pops(a)
    Hfit, Hcal = hfit_hcal(a, P["H"])
    E = P["E"]
    nL = len(blocks(model))
    t0 = time.time()
    pr = prov(a, model, dev)
    J, T = {"provenance": pr}, {}
    # ---- frames (H_cal, clean B, S, X runs in P1, POST, NONE)
    recs, gens = [], []
    stop = candidate_stop(tok)
    for c in Hcal:
        d = edits.prep(tok, c, {"B": c["base"], "S": c["source"], "X": pick_x(c)})
        for f in CAL_FORMATS:
            ids = torch.cat([d["ids"][f][k] for k in ("B", "S", "X")])
            out = greedy(model, tok, ids, MAX_NEW, stop=stop)
            for k, g in zip(("B", "S", "X"), out):
                text = tok.decode(g)
                fr = extract_frame(text, edits.names_of(c))
                recs.append((f, fr))
                gens.append({"format": f, "run": k, "text": text, "frame": fr})
    cand = discover_frames(recs, FRAMES_E_FIXED)
    frames, dropped = [], []
    for fr in cand:   # admitted in order only if every H_cal and E form set builds with it
        try:
            for c in list(Hcal) + list(E):
                for f in ("P1", "POST", "NONE"):
                    edits.form_set(tok, f, c, frames + [fr])
            frames.append(fr)
        except ValueError as ex:
            dropped.append({"frame": fr, "reason": str(ex)[:200]})
    J["frames"], J["frames_candidates"], J["frames_dropped"] = frames, cand, dropped
    J["frame_census"] = {f: {} for f in CAL_FORMATS}
    for f, fr in recs:
        J["frame_census"][f][str(fr)] = J["frame_census"][f].get(str(fr), 0) + 1
    J["generations"] = gens
    pr["timings"]["frames"] = round(time.time() - t0, 1)
    log(f"frames: {len(frames)} admitted {frames}, dropped {len(dropped)} ({time.time() - t0:.0f}s)")
    # ---- means, spans, random directions
    T["mu1"] = edits.class_means(model, tok, Hfit, a.depths)
    T["mu5"] = {}
    for fam in neutral.FORMS:
        idx = neutral.clean_sentences(tok, fam)[:TEST_SENTENCES if a.test else None]
        if len(idx) < (TEST_SENTENCES if a.test else neutral.MIN_SENTENCES):
            J.setdefault("e5_unusable", []).append(fam)
            continue
        T["mu5"][fam] = edits.lexical_means(model, tok, fam, a.depths, idx, min_sentences=len(idx) if a.test else None)
        J.setdefault("sentences_used", {})[fam] = idx
    T["Q"] = {l: edits.lexical_span(T["mu5"]["EN"][l]) for l in a.depths}
    T["R"] = edits.random_dirs(a.hidden, a.depths)
    pr["timings"]["means"] = round(time.time() - t0, 1)
    log(f"means and spans ({time.time() - t0:.0f}s)")
    # ---- H_cal items, natural runs (for the E5 alpha rule and the E3 k_F / beta rule)
    cal = []
    for c in Hcal:
        X = pick_x(c)
        d = edits.prep(tok, c, {"B": c["base"], "S": c["source"], "X": X})
        pre = torch.cat([d["ids"]["NONE"][k][:, :d["p"] + 1] for k in ("B", "S", "X")])
        res, kv = edits.prefix_pass(model, pre, d["p"], resid_layers=a.depths, kv_layers=range(min(a.depths) + 1, nL))
        cal.append((c, d, res, kv, edits.form_set(tok, "NONE", c, frames)))
    # ---- E3: SAE gates, selection, the k_F / beta rule
    if "E3" in FAMILIES[a.key]:
        J["sae"], T["sae"] = {}, {}
        for l in a.depths:
            S, rec = load_sae(a, l, dev)
            g = {"files": rec}
            g["fve"] = sae_fve(model, tok, S, P["H"], l, nL)
            g["published_fve"] = sae.PINS[l]["fve"] if not a.test else None
            # selection on H_fit: abar [6, m] over h_x,l(p)
            hx = h_values(model, tok, Hfit, l)                       # [n, 6, D]
            acts = S.encode(hx.reshape(-1, hx.shape[-1]).to(dev)).reshape(hx.shape[0], 6, -1)
            abar = acts.mean(0).float().cpu()
            g["fve_at_p"] = sae.fve(S, hx.reshape(-1, hx.shape[-1]).to(dev))
            choice, table = None, []
            for kF, beta in [(k, 1.0) for k in sae.KF_GRID] + [(64, float(b)) for b in sae.BETA_GRID]:
                F = sae.select(abar, kF)
                fr = e3_cal_flip(model, S, F, abar, beta, l, cal, nL, a)
                table.append({"kF": kF, "beta": beta, "flip": fr, "sizes": [len(x) for x in F]})
                log(f"  E3 l={l} kF={kF} beta={beta}: H_cal flip-to-target {fr:.3f}")
                if fr >= KF_MIN_FLIP:
                    choice = (kF, beta)
                    break
            g["rule"] = table
            g["ineffective_at_calibration"] = choice is None
            kF, beta = choice or (64, float(sae.BETA_GRID[-1]))
            F = sae.select(abar, kF)
            g["kF"], g["beta"] = kF, beta
            T["sae"][l] = {"F": [x.tolist() for x in F], "abar": abar.half(), "beta": beta, "kF": kF}
            J["sae"][str(l)] = g
            del S
            if dev == "cuda":
                torch.cuda.empty_cache()
            log(f"SAE l={l}: FVE {g['fve']}, at p {g['fve_at_p']:.3f}; kF {kF} beta {beta} ({time.time() - t0:.0f}s)")
    # ---- E5: alpha in {1, 2} on H_cal under NONE (phi >= 0.3 with alpha = 1, else 2)
    J["e5_alpha"], J["e5_cal"] = {}, {}
    for sub in E5_SUB:
        if sub != "NL" and sub not in T["mu5"]:
            continue
        J["e5_alpha"][sub], J["e5_cal"][sub] = {}, {}
        for l in a.depths:
            phis = {}
            for alpha in (1.0, 2.0):
                phis[alpha] = e5_cal_phi(model, T, sub, alpha, l, cal, nL)
                if phis[alpha] >= E5_MIN_PHI:
                    break
            alpha = 1.0 if phis.get(1.0, -1) >= E5_MIN_PHI else 2.0
            J["e5_alpha"][sub][str(l)] = alpha
            J["e5_cal"][sub][str(l)] = {str(k): v for k, v in phis.items()}
        log(f"E5 {sub}: alpha {J['e5_alpha'][sub]} phi {J['e5_cal'][sub]}")
    pr["timings"]["total"] = round(time.time() - t0, 1)
    f = out_path(a, "calib", "pt")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".pt.tmp")
    torch.save(T, tmp)
    os.replace(tmp, f)
    J["calib_pt_sha256"] = sha_file(f)
    pr["wrapper"] = dict(WRAPPER_USED)
    write_atomic(J, out_path(a, "calib"))
    log(f"wrote {f} sha256 {J['calib_pt_sha256']} and calib/{a.tag}.json ({time.time() - t0:.0f}s)")


def h_values(model, tok, cores, l):
    """[n, 6, D] h_x,l(p) for every value x (LOCATIONS order) written at p."""
    out = []
    for c in cores:
        d = edits.prep(tok, c, {"B": c["base"]} | {f"v{j}": x for j, x in enumerate(LOCATIONS) if x != c["base"]})
        pre = torch.cat([d["ids"]["NONE"][k][:, :d["p"] + 1] for k in d["vals"]])
        res, _ = edits.prefix_pass(model, pre, d["p"], resid_layers=[l])
        h = torch.zeros(6, res[l].shape[-1])
        h[[LOCATIONS.index(v) for v in d["vals"].values()]] = res[l]
        out.append(h)
    return torch.stack(out)


@torch.no_grad()
def sae_fve(model, tok, S, H, l, nL):
    """J-C-G3: FVE of the block-l dictionary on the block outputs l-1, l and l+1 over every prompt position >= 1 of the
    clean B NONE prompts of H (position 0, the attention sink, is left out)."""
    dev = next(model.parameters()).device
    acc = {q: sae.FVE() for q in (l - 1, l, l + 1) if 0 <= q < nL}
    from ckeys.interventions import capture
    for c in H:
        ids = edits.prompt_ids(tok, c, "NONE", c["base"]).to(dev)
        with capture(model, list(acc), "resid") as st:
            model(ids, use_cache=False, logits_to_keep=1)
        for q, f in acc.items():
            h = st[q][0, 1:].float().to(S.W_dec.device)
            f.add(h, S.decode(S.encode(h)))
    return {str(q - l): f.value() for q, f in acc.items()}


def cal_rows(model, c, d, res, kv, fs, l, vecs, nL):
    """KV rows of calibration vectors (name -> [D]) at depth l on one H_cal story under NONE (with the natural rows)."""
    layers = list(range(l + 1, nL))
    tabB = split_tables(kv, 0, layers)
    names = list(vecs)
    _, kvE = edits.prefix_pass(model, d["prefix"], d["p"], kv_layers=layers, write=(l, torch.stack([vecs[n] for n in names])))
    rows = [("self", {q: tabB[(q, "k")] for q in layers}, {q: tabB[(q, "v")] for q in layers})]
    for i, t in ((1, "S"), (2, "X")):
        rows.append(kv_rows(f"nat|{t}", split_tables(kv, i, layers), tabB, layers))
    for i, n in enumerate(names):
        rows.append(kv_rows(n, split_tables(kvE, i, layers), tabB, layers))
    return edits.score_rows(model, d["ids"]["NONE"]["B"], fs, d["p"], rows, layers, chunk=64)


def e3_cal_flip(model, S, F, abar, beta, l, cal, nL, a):
    hits, n = 0, 0
    dev = S.W_dec.device
    for c, d, res, kv, fs in cal:
        hB = res[l][0]
        it, ix, ib = d["iS"], d["iX"], d["iB"]
        vecs = {f"E3|{t}": sae.edit(S, hB.float().to(dev), j, ib, [x.to(dev) for x in F], abar.to(dev), beta).float().cpu()
                for t, j in (("S", it), ("X", ix))}
        out = cal_rows(model, c, d, res, kv, fs, l, vecs, nL)
        hits += (out["E3|S"]["argmax"] == it) + (out["E3|X"]["argmax"] == ix)
        n += 2
    return hits / max(1, n)


def e5_vec(T, sub, alpha, l, hB, it, ib):
    if sub == "NL":
        d = T["mu1"][l][it] - T["mu1"][l][ib]
        return hB + alpha * edits.lex_split(d, T["Q"][l])[1]
    mu = T["mu5"][sub][l]
    return hB + alpha * (mu[it] - mu[ib])


def e5_cal_phi(model, T, sub, alpha, l, cal, nL):
    num, den = 0.0, 0.0
    for c, d, res, kv, fs in cal:
        hB, it, ix, ib = res[l][0], d["iS"], d["iX"], d["iB"]
        vecs = {f"E5|{t}": e5_vec(T, sub, alpha, l, hB, j, ib) for t, j in (("S", it), ("X", ix))}
        out = cal_rows(model, c, d, res, kv, fs, l, vecs, nL)
        num += id_contrast(out, "E5", it, ix)
        den += id_contrast(out, "nat", it, ix)
    return num / den if den else float("nan")


# --------------------------------------------------------------------------- stage dasfit
def das_items(model, tok, cores, l):
    """Pairs (NONE prompt of B', p, target pi(S')) with the block-l residuals of B' and S' at p."""
    pairs, hb, hs = [], [], []
    for c in cores:
        d = edits.prep(tok, c, {"B": c["base"], "S": c["source"]})
        pre = torch.cat([d["ids"]["NONE"][k][:, :d["p"] + 1] for k in ("B", "S")])
        res, _ = edits.prefix_pass(model, pre, d["p"], resid_layers=[l])
        pairs.append(das_at.Pair(base_ids=d["ids"]["NONE"]["B"], src_ids=d["ids"]["NONE"]["S"], base_pos=[d["p"]],
                                 src_pos=[d["p"]], target=LOCATIONS.index(PAIR_SWAP[c["source"]])))
        hb.append(res[l][0])
        hs.append(res[l][1])
    return pairs, hb, hs


def stage_dasfit(a):
    if "E4" not in FAMILIES[a.key]:
        log(f"dasfit: E4 is not run at {a.key}")
        return
    model, tok, dev = load_model(a)
    P = pops(a)
    t0 = time.time()
    pr = prov(a, model, dev)
    forms = das_at.form_ids(tok)
    J, bases = {"provenance": pr, "fits": {}}, {}
    for l in a.depths:
        with torch.no_grad():
            tr = das_items(model, tok, P["TSET"], l)
            ho = das_items(model, tok, P["THOLD"], l)
        for s in DAS_SEEDS:
            U, losses = das_at.train_remap_at(model, *tr, l, forms, seed=s, log=log, log_every=max(1, len(tr[0]) // 5))
            with torch.no_grad():
                fl = das_at.flip_rate(model, *ho, l, U, forms)
            bases[(l, s)] = U
            J["fits"][f"{l}_{s}"] = {"init": das_at.SEED_INIT[s], "n_train": len(tr[0]), "loss_first50": float(np.mean(losses[:50])),
                                    "loss_last50": float(np.mean(losses[-50:])), "losses": r5(losses), "flip_THOLD": fl,
                                    "n_hold": len(ho[0])}
            log(f"E4 l={l} seed={s} ({das_at.SEED_INIT[s]}): loss {np.mean(losses[:50]):.3f} -> {np.mean(losses[-50:]):.3f}; "
                f"THOLD flip {fl:.3f} ({time.time() - t0:.0f}s)")
    f = out_path(a, "das", "pt")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".pt.tmp")
    torch.save(bases, tmp)
    os.replace(tmp, f)
    J["das_pt_sha256"] = sha_file(f)
    pr["timings"]["total"] = round(time.time() - t0, 1)
    write_atomic(J, out_path(a, "das"))
    log(f"wrote {f} sha256 {J['das_pt_sha256']}")


def load_das(a):
    J = json.loads(out_path(a, "das").read_text())
    f = out_path(a, "das", "pt")
    h = sha_file(f)
    assert h == J["das_pt_sha256"], f"{f}: sha256 {h} != the one das.json recorded"
    return torch.load(f, weights_only=False), h


# --------------------------------------------------------------------------- reader heads (stage 6 files)
def reader_sets(a, nL, H, l=READ_L):
    """H*_{>l} (the stage-6 top-k* a3 heads under OPTIONS-AFTER in blocks > l) and three size-matched random sets of
    blocks > l (numpy default_rng(2)). TEST_MODE: a pseudo-ranking (default_rng(0)) at 5 % of the heads."""
    if a.test or a.key not in HEADS_FILES:
        rk = [list(map(int, c)) for c in np.random.default_rng(0).permutation([[x, h] for x in range(nL) for h in range(H)])]
        kstar, src = math.ceil(0.05 * nL * H), {"file": None, "note": "TEST_MODE pseudo-ranking"}
    else:
        f = Path(a.heads_dir) / HEADS_FILES[a.key]
        J = json.loads(f.read_text())
        rk, kstar = J["arms"]["P1"]["rankings"]["a3"], J["provenance"]["kstar"]
        assert J["provenance"]["n_layers"] == nL and J["provenance"]["heads_per_layer"] == H, "stage-6 heads of another geometry"
        src = {"file": str(f), "sha256": sha_file(f), "kstar": kstar}
    Hs = [tuple(c) for c in rk[:kstar] if c[0] > l]
    elig = [(x, h) for x in range(l + 1, nL) for h in range(H)]
    rng = np.random.default_rng(2)
    rand = [[elig[i] for i in rng.permutation(len(elig))[:len(Hs)]] for _ in range(3)]
    return Hs, rand, elig, src


def reader_kv_columns(Hs, H, KVH, hd):
    """{layer: column indices of the k_proj output} of the KV groups the heads Hs read (GQA: group = h // (H / KVH))."""
    per = H // KVH
    out = {}
    for l, h in Hs:
        g = h // per
        out.setdefault(l, set()).update(range(g * hd, (g + 1) * hd))
    return {l: torch.tensor(sorted(v)) for l, v in out.items()}


# --------------------------------------------------------------------------- edit vectors of one story
class Families:
    """Every edit vector of the model's families at one depth for one story (FP32 [D] on the CPU)."""

    def __init__(self, a, cal: Calib, das, e3vec):
        self.a, self.cal, self.das, self.e3vec = a, cal, das, e3vec
        self.fams = FAMILIES[a.key]
        self.alpha = cal.J.get("e5_alpha", {})
        self.e5 = [s for s in E5_SUB if s in self.alpha] if "E5" in self.fams else []

    def instances(self):
        """Instance names written as edits (the components follow from them)."""
        out = ["T", "R"]
        out += [f for f in ("E1", "E2", "E3") if f in self.fams]
        out += ["E4a", "E4b"] if "E4" in self.fams else []
        out += [f"E5{s}" for s in self.e5]
        return out

    def decomp(self):
        return [z for z in ("E1", "E2", "E3", "E4a", "E4b") if z in self.instances()]

    def vectors(self, si, l, h, ib, targets):
        """{f"{inst}|{t}": vec}; ``h`` {name: {l: [D]}} the natural residuals (B, S, X, piS, piX); targets {t: index}."""
        T, out = self.cal.T, {}
        hB = h["B"][l]
        for t, it in targets.items():
            dn = h[t][l] - hB
            out[f"T|{t}"] = h[t][l]
            out[f"R|{t}"] = hB + (T["mu1"][l][it] - T["mu1"][l][ib]).norm() * T["R"][l][it]
            if "E1" in self.fams:
                out[f"E1|{t}"] = hB + T["mu1"][l][it] - T["mu1"][l][ib]
            if "E2" in self.fams:
                out[f"E2|{t}"] = hB + T["mu5"]["EN"][l][it] - T["mu5"]["EN"][l][ib]
            if "E3" in self.fams:
                out[f"E3|{t}"] = self.e3vec[(si, l, t)]
            if "E4" in self.fams:
                for tag, s in (("E4a", 101), ("E4b", 102)):
                    U = self.das[(l, s)]
                    out[f"E4{tag[-1]}|{t}"] = fixed_subspace_interchange(U)(hB[None], h["pi" + t][l][None])[0]
            for sub in self.e5:
                out[f"E5{sub}|{t}"] = e5_vec(T, sub, self.alpha[sub][str(l)], l, hB, it, ib)
            for z in self.decomp():
                dz = out[f"{z}|{t}"] - hB
                par, perp, _ = edits.par_perp(dz, dn)
                lex, nonlex = edits.lex_split(dz, T["Q"][l])
                out[f"PAR:{z}|{t}"], out[f"PERP:{z}|{t}"] = hB + par, hB + perp
                out[f"LEX:{z}|{t}"], out[f"NONLEX:{z}|{t}"] = hB + lex, hB + nonlex
            lex, nonlex = edits.lex_split(dn, T["Q"][l])
            out[f"LEX:nat|{t}"], out[f"NONLEX:nat|{t}"] = hB + lex, hB + nonlex
        return out


def rows_for(fmt, inst, decomp, tab, tabB, tabT, layers):
    """The clamp rows of one cell. ``tab`` {f"{inst}|{t}": tables}; ``tabT`` {t: natural tables}. Returns a list of
    (name, ktab, vtab), the self row first."""
    K = lambda T_: {q: T_[(q, "k")] for q in layers}  # noqa: E731
    V = lambda T_: {q: T_[(q, "v")] for q in layers}  # noqa: E731
    rows = [("self", K(tabB), V(tabB))]

    def kvk(name, T_, chans=("K", "V", "KV")):
        for t in ("S", "X"):
            Z = T_[t]
            if "K" in chans:
                rows.append((f"{name}|K|{t}", K(Z), V(tabB)))
            if "V" in chans:
                rows.append((f"{name}|V|{t}", K(tabB), V(Z)))
            if "KV" in chans:
                rows.append((f"{name}|KV|{t}", K(Z), V(Z)))
    kvk("nat", tabT)
    for z in inst:
        kvk(z, {t: tab[f"{z}|{t}"] for t in ("S", "X")}, ("KV",) if z == "R" else ("K", "V", "KV"))
    for kind, lam in SYNTH:
        for t in ("S", "X"):
            if kind == "SK":
                kz = edits.interp(tabB, tabT[t], lam, layers, "k")
                rows.append((f"{kind}{int(lam * 100)}|K|{t}", kz, V(tabB)))
                rows.append((f"{kind}{int(lam * 100)}|KV|{t}", kz, V(tabT[t])))
            else:
                vz = edits.interp(tabB, tabT[t], lam, layers, "v")
                rows.append((f"{kind}{int(lam * 100)}|V|{t}", K(tabB), vz))
                rows.append((f"{kind}{int(lam * 100)}|KV|{t}", K(tabT[t]), vz))
    if fmt in ("P1", "NONE"):
        for z in decomp:
            for comp in ("PERP", "NONLEX"):
                kvk(f"{comp}:{z}", {t: tab[f"{comp}:{z}|{t}"] for t in ("S", "X")})
        kvk("NONLEX:nat", {t: tab[f"NONLEX:nat|{t}"] for t in ("S", "X")})
    if fmt == "NONE":
        for z in decomp:
            for comp in ("PAR", "LEX"):
                kvk(f"{comp}:{z}", {t: tab[f"{comp}:{z}|{t}"] for t in ("S", "X")}, ("KV",))
        kvk("LEX:nat", {t: tab[f"LEX:nat|{t}"] for t in ("S", "X")}, ("KV",))
    return rows


def compact(out):
    """score_rows output -> {"self": [E6, L6, mass, argmax], name: [dE6 + dL6 + [mass, argmax]]} rounded."""
    s = out["self"]
    res = {"self": r5(s["E"]) + r5(s["L"]) + [r5(s["mass"]), s["argmax"]]}
    for n, v in out.items():
        if n != "self":
            res[n] = r5(v["dE"]) + r5(v["dL"]) + [r5(v["mass"]), v["argmax"]]
    return res


@torch.no_grad()
def e3_precompute(a, model, tok, E, cal, dev, depths):
    """E3 vectors for every E story (and target) at each depth, one dictionary on the device at a time. h_B is read from
    the natural pass's shape (NAT_ROWS), so it is the h_B of eval's natural pass bitwise and E3 = h_B + its SAE edit."""
    vec = {}
    for l in depths:
        S, _ = load_sae(a, l, dev)
        cl = cal.T["sae"][l]
        for si, c in enumerate(E):
            d = edits.prep(tok, c)
            pre = torch.cat([d["ids"]["NONE"][k][:, :d["p"] + 1] for k in NAT_ROWS])
            res, _ = edits.prefix_pass(model, pre, d["p"], resid_layers=[l])
            for t, it in (("S", d["iS"]), ("X", d["iX"])):
                vec[(si, l, t)] = e3_vectors_for(S, res[l][0], it, d["iB"], cl)
        del S
        if dev == "cuda":
            torch.cuda.empty_cache()
    return vec


# --------------------------------------------------------------------------- stage eval
@torch.no_grad()
def stage_eval(a):
    model, tok, dev = load_model(a)
    a.hidden = model.config.hidden_size
    cfg = model.config
    nL = len(blocks(model))
    P = pops(a)
    E = P["E"]
    cal = Calib(a)
    das, das_sha = load_das(a) if "E4" in FAMILIES[a.key] else (None, None)
    path = out_path(a, "eval")
    old = json.loads(path.read_text()) if path.exists() else None
    pr = prov(a, model, dev, {"calib_sha256": cal.sha, "das_sha256": das_sha, "frames": cal.frames, "chunk": a.chunk,
                              "e5_alpha": cal.J.get("e5_alpha"), "explo_skipped": []})
    stories = []
    if old and old["provenance"].get("calib_sha256") == cal.sha and old["provenance"].get("das_sha256") == das_sha:
        stories = old["stories"]
        log(f"resuming: {len(stories)} stories done in {path}")
    done = {s["index"] for s in stories}
    t0 = time.time()
    e3vec = e3_precompute(a, model, tok, E, cal, dev, a.depths) if "E3" in FAMILIES[a.key] else {}
    pr["timings"]["e3_precompute"] = round(time.time() - t0, 1)
    fam = Families(a, cal, das, e3vec)
    inst = fam.instances()
    decomp = fam.decomp()
    Hs, _, _, hsrc = reader_sets(a, nL, cfg.num_attention_heads) if a.key in HEADS_FILES or a.test else ([], None, None, None)
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    cols = reader_kv_columns(Hs, cfg.num_attention_heads, cfg.num_key_value_heads, hd) if Hs else None
    pr |= {"instances": inst, "decomposition": decomp, "reader_heads": hsrc}
    J = {"provenance": pr, "depths": list(a.depths), "formats": list(edits.FORMATS), "stories": stories}
    per_story, status = None, 0
    for si, c in enumerate(E):
        if si in done:
            continue
        if per_story is not None and deadline_left(a) < 1.2 * per_story:
            log(f"deadline: {len(J['stories'])} of {len(E)} stories done; stopping (exit 3, the next session goes on)")
            status = 3
            break
        ts = time.time()
        J["stories"].append(eval_story(a, model, tok, si, c, cal, fam, inst, decomp, nL, cols))
        per_story = time.time() - ts if per_story is None else 0.8 * per_story + 0.2 * (time.time() - ts)
        if len(J["stories"]) % 5 == 0 or si == len(E) - 1:
            J["stories"].sort(key=lambda s: s["index"])
            pr["timings"]["eval"] = round(time.time() - t0, 1)
            write_atomic(J, path)
        log(f"story {si}: {time.time() - ts:.1f}s ({len(J['stories'])}/{len(E)}; {time.time() - t0:.0f}s)")
    J["stories"].sort(key=lambda s: s["index"])
    pr["timings"]["eval"] = round(time.time() - t0, 1)
    pr["n_done"] = len(J["stories"])
    pr["wrapper"] = dict(WRAPPER_USED)
    write_atomic(J, path)
    log(f"wrote {path}: {len(J['stories'])} stories")
    return status


NAT_ROWS = ("B", "S", "X", "piS", "piX")     # the rows of the natural prefix pass of an E story


def natural_pass(model, d, depths, nL):
    """The natural prefix pass of one E story (NAT_ROWS, NONE prefix): block outputs at p at ``depths`` and K/V at p from
    block min(depths) + 1."""
    pre = torch.cat([d["ids"]["NONE"][k][:, :d["p"] + 1] for k in NAT_ROWS])
    return edits.prefix_pass(model, pre, d["p"], resid_layers=depths, kv_layers=range(min(depths) + 1, nL))


def edit_pass(model, d, l, vecs, layers):
    """The edit vectors ``vecs`` [R, D] written at (p, l) of the B prefix: K/V at p in ``layers``. Run in forwards of
    exactly len(NAT_ROWS) rows (the last padded), the natural pass's shape, so that an edit's tables are computed by the
    natural pass's kernels: T then equals the natural tables bitwise (BF16 included) and an edit differs from them only
    through what it writes (J-C-G1)."""
    return edits.prefix_pass(model, d["prefix"], d["p"], kv_layers=layers, write=(l, vecs), chunk=len(NAT_ROWS), pad=True)[1]


def eval_story(a, model, tok, si, c, cal, fam, inst, decomp, nL, cols):
    d = edits.prep(tok, c)
    p = d["p"]
    names = NAT_ROWS
    res, kv = natural_pass(model, d, a.depths, nL)
    h = {k: {l: res[l][i] for l in a.depths} for i, k in enumerate(names)}
    targets = {"S": d["iS"], "X": d["iX"]}
    fs = {f: edits.form_set(tok, f, c, cal.frames) for f in edits.FORMATS}
    rec = {"index": si, "core": c, "X": d["X"], "p": p, "iB": d["iB"], "iS": d["iS"], "iX": d["iX"],
           "iPiS": LOCATIONS.index(d["vals"]["piS"]), "iPiX": LOCATIONS.index(d["vals"]["piX"]), "cells": {}, "stats": {}}
    for l in a.depths:
        layers = list(range(l + 1, nL))
        vec = fam.vectors(si, l, h, d["iB"], targets)
        vn = list(vec)
        kvE = edit_pass(model, d, l, torch.stack([vec[n] for n in vn]), layers)
        tab = {n: split_tables(kvE, i, layers) for i, n in enumerate(vn)}
        tabB = split_tables(kv, 0, layers)
        tabT = {"S": split_tables(kv, 1, layers), "X": split_tables(kv, 2, layers)}
        st = {}
        for n in vn:
            t = n.split("|")[1]
            s = edits.kv_stats(tab[n], tabT[t], tabB, layers, {q: v for q, v in cols.items() if q in layers} if cols else None)
            dz, dn = vec[n] - h["B"][l], h[t][l] - h["B"][l]
            s |= {"cos_resid": edits.cos(dz, dn), "ratio_resid": float(dz.norm() / dn.norm().clamp_min(1e-30))}
            st[n] = {k: r5(v) for k, v in s.items()}
        rec["stats"][str(l)] = st
        for f in edits.FORMATS:
            rows = rows_for(f, inst, decomp, tab, tabB, tabT, layers)
            out = edits.score_rows(model, d["ids"][f]["B"], fs[f], p, rows, layers, chunk=a.chunk, letters=f == "LETTER")
            rec["cells"][f"{f}@{l}"] = compact(out)
    return rec


# --------------------------------------------------------------------------- stage readers (J-C-READ)
def g_rows(tok, c):
    """The six location-word rows of the "Choices:" list of the P1 B prompt (canonical order)."""
    rec = record(c, "direct", c["base"])
    text = chat_text(tok, raw_prompt("P1", rec["story"], rec["query"]), prefill="Answer:")
    e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert text.count(LISTING) == 1
    c0 = text.index(LISTING)
    rows = rows_in(e["offset_mapping"], c0, c0 + len(LISTING))
    loc = candidate_ids(tok, "P1")
    G = [i for i in rows if e["input_ids"][i] in set(loc)]
    assert len(G) == 6 and [e["input_ids"][i] for i in G] == loc, G
    return G, torch.tensor([e["input_ids"]])


@torch.no_grad()
def reader_logprobs(model, hooks3, ids, p, G, rows, KB, VB, on, cid, chunk):
    """The reader rows of one story: ``rows`` = [(name, ks {layer: [D]}, (cells in G, outside))], rows[0] the self row,
    led into every chunk. HeadSplice ("splice") gives the heads of ``cells`` in the rows G, and every head in every other
    row when ``outside``, the key ks at p from layer ``on``; KeyPin / value pin write B's K and V (KB, VB) at p from ``on``
    in every pass, so every other head sees B's key and every head B's value. Returns {"self": six L log-probs,
    name: six L log-probs minus the in-chunk self row's}."""
    hsp, kpin, vpin = hooks3
    dev = next(model.parameters()).device
    nL, H = len(blocks(model)), model.config.num_attention_heads
    out = {}
    step = max(1, chunk - 1)
    for b0 in range(0, max(1, len(rows) - 1), step):
        ch = [rows[0]] + rows[1 + b0:1 + b0 + step]
        B = len(ch)
        masks = blind_masks([r[2] for r in ch], G, ids.shape[1], nL, H, on)
        ks = {q: torch.stack([r[1][q] for r in ch]).to(dev) for q in range(on, nL)}
        one = torch.ones(B, dtype=torch.bool)
        try:
            kpin.tab, kpin.use, kpin.pos, kpin.active = {q: KB[q][None].expand(B, -1).to(dev) for q in range(on, nL)}, one, torch.full((B,), p), True
            vpin.tab, vpin.use, vpin.pos, vpin.active = {q: VB[q][None].expand(B, -1).to(dev) for q in range(on, nL)}, one, torch.full((B,), p), True
            hsp.mode, hsp.ks, hsp.pos, hsp.masks, hsp.mu, hsp.active = "splice", ks, p, masks, None, True
            lg = model(ids.expand(B, -1).to(dev), use_cache=False, logits_to_keep=1).logits[:, -1].float()
        finally:
            kpin.active = vpin.active = hsp.active = False
            kpin.tab = vpin.tab = None
            hsp.masks = None
        lp = torch.log_softmax(lg, -1)[:, cid].double().cpu()
        for i, r in enumerate(ch):
            if i == 0:
                out.setdefault("self", r5(lp[0].tolist()))
            else:
                out[r[0]] = r5((lp[i] - lp[0]).tolist())
    return out


@torch.no_grad()
def stage_readers(a):
    if a.key not in HEADS_FILES and not a.test:
        log(f"readers: not run at {a.key}")
        return
    model, tok, dev = load_model(a)
    a.hidden = model.config.hidden_size
    cfg = model.config
    nL, H = len(blocks(model)), cfg.num_attention_heads
    P = pops(a)
    E = P["E"][:a.n_readers] if a.n_readers else P["E"]
    cal = Calib(a)
    das, das_sha = load_das(a) if "E4" in FAMILIES[a.key] else (None, None)
    l = READ_L
    on = l + 1
    Hs, rand, elig, hsrc = reader_sets(a, nL, H, l)
    t0 = time.time()
    e3vec = e3_precompute(a, model, tok, E, cal, dev, [l]) if "E3" in FAMILIES[a.key] else {}
    fam = Families(a, cal, das, e3vec)
    donors = ["nat"] + [z for z in ("E1", "E2", "E3", "E4a") if z in fam.instances()]
    hsp = HeadSplice(model)
    kpin, vpin = KeyPin(model, range(on, nL), "k"), KeyPin(model, range(on, nL), "v")
    comp = lambda C: sorted(set(elig) - set(C))  # noqa: E731
    specs = {"all": (elig, True), "H": (comp(Hs), True), "r0": (comp(rand[0]), True), "r1": (comp(rand[1]), True),
             "r2": (comp(rand[2]), True)}
    cid = candidate_ids(tok, "P1")
    pr = prov(a, model, dev, {"calib_sha256": cal.sha, "das_sha256": das_sha, "reader_heads": hsrc, "layer": l, "onset": on,
                              "H_star": [list(x) for x in Hs], "rand": [[list(x) for x in r] for r in rand]})
    J = {"provenance": pr, "donors": donors, "specs": list(specs), "stories": []}
    for si, c in enumerate(E):
        d = edits.prep(tok, c)
        p = d["p"]
        G, ids = g_rows(tok, c)
        assert torch.equal(ids, d["ids"]["P1"]["B"]), "offset encoding differs from the prompt ids"
        names = ("B", "S", "X", "piS", "piX")
        pre = torch.cat([d["ids"]["NONE"][k][:, :p + 1] for k in names])
        res, kv = edits.prefix_pass(model, pre, p, resid_layers=[l], kv_layers=range(on, nL))
        h = {k: {l: res[l][i]} for i, k in enumerate(names)}
        vec = fam.vectors(si, l, h, d["iB"], {"S": d["iS"], "X": d["iX"]})
        vn = [f"{z}|{t}" for z in donors if z != "nat" for t in ("S", "X")]
        _, kvE = edits.prefix_pass(model, d["prefix"], p, kv_layers=range(on, nL), write=(l, torch.stack([vec[n] for n in vn])))
        K = {f"nat|S": {q: kv[(q, "k")][1] for q in range(on, nL)}, f"nat|X": {q: kv[(q, "k")][2] for q in range(on, nL)}}
        K |= {n: {q: kvE[(q, "k")][i] for q in range(on, nL)} for i, n in enumerate(vn)}
        KB = {q: kv[(q, "k")][0] for q in range(on, nL)}
        VB = {q: kv[(q, "v")][0] for q in range(on, nL)}
        rows = [("self", KB, ([], False))] + [(f"{z}|{t}|{s}", K[f"{z}|{t}"], specs[s]) for z in donors for t in ("S", "X") for s in specs]
        out = reader_logprobs(model, (hsp, kpin, vpin), ids, p, G, rows, KB, VB, on, cid, a.chunk_readers)
        J["stories"].append({"index": si, "core": c, "p": p, "G": G, "iB": d["iB"], "iS": d["iS"], "iX": d["iX"], "rows": out})
        if si % 10 == 9 or si == len(E) - 1:
            pr["timings"]["readers"] = round(time.time() - t0, 1)
            pr["n_double_passes"] = hsp.n_double
            write_atomic(J, out_path(a, "readers"))
        log(f"readers story {si} ({time.time() - t0:.0f}s)")
    write_atomic(J, out_path(a, "readers"))


# --------------------------------------------------------------------------- stage explore
@torch.no_grad()
def stage_explore(a):
    """Exploratory: greedy-generation flip rates (O4) of the KV rows of every instance at l = 7, NONE and P1."""
    path = out_path(a, "explore")
    if deadline_left(a) < 0:
        write_atomic({"provenance": {"skipped": "deadline passed"}}, path)
        log("explore skipped: deadline")
        return
    model, tok, dev = load_model(a)
    a.hidden = model.config.hidden_size
    nL = len(blocks(model))
    P = pops(a)
    E = P["E"][:(2 if a.test else EXPLORE_N)]
    cal = Calib(a)
    das, das_sha = load_das(a) if "E4" in FAMILIES[a.key] else (None, None)
    l = READ_L
    t0 = time.time()
    e3vec = e3_precompute(a, model, tok, E, cal, dev, [l]) if "E3" in FAMILIES[a.key] else {}
    fam = Families(a, cal, das, e3vec)
    inst = [z for z in fam.instances() if z != "R"]
    layers = list(range(l + 1, nL))
    stop = candidate_stop(tok)
    pr = prov(a, model, dev, {"calib_sha256": cal.sha, "das_sha256": das_sha, "layer": l})
    J = {"provenance": pr, "instances": inst, "stories": []}
    for si, c in enumerate(E):
        if deadline_left(a) < 0:
            pr["explo_skipped"] = f"stopped at the deadline after {si} stories"
            break
        d = edits.prep(tok, c)
        p = d["p"]
        names = ("B", "S", "X", "piS", "piX")
        pre = torch.cat([d["ids"]["NONE"][k][:, :p + 1] for k in names])
        res, kv = edits.prefix_pass(model, pre, p, resid_layers=[l], kv_layers=layers)
        h = {k: {l: res[l][i]} for i, k in enumerate(names)}
        vec = fam.vectors(si, l, h, d["iB"], {"S": d["iS"], "X": d["iX"]})
        vn = [f"{z}|{t}" for z in inst for t in ("S", "X")]
        _, kvE = edits.prefix_pass(model, d["prefix"], p, kv_layers=layers, write=(l, torch.stack([vec[n] for n in vn])))
        tabs = [split_tables(kv, 0, layers), split_tables(kv, 1, layers), split_tables(kv, 2, layers)] + \
               [split_tables(kvE, i, layers) for i in range(len(vn))]
        rn = ["self", "nat|S", "nat|X"] + vn
        T = {(q, ch): torch.stack([t[(q, ch)] for t in tabs])[:, None] for q in layers for ch in "kv"}
        r = {}
        for f in EXPLORE_FORMATS:
            with clamp_kv(model, [p], T, layers):
                gen = greedy(model, tok, d["ids"][f]["B"].expand(len(rn), -1), MAX_NEW, stop=stop)
            r[f] = {n: parse_answer(tok.decode(g)) for n, g in zip(rn, gen)}
        J["stories"].append({"index": si, "core": c, "S": c["source"], "X": d["X"], "B": c["base"], "answers": r})
    pr["timings"]["total"] = round(time.time() - t0, 1)
    write_atomic(J, path)
    log(f"wrote {path}")


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "calib", "dasfit", "eval", "readers", "explore"])
    ap.add_argument("--model", default=None, help="a local verified directory (s8_fetch) or a Hub id")
    ap.add_argument("--key", required=True, choices=sorted(FAMILIES), help="the model key of scripts/stage8_models.json")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="auto")
    ap.add_argument("--n", type=int, default=0, help="0 = every E story")
    ap.add_argument("--n-readers", type=int, default=0, help="0 = every E story")
    ap.add_argument("--chunk", type=int, default=64, help="rows per scoring forward (the self row included)")
    ap.add_argument("--chunk-readers", type=int, default=26, help="rows per HeadSplice forward (the self row included)")
    ap.add_argument("--sae-dir", default=os.path.expanduser("~/stage8_models/sae_qwen7"))
    ap.add_argument("--heads-dir", default="results/gpu_stage6/heads")
    ap.add_argument("--reserve-min", type=float, default=0.0, help="minutes the pipeline needs after this step (deadline)")
    ap.add_argument("--out", default="results/gpu_stage8c")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B FP32 on the CPU, n = 2")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    if a.test:
        a.model, a.revision, a.dtype = TINY, None, "float32"
        a.n, a.chunk = a.n or 2, min(a.chunk, 24)
    assert a.model, "--model is required"
    a.depths = TEST_DEPTHS if a.test else DEPTHS
    a.tag = ("TEST_" if a.test else "") + a.key
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    rc = {"preflight": stage_preflight, "calib": stage_calib, "dasfit": stage_dasfit, "eval": stage_eval,
          "readers": stage_readers, "explore": stage_explore}[a.stage](a)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
