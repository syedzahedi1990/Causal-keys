"""Stage 8 part B of P-2026-10-10-J (docs/PREREGISTRATION.md): fresh-sample replication of the key/value format
factorial in new model families, scored on the forms the models emit.

Populations, lexicons, sentences, prompts and parsers: ckeys/fresh.py (F, C, S0). Per model (BF16; sdpa, eager for
Gemma-2; use_cache=False in every scoring pass; transformers 5.18.0) and item, the B, S and X prompts (moved-to location
base, source and the third location X = pick_x) differ at exactly one token, p. Rows (experiments/format_factorial.
row_specs, unchanged): the self-clamp ID; K_S, V_S, KV_S from l0 in {0, round(0.0625 L), round(0.3 L)}; K_X, V_X, KV_X
from 0; plus the clean B, S and X runs. Every row is scored by ONE forward over the prompt and the token trie of every
form of the six candidates (ckeys.surface.score): L = log p(" w"), Sigma = logsumexp over the 12 reviewer-named forms,
E = logsumexp over the 32 fixed forms and the model's discovered frames. Generation (ckeys.generate.greedy, the KV cache
on, the clamps active in the prompt pass; at most MAX_NEW new tokens, stopping at a newline, EOS or once a candidate of
the item's lexicon is complete) in the clean B, S, X runs and the rows ID, K_S, K_X, V_S, V_X (l0 = 0), as one batch.
Stages (each a separate process and results file; the pipeline is scripts/gpu_stage8b.sh):
  tokcheck  JB-G0b, tokenizer only (no weights): both lexicons single-token and stable in context; the FormSet of every
            F, C and S0 item builds (no form a proper prefix of another) and decodes back; every arm of F, C and S0 gives
            B, S, X prompts of one length differing at one position; every sentence (and null sentence) within +-3 tokens
            of ROOM. -> OUT/tokcheck/<tag>.json; exit 1 if any check fails
  calib     R0 (frame discovery on C; no statistic): greedy generations of the clean B, S, X runs of the 30 C cores in
            the arms ARMS_CAL; frames = the text before the first candidate with the names as {a} {b} {o} {d}; admitted if
            not a fixed E frame and in >= 2 % (and >= 2) of some arm's generations, at most 16, most frequent first
            (ckeys.generate.discover_frames), and only if the FormSet of every C item builds with it.
            -> OUT/frames/<tag>.json (its sha256 is printed, and recorded by every later file of the model)
  g3        JB-G3: on the first 30 F cores in NONE, POST and AFTER, the trie pass (no generation) against the plain
            published path (format_factorial.run_item's passes on the item's prompts): per arm |s_ID^L(trie) -
            s_ID^L(plain)| <= 0.02 and mean |L(trie) - L(plain)| <= 0.05 x mean ID_KV^L(plain). Pass: the evaluation
            scores with the trie; fail: with score_cached (the exact per-node path: the prompt with the plain causal
            kernel and the KV cache, every trie node as a cached continuation). -> OUT/g3/<tag>.json
  eval      R1 (S0: P1, AFTER, BEFORE, POST, PRE, NONE, plus format_factorial.run_item itself on every item, the
            published path for JB-G2) or R2/R3 (F: AFTER, BEFORE, NONE, POST, PRE, POST-NULL); the deadline
            (STAGE8_DEADLINE minus --reserve-min) is checked before each arm; arms not run are recorded and the exit
            status is 3. -> OUT/eval/<tag>_<population>.json
Batch sizes (A100-80GB): per item one 3-row capture/scoring pass (clean B, S, X + trie), one 13-row scoring pass, one
8-row generation batch (<= 16 decode steps); prompts <= 160 tokens plus <= ~400 trie nodes: under 10 GB beyond the
weights at 14B. score_cached splits the node continuations into chunks of 256 rows.
TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B-Instruct FP32 on the CPU, n = 2 items (C: 2, G3: 2), outputs tagged
TEST_<key>.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ckeys import fresh
from ckeys.clamp import capture_kv, clamp_kv, stack_rows
from ckeys.encoding import WRAPPER_USED, encode
from ckeys.generate import discover_frames, greedy
from ckeys.interventions import blocks
from ckeys.surface import FRAMES_E_FIXED, FormSet, _fields, score, score_reference
from experiments.format_factorial import LABEL, row_specs
from experiments.format_factorial import provenance as ff_provenance
from experiments.format_factorial import run_item as ff_run_item
from experiments.ioi_factorial import device_name
from experiments.stage6_heads import write_atomic

TINY = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 16
N_G3, G3_ARMS = 30, ("NONE", "POST", "AFTER")
G3_DS, G3_DL = 0.02, 0.05
GEN_ROWS = (("B", "B", None, "B"), ("S", "S", None, "S"), ("X", "X", None, "X"), ("B", "B", 0, "ID@0"),
            ("S", "B", 0, "K_S@0"), ("X", "B", 0, "K_X@0"), ("B", "S", 0, "V_S@0"), ("B", "X", 0, "V_X@0"))
SCORE_SETS = ("L", "sigma", "E")
CHUNK = 256


def log(s):
    print(s, flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def deadline_passed(reserve_min=0.0):
    d = os.environ.get("STAGE8_DEADLINE")
    return bool(d) and time.time() > float(d) - 60 * reserve_min


class Invalid(Exception):
    """An item that is not valid for a tokenizer in an arm (skipped and counted, with the reason)."""


# --------------------------------------------------------------------------- scorers
@torch.no_grad()
def score_cached(model, ids: torch.Tensor, fs: FormSet, topk: int = 10, chunk: int = CHUNK):
    """The fields of ckeys.surface.score computed without the tree mask: the prompt by one plain causal pass with the
    KV cache (so the answer position is the plain path's), then every trie node as a cached continuation of its own
    path (rows chunked; nodes grouped by depth). Exact in FP32 (tests/test_fresh_factorial.py); the JB-G3 fallback.
    Clamps (ckeys.clamp) act in the prompt pass and pass through on the continuations, which are shorter than p."""
    dev = next(model.parameters()).device
    R = ids.shape[0]
    out = model(ids.to(dev), use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    first = torch.log_softmax(out.logits[:, -1].float(), -1)
    lp = torch.empty(R, 1 + len(fs), first.shape[-1], device=first.device)
    lp[:, 0] = first
    depths = sorted({len(n) for n in fs.nodes})
    for d in depths:
        nodes = [n for n in fs.nodes if len(n) == d]
        per = max(1, chunk // R)
        for j in range(0, len(nodes), per):
            ch = nodes[j:j + per]
            c = copy.deepcopy(past)
            c.batch_repeat_interleave(len(ch))
            x = torch.tensor([list(n) for _ in range(R) for n in ch], dtype=ids.dtype, device=dev)
            o = model(x, past_key_values=c, use_cache=True, logits_to_keep=1)
            lo = torch.log_softmax(o.logits[:, -1].float(), -1)
            for k, n in enumerate(ch):
                lp[:, 1 + fs.index[n]] = lo[k::len(ch)]
            del c, o
    return _fields(lp, fs, topk)


SCORERS = {"trie": score, "cached": score_cached, "reference": score_reference}


# --------------------------------------------------------------------------- items
def encode_item(tok, it: fresh.Item, arm: str):
    """{B, S, X: ids [1, T]} and the one position p where they differ; raises Invalid."""
    locs = {"B": it.core["base"], "S": it.core["source"], "X": it.x_canon}
    ids = {n: encode(tok, fresh.raw_prompt(it, arm, loc)) for n, loc in locs.items()}
    if len({tuple(v.shape) for v in ids.values()}) > 1:
        raise Invalid("B, S, X prompts of different lengths")
    diff = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()
    if len(diff) != 1 or (ids["B"][0] != ids["X"][0]).nonzero().flatten().tolist() != diff:
        raise Invalid("B, S, X prompts do not differ at exactly one position")
    return ids, diff[0]


def _vec(r, name, words, i):
    return [float(r[name][w][i]) for w in words]


def unpack(r, fs: FormSet, i: int, top=False, tok=None) -> dict:
    """One row of a scorer result: L, sigma and E of the six words (canonical order), the uncovered first-token mass."""
    out = {k: _vec(r, k, fs.words, i) for k in SCORE_SETS}
    out["gap"] = float(r["gap"][i])
    if top:
        ids, ps = r["top"]
        out["top"] = [[tok.decode([int(t)]), float(p)] for t, p in zip(ids[i], ps[i])]
    return out


def ff_view(it: fresh.Item, clean: dict, rows: dict) -> dict:
    """format_factorial.run_item's lower-case fields from the trie pass's L: clean lp / m / mass / argmax_cand, row m / lp
    (for the tracked S, B, X and the initial location)."""
    W = list(it.words)
    tid = {k: W.index(v) for k, v in it.track.items()}
    view = {"clean": {}, "m": {}}
    for n, c in clean.items():
        L = c["L"]
        view["clean"][n] = {"lp": {k: L[j] for k, j in tid.items()}, "m": L[tid["S"]] - L[tid["B"]],
                            "mass": float(np.exp(L).sum()), "argmax_cand": W[int(np.argmax(L))]}
    for lab, c in rows.items():
        L = c["L"]
        view["m"][lab] = {"m": L[tid["S"]] - L[tid["B"]], "lp": {k: L[j] for k, j in tid.items()}}
    return view


@torch.no_grad()
def run_item(model, tok, it: fresh.Item, arm: str, fs: FormSet, nL: int, dev, gen: bool = True, scorer: str = "trie",
             max_new: int = MAX_NEW) -> dict:
    """The trie-scored rows, the clean runs and (gen) the generation batch of one item in one arm."""
    ids, pos = encode_item(tok, it, arm)
    assert max(len(n) for n in fs.nodes) + 1 < pos if fs.nodes else True, "a trie path as long as p"
    sc = SCORERS[scorer]
    with capture_kv(model, [pos], range(nL)) as t:
        rc = sc(model, torch.cat([ids[n] for n in "BSX"]).to(dev), fs)
    kv = {n: {k: v[i] for k, v in t.items()} for i, n in enumerate("BSX")}
    clean = {n: unpack(rc, fs, i, top=True, tok=tok) for i, n in enumerate("BSX")}
    rows = row_specs(nL)
    tabs = stack_rows(kv, [lambda l, ch, r=r: (r[0] if ch == "k" else r[1]) if l >= r[2] else "B" for r in rows], range(nL))
    with clamp_kv(model, [pos], tabs, range(nL)):
        rr = sc(model, ids["B"].to(dev).expand(len(rows), -1), fs)
    rowd = {f"{LABEL[(k, v)]}@{l0}": unpack(rr, fs, i) for i, (k, v, l0) in enumerate(rows)}
    item = {**it.meta(), "arm": arm, "pos": pos, "len": ids["B"].shape[1], "n_layers": nL, "scorer": scorer,
            "nodes": len(fs), "clean": clean, "rows": rowd, "ff": ff_view(it, clean, rowd)}
    if gen:
        gids = torch.cat([ids[g[0] if g[2] is None else "B"] for g in GEN_ROWS]).to(dev)
        gt = stack_rows(kv, [lambda l, ch, g=g: "B" if g[2] is None else (g[0] if ch == "k" else g[1]) for g in GEN_ROWS], range(nL))
        sel = torch.tensor([g[2] is not None for g in GEN_ROWS])
        with clamp_kv(model, [pos], gt, range(nL), per_row=sel):
            out = greedy(model, tok, gids, max_new=max_new, stop=fresh.stop_fn(tok, it.words))
        txt = {g[3]: fresh.gen_text(tok, gids[i].tolist(), out[i]) for i, g in enumerate(GEN_ROWS)}
        item["gen"] = txt
        item["ans"] = {k: fresh.parse(v, it.words) for k, v in txt.items()}
        item["gen_len"] = {g[3]: len(out[i]) for i, g in enumerate(GEN_ROWS)}
    return item


@torch.no_grad()
def plain_item(model, tok, it: fresh.Item, arm: str, nL: int, dev) -> dict:
    """format_factorial.run_item's passes on the item's prompts (each clean run alone, then the 13-row clamp batch,
    logits_to_keep=1): L of the six words per run and row. On S0 it equals format_factorial.run_item."""
    ids, pos = encode_item(tok, it, arm)
    cid = []
    for w in it.words:
        t = tok.encode(" " + w, add_special_tokens=False)
        if len(t) != 1:
            raise Invalid(f"candidate {w!r} is not a single token")
        cid.append(t[0])
    kv, clean = {}, {}
    for n in "BSX":
        with capture_kv(model, [pos], range(nL)) as t:
            lp = torch.log_softmax(model(ids[n].to(dev), use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1)[0]
        kv[n] = {k: v[0] for k, v in t.items()}
        clean[n] = {"L": [float(lp[i]) for i in cid]}
    rows = row_specs(nL)
    tabs = stack_rows(kv, [lambda l, ch, r=r: (r[0] if ch == "k" else r[1]) if l >= r[2] else "B" for r in rows], range(nL))
    with clamp_kv(model, [pos], tabs, range(nL)):
        lp = torch.log_softmax(model(ids["B"].to(dev).expand(len(rows), -1), use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1)
    rowd = {f"{LABEL[(k, v)]}@{l0}": {"L": [float(lp[i, j]) for j in cid]} for i, (k, v, l0) in enumerate(rows)}
    return {"key": it.key, "arm": arm, "pos": pos, "clean": clean, "rows": rowd}


# --------------------------------------------------------------------------- G3 statistics (also used by the scorer)
def idk_idv(rec: dict, score: str = "L") -> tuple[float, float, float]:
    """ID_K, ID_V and ID_KV of one item (l0 = 0) under one scoring, from the rows' six-word vectors."""
    W = list(rec["words"]) if "words" in rec else None
    tr = rec["track"]
    iS, iX = W.index(tr["S"]), W.index(tr["X"])
    R = rec["rows"]
    d = lambda row, i: R[row][score][i] - R["ID@0"][score][i]  # noqa: E731
    out = []
    for ch in ("K", "V", "KV"):
        out.append(0.5 * ((d(f"{ch}_S@0", iS) - d(f"{ch}_X@0", iS)) + (d(f"{ch}_X@0", iX) - d(f"{ch}_S@0", iX))))
    return tuple(out)


def g3_compare(trie: list[dict], plain: list[dict]) -> dict:
    """JB-G3 for one arm: trie and plain records of the same items (L only)."""
    P = {p["key"]: p for p in plain}
    K, V, KV, k2, v2, dl = [], [], [], [], [], []
    for t in trie:
        p = dict(P[t["key"]], words=t["words"], track=t["track"])
        a, b = idk_idv(t), idk_idv(p)
        K.append(a[0]); V.append(a[1]); k2.append(b[0]); v2.append(b[1]); KV.append(b[2])
        for run in ("B", "S", "X"):
            dl += list(np.abs(np.subtract(t["clean"][run]["L"], p["clean"][run]["L"])))
        for row in t["rows"]:
            dl += list(np.abs(np.subtract(t["rows"][row]["L"], p["rows"][row]["L"])))
    sid = lambda k, v: float(np.mean(k) / (np.mean(k) + np.mean(v)))  # noqa: E731
    s_t, s_p = sid(K, V), sid(k2, v2)
    mdl, kv = float(np.mean(dl)), float(np.mean(KV))
    ok = abs(s_t - s_p) <= G3_DS and mdl <= G3_DL * kv
    return {"n": len(trie), "s_ID_trie": s_t, "s_ID_plain": s_p, "abs_ds": abs(s_t - s_p), "mean_abs_dL": mdl,
            "ID_KV_plain": kv, "tol_dL": G3_DL * kv, "pass": bool(ok)}


# --------------------------------------------------------------------------- model
def load_model(a):
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    attn = a.attn if a.attn != "auto" else ("eager" if cfg.model_type.startswith("gemma2") else "sdpa")
    dev = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    tok, tokrec = load_tokenizer(a)
    if a.stage == "tokcheck":
        return None, tok, dev, tokrec
    kw = dict(dtype=getattr(torch, a.dtype), revision=a.revision, attn_implementation=attn)
    if dev == "cuda":
        kw["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if dev == "cpu":
        model = model.to(dev)
    assert model.config._attn_implementation == attn
    return model, tok, dev, tokrec


def load_tokenizer(a):
    """The tokenizer; at Mistral-Small-24B (key mistral24) with fix_mistral_regex, as experiments/paper1_frames.py
    checks it (the fix must be installed, strict outside TEST_MODE)."""
    if a.key == "mistral24":
        from experiments.paper1_frames import fixed_tokenizer
        return fixed_tokenizer(a.model, a.revision, strict=not a.test)
    return AutoTokenizer.from_pretrained(a.model, revision=a.revision), None


def verified(model_dir):
    f = Path(model_dir) / "VERIFIED.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    return {"sha256": sha_file(f), "repo": j.get("repo"), "revision": j.get("revision"), "key": j.get("key"),
            "attn": j.get("attn"), "token_used": j.get("token_used")}


def prov(a, model, dev, extra=None):
    p = ff_provenance(a) | {"model": a.model, "model_key": a.key, "revision": a.revision, "verified": verified(a.model),
                            "test_mode": a.test, "max_new": a.max_new, "stage": a.stage, "timings": {}}
    if model is not None:
        p |= {"dtype": str(next(model.parameters()).dtype), "attn_implementation": model.config._attn_implementation,
              "device": device_name(dev), "n_layers": len(blocks(model))}
    return p | (extra or {})


def pop_items(a, name):
    items = list(fresh.population(name))
    assert fresh.pop_hash(items) == fresh.POP_SHA256[name], f"population {name} differs from its pinned hash"
    n = a.n_cal if name == "C" else a.n
    return items[:n] if n else items


def read_frames(a):
    f = Path(a.out) / "frames" / f"{a.tag_base}.json"
    j = json.loads(f.read_text())
    h = sha_file(f)
    log(f"frames file {f} sha256 {h}: {len(j['frames'])} discovered frames {j['frames']}")
    return j["frames"], str(f), h


# --------------------------------------------------------------------------- stage tokcheck (JB-G0b)
def stage_tokcheck(a):
    _, tok, _, tokrec = load_model(a)
    t0, fails, rep = time.time(), [], {}
    for lex, words in fresh.LEXICONS.items():
        for w in words + (fresh.NULL_NOUNS if lex == 1 else ()):
            x = tok(" " + w, add_special_tokens=False).input_ids
            ctx = [tok(f"the {w}{p}", add_special_tokens=False).input_ids for p in (",", ".", " and")]
            if len(x) != 1 or not all(x[0] in c for c in ctx):
                fails.append(f"' {w}' is not one token stable before ',' '.' ' and'")
    n = lambda s: len(tok(" " + s, add_special_tokens=False).input_ids)  # noqa: E731
    room = n(fresh.ROOM_TEMPLATE.format(*fresh.LEX1))
    lens = {}
    for k, t in enumerate(fresh.SENTENCES):
        ls = {n(t.format(*[nouns[i] for i in o])) for nouns in (fresh.LEX1, fresh.LEX2, fresh.NULL_NOUNS)
              for o in (tuple(range(6)), (5, 4, 3, 2, 1, 0), (2, 5, 0, 3, 1, 4))}
        lens[k] = sorted(ls)
        if min(ls) < room - 3 or max(ls) > room + 3:
            fails.append(f"sentence {k}: {sorted(ls)} tokens against ROOM {room}")
    rep["sentence_tokens"], rep["room_tokens"] = lens, room
    skipped, nodes, bad_decode = {}, {}, []
    for name in ("F", "C", "S0"):
        for it in fresh.population(name):
            try:
                fs, dropped = fresh.form_set(tok, it)
                nodes.setdefault(name, []).append(len(fs))
                bad_decode += [(it.key, x) for x in fs.check_decode()]
            except ValueError as ex:
                fails.append(f"{it.key}: FormSet: {ex}")
            for arm in (fresh.ARMS_S0 if name == "S0" else fresh.ARMS_CAL if name == "C" else fresh.ARMS_F):
                try:
                    encode_item(tok, it, arm)
                except Invalid as ex:
                    skipped.setdefault(f"{name}/{arm}", []).append([it.key, str(ex)])
    if bad_decode:
        fails.append(f"{len(bad_decode)} forms do not decode back: {bad_decode[:5]}")
    if skipped:
        fails.append(f"skipped items: { {k: len(v) for k, v in skipped.items()} }")
    rep |= {"skipped": skipped, "trie_nodes": {k: [min(v), max(v)] for k, v in nodes.items()}, "fails": fails,
            "pass": not fails, "wrapper": dict(WRAPPER_USED), "tokenizer": tokrec,
            "provenance": prov(a, None, None, {"seconds": round(time.time() - t0, 1)})}
    write_atomic(rep, Path(a.out) / "tokcheck" / f"{a.tag}.json")
    log(f"tokcheck {'PASS' if not fails else 'FAIL'}: ROOM {room} tokens, sentences {lens}; trie nodes {rep['trie_nodes']}; "
        f"{len(fails)} failures {fails[:3]}")
    return 0 if not fails else 1


# --------------------------------------------------------------------------- stage calib (R0)
def admit_frames(tok, items, cand):
    """The discovery rule's last clause: in order, a candidate frame is admitted only if the FormSet of every item builds
    with the fixed frames, the frames admitted so far and it (no form a proper prefix of another). (admitted, rejected)."""
    admitted, rejected = [], {}
    for fr in cand:
        try:
            for it in items:
                FormSet(tok, it.words, fresh.frame_sets(admitted + [fr]), names=it.names)
            admitted.append(fr)
        except ValueError as ex:
            rejected[fr] = str(ex)
    return admitted, rejected


@torch.no_grad()
def stage_calib(a):
    model, tok, dev, tokrec = load_model(a)
    items = pop_items(a, "C")
    t0, recs, skipped = time.time(), [], []
    for arm in fresh.ARMS_CAL:
        for it in items:
            try:
                ids, _ = encode_item(tok, it, arm)
            except Invalid as ex:
                skipped.append({"key": it.key, "arm": arm, "reason": str(ex)})
                continue
            x = torch.cat([ids[n] for n in "BSX"]).to(dev)
            out = greedy(model, tok, x, max_new=a.max_new, stop=fresh.stop_fn(tok, it.words))
            for i, n in enumerate("BSX"):
                txt = fresh.gen_text(tok, x[i].tolist(), out[i])
                recs.append({"key": it.key, "arm": arm, "run": n, "lex": it.lex, "text": txt,
                             "frame": fresh.extract_frame(txt, it.names, it.words), "ans": fresh.parse(txt, it.words)})
        log(f"  calib {arm} done ({time.time() - t0:.0f}s)")
    cand = discover_frames([(r["arm"], r["frame"]) for r in recs], fixed=FRAMES_E_FIXED)
    admitted, rejected = admit_frames(tok, items, cand)
    counts = {}
    for r in recs:
        c = counts.setdefault(r["arm"], {})
        k = "(no candidate)" if r["frame"] is None else r["frame"]
        c[k] = c.get(k, 0) + 1
    res = {"provenance": prov(a, model, dev, {"population_sha256": fresh.POP_SHA256["C"], "n_items": len(items),
                                              "skipped_items": skipped, "wrapper": dict(WRAPPER_USED), "tokenizer": tokrec,
                                              "seconds": round(time.time() - t0, 1)}),
           "frames": admitted, "candidates": cand, "rejected": rejected, "counts": counts,
           "rule": "not a fixed E frame; >= 2% (and >= 2) of some arm's generations; at most 16, most frequent first; "
                   "the FormSet of every C item builds with it", "records": recs}
    f = Path(a.out) / "frames" / f"{a.tag}.json"
    write_atomic(res, f)
    log(f"frames {admitted} (rejected {list(rejected)}); file {f} sha256 {sha_file(f)}")


# --------------------------------------------------------------------------- stage g3 (JB-G3)
@torch.no_grad()
def stage_g3(a):
    model, tok, dev, tokrec = load_model(a)
    disc, ffile, fsha = read_frames(a)
    nL = len(blocks(model))
    items = pop_items(a, "F")[:a.n_g3]
    t0, arms, skipped = time.time(), {}, []
    for arm in G3_ARMS:
        T, P = [], []
        for it in items:
            try:
                fs, _ = fresh.form_set(tok, it, disc)
                T.append(run_item(model, tok, it, arm, fs, nL, dev, gen=False, scorer="trie"))
                P.append(plain_item(model, tok, it, arm, nL, dev))
            except Invalid as ex:
                skipped.append({"key": it.key, "arm": arm, "reason": str(ex)})
        arms[arm] = {"stat": g3_compare(T, P), "trie": [{k: t[k] for k in ("key", "words", "track", "clean", "rows")} for t in T],
                     "plain": P}
        log(f"  g3 {arm}: {arms[arm]['stat']} ({time.time() - t0:.0f}s)")
    ok = all(v["stat"]["pass"] for v in arms.values())
    res = {"provenance": prov(a, model, dev, {"population_sha256": fresh.POP_SHA256["F"], "frames_file": ffile,
                                              "frames_sha256": fsha, "skipped_items": skipped, "wrapper": dict(WRAPPER_USED),
                                              "tokenizer": tokrec, "seconds": round(time.time() - t0, 1)}),
           "arms": arms, "pass": ok, "scorer": "trie" if ok else "cached",
           "rule": f"per arm |s_ID^L(trie) - s_ID^L(plain)| <= {G3_DS} and mean |dL| <= {G3_DL} x mean ID_KV^L(plain)"}
    write_atomic(res, Path(a.out) / "g3" / f"{a.tag}.json")
    log(f"JB-G3 {'PASS: the evaluation scores with the trie' if ok else 'FAIL: the evaluation scores with score_cached'}")


# --------------------------------------------------------------------------- stage eval (R1, R2, R3)
@torch.no_grad()
def stage_eval(a):
    disc, ffile, fsha = read_frames(a)   # the frames' hash is printed before the first evaluation item
    g3f = Path(a.out) / "g3" / f"{a.tag_base}.json"
    g3 = json.loads(g3f.read_text())
    scorer = a.scorer or g3["scorer"]
    model, tok, dev, tokrec = load_model(a)
    nL = len(blocks(model))
    items = pop_items(a, a.population)
    arms = a.arms.split(",") if a.arms else list(fresh.ARMS_S0 if a.population == "S0" else fresh.ARMS_F)
    t0, res, skipped, not_run, dropped = time.time(), [], [], [], {}
    nodes, fsets = [], {}
    for arm in arms:
        if deadline_passed(a.reserve_min):
            not_run.append(arm)
            log(f"  {arm} not run: the deadline (minus {a.reserve_min} min reserve) has passed")
            continue
        ta = time.time()
        for it in items:
            try:
                if it.key not in fsets:   # one form set per item, the same in every arm
                    fsets[it.key] = fresh.form_set(tok, it, disc)
                fs, drop = fsets[it.key]
                if drop:
                    dropped[it.key] = drop
                r = run_item(model, tok, it, arm, fs, nL, dev, gen=True, scorer=scorer, max_new=a.max_new)
                nodes.append(len(fs))
                if a.population == "S0" and not a.no_plain:
                    pr = ff_run_item(model, tok, it.core, arm, "direct", dev)
                    if pr is None:
                        raise Invalid("format_factorial.run_item skipped the item")
                    r["plain"] = {"clean": pr["clean"], "m": pr["m"], "pos": pr["pos"], "len": pr["len"]}
                res.append(r)
            except Invalid as ex:
                skipped.append({"key": it.key, "arm": arm, "reason": str(ex)})
        log(f"  {arm} done: {sum(r['arm'] == arm for r in res)} items ({time.time() - ta:.0f}s; total {time.time() - t0:.0f}s)")
    P = prov(a, model, dev, {"population": a.population, "population_sha256": fresh.POP_SHA256[a.population],
                             "n_items": len(items), "arms": arms, "arms_not_run": not_run, "skipped_items": skipped,
                             "frames_file": ffile, "frames_sha256": fsha, "frames": disc, "frames_dropped": dropped,
                             "g3_file": str(g3f), "g3_sha256": sha_file(g3f), "g3_pass": g3["pass"], "scorer": scorer,
                             "trie_nodes": [min(nodes), max(nodes)] if nodes else None, "wrapper": dict(WRAPPER_USED),
                             "tokenizer": tokrec, "plain_pass": a.population == "S0" and not a.no_plain,
                             "seconds": round(time.time() - t0, 1)})
    path = Path(a.out) / "eval" / f"{a.tag}_{a.population}.json"
    write_atomic({"provenance": P, "results": res}, path)
    log(f"wrote {path} ({len(res)} items, {len(skipped)} skipped, arms not run {not_run}; {time.time() - t0:.0f}s)")
    return 3 if not_run else 0


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tokcheck", "calib", "g3", "eval"])
    ap.add_argument("--model", default=None, help="a local verified directory (scripts/stage8_common.sh s8_fetch) or a Hub id")
    ap.add_argument("--key", default="model", help="the model key of scripts/stage8_models.json (the output tag)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="auto", help="auto = eager for Gemma-2, sdpa otherwise (exploratory X1: eager)")
    ap.add_argument("--population", default="F", choices=["F", "S0"])
    ap.add_argument("--arms", default="", help="default: the population's arms (fresh.ARMS_F or fresh.ARMS_S0)")
    ap.add_argument("--n", type=int, default=0, help="0 = every item of the population")
    ap.add_argument("--n-cal", type=int, default=0, help="0 = every C item")
    ap.add_argument("--n-g3", type=int, default=N_G3)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--scorer", default=None, choices=["trie", "cached", "reference"], help="override the G3 choice (tests)")
    ap.add_argument("--no-plain", action="store_true", help="S0: skip the published-path pass (exploratory runs only)")
    ap.add_argument("--suffix", default="", help="output tag suffix of an exploratory run (X1, X2); frames and G3 of the base tag")
    ap.add_argument("--reserve-min", type=float, default=0.0, help="minutes the pipeline needs after this step (deadline)")
    ap.add_argument("--out", default="results/gpu_stage8b")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B FP32 on the CPU, n = 2")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    if a.test:
        a.model, a.revision, a.dtype = TINY, None, "float32"
        a.n, a.n_cal, a.n_g3 = a.n or 2, a.n_cal or 2, min(a.n_g3, 2)
    assert a.model, "--model is required"
    a.tag_base = ("TEST_" if a.test else "") + a.key
    a.tag = a.tag_base + a.suffix
    if a.stage == "calib" or a.stage == "g3":
        a.tag = a.tag_base
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    rc = {"tokcheck": stage_tokcheck, "calib": stage_calib, "g3": stage_g3, "eval": stage_eval}[a.stage](a)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())   # 3: complete up to the deadline, arms not run (scripts/stage8_common.sh); 1: tokcheck failed
