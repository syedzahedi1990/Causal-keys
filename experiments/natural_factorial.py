"""Stage 8 part A of P-2026-10-10-J (docs/PREREGISTRATION.md): the key/value format law on natural reading comprehension
(counterfactual SQuAD v1.1 dev items, data/stage8a_items.json, built by scripts/build_stage8a_items.py).

Per model (BF16; sdpa, eager for Gemma-2; use_cache=False in every scoring pass) and E item, six prompt formats
(ckeys.natural_formats: NOM, OPTA, OPTB, MENA, MENB, LETA; chat wrapper with the "Answer:" prefill). The B, S, X, Z
prompts differ only inside the answer entity's token span P; rows on the B prompt clamp the key and/or value at every
position of P in every layer from 0 (ckeys.natural_rows, ckeys.clamp). Stages (each a separate process, one results
file each; the pipeline is scripts/gpu_stage8a.sh):
  preflight  no model: the SQuAD file's sha256 (ckeys.squad_items.load_squad), the item rebuild with the build's
             tokenizers byte-identical to data/stage8a_items.json -> OUT/preflight.json
  frames     the model's answer frame (A-8): greedy ID generations (16 tokens) of the first 30 R items in NOM and OPTA;
             the text between "Answer:" and the entity, counted over FRAMES; the frame is the most frequent one under
             which >= 80 % of the E items are valid in every format (ties: FRAMES order; none: " ") -> OUT/frames/<tag>.json
  factorial  every E item valid for this tokenizer and frame in every format (the others are skipped with the reason):
             the closed-book prompts CBOPT / CBLET (decision log-probs of the four options after prompt + frame); per
             format, in the order NOM, OPTA, OPTB, MENA, LETA, MENB: the capture batch (B, S, X, Z prompts + w: K/V at P,
             each run's decision log-probs), the scoring batch (every core row on B prompt + c_S, teacher-forced:
             decision log-probs of B, S, X, Z, D (LETA: the letters of B, S, X, D), option mass, argmax, the log-probs of
             c_S after the decision token), the generation batch (greedy, 16 new tokens, stop at newline or EOS; NOM,
             OPTA, LETA: every core row; OPTB, MENA, MENB: ID, KV_S, KV_X; plus the unclamped S run) and, in NOM, OPTA
             and LETA, the KIVI batch (J-A7: the S prompt with none / keys / values of every passage token fake-quantized
             at 2 bits in every layer, ckeys.kvquant; greedy; relative reconstruction errors) -> OUT/factorial/<tag>.json
  explore    exploratory (deadline-gated by part): E1 onset and E2 piece rows (NOM, OPTA), E6 the X-target batch
             (NOM, OPTA), E4 the YEAR stratum and the LEAK stratum (NOM, OPTA), E5 extras (KIVI of both channels and at
             3 bits on the S prompt; none / keys / values at 2 bits on the B prompt) -> OUT/explore/<tag>.json
Batch sizes (A100-80GB; prompts up to ~780 tokens): each item's capture batch (4 rows), scoring batch (<= 14 rows) and
generation batch (<= 15 rows) is one forward; at 9B with eager attention the largest (15 rows x ~800 tokens) needs
< 6 GB beyond the weights.
Deadline (env STAGE8_DEADLINE, epoch seconds; --reserve-min, the minutes the pipeline still needs after this step):
before LETA and before MENB the factorial projects that format's time from the measured time of OPTA (for LETA) or
MENA (for MENB); if it would end later than the deadline minus the reserve, LETA runs with the reduced row set
(ckeys.natural_rows.LETA_REDUCED) and MENB is skipped; each decision is recorded in provenance.reduced. explore checks
the deadline at the start of each part and records each skipped part.
Exit status 3 when the factorial was reduced or explore skipped parts at the deadline (the pipeline's "partial").
TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B-Instruct FP32 on the CPU, passages windowed to <= 320 characters,
n = 2 E items (2 R items for the frames, 2 YEAR and 2 LEAK items), every output tagged TEST_<key>.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ckeys.encoding import WRAPPER_USED
from ckeys.generate import greedy
from ckeys.interventions import blocks
from ckeys.kvquant import quantize_kv
from ckeys.natural_formats import (CB_FORMATS, FORMATS, FRAMES, answer_of, choose_frame, decision_ids, encode_cb,
                                   encode_item, frame_of)
from ckeys.natural_rows import (BASE, COMPETENCE, FULL_FORMATS, Invalid, capture, core_rows, decided_stop, explore_rows,
                                gen_text, generate_rows, passage_positions, prep, score_rows, valid_all)
from ckeys.squad_items import SQUAD_DEV_SHA256, load_squad
from experiments.format_factorial import provenance as ff_provenance
from experiments.ioi_factorial import device_name
from experiments.stage6_heads import write_atomic

ROOT = Path(__file__).resolve().parents[1]
TINY = "Qwen/Qwen2.5-0.5B-Instruct"
ITEMS = ROOT / "data" / "stage8a_items.json"
FORMAT_ORDER = ("NOM", "OPTA", "OPTB", "MENA", "LETA", "MENB")
assert sorted(FORMAT_ORDER) == sorted(FORMATS)
N_FRAME_ITEMS, FRAME_VALID_MIN, MAX_NEW, QBITS = 30, 0.8, 16, 2
QUANT_FORMATS = ("NOM", "OPTA", "LETA")


def log(s):
    print(s, flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def deadline_passed(extra_s=0.0, reserve_min=0.0):
    d = os.environ.get("STAGE8_DEADLINE")
    return bool(d) and time.time() + extra_s > float(d) - 60 * reserve_min


# --------------------------------------------------------------------------- items
def window(it, maxc=320):
    """TEST_MODE only: the entity's sentence and, if the two fit in maxc characters, the one before it."""
    ctx, s0 = it["context"], it["start"]
    bounds = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\"'(])", ctx)] + [len(ctx)]
    k = max(i for i, b in enumerate(bounds[:-1]) if b <= s0)
    a, b = bounds[k], bounds[k + 1]
    if k > 0 and (b - bounds[k - 1]) <= maxc:
        a = bounds[k - 1]
    w = ctx[a:b].strip()
    off = ctx[a:b].index(w)
    return dict(it, context=w, start=s0 - a - off)


def load_items(a):
    """The committed items with their passages from the SQuAD file (sha256 asserted); windowed in TEST_MODE."""
    data = load_squad(a.squad)
    items = json.load(open(a.items))
    for it in items:
        it["context"] = data[it["art"]]["paragraphs"][it["par"]]["context"]
        assert it["context"][it["start"]:it["start"] + len(it["answer"])] == it["answer"], it["id"]
    if a.test:
        items = [window(it) for it in items]
    return items


def split(items, name):
    return sorted([i for i in items if i["split"] == name], key=lambda i: i["rank"])


# --------------------------------------------------------------------------- model
def load_model(a):
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    attn = "eager" if cfg.model_type.startswith("gemma2") else "sdpa"
    dev = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    kw = dict(dtype=getattr(torch, a.dtype), revision=a.revision, attn_implementation=attn)
    if dev == "cuda":
        kw["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if dev == "cpu":
        model = model.to(dev)
    assert model.config._attn_implementation == attn
    return model, tok, dev


def verified(model_dir):
    """The VERIFIED.json of a fetched model directory (scripts/fetch_verified.py), if any: its sha256 and revision."""
    f = Path(model_dir) / "VERIFIED.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    return {"sha256": sha_file(f), "revision": j.get("revision"), "repo": j.get("repo"), "source": j.get("source")}


def prov(a, model, dev, extra=None):
    cfg = model.config
    return ff_provenance(a) | {"model": a.model, "model_key": a.key, "revision": a.revision, "verified": verified(a.model),
                               "dtype": str(next(model.parameters()).dtype), "attn_implementation": cfg._attn_implementation,
                               "device": device_name(dev), "test_mode": a.test, "n_layers": len(blocks(model)),
                               "items_sha256": sha_file(a.items), "squad_sha256": SQUAD_DEV_SHA256, "max_new": MAX_NEW,
                               "timings": {}} | (extra or {})


def read_frame(a):
    f = Path(a.out) / "frames" / f"{a.tag}.json"
    j = json.load(open(f))
    return j["frame"], sha_file(f)


# --------------------------------------------------------------------------- stage preflight (no model)
def stage_preflight(a):
    t0 = time.time()
    data = load_squad(a.squad)   # asserts the sha256
    spec = importlib.util.spec_from_file_location("build_stage8a_items", ROOT / "scripts" / "build_stage8a_items.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    out, cnt, why, nft, R_titles = b.build(a.squad)
    blob = json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    same = Path(a.items).read_text() == blob
    by = collections.Counter(f"{i['split']}/{i['sub']}" for i in out)
    res = {"provenance": ff_provenance(a) | {"test_mode": a.test}, "squad_sha256": SQUAD_DEV_SHA256, "n_articles": len(data),
           "items_sha256": sha_file(a.items), "rebuild_sha256": hashlib.sha256(blob.encode()).hexdigest(), "rebuild_identical": same,
           "tokenizers": b.TOKENIZERS, "counts": dict(sorted(by.items())), "valid_ft": nft, "R_titles": R_titles,
           "build_counts": dict(sorted(cnt.items())), "invalid": dict(why), "seconds": round(time.time() - t0, 1)}
    write_atomic(res, Path(a.out) / "preflight.json")
    assert same, "the rebuilt items differ from data/stage8a_items.json"
    log(f"preflight OK: SQuAD sha256 {SQUAD_DEV_SHA256[:12]}, rebuild identical ({res['rebuild_sha256'][:12]}), {dict(by)}")


# --------------------------------------------------------------------------- stage frames
@torch.no_grad()
def stage_frames(a):
    model, tok, dev = load_model(a)
    items = load_items(a)
    R = split(items, "R")[:a.n_frame]
    E = split(items, "E")[:a.n] if a.n else split(items, "E")
    recs = []
    for fmt in ("NOM", "OPTA"):
        for it in R:
            _, ids, _, _ = encode_item(tok, it, fmt, it["answer"])
            g = greedy(model, tok, torch.tensor([ids]), max_new=MAX_NEW)[0]
            txt = gen_text(tok, ids, g)
            recs.append({"id": it["id"], "fmt": fmt, "text": txt, "prefix": frame_of(txt, it["answer"])})
    _, counts = choose_frame([r["prefix"] for r in recs])
    order = sorted([f for f in FRAMES if counts[f]], key=lambda f: (-counts[f], FRAMES.index(f)))
    rate = {}
    for f in order + [" "]:
        if f not in rate:
            rate[f] = sum(valid_all(tok, it, FORMATS, f) is None for it in E) / max(1, len(E))
    frame = next((f for f in order if rate[f] >= FRAME_VALID_MIN), " ")
    res = {"provenance": prov(a, model, dev), "frame": frame, "counts": counts, "candidates": order, "valid_rate": rate,
           "rule": f"most frequent of FRAMES under which >= {FRAME_VALID_MIN} of the E items are valid; ties FRAMES order; else ' '",
           "n_items": len(R), "records": recs}
    write_atomic(res, Path(a.out) / "frames" / f"{a.tag}.json")
    log(f"frame {frame!r} (counts {counts}; valid rates {rate})")


# --------------------------------------------------------------------------- stage factorial
def closed_book(model, tok, it, frame, dev):
    out = {}
    for fmt in CB_FORMATS:
        text, ids = encode_cb(tok, it, fmt)
        try:
            dd = decision_ids(tok, text, ids, it, fmt, frame)
        except AssertionError as ex:
            out[fmt] = {"invalid": str(ex)}
            continue
        lg = model(torch.tensor([ids + dd["w"]]).to(dev), use_cache=False, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
        lp = {Y: float(lg[t]) for Y, t in dd["dec"].items()}
        out[fmt] = {"lp": lp, "win": max(lp, key=lp.get), "T": len(ids)}
    return out


def run_quant(model, tok, it, d, nL, run, rows, bits, max_new):
    """Greedy generation of the ``run`` prompt with the passage keys / values fake-quantized: rows {name: channels}."""
    pos = passage_positions(tok, d, it, run)
    names = list(rows)
    ids = torch.tensor([d["ids"][run]] * len(names))
    sel = {ch: [ch in rows[n] for n in names] for ch in "kv"}
    stats = {}
    with quantize_kv(model, pos, bits, sel, stats=stats):
        gens = greedy(model, tok, ids, max_new=max_new, stop=decided_stop(tok, it, d["fmt"]))
    out = {"n_pos": len(pos), "bits": bits, "rows": {}, "relerr": {}}
    for r, n in enumerate(names):
        txt = gen_text(tok, d["ids"][run], gens[r])
        g = gens[r]
        out["rows"][n] = {"ids": g, "text": txt, "who": answer_of(txt, it, d["fmt"]),
                          "g1": g[len(d["w"])] if len(g) > len(d["w"]) and g[:len(d["w"])] == d["w"] else None}
    for ch in "kv":
        k = [n for n in names if ch in rows[n]]
        for i, n in enumerate(k):
            out["relerr"].setdefault(n, {})[ch] = [float(stats[(l, ch)][i]) for l in range(nL)]
    return out


def diff_mask(d, target="S"):
    """Per token of c_target after the decision token: whether it differs from c_B at the same index."""
    c, cb, j = d["c"][target], d["c"]["B"], d["j"]
    return [t >= len(cb) or c[t] != cb[t] for t in range(j + 1, len(c))]


@torch.no_grad()
def run_item(model, tok, it, fmt, frame, nL, dev, reduced=False, quant=True):
    d = prep(tok, it, fmt, frame)
    kv, runs = capture(model, d, range(nL), dev)
    rows = core_rows(fmt, reduced)
    sc = score_rows(model, d, kv, rows, nL, dev, "S")
    grows = rows if fmt in FULL_FORMATS else {n: rows[n] for n in COMPETENCE}
    gens = generate_rows(model, tok, it, d, kv, grows, nL, MAX_NEW, extra={"S_run": "S"})
    rec = dict(id=it["id"], art=it["art"], title=it["title"], sub=it["sub"], D_in=it["D_in"], stratum=it["stratum"],
               P=d["P"], nP=len(d["P"]), T=d["T"], j=d["j"], w=d["w"], dec=d["dec"], c=d["c"], diff_S=diff_mask(d),
               rows=sc, runs=runs, gen=gens)
    if quant and fmt in QUANT_FORMATS:
        rec["quant"] = run_quant(model, tok, it, d, nL, "S", {"none": "", "K2": "k", "V2": "v"}, QBITS, MAX_NEW)
    return rec


@torch.no_grad()
def stage_factorial(a):
    frame, fsha = read_frame(a)
    model, tok, dev = load_model(a)
    nL = len(blocks(model))
    items = load_items(a)
    E = split(items, "E")
    valid, skipped = [], []
    for it in E:
        why = valid_all(tok, it, FORMATS, frame)
        (skipped.append({"id": it["id"], "reason": why}) if why else valid.append(it))
        if a.n and len(valid) == a.n:
            break
    P = prov(a, model, dev, {"frame": frame, "frames_sha256": fsha, "skipped_items": skipped, "reduced": {},
                             "population_sha256": sha_ids([i["id"] for i in valid]), "n_valid": len(valid), "qbits": QBITS,
                             "format_order": list(FORMAT_ORDER), "reserve_min": a.reserve_min})
    path = Path(a.out) / "factorial" / f"{a.tag}.json"
    out, t0 = {"provenance": P, "frame": frame, "closed_book": {}, "formats": {}}, time.time()
    for it in valid:
        out["closed_book"][it["id"]] = closed_book(model, tok, it, frame, dev)
    P["timings"]["closed_book"] = round(time.time() - t0, 1)
    log(f"[{a.tag}] frame {frame!r}; {len(valid)} valid E items, {len(skipped)} skipped; closed book ({time.time() - t0:.0f}s)")
    sec = {}
    for fmt in FORMAT_ORDER:
        reduced = False
        if fmt in ("LETA", "MENB"):
            ref = sec.get("OPTA" if fmt == "LETA" else "MENA")
            if ref is not None and deadline_passed(ref, a.reserve_min):
                if fmt == "MENB":
                    P["reduced"][fmt] = f"skipped: projected {ref:.0f}s past the deadline minus the reserve of {a.reserve_min} min"
                    log(f"[{a.tag}] {fmt} skipped at the deadline")
                    continue
                reduced = True
                P["reduced"][fmt] = f"rows reduced to {'/'.join(core_rows('LETA', True))}: projected {ref:.0f}s past the deadline minus the reserve"
        t1, recs, bad = time.time(), [], []
        for it in valid:
            try:
                recs.append(run_item(model, tok, it, fmt, frame, nL, dev, reduced))
            except Invalid as ex:   # cannot happen after valid_all; kept for safety and recorded
                bad.append({"id": it["id"], "reason": str(ex)})
        sec[fmt] = time.time() - t1
        out["formats"][fmt] = {"items": recs, "skipped": bad, "rows": list(core_rows(fmt, reduced)), "reduced": reduced,
                               "seconds": round(sec[fmt], 1)}
        P["timings"][fmt] = round(sec[fmt], 1)
        P["wrapper"] = dict(WRAPPER_USED)
        write_atomic(out, path)
        log(f"[{a.tag}] {fmt}: {len(recs)} items ({sec[fmt]:.0f}s)")
    P["complete"] = True
    write_atomic(out, path)
    log(f"wrote {path} ({time.time() - t0:.0f}s)" + (f"; reduced at the deadline: {P['reduced']}" if P["reduced"] else ""))
    return 3 if P["reduced"] else 0


# --------------------------------------------------------------------------- stage explore
@torch.no_grad()
def stage_explore(a):
    frame, fsha = read_frame(a)
    model, tok, dev = load_model(a)
    nL = len(blocks(model))
    items = load_items(a)
    pick = lambda name: [i for i in split(items, name) if valid_all(tok, i, ("NOM", "OPTA"), frame) is None][:a.n or None]  # noqa: E731
    E = [i for i in split(items, "E") if valid_all(tok, i, FORMATS, frame) is None][:a.n or None]
    P = prov(a, model, dev, {"frame": frame, "frames_sha256": fsha, "skipped_parts": [], "onset": round(0.3 * nL)})
    path = Path(a.out) / "explore" / f"{a.tag}.json"
    out, t0 = {"provenance": P, "parts": {}}, time.time()

    def part(name, fn):
        if deadline_passed():
            P["skipped_parts"].append(name)
            log(f"[{a.tag}] explore {name} skipped: the deadline has passed")
            return
        t1 = time.time()
        out["parts"][name] = fn()
        P["timings"][name] = round(time.time() - t1, 1)
        write_atomic(out, path)
        log(f"[{a.tag}] explore {name} ({time.time() - t1:.0f}s)")

    def rows_part(its, fmts, rows_of, target="S", gen=None):
        res = {}
        for fmt in fmts:
            recs = []
            for it in its:
                d = prep(tok, it, fmt, frame)
                kv, runs = capture(model, d, range(nL), dev)
                rows = rows_of(fmt)
                r = dict(id=it["id"], art=it["art"], title=it["title"], sub=it["sub"], nP=len(d["P"]), j=d["j"], w=d["w"],
                         dec=d["dec"], c=d["c"], diff_S=diff_mask(d), rows=score_rows(model, d, kv, rows, nL, dev, target))
                if gen:
                    r["gen"] = generate_rows(model, tok, it, d, kv, {n: rows[n] for n in gen}, nL, MAX_NEW)
                recs.append(r)
            res[fmt] = recs
        return res

    part("onset_piece", lambda: rows_part(E, ("NOM", "OPTA"), lambda f: explore_rows(f, nL)))
    part("x_target", lambda: rows_part(E, ("NOM", "OPTA"), lambda f: dict(BASE), target="X"))
    part("year", lambda: rows_part(pick("YEAR"), ("NOM", "OPTA"), lambda f: {n: BASE[n] for n in ("ID", "K_S", "V_S", "KV_S")}))
    part("leak", lambda: rows_part(pick("LEAK"), ("NOM", "OPTA"), lambda f: dict(BASE), gen=COMPETENCE))

    def quant_extra():
        res = {}
        for fmt in QUANT_FORMATS:
            recs = []
            for it in E:
                d = prep(tok, it, fmt, frame)
                r = {"id": it["id"], "art": it["art"], "title": it["title"],
                     "B2": run_quant(model, tok, it, d, nL, "B", {"none": "", "K2": "k", "V2": "v"}, 2, MAX_NEW)}
                if fmt != "LETA":
                    r["S3"] = run_quant(model, tok, it, d, nL, "S", {"K3": "k", "V3": "v"}, 3, MAX_NEW)
                recs.append(r)
            res[fmt] = recs
        return res

    part("quant_extra", quant_extra)
    P["complete"] = True
    write_atomic(out, path)
    log(f"wrote {path} ({time.time() - t0:.0f}s)")
    return 3 if P["skipped_parts"] else 0


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["preflight", "frames", "factorial", "explore"])
    ap.add_argument("--model", default=None, help="a Hub name or a local directory (scripts/stage8_common.sh s8_fetch)")
    ap.add_argument("--key", default="model", help="the model key of scripts/stage8_models.json (the output tag)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--squad", required=True, help="SQuAD v1.1 dev-v1.1.json (sha256 asserted)")
    ap.add_argument("--items", default=str(ITEMS))
    ap.add_argument("--out", default="results/gpu_stage8a")
    ap.add_argument("--n", type=int, default=0, help="0 = every E item")
    ap.add_argument("--n-frame", type=int, default=N_FRAME_ITEMS)
    ap.add_argument("--reserve-min", type=float, default=0.0, help="minutes the pipeline needs after this step (deadline)")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B FP32 on the CPU, windowed passages, n = 2")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    if a.test:
        a.model, a.revision, a.dtype, a.n, a.n_frame = TINY, None, "float32", 2, 2
    assert a.model or a.stage == "preflight", "--model is required"
    a.tag = ("TEST_" if a.test else "") + a.key
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    rc = {"preflight": stage_preflight, "frames": stage_frames, "factorial": stage_factorial, "explore": stage_explore}[a.stage](a)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())   # 3: complete up to the deadline, parts skipped or reduced (scripts/stage8_common.sh)
