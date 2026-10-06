"""Part (b) of preregistration P-2026-10-05-H: the key-only / value-only exchange on Prakash et al.'s (2026) reversed-
sentence binding swap (BIND) and on an identity edit at the same positions (ID), on their stories, raw wrapper and seed-10
pool (ckeys.causaltom), Qwen2.5-14B-Instruct (BF16, sdpa, use_cache=False, logits_to_keep=1); MODEL=llama70 optional.

Stages (--stage, comma-separated or 'all'; each writes one file under {out}/{model}/ and reads the earlier ones):
  preflight  (tokenizer only, before any weights) pool hash, release sha256s, single-token check of every entity word
             with a leading space, lengths and positions of all 320 pairs in every format, B/S/X differing at exactly p,
             the cf spans at the swapped positions -> preflight.json
  filter     their LM filter under NO-MENTION as unpadded single forwards (argmax decode lower/strip == answer) on both
             prompts of every pair -> filter.json; the population is the first --n passing pairs in pool order
  sweep      Stage A, NO-MENTION: per layer l, batches of up to 8 pairs = the patched rows + the same pairs unpatched (in-
             batch self rows); IIA(l) and Phi(l) = mean m(M_l) - m(B); BIND: block-l output at [s2, s2+1, s1, s1+1] <- the
             cf's at [s1, s1+1, s2, s2+1]; ID: at [p, p+1] <- the donor's (s_q -> S); m = logp(target) - logp(s_q)
             -> sweep_{BIND,ID}_{format}.json; l* / l*_ID = the earliest layer with IIA >= max - 0.01 -> lstar.json
  exchange   Stage B, one batch per pair, arm, format and depth (BIND at l*, ID at l*_ID and at l*): r0 B+KV_B, r1 M,
             r2 B+K_M, r3 B+V_M, r4 B+KV_M, r5 M+K_B, r6 M+V_B (the resid patch active only in r1, r5, r6), words-only
             r2w/r3w/r4w (K/V of the state words only); K/V of the patched positions in blocks > l_patch; Gate b0
             (|m(r4) - m(r1)|, |m(r0) - m(B unbatched)|) logged per batch, asserted <= 1e-3 in FP32 -> exchange.json
  clamp      Stage C, natural K/V clamps at p = the queried state word on the same clean stories, one batch per onset
             l0 (rows ID, K_S, V_S, KV_S, K_X, V_X, KV_X in every block >= l0) from one capture run per prompt (B, S, X)
             shared by every onset; onsets {l*+1, l*_ID+1, 0, 3, 14} + the exploratory sweep -> clamp.json
TEST_MODE (--test, or TEST_MODE=1 which keeps --n): Qwen2.5-0.5B-Instruct, FP32, CPU, no LM filter, n = 2, NO-MENTION and
OPTIONS-AFTER, both arms, sweep over layers 0,4,...,20,23, the primary onsets only.
--model llama70 (= meta-llama/Meta-Llama-3-70B-Instruct, --device-map auto on 2 GPUs): BIND sweep 25..44 (the release's
list), ID sweep 0..78 step 2 plus 25..44, BOS-shifted positions; without access to the gated weights it writes SKIPPED.txt, exit 0.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys import causaltom as ct
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.interventions import _out_tensor, _with_tensor, blocks, hooks
from experiments.ioi_factorial import device_name

MODELS = {"qwen14": "Qwen/Qwen2.5-14B-Instruct", "llama70": "meta-llama/Meta-Llama-3-70B-Instruct"}
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIRM = ("NO-MENTION", "QNAMES", "OPTIONS-AFTER")
ROWS = ("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r2w", "r3w", "r4w")
# row -> (key donor, value donor, resid patch active, words only)
ROWDEF = {"r0": ("B", "B", 0, 0), "r1": (None, None, 1, 0), "r2": ("M", "B", 0, 0), "r3": ("B", "M", 0, 0), "r4": ("M", "M", 0, 0),
          "r5": ("B", "M", 1, 0), "r6": ("M", "B", 1, 0), "r2w": ("M", "B", 0, 1), "r3w": ("B", "M", 0, 1), "r4w": ("M", "M", 0, 1)}
CROWS = (("ID", "B", "B"), ("K_S", "S", "B"), ("V_S", "B", "S"), ("KV_S", "S", "S"), ("K_X", "X", "B"), ("V_X", "B", "X"), ("KV_X", "X", "X"))
ONSETS_FIXED = (0, 3, 14)
ONSET_SWEEP_14B = (0, 6, 12, 18, 24, 27, 30, 33, 36, 42)
ROLES = ("s_q", "other", "S", "X")


def tie_rule(iia: dict) -> int:
    """The earliest layer maximising IIA, ties within 0.01 of the maximum going to the earliest."""
    mx = max(iia.values())
    return min(int(l) for l, v in iia.items() if v >= mx - 0.01 - 1e-9)


# --------------------------------------------------------------------------- hooks
@contextlib.contextmanager
def capture_resid(model, layers, pos_rows):
    """{layer: [R, P, D]} block outputs at per-row positions ``pos_rows[r]``."""
    store, hs = {}, []
    for l in layers:
        def fn(_m, _i, out, l=l):
            h = _out_tensor(out)
            store[l] = torch.stack([h[r, list(p)] for r, p in enumerate(pos_rows)]).detach().clone()
        hs.append(blocks(model)[l].register_forward_hook(fn))
    with hooks(hs):
        yield store


@contextlib.contextmanager
def resid_patch(model, layer, dst_rows, src_rows):
    """At the output of block ``layer`` set row r's residual at ``dst_rows[r]`` to ``src_rows[r]`` ([P, D]); a row whose
    destination is None is left alone (the patch is inactive there)."""
    def fn(_m, _i, out):
        h = _out_tensor(out)
        if h.shape[1] <= max(max(d) for d in dst_rows if d is not None):
            return out
        h = h.clone()
        for r, (d, s) in enumerate(zip(dst_rows, src_rows)):
            if d is not None:
                h[r, list(d)] = s.to(h.device, h.dtype)
        return _with_tensor(out, h)
    with hooks([blocks(model)[layer].register_forward_hook(fn)]):
        yield


@torch.no_grad()
def logprobs(model, ids):
    dev = model.get_input_embeddings().weight.device
    x = torch.tensor(ids, device=dev) if not isinstance(ids, torch.Tensor) else ids.to(dev)
    x = x[None] if x.dim() == 1 else x
    return torch.log_softmax(model(x, use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1).cpu()


# --------------------------------------------------------------------------- items
def item(tok, rel, pr, fmt, idx):
    L = ct.locate(tok, rel, pr, fmt)
    m = L["meta"]
    w = ct.readout_words(fmt, m)
    L["tid"] = {k: ct.single_id(tok, v) for k, v in w.items()}
    L["word"] = w
    L["arm"] = {"BIND": dict(dst=L["bind_dst"], src=L["bind_src"], donor="C", P=L["P"], Pw=L["Pw"], target="other"),
                "ID": dict(dst=L["id_span"], src=L["id_span"], donor="S", P=L["id_span"], Pw=[L["p"]], target="S")}
    L["i"] = idx
    return L


def mval(lp, L, arm):
    t = L["tid"]
    return (lp[..., t[L["arm"][arm]["target"]]] - lp[..., t["s_q"]]).tolist()


def roles(lp, L):
    return {k: lp[L["tid"][k]].item() for k in ROLES}


# --------------------------------------------------------------------------- stages
def preflight(tok, rel, pairs, formats):
    t0 = time.time()
    bad = [w for w in rel.characters + rel.objects + rel.states if len(tok.encode(" " + w, add_special_tokens=False)) != 1]
    assert not bad, f"not single tokens with a leading space: {bad}"
    letters = {L: tok.encode(" " + L, add_special_tokens=False) for L in ct.LETTERS}
    assert all(len(v) == 1 for v in letters.values()), letters
    fm, n_bos = {}, set()
    for f in formats:
        lens, pos = set(), set()
        for pr in pairs:
            L = ct.locate(tok, rel, pr, f)   # asserts lengths, positions, punctuation, the cf spans, B/S/X diffs at p
            lens.add(L["T"]); pos.add(tuple(L["P"])); n_bos.add(L["n_bos"])
        assert len(lens) == 1 and len(pos) == 1, (f, lens, pos)
        fm[f] = {"length": lens.pop(), "positions": list(pos.pop())}
    nb = n_bos.pop()
    return {"pool_sha256": ct.pool_hash(pairs), "n_pool": len(pairs), "release": {"url": ct.RELEASE_URL, "sha": ct.RELEASE_SHA, "files": rel.hashes},
            "n_words_single_token": len(rel.characters + rel.objects + rel.states), "letters": letters, "n_bos": nb,
            "punct_ids": list(ct.PUNCT[nb]), "formats": fm, "seconds": round(time.time() - t0, 1)}


def lm_filter(model, tok, pairs):
    out = []
    for i, pr in enumerate(pairs):
        pc = logprobs(model, ct.encode(tok, pr["clean_prompt"])["input_ids"]).argmax(-1).item()
        pf = logprobs(model, ct.encode(tok, pr["counterfactual_prompt"])["input_ids"]).argmax(-1).item()
        ok = ct.correct(tok, pc, pr["clean_ans"]) and ct.correct(tok, pf, pr["counterfactual_ans"])
        out.append({"i": i, "clean_pred": tok.decode([pc]), "cf_pred": tok.decode([pf]), "ok": ok})
        if i % 40 == 39:
            print(f"  filter {i + 1}/{len(pairs)}: {sum(o['ok'] for o in out)} pass", flush=True)
    return out


def sweep(model, tok, items, arm, layers, bs=8):
    rows = {l: [] for l in layers}
    for c0 in range(0, len(items), bs):
        ch = items[c0:c0 + bs]
        n = len(ch)
        A = [L["arm"][arm] for L in ch]
        with capture_resid(model, layers, [a["src"] for a in A]) as R:
            logprobs(model, [L["ids"][a["donor"]] for L, a in zip(ch, A)])
        ids = [L["ids"]["B"] for L in ch] * 2
        for l in layers:
            with resid_patch(model, l, [a["dst"] for a in A] + [None] * n, list(R[l]) + [None] * n):
                lp = logprobs(model, ids)
            for j, L in enumerate(ch):
                pj, sj = lp[j], lp[n + j]
                rows[l].append({"i": L["i"], "m_patch": mval(pj, L, arm), "m_self": mval(sj, L, arm),
                                "pred": tok.decode([pj.argmax().item()]), "pred_self": tok.decode([sj.argmax().item()]),
                                "ok": ct.correct(tok, pj.argmax().item(), L["word"][A[j]["target"]])})
        print(f"  sweep {arm}: pairs {c0 + n}/{len(items)} " + " ".join(
            f"{l}:{sum(r['ok'] for r in rows[l]) / len(rows[l]):.2f}" for l in layers), flush=True)
    summ = {l: {"IIA": sum(r["ok"] for r in v) / len(v), "Phi": sum(r["m_patch"] - r["m_self"] for r in v) / len(v)} for l, v in rows.items()}
    return rows, summ


def run_pair(model, tok, L, depths, onsets, do_ex, do_cl, nL, exact, log):
    """Stages B and C of one pair in one format: four capture runs (B, C, S, X), one M run and one exchange batch per
    (arm, depth), one clamp batch per onset."""
    P, p = L["P"], L["p"]
    allL = list(range(nL))
    need = sorted({d for _, d in depths})
    with capture_kv(model, P, allL) as kvB:
        lpB = logprobs(model, L["ids"]["B"])[0]
    with capture_resid(model, need, [L["bind_src"]]) as rC:
        lpC = logprobs(model, L["ids"]["C"])[0]
    with capture_kv(model, P, allL) as kvS, capture_resid(model, need, [L["id_span"]]) as rS:
        logprobs(model, L["ids"]["S"])
    with capture_kv(model, P, allL) as kvX:
        logprobs(model, L["ids"]["X"])
    base = {"i": L["i"], "q": L["meta"]["q"], "pred_B": tok.decode([lpB.argmax().item()]), "pred_C": tok.decode([lpC.argmax().item()]),
            "ok_B": ct.correct(tok, lpB.argmax().item(), L["word"]["s_q"]), "ok_C": ct.correct(tok, lpC.argmax().item(), L["word"]["s_q"]),
            "lp_B": roles(lpB, L)}
    ex, cl = [], []
    for arm, d in depths if do_ex else ():
        a = L["arm"][arm]
        cols = [P.index(x) for x in a["P"]]
        wcols = torch.tensor([x in a["Pw"] for x in a["P"]])
        src = (rC if arm == "BIND" else rS)[d][0]
        lay = list(range(d + 1, nL))
        with resid_patch(model, d, [a["dst"]], [src]), capture_kv(model, a["P"], lay) as kvM:
            lpM = logprobs(model, L["ids"]["B"])[0]
        tab = {"B": {k: v[0][cols] for k, v in kvB.items()}, "M": {k: v[0] for k, v in kvM.items()}}
        tables = {(l, ch): torch.stack([tab[ROWDEF[r]["kv".index(ch)] or "M"][(l, ch)] for r in ROWS]) for l in lay for ch in "kv"}
        per_row = torch.stack([torch.zeros(len(cols), dtype=torch.bool) if r == "r1" else wcols if ROWDEF[r][3]
                               else torch.ones(len(cols), dtype=torch.bool) for r in ROWS])
        act = [a["dst"] if ROWDEF[r][2] else None for r in ROWS]
        with resid_patch(model, d, act, [src if x else None for x in act]), clamp_kv(model, a["P"], tables, lay, "kv", per_row):
            lp = logprobs(model, [L["ids"]["B"]] * len(ROWS))
        m = dict(zip(ROWS, mval(lp, L, arm)))
        mB, mM = mval(lpB, L, arm), mval(lpM, L, arm)
        b0 = {"r4_r1": abs(m["r4"] - m["r1"]), "r0_B": abs(m["r0"] - mB), "r1_M": abs(m["r1"] - mM)}
        log(f"    b0 {arm}@{d} pair {L['i']}: |m(r4)-m(r1)| {b0['r4_r1']:.2e} |m(r0)-m(B)| {b0['r0_B']:.2e} |m(r1)-m(M unbatched)| {b0['r1_M']:.2e}")
        if exact:
            assert b0["r4_r1"] <= 1e-3 and b0["r0_B"] <= 1e-3, f"Gate b0 (FP32) failed: {b0}"
        top = lp.argmax(-1)
        ex.append(base | {"arm": arm, "depth": d, "m": m, "m_B": mB, "m_M": mM, "b0": b0,
                          "pred": {r: tok.decode([top[k].item()]) for k, r in enumerate(ROWS)},
                          "ok_r1": ct.correct(tok, top[1].item(), L["word"][a["target"]]),
                          "lp": {r: roles(lp[k], L) for k, r in enumerate(ROWS)}})
    if do_cl:
        k = P.index(p)
        kv = {w: {key: v[0][[k]] for key, v in src.items()} for w, src in (("B", kvB), ("S", kvS), ("X", kvX))}
        for l0 in onsets:
            lay = list(range(l0, nL))
            tables = {(l, ch): torch.stack([kv[(kd, vd)["kv".index(ch)]][(l, ch)] for _, kd, vd in CROWS]) for l in lay for ch in "kv"}
            with clamp_kv(model, [p], tables, lay, "kv"):
                lp = logprobs(model, [L["ids"]["B"]] * len(CROWS))
            cl.append(base | {"l0": l0, "lp": {name: roles(lp[j], L) for j, (name, _, _) in enumerate(CROWS)},
                              "pred": {name: tok.decode([lp[j].argmax().item()]) for j, (name, _, _) in enumerate(CROWS)}})
    return ex, cl


# --------------------------------------------------------------------------- provenance, access
def access(model_id, revision=None):
    """(ok, reason): the gated weights are reachable with the current token."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(model_id, "config.json", revision=revision)
        return True, ""
    except Exception as ex:
        return False, f"{type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ''}"


def provenance(a, model=None):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    p = {"args": vars(a), "git_commit": commit, "torch": torch.__version__, "transformers": transformers.__version__,
         "python": platform.python_version(), "release_sha": ct.RELEASE_SHA, "pool_sha256": ct.POOL_SHA256}
    if model is not None:
        p |= {"dtype": str(next(model.parameters()).dtype), "device": device_name(next(model.parameters()).device),
              "attn_implementation": model.config._attn_implementation, "n_layers": len(blocks(model)),
              "hf_device_map": {str(k): str(v) for k, v in (getattr(model, "hf_device_map", None) or {}).items()} or None}
    return p


def dump(path, prov, body):
    tmp = Path(str(path) + ".tmp")
    json.dump({"provenance": prov} | body, open(tmp, "w"))
    tmp.replace(path)
    print(f"wrote {path}", flush=True)


def ints(s):
    return [int(x) for x in s.split(",")] if s else []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen14", help="qwen14, llama70 or a Hub id")
    ap.add_argument("--stage", default="all", help="preflight,filter,sweep,exchange,clamp or all")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--formats", default="NO-MENTION,QNAMES,OPTIONS-AFTER,LETTERS-AFTER,QNAMES2")
    ap.add_argument("--sweep-layers", default=None, help="BIND sweep layers (default all; llama70 25..44)")
    ap.add_argument("--sweep-layers-id", default=None, help="ID sweep layers (default all; llama70 0..78 step 2 and 25..44)")
    ap.add_argument("--explore-sweeps", default="QNAMES,OPTIONS-AFTER,LETTERS-AFTER", help="exploratory sweep formats ('' = none)")
    ap.add_argument("--explore-layers", default=None, help="exploratory BIND sweep layers (default 18..40 scaled to depth)")
    ap.add_argument("--onset-sweep", default=None, help="exploratory onsets under NO-MENTION and OPTIONS-AFTER (default 0,6,...,42 scaled)")
    ap.add_argument("--batch", type=int, default=8, help="pairs per sweep batch (2 x rows)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prakash-repo", default=None, help="checkout of the release (default $PRAKASH_REPO)")
    ap.add_argument("--out", default="results/gpu_stage6/prakash")
    ap.add_argument("--label", default="", help="exploratory sweep re-run (e.g. _fp32 with --dtype float32 --stage sweep --sweep-layers ...): "
                    "files sweep_*<label>.json, lstar.json untouched, ignored by the scorer")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B, FP32, CPU, n = 2, no LM filter")
    a = ap.parse_args(argv)
    a.model = MODELS.get(a.model, a.model)
    if a.test or os.environ.get("TEST_MODE"):
        a.n = min(a.n, 2) if a.test else a.n
        a.test, a.model, a.dtype = True, TEST_MODEL, "float32"
        a.formats = "NO-MENTION,OPTIONS-AFTER"
        a.sweep_layers = a.sweep_layers_id = a.sweep_layers or "0,4,8,12,16,20,23"
        a.explore_sweeps, a.onset_sweep = "", a.onset_sweep or "0"
        a.out = a.out if a.out != ap.get_default("out") else tempfile.mkdtemp(prefix="stage6_prakash_test_")
        print(f"TEST_MODE: {a.model}, float32, CPU, n = {a.n}, formats {a.formats}, no LM filter", flush=True)
    llama = "llama" in a.model.lower()
    if llama and a.device_map is None and not a.test:
        a.device_map = "auto"
    stages = ["preflight", "filter", "sweep", "exchange", "clamp"] if a.stage == "all" else a.stage.split(",")
    out = Path(a.out) / a.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    if "meta-llama" in a.model:
        ok, why = access(a.model, a.revision)
        if not ok:
            (out / "SKIPPED.txt").write_text(f"no access to the gated weights of {a.model}: {why}\n")
            print(f"SKIP {a.model}: no access ({why}); wrote {out / 'SKIPPED.txt'}")
            return
    formats = a.formats.split(",")
    rel = ct.load(a.prakash_repo)
    pairs = ct.pool(rel)   # asserts the pool hash
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    pf = preflight(tok, rel, pairs, sorted(set(formats) | set(ct.FORMATS)))
    print(f"preflight OK: pool {pf['pool_sha256'][:16]}, {pf['n_words_single_token']} single-token words, n_bos {pf['n_bos']}, "
          + ", ".join(f"{f} T={v['length']} P={v['positions']}" for f, v in pf["formats"].items()), flush=True)
    dump(out / "preflight.json", provenance(a), pf)
    if stages == ["preflight"]:
        return
    t0 = time.time()
    kw = {"dtype": getattr(torch, a.dtype), "revision": a.revision, "attn_implementation": a.attn}
    if a.device_map:
        kw["device_map"] = a.device_map
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if not a.device_map:
        model = model.to("cuda" if torch.cuda.is_available() and not a.test else "cpu")
    nL = len(blocks(model))
    prov = provenance(a, model)
    print(f"loaded {a.model} ({nL} layers, {prov['dtype']}, {prov['device']}) in {time.time() - t0:.0f}s", flush=True)

    if "filter" in stages and not a.test:
        fl = lm_filter(model, tok, pairs)
        dump(out / "filter.json", prov, {"pairs": fl, "n_pass": sum(f["ok"] for f in fl), "accuracy": sum(f["ok"] for f in fl) / len(fl)})
    if a.test:
        pop = list(range(a.n))
        dump(out / "filter.json", prov, {"pairs": [], "n_pass": None, "accuracy": None, "test_no_filter": True, "population": pop})
    else:
        fl = json.load(open(out / "filter.json"))["pairs"]
        pop = [f["i"] for f in fl if f["ok"]][:a.n]
    print(f"population: {len(pop)} pairs (first {pop[:5]} ...)", flush=True)

    full = list(range(nL))
    scale = lambda xs: sorted({min(nL - 1, round(x * nL / 48)) for x in xs})  # noqa: E731
    lay_b = ints(a.sweep_layers) or (list(range(25, 45)) if llama else full)
    lay_i = ints(a.sweep_layers_id) or (sorted(set(range(0, nL, 2)) | set(lay_b)) if llama else full)   # l* swept for H11 Part 2
    if "sweep" in stages:
        items = [item(tok, rel, pairs[i], "NO-MENTION", i) for i in pop]
        ls = {}
        for arm, lay in (("BIND", lay_b), ("ID", lay_i)):
            rows, summ = sweep(model, tok, items, arm, lay, a.batch)
            ls[arm] = tie_rule({l: s["IIA"] for l, s in summ.items()})
            dump(out / f"sweep_{arm}_NO-MENTION{a.label}.json", prov, {"arm": arm, "format": "NO-MENTION", "layers": lay, "population": pop,
                                                                        "summary": summ, "rows": rows} | ({"label": a.label} if a.label else {}))
        if a.label:   # an exploratory re-run (e.g. FP32 at l* +- 2): never selects l*
            return
        dump(out / "lstar.json", prov, {"lstar": ls["BIND"], "lstar_ID": ls["ID"], "rule": "earliest layer with IIA >= max IIA - 0.01 (NO-MENTION sweep)",
                                         "population": pop})
        print(f"l* = {ls['BIND']}, l*_ID = {ls['ID']}", flush=True)
        for f in filter(None, a.explore_sweeps.split(",")):   # exploratory: the swap's layer band per format
            it = [item(tok, rel, pairs[i], f, i) for i in pop]
            for arm, lay in (("BIND", ints(a.explore_layers) or (lay_b if llama else scale(range(18, 41)))), ("ID", lay_i)):
                rows, summ = sweep(model, tok, it, arm, lay, a.batch)
                dump(out / f"sweep_{arm}_{f}.json", prov, {"arm": arm, "format": f, "layers": lay, "population": pop, "summary": summ,
                                                          "rows": rows, "exploratory": True})
    if "exchange" in stages or "clamp" in stages:
        lj = json.load(open(out / "lstar.json"))
        ls, li = lj["lstar"], lj["lstar_ID"]
        assert lj["population"] == pop, "lstar.json was selected on another population"
        sweep_on = ints(a.onset_sweep) or scale(ONSET_SWEEP_14B)
        prim = sorted({ls + 1, li + 1, *ONSETS_FIXED})
        ex, cl, logs = [], [], []
        log = lambda s: (logs.append(s), print(s, flush=True))  # noqa: E731
        for f in formats:
            depths = [("BIND", ls)] + ([] if f == "QNAMES2" else [("ID", li)] + ([("ID", ls)] if ls != li else []))
            ons = [] if f == "QNAMES2" else sorted({o for o in (prim + (sweep_on if f in ("NO-MENTION", "OPTIONS-AFTER") else [])) if o < nL})
            t1 = time.time()
            for i in pop:
                e, c = run_pair(model, tok, item(tok, rel, pairs[i], f, i), depths, ons, "exchange" in stages, "clamp" in stages,
                                nL, a.test, log)
                ex += [x | {"format": f} for x in e]
                cl += [x | {"format": f} for x in c]
            print(f"  {f}: {len(pop)} pairs, depths {depths}, onsets {ons} ({time.time() - t1:.0f}s)", flush=True)
        body = {"lstar": ls, "lstar_ID": li, "population": pop, "formats": formats}
        if "exchange" in stages:
            dump(out / "exchange.json", prov, body | {"rows": ROWS, "rowdef": ROWDEF, "cells": ex, "b0_log": logs})
        if "clamp" in stages:
            dump(out / "clamp.json", prov, body | {"rows": [r[0] for r in CROWS], "onsets_primary": prim, "onset_sweep": sweep_on, "cells": cl})
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
