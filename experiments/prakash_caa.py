"""J-C6 of P-2026-10-10-J part C (docs/PREREGISTRATION.md): a CAA identity edit at the depth of the binding swap on
Prakash et al.'s (2026) material at Qwen2.5-14B-Instruct, set against the binding swap (BIND) and the identity edit (ID),
re-run in the same job. Extends experiments/prakash_swap.py (imported, not edited) with the CAA arm.

Material: their template-2 stories, raw wrapper and seed-10 pool (ckeys.causaltom, release at 0579347 with every file
hash and the pool hash asserted); the population is the first 150 pairs that pass their LM filter under NO-MENTION
(re-run here; the stage-6 population file, when present, is compared and the comparison recorded); the CAA means use the
other 170 pool pairs (every pool pair outside the population).
CAA at l* = 28 (stage 6's l*): mu_28(x) [2, D] = the mean block-28 output at the queried state word p and the token after
it (p + 1) over the held-out pairs, with drink x (each of the 23) written at the queried state. Edit toward S: the clean
residual at [p, p+1] plus mu_28(S) - mu_28(s_q), written at the output of block 28 (S is the pair's ID donor drink).
Rows per arm, pair and format (H's exchange, prakash_swap.ROWS): r0 B+KV_B, r1 M, r2 B+K_M, r3 B+V_M, r4 B+KV_M, r5 M+K_B,
r6 M+V_B and the words-only r2w-r4w, K/V of the patched positions from block 29; m = log p(target) - log p(s_q) (target:
the other story state for BIND, S for ID and CAA). nu_CAA = |[K;V]_CAA - [K;V]_ID| / |[K;V]_ID - [K;V]_B| over blocks 29..L-1
and positions [p, p+1] (one pooled ratio). Formats NO-MENTION, QNAMES, OPTIONS-AFTER.
Stages: filter -> filter.json; means -> means.pt + means.json (sha256); exchange -> exchange.json. TEST_MODE: Qwen2.5-0.5B
FP32 on the CPU, no LM filter, n = 2, 4 held-out pairs, l* = 12 (the 0.5B model has 24 blocks), NO-MENTION and OPTIONS-AFTER.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys import causaltom as ct
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.interventions import blocks
from experiments import prakash_swap as ps
from experiments.stage6_heads import write_atomic
from experiments.stage8_edits import verified

LSTAR, LSTAR_TEST = 28, 12
FORMATS = ("NO-MENTION", "QNAMES", "OPTIONS-AFTER")
ANCHOR = {"NO-MENTION": 0.618, "QNAMES": 0.906, "OPTIONS-AFTER": 0.864}     # stage 6, BIND@28 kappa (Gate J-C-G7)
STAGE6_FILTER = "results/gpu_stage6/prakash/Qwen2.5-14B-Instruct/filter.json"


def log(s):
    print(s, flush=True)


def caa_item(L):
    """Add the CAA arm (destination and readout as ID's) to a prakash_swap.item record."""
    L["arm"]["CAA"] = dict(dst=L["id_span"], src=L["id_span"], donor=None, P=L["id_span"], Pw=[L["p"]], target="S")
    return L


@torch.no_grad()
def caa_means(model, tok, rel, pairs, idx, layer):
    """{drink: [2, D] float32}: mean block-``layer`` output at [p, p+1] over the pairs ``idx`` with each drink written at
    the queried state (NO-MENTION prompt; the prefix through p + 1 is the same in every format)."""
    acc = {x: None for x in rel.states}
    for i in idx:
        pr = pairs[i]
        m = ct.meta(rel, pr)
        L = ct.locate(tok, rel, pr, "NO-MENTION", m)
        p = L["p"]
        ids = []
        for x in rel.states:
            story = ct.swap_state(pr["clean_story"], m["q"], x)
            e = ct.encode(tok, ct.prompt(rel, story, pr["clean_question"], "NO-MENTION", m))["input_ids"][:p + 2]
            assert len(e) == p + 2 and e[:p] == L["ids"]["B"][:p], "a drink changed the prefix"
            ids.append(e)
        with ps.capture_resid(model, [layer], [[p, p + 1]] * len(ids)) as R:
            ps.logprobs(model, torch.tensor(ids))
        for j, x in enumerate(rel.states):
            v = R[layer][j].float().cpu()
            acc[x] = v if acc[x] is None else acc[x] + v
    return {x: v / len(idx) for x, v in acc.items()}


def exchange_arm(model, tok, L, arm, src, d, nL, kvB, lpB, exact, logf):
    """prakash_swap.run_pair's exchange for one arm with the patched source ``src`` [len(dst), D] at block d. Returns
    (cell record, the M run's K/V at the arm's positions in blocks > d)."""
    a = L["arm"][arm]
    P = L["P"]
    cols = [P.index(x) for x in a["P"]]
    wcols = torch.tensor([x in a["Pw"] for x in a["P"]])
    lay = list(range(d + 1, nL))
    with ps.resid_patch(model, d, [a["dst"]], [src]), capture_kv(model, a["P"], lay) as kvM:
        lpM = ps.logprobs(model, L["ids"]["B"])[0]
    tab = {"B": {k: v[0][cols] for k, v in kvB.items()}, "M": {k: v[0] for k, v in kvM.items()}}
    tables = {(l, ch): torch.stack([tab[ps.ROWDEF[r]["kv".index(ch)] or "M"][(l, ch)] for r in ps.ROWS]) for l in lay for ch in "kv"}
    per_row = torch.stack([torch.zeros(len(cols), dtype=torch.bool) if r == "r1" else wcols if ps.ROWDEF[r][3]
                           else torch.ones(len(cols), dtype=torch.bool) for r in ps.ROWS])
    act = [a["dst"] if ps.ROWDEF[r][2] else None for r in ps.ROWS]
    with ps.resid_patch(model, d, act, [src if x else None for x in act]), clamp_kv(model, a["P"], tables, lay, "kv", per_row):
        lp = ps.logprobs(model, [L["ids"]["B"]] * len(ps.ROWS))
    m = dict(zip(ps.ROWS, ps.mval(lp, L, arm)))
    mB, mM = ps.mval(lpB, L, arm), ps.mval(lpM, L, arm)
    b0 = {"r4_r1": abs(m["r4"] - m["r1"]), "r0_B": abs(m["r0"] - mB), "r1_M": abs(m["r1"] - mM)}
    logf(f"    b0 {arm}@{d} pair {L['i']}: |m(r4)-m(r1)| {b0['r4_r1']:.2e} |m(r0)-m(B)| {b0['r0_B']:.2e}")
    if exact:
        assert b0["r4_r1"] <= 1e-3 and b0["r0_B"] <= 1e-3, f"Gate b0 (FP32) failed: {b0}"
    top = lp.argmax(-1)
    cell = {"arm": arm, "depth": d, "m": m, "m_B": mB, "m_M": mM, "b0": b0,
            "ok_r1": ct.correct(tok, top[1].item(), L["word"][a["target"]]), "lp": {r: ps.roles(lp[k], L) for k, r in enumerate(ps.ROWS)}}
    return cell, {k: v[0] for k, v in kvM.items()}


def nu_of(kvE, kvID, kvB_id, lay):
    num = sum(float(((kvE[(l, ch)] - kvID[(l, ch)]).double() ** 2).sum()) for l in lay for ch in "kv")
    den = sum(float(((kvID[(l, ch)] - kvB_id[(l, ch)]).double() ** 2).sum()) for l in lay for ch in "kv")
    return (num / max(den, 1e-30)) ** 0.5


@torch.no_grad()
def run_pair(model, tok, L, d, mu, nL, exact, logf):
    """BIND, ID and CAA (toward S) exchanges of one pair in one format at block d; nu of CAA against ID."""
    P, p = L["P"], L["p"]
    allL = list(range(nL))
    with capture_kv(model, P, allL) as kvB, ps.capture_resid(model, [d], [L["id_span"]]) as rB:
        lpB = ps.logprobs(model, L["ids"]["B"])[0]
    with ps.capture_resid(model, [d], [L["bind_src"]]) as rC:
        ps.logprobs(model, L["ids"]["C"])
    with ps.capture_resid(model, [d], [L["id_span"]]) as rS:
        ps.logprobs(model, L["ids"]["S"])
    m_ = L["meta"]
    hB = rB[d][0]
    caa = (hB.float() + mu[m_["S"]].to(hB.device) - mu[m_["s_q"]].to(hB.device)).to(hB.dtype)
    out, kvs = {}, {}
    for arm, src in (("BIND", rC[d][0]), ("ID", rS[d][0]), ("CAA", caa)):
        out[arm], kvs[arm] = exchange_arm(model, tok, L, arm, src, d, nL, kvB, lpB, exact, logf)
    lay = list(range(d + 1, nL))
    cols = [P.index(x) for x in L["id_span"]]
    kvB_id = {(l, ch): kvB[(l, ch)][0][cols] for l in lay for ch in "kv"}
    nu = nu_of(kvs["CAA"], kvs["ID"], kvB_id, lay)
    base = {"i": L["i"], "q": m_["q"], "lp_B": ps.roles(lpB, L), "nu_CAA_ID": nu,
            "pred_B": tok.decode([lpB.argmax().item()]), "ok_B": ct.correct(tok, lpB.argmax().item(), L["word"]["s_q"])}
    return [base | c for c in out.values()]


def load_model(a):
    dev = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(dtype=getattr(torch, a.dtype), attn_implementation="sdpa")
    if dev == "cuda":
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if dev == "cpu":
        model = model.to(dev)
    return model, tok, dev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="filter,means,exchange")
    ap.add_argument("--model", default=None, help="the verified local directory of Qwen2.5-14B-Instruct (or a Hub id)")
    ap.add_argument("--key", default="qwen14")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prakash-repo", default=None)
    ap.add_argument("--out", default="results/gpu_stage8c")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    formats, d = FORMATS, LSTAR
    if a.test:
        a.model, a.dtype, a.n = "Qwen/Qwen2.5-0.5B-Instruct", "float32", min(a.n, 2)
        formats, d = ("NO-MENTION", "OPTIONS-AFTER"), LSTAR_TEST
    tag = ("TEST_" if a.test else "") + a.key
    out = Path(a.out) / "jc6" / tag
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    rel = ct.load(a.prakash_repo)
    pairs = ct.pool(rel)
    model, tok, dev = load_model(a)
    nL = len(blocks(model))
    prov = ps.provenance(a, model) | {"lstar": d, "formats": list(formats), "test_mode": a.test, "model_key": a.key,
                                      "verified": verified(a.model)}
    stages = a.stage.split(",")
    t0 = time.time()
    if "filter" in stages:
        if a.test:
            pop, fl = list(range(a.n)), []
        else:
            fl = ps.lm_filter(model, tok, pairs)
            pop = [f["i"] for f in fl if f["ok"]][:a.n]
        same = None
        if not a.test and Path(STAGE6_FILTER).exists():
            s6 = json.load(open(STAGE6_FILTER))["pairs"]
            same = [f["i"] for f in s6 if f["ok"]][:a.n] == pop
        write_atomic({"provenance": prov, "pairs": fl, "population": pop, "same_as_stage6": same,
                      "n_pass": None if a.test else sum(f["ok"] for f in fl)}, out / "filter.json")
        log(f"filter: population {len(pop)} pairs; equal to stage 6's: {same} ({time.time() - t0:.0f}s)")
    pop = json.load(open(out / "filter.json"))["population"]
    held = [i for i in range(len(pairs)) if i not in set(pop)]
    if a.test:
        held = held[:4]
    if "means" in stages:
        mu = caa_means(model, tok, rel, pairs, held, d)
        f = out / "means.pt"
        torch.save(mu, f.with_suffix(".pt.tmp"))
        os.replace(f.with_suffix(".pt.tmp"), f)
        write_atomic({"provenance": prov, "held_out": held, "layer": d, "means_sha256": ps_sha(f)}, out / "means.json")
        log(f"means over {len(held)} held-out pairs ({time.time() - t0:.0f}s)")
    if "exchange" in stages:
        mj = json.load(open(out / "means.json"))
        assert ps_sha(out / "means.pt") == mj["means_sha256"], "means.pt differs from the one means.json recorded"
        mu = torch.load(out / "means.pt", weights_only=False)
        cells, logs = [], []
        logf = lambda s: (logs.append(s), print(s, flush=True))  # noqa: E731
        for f in formats:
            for i in pop:
                L = caa_item(ps.item(tok, rel, pairs[i], f, i))
                cells += [c | {"format": f} for c in run_pair(model, tok, L, d, mu, nL, a.test, logf)]
            log(f"  {f}: {len(pop)} pairs ({time.time() - t0:.0f}s)")
        write_atomic({"provenance": prov | {"means_sha256": mj["means_sha256"]}, "population": pop, "formats": list(formats),
                      "lstar": d, "rows": list(ps.ROWS), "cells": cells, "b0_log": logs}, out / "exchange.json")
    log(f"done ({time.time() - t0:.0f}s)")


def ps_sha(f):
    from ckeys.sae import sha_file
    return sha_file(f)


if __name__ == "__main__":
    main()
