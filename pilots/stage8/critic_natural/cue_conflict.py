"""Critic pilot (disclosed; CPU, FP32, windowed passages <= 320 chars): prior-neutral behavioural rows.

Rows (K source, V source) on the B prompt, clamped at every position of the entity span P in every layer from 0:
  ID=(B,B), KV_S, KV_X, KV_Z (= the Z passage: is B still chosen? -> question/memory prior),
  K_S=(S,B), V_S=(B,S), cue-conflict (S,X) and (X,S) (neither source is B, so the B prior is neutral),
  copy-fallback (Z,S) (no option matches the key; the value says S), flag-only (S,Z) (value is a non-option).
Greedy generation (8 tokens). For each row we record which entity the answer matches (B/S/X/Z/D/other).
Usage: cue_conflict.py MODEL N FORMATS OUT
"""
from __future__ import annotations

import collections
import json
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

A = "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partA/"
sys.path.insert(0, "/home/user/Causal-keys")
sys.path.insert(0, A)
from ckeys.clamp import capture_kv, clamp_kv, stack_rows  # noqa: E402
from ckeys.interventions import blocks  # noqa: E402
from pilot_fast import window  # noqa: E402
from pilot_natural import encode, matches  # noqa: E402

ROWS = {"ID": ("B", "B"), "KV_S": ("S", "S"), "KV_X": ("X", "X"), "KV_Z": ("Z", "Z"), "K_S": ("S", "B"),
        "V_S": ("B", "S"), "KS_VX": ("S", "X"), "KX_VS": ("X", "S"), "KZ_VS": ("Z", "S"), "KS_VZ": ("S", "Z")}


def who(gen, it, letters=False):
    ents = {"B": it["answer"], "S": it["S"], "X": it["X"], "Z": it["Y"], "D": it["D"]}
    if letters:
        g = gen.strip()[:1]
        for Y, e in ents.items():
            if e in it["options"] and "ABCD"[it["options"].index(e)] == g:
                return Y
        return "other"
    for Y, e in ents.items():
        if matches(gen, e):
            return Y
    return "other"


@torch.no_grad()
def run(model, tok, it, fmt):
    nL = len(blocks(model))
    ents = {"B": it["answer"], "S": it["S"], "X": it["X"], "Z": it["Y"]}
    enc = {Y: encode(tok, it, fmt, e) for Y, e in ents.items()}
    tb, ib, ob, (c0, c1) = enc["B"]
    P = [i for i, (s, e) in enumerate(ob) if e > c0 and s < c1]
    for Y in ("S", "X", "Z"):
        iy = enc[Y][1]
        if len(iy) != len(ib):
            return None
        diff = [i for i, (x, y) in enumerate(zip(ib, iy)) if x != y]
        if not diff or not set(diff) <= set(P):
            return None
    kv = {}
    for Y in ents:
        with capture_kv(model, P, range(nL)) as t:
            model(torch.tensor([enc[Y][1]]), use_cache=False)
        kv[Y] = {k: v[0] for k, v in t.items()}
    names = list(ROWS)
    tabs = stack_rows(kv, [lambda l, ch, r=ROWS[n]: r[0] if ch == "k" else r[1] for n in names], range(nL))
    x = torch.tensor([ib]).expand(len(names), -1)
    with clamp_kv(model, P, tabs, range(nL)):
        g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    gens = {n: tok.decode(r, skip_special_tokens=True) for n, r in zip(names, g[:, len(ib):])}
    return {n: who(gens[n], it, letters=(fmt == "LETA")) for n in names}, gens


def main():
    mname, n, fmts, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4]
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.float32).eval()
    items = json.load(open(A + "items_ft5.json"))
    sp = json.load(open(A + "split_v5.json"))
    E = set(sp["E"])
    items = [window(i) for i in items if i["id"] in E and i["sub"] in ("PERSON", "PLACE")]
    items = [i for i in items if len(i["context"]) <= 330]
    random.Random(3).shuffle(items)
    res, t0 = [], time.time()
    for it in items:
        if len(res) >= n:
            break
        row = {"id": it["id"], "sub": it["sub"]}
        ok = True
        for f in fmts:
            r = run(model, tok, it, f)
            if r is None:
                ok = False
                break
            row[f], row[f + "_gen"] = r
        if not ok:
            continue
        res.append(row)
        print(len(res), f"{time.time() - t0:.0f}s", {f: row[f] for f in fmts}, flush=True)
        json.dump(res, open(outp, "w"))
    for f in fmts:
        comp = [r for r in res if r[f]["ID"] == "B" and r[f]["KV_S"] == "S" and r[f]["KV_X"] == "X"]
        print(f"\n{f}: n={len(res)} competent={len(comp)}")
        for nme in ROWS:
            c = collections.Counter(r[f][nme] for r in comp)
            print(f"   {nme:6s}", dict(c))


if __name__ == "__main__":
    main()
