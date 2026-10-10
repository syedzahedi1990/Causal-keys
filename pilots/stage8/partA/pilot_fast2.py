"""Fast CPU pilot (disclosed): passage windowed to the entity's sentence and its predecessor (<= 320 chars), decision-token
scoring only, one 7-row clamp batch per item and format, greedy generation (6 tokens) for rows ID, K_S, V_S.
Usage: pilot_fast.py MODEL N FORMATS ITEMS VALID_IDS OUT"""
from __future__ import annotations

import json
import random
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/user/Causal-keys")
sys.path.insert(0, "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partA")
from ckeys.clamp import capture_kv, clamp_kv, stack_rows  # noqa: E402
from ckeys.interventions import blocks  # noqa: E402
from pilot_natural import NAMES, ROWS, cont_ids, encode, matches  # noqa: E402


def window(it, maxc=320):
    ctx, s0 = it["context"], it["start"]
    bounds = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\"'(])", ctx)] + [len(ctx)]
    k = max(i for i, b in enumerate(bounds[:-1]) if b <= s0)
    a, b = bounds[k], bounds[k + 1]
    if k > 0 and (b - bounds[k - 1]) <= maxc:
        a = bounds[k - 1]
    w = ctx[a:b].strip()
    off = ctx[a:b].index(w)
    return dict(it, context=w, start=s0 - a - off)


@torch.no_grad()
def run(model, tok, it, fmt, check_exact=False):
    nL = len(blocks(model))
    enc = {Y: encode(tok, it, fmt, e) for Y, e in (("B", it["answer"]), ("S", it["S"]), ("X", it["X"]))}
    tb, ib, ob, (c0, c1) = enc["B"]
    P = [i for i, (s, e) in enumerate(ob) if e > c0 and s < c1]
    for Y in ("S", "X"):
        iy = enc[Y][1]
        if len(iy) != len(ib):
            return None
        diff = [i for i, (x, y) in enumerate(zip(ib, iy)) if x != y]
        if not diff or not set(diff) <= set(P):
            return None
    kv = {}
    for Y in ("B", "S", "X"):
        with capture_kv(model, P, range(nL)) as t:
            model(torch.tensor([enc[Y][1]]), use_cache=False)
        kv[Y] = {k: v[0] for k, v in t.items()}
    tabs = stack_rows(kv, [lambda l, ch, r=r: r[0] if ch == "k" else r[1] for r in ROWS], range(nL))
    cands = {"B": it["answer"], "S": it["S"], "X": it["X"], "D": it["D"]}
    dec = {}
    pre = None
    for Y, e in cands.items():
        c, k = cont_ids(tok, tb, ib, e)
        dec[Y] = c[k]
        pre = c[:k] if pre is None else pre
        assert c[:k] == pre
    x = torch.tensor([ib + pre]).expand(len(ROWS), -1)
    with clamp_kv(model, P, tabs, range(nL)):
        lp = model(x, use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
    out = dict(P=P, T=len(ib), lp={n: {Y: float(lp[i, dec[Y]]) for Y in cands} for i, n in enumerate(NAMES)},
               mass={n: float(lp[i, list(dec.values())].exp().sum()) for i, n in enumerate(NAMES)})
    gi = [NAMES.index(n) for n in ("ID", "KV_S", "K_S", "V_S")]
    sub = {key: v[gi] for key, v in tabs.items()}
    xg = torch.tensor([ib]).expand(len(gi), -1)
    with clamp_kv(model, P, sub, range(nL)):
        g = model.generate(xg, attention_mask=torch.ones_like(xg), max_new_tokens=6, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    out["gen"] = {n: tok.decode(r, skip_special_tokens=True) for n, r in zip(("ID", "KV_S", "K_S", "V_S"), g[:, len(ib):])}
    if check_exact:
        xs = torch.tensor([enc["S"][1] + pre])
        lps = model(xs, use_cache=False, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
        out["exact"] = max(abs(float(lps[dec[Y]]) - out["lp"]["KV_S"][Y]) for Y in cands)
    return out


def main():
    mname, n, fmts, ipath, vpath, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4], sys.argv[5], sys.argv[6]
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(mname)
    import os
    dt = getattr(torch, os.environ.get("DTYPE", "float32"))
    model = AutoModelForCausalLM.from_pretrained(mname, dtype=dt).eval()
    skip = int(os.environ.get("SKIP", "0"))
    valid = set(json.load(open(vpath)))
    items = [window(i) for i in json.load(open(ipath)) if i["id"] in valid]
    items = [i for i in items if len(i["context"]) <= 330]
    random.Random(11).shuffle(items)
    # balance subtypes a little
    pick, seen = [], {}
    for i in items:
        if seen.get(i["sub"], 0) < (n + 3) // 4 + 1:
            pick.append(i)
            seen[i["sub"]] = seen.get(i["sub"], 0) + 1
        if len(pick) >= n:
            break
    res, t0 = [], time.time()
    pick = pick[skip:]
    for j, it in enumerate(pick):
        for f in fmts:
            r = run(model, tok, it, f, check_exact=False)
            if r is None:
                continue
            r.update(id=it["id"], fmt=f, sub=it["sub"], art=it["art"], ans=it["answer"], S=it["S"], X=it["X"], D=it["D"])
            r["ok_B"] = matches(r["gen"]["ID"], it["answer"]) if f != "LETA" else None
            r["ok_S"] = matches(r["gen"]["KV_S"], it["S"]) if f != "LETA" else None
            res.append(r)
        print(f"item {j + 1}/{len(pick)} rows {len(res)} {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(outp, "w"))


if __name__ == "__main__":
    main()
