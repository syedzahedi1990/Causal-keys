"""Natural key/value factorial at the critical token (no training, no patch).

For each story core, run the unpatched base story (critical location B) and source story (S).
At the critical token, for every layer >= l0, set the cached key and/or value that later tokens
read to the source run's (S) or keep the base run's (B). Because later tokens see the critical
token only through its keys and values, clamping both reproduces the source run exactly for l0=0.

Question: when keys and values disagree, does the answer follow the keys or the values?
Variant ``--no-choices`` removes the answer-choices line so S appears nowhere in the base context.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.interventions import blocks, capture, hooks
from ckeys.story import LOCATIONS, make_cores, prompt, record

CONDS = ("base", "K_S", "V_S", "KV_S")


def encode(tok, text, chat):
    if chat:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      return_tensors="pt")
        ids = ids if torch.is_tensor(ids) else ids["input_ids"]
    else:
        ids = tok(text, return_tensors="pt").input_ids
    return ids


def crit_position(tok, ids_b, ids_s):
    diff = (ids_b[0] != ids_s[0]).nonzero().flatten().tolist()
    assert len(diff) == 1, f"base/source differ at {diff}"
    return diff[0]


@torch.no_grad()
def run_core(model, tok, core, view, choices, chat, l0s, cand, device):
    rb = record(core, view, core["base"]); rs = record(core, view, core["source"])
    ib = encode(tok, prompt(rb, choices), chat).to(device)
    is_ = encode(tok, prompt(rs, choices), chat).to(device)
    if ib.shape != is_.shape:
        return None
    pos = crit_position(tok, ib, is_)
    L = range(len(blocks(model)))
    kv = {}
    for name, ids in (("B", ib), ("S", is_)):
        with capture(model, L, "k") as K, capture(model, L, "v") as V:
            lg = model(ids).logits[0, -1, cand]
        kv[name] = ({l: K[l][0, pos] for l in L}, {l: V[l][0, pos] for l in L}, lg)
    rows = [(c, l0) for l0 in l0s for c in CONDS[1:]]
    n = len(rows)

    def src(c, l0, l, chan):  # which run supplies channel ``chan`` at layer l for row (c, l0)
        return "S" if (l >= l0 and (c == "KV_S" or c == f"{chan}_S")) else "B"

    Ktab = {l: torch.stack([kv[src(c, l0, l, "K")][0][l] for c, l0 in rows]) for l in L}
    Vtab = {l: torch.stack([kv[src(c, l0, l, "V")][1][l] for c, l0 in rows]) for l in L}
    handles = []
    for l in L:
        attn = blocks(model)[l].self_attn
        for mod, tab in ((attn.k_proj, Ktab[l]), (attn.v_proj, Vtab[l])):
            def hk(_m, _i, out, tab=tab):
                out = out.clone()
                out[:, pos] = tab.to(out.dtype)
                return out
            handles.append(mod.register_forward_hook(hk))
    with hooks(handles):
        lg = model(ib.expand(n, -1)).logits[:, -1][:, cand]
    pred = lg.argmax(-1).tolist()
    out = {"core": core, "view": view, "pos": pos,
           "clean_B": LOCATIONS[int(kv["B"][2].argmax())], "clean_S": LOCATIONS[int(kv["S"][2].argmax())],
           "rows": [{"cond": c, "l0": l0, "pred": LOCATIONS[p], "logp": torch.log_softmax(lg[i], -1).tolist()}
                    for i, ((c, l0), p) in enumerate(zip(rows, pred))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--views", default="world,direct")
    ap.add_argument("--l0", default="0,2,4,8,12")
    ap.add_argument("--no-choices", action="store_true")
    ap.add_argument("--choices-first", action="store_true", help="list choices before the story")
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default="results/kv_factorial")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(device).eval()
    cand = torch.tensor([tok.encode(" " + l, add_special_tokens=False)[0] for l in LOCATIONS], device=device)
    nL = len(blocks(model))
    l0s = [int(x) for x in a.l0.split(",") if int(x) < nL]
    cores = make_cores(a.n, random.Random(a.seed))
    res = []
    t0 = time.time()
    for view in a.views.split(","):
        for i, core in enumerate(cores):
            ch = False if a.no_choices else ("before" if a.choices_first else True)
            r = run_core(model, tok, core, view, ch, not a.no_chat, l0s, cand, device)
            if r is not None:
                res.append(r)
        print(f"  view {view} done ({time.time() - t0:.0f}s)", flush=True)
    tag = f"{a.model.split('/')[-1]}_{'nochoices' if a.no_choices else 'choicesfirst' if a.choices_first else 'choices'}_s{a.seed}"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(a), "results": res}, open(f"{a.out}/{tag}.json", "w"))
    summarize(res, l0s)


def summarize(res, l0s):
    for view in sorted({r["view"] for r in res}):
        R = [r for r in res if r["view"] == view and r["clean_B"] == r["core"]["base"] and r["clean_S"] == r["core"]["source"]]
        tot = len([r for r in res if r["view"] == view])
        print(f"\n view={view}: competent {len(R)}/{tot} (base->B and source->S)")
        if not R:
            continue
        print(f"   {'cond':6s} {'l0':>3s} | {'=S':>6s} {'=B':>6s} {'other':>6s} | mean logp(S)-logp(B)")
        for l0 in l0s:
            for c in CONDS[1:]:
                xs = [next(x for x in r["rows"] if x["cond"] == c and x["l0"] == l0) for r in R]
                s = sum(x["pred"] == r["core"]["source"] for x, r in zip(xs, R)) / len(R)
                b = sum(x["pred"] == r["core"]["base"] for x, r in zip(xs, R)) / len(R)
                m = sum(x["logp"][LOCATIONS.index(r["core"]["source"])] - x["logp"][LOCATIONS.index(r["core"]["base"])]
                        for x, r in zip(xs, R)) / len(R)
                print(f"   {c:6s} {l0:3d} | {s:6.1%} {b:6.1%} {1 - s - b:6.1%} | {m:+.2f}")


if __name__ == "__main__":
    main()
