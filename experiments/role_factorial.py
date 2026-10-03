"""Identity or state? Clamp the critical token's key/value from a role-swapped story with the SAME location word.

Base B:  "... {A} watches as the {o} is moved to the {LOC}. {B} does not see this happen."   (A believes LOC)
Role R:  "... {B} watches as the {o} is moved to the {LOC}. {A} does not see this happen."   (A believes init)
The critical token (LOC) is lexically identical in B and R; only its context (who observed) differs. Clamping
its key and/or value (all layers) from R into B asks which channel carries the belief state rather than the
word's identity. Metric: m = logp(init) - logp(LOC) on the question about A's belief; f_C = d_C / (m(R) - m(B)).
Control view "world" (where the object actually is): the role swap should not matter there.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.encoding import candidate_ids, encode, raw_prompt
from ckeys.interventions import blocks, capture, hooks
from ckeys.story import LOCATIONS, QUERIES, make_cores
from experiments.format_factorial import last_logprobs, provenance

ROWS = [("B", "B"), ("R", "B"), ("B", "R"), ("R", "R")]   # (key source, value source) at the critical token
LABEL = {("B", "B"): "ID", ("R", "B"): "K_R", ("B", "R"): "V_R", ("R", "R"): "KV_R"}


def story(core, observer, other):
    initial = " ".join(f"Everyone initially sees that the {obj} is in the {where}."
                       for obj, where in sorted([(core["object"], core["initial"]),
                                                 (core["distractor"], core["distractor_location"])]))
    return (initial + f" {observer} watches as the {core['object']} is moved to the {core['base']}. "
            f"{other} does not see this happen.")


@torch.no_grad()
def run_item(model, tok, core, arm, view, device):
    a, b = core["agent"], core["other"]
    q = QUERIES[view].format(a=a, b=b, o=core["object"], d=core["distractor"])
    ids = {"B": encode(tok, raw_prompt(arm, story(core, a, b), q)).to(device),
           "R": encode(tok, raw_prompt(arm, story(core, b, a), q)).to(device)}
    if ids["B"].shape != ids["R"].shape:
        return None
    diff = (ids["B"][0] != ids["R"][0]).nonzero().flatten().tolist()
    loc_tok = tok.encode(" " + core["base"], add_special_tokens=False)[0]
    pos = [i for i in range(diff[0], diff[-1]) if ids["B"][0, i].item() == loc_tok]
    if len(diff) != 2 or len(pos) != 1:
        return None
    pos = pos[0]
    nL = len(blocks(model))
    cid = candidate_ids(tok, arm)
    iL, iI = cid[LOCATIONS.index(core["base"])], cid[LOCATIONS.index(core["initial"])]
    kv, clean = {}, {}
    for name in ("B", "R"):
        with capture(model, range(nL), "k") as K, capture(model, range(nL), "v") as V:
            lp = last_logprobs(model, ids[name])[0]
        kv[name] = ({l: K[l][0, pos] for l in range(nL)}, {l: V[l][0, pos] for l in range(nL)})
        clean[name] = {"m": (lp[iI] - lp[iL]).item(), "argmax": LOCATIONS[int(lp[cid].argmax())]}
    tabs = {(l, ch): torch.stack([kv[r[0] if ch == "K" else r[1]][0 if ch == "K" else 1][l] for r in ROWS])
            for l in range(nL) for ch in "KV"}
    hs = []
    for l in range(nL):
        at = blocks(model)[l].self_attn
        for mod, ch in ((at.k_proj, "K"), (at.v_proj, "V")):
            def hk(_m, _i, out, t=tabs[(l, ch)]):
                out = out.clone()
                out[:, pos] = t.to(out.device, out.dtype)
                return out
            hs.append(mod.register_forward_hook(hk))
    with hooks(hs):
        lp = last_logprobs(model, ids["B"].expand(len(ROWS), -1))
    m = {LABEL[r]: (lp[i, iI] - lp[i, iL]).item() for i, r in enumerate(ROWS)}
    return {"core": core, "arm": arm, "view": view, "pos": pos, "clean": clean, "m": m}


def summarize(res):
    rng = np.random.default_rng(0)
    lines = []
    for view in ("direct", "world"):
        for arm in sorted({r["arm"] for r in res}):
            R = [r for r in res if r["arm"] == arm and r["view"] == view]
            if len(R) < 5:
                continue
            full = np.array([r["clean"]["R"]["m"] - r["m"]["ID"] for r in R])
            parts = []
            for k in ("K_R", "V_R", "KV_R"):
                d = np.array([r["m"][k] - r["m"]["ID"] for r in R])
                idx = rng.integers(0, len(R), (10000, len(R)))
                f = d[idx].mean(1) / full[idx].mean(1)
                parts.append(f"{k} d={d.mean():+6.2f} frac={d.mean() / full.mean():+.2f} [{np.percentile(f, 2.5):+.2f},{np.percentile(f, 97.5):+.2f}]")
            lines.append(f"{view:6s} {arm:7s} n={len(R):3d} full role effect {full.mean():+6.2f} | " + " | ".join(parts))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--arms", default="P1,NONE,LETTER")
    ap.add_argument("--views", default="direct,world")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/role_factorial")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {"dtype": getattr(torch, a.dtype)}
    if a.device_map:
        kw["device_map"] = a.device_map
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if not a.device_map and torch.cuda.is_available():
        model = model.to("cuda")
    device = next(model.parameters()).device
    cores = make_cores(a.n, random.Random(a.seed))
    res, skipped, t0 = [], 0, time.time()
    for view in a.views.split(","):
        for arm in a.arms.split(","):
            for core in cores:
                r = run_item(model, tok, core, arm, view, device)
                if r is None:
                    skipped += 1
                else:
                    res.append(r)
            print(f"  {view}/{arm} done ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_s{a.seed}"
    prov = provenance(a) | {"skipped_items": skipped}
    json.dump({"provenance": prov, "results": res}, open(f"{a.out}/{tag}.json", "w"))
    s = json.dumps(prov) + "\n" + summarize(res)
    open(f"{a.out}/{tag}_summary.txt", "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
