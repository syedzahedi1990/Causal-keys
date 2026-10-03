"""Key/value factorial at the critical state token for the stage-3 generality tasks (ckeys.tasks).

Same estimands, rows, identity double difference and bootstrap summaries as experiments/format_factorial.py;
only the story, candidate alphabet and prompt arms come from the task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.encoding import encode
from ckeys.interventions import blocks, capture, hooks
from ckeys.tasks import TASKS
from experiments.format_factorial import LABEL, last_logprobs, provenance, row_specs, summarize


def cand_ids(tok, task, arm):
    ids = []
    for c in task.alphabet(arm):
        t = tok.encode(" " + c, add_special_tokens=False)
        assert len(t) == 1, f"candidate {c!r} is not a single token"
        ids.append(t[0])
    return ids


def pick_x(task, core):
    used = {core["base"], core["source"], core["initial"], core["distractor_location"]}
    pool = [v for v in task.values if v not in used]
    seed = int(hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8], 16)
    return random.Random(seed).choice(pool)


@torch.no_grad()
def run_item(model, tok, task, core, arm, device):
    X = pick_x(task, core)
    vals = {"B": core["base"], "S": core["source"], "X": X}
    q = task.question(core)
    ids = {k: encode(tok, task.raw_prompt(arm, task.story(core, v), q)).to(device) for k, v in vals.items()}
    if len({tuple(v.shape) for v in ids.values()}) > 1:
        return None
    diff = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()
    if len(diff) != 1 or (ids["B"][0] != ids["X"][0]).nonzero().flatten().tolist() != diff:
        return None
    pos = diff[0]
    nL = len(blocks(model))
    cid = cand_ids(tok, task, arm)
    track = {"S": core["source"], "B": core["base"], "X": X, "init": core["initial"]}
    tid = {k: cid[task.values.index(v)] for k, v in track.items()}
    kv, clean = {}, {}
    for name in ("B", "S", "X"):
        with capture(model, range(nL), "k") as K, capture(model, range(nL), "v") as V:
            lp = last_logprobs(model, ids[name])[0]
        kv[name] = ({l: K[l][0, pos] for l in range(nL)}, {l: V[l][0, pos] for l in range(nL)})
        clean[name] = {"lp": {k: lp[i].item() for k, i in tid.items()}, "m": (lp[tid["S"]] - lp[tid["B"]]).item(),
                       "mass": lp[cid].exp().sum().item(), "argmax_cand": task.values[int(lp[cid].argmax())]}
    rows = row_specs(nL)

    def who(row, l, ch):
        ks, vs, l0 = row
        return (ks if ch == "K" else vs) if l >= l0 else "B"

    tabs = {(l, ch): torch.stack([kv[who(r, l, ch)][0 if ch == "K" else 1][l] for r in rows])
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
        lp = last_logprobs(model, ids["B"].expand(len(rows), -1))
    out = {f"{LABEL[(k, v)]}@{l0}": {"m": (lp[i, tid["S"]] - lp[i, tid["B"]]).item(),
                                      "lp": {t: lp[i, j].item() for t, j in tid.items()}}
           for i, (k, v, l0) in enumerate(rows)}
    return {"core": core, "X": X, "arm": arm, "view": "direct", "task": task.name, "pos": pos,
            "len": ids["B"].shape[1], "n_layers": nL, "clean": clean, "m": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--arms", default="P1,NONE,BEFORE,POST,LETTER")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--out", default="results/task_factorial")
    a = ap.parse_args()
    task = TASKS[a.task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {"dtype": getattr(torch, a.dtype)}
    if a.device_map:
        kw["device_map"] = a.device_map
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if not a.device_map:
        model = model.to(device)
    device = next(model.parameters()).device
    cores = task.make_cores(a.n, random.Random(a.seed))
    res, skipped, t0 = [], 0, time.time()
    for arm in a.arms.split(","):
        for core in cores:
            r = run_item(model, tok, task, core, arm, device)
            if r is None:
                skipped += 1
            else:
                res.append(r)
        print(f"  {a.task}/{arm} done ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"{a.task}_{a.model.split('/')[-1]}_s{a.seed}"
    prov = provenance(a) | {"skipped_items": skipped, "task": a.task}
    json.dump({"provenance": prov, "results": res}, open(f"{a.out}/{tag}.json", "w"))
    s = json.dumps(prov) + "\n" + summarize(res)
    open(f"{a.out}/{tag}_summary.txt", "w").write(s + "\n")
    print(s)
    if not res:
        raise SystemExit("no valid items")


if __name__ == "__main__":
    main()
