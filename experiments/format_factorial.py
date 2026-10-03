"""Key/value factorial at the critical state token across answer-format arms.

For each arm and story core, the cached key and/or value that later tokens read at the critical token is set
to another run's (B base, S source, X third location absent from the story body), for all layers >= l0, in
the unpatched base story. l0 is 0 or a depth fraction (0.0625 matches Paper 1's exchange from 1-based block 6
of 80 in Qwen2.5-72B; 0.3 is a mid-depth control).

Metrics (full-vocabulary log-probs at the answer position; letters for LETTER):
  m = logp(S) - logp(B);  d_C = m(C) - m(clean B)
  key share = mean dK / (mean dK + mean dV)
  interaction = dKV - dK - dV  (also as a fraction of dKV)
  identity(K) = 0.5 * [(dlogp_S(K_S) - dlogp_S(K_X)) + (dlogp_X(K_X) - dlogp_X(K_S))]
      a double difference that cancels effects shared by any foreign key (e.g. weakening B); absolute
      log-prob changes, not log-odds. identity(V) likewise for values.
All statistics are reported on all items (primary) and on items both clean runs answer correctly (secondary),
with core-bootstrap 95% CIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.encoding import ARMS, WRAPPER_USED, alphabet, candidate_ids, encode, raw_prompt
from ckeys.interventions import blocks, capture, hooks
from ckeys.story import LOCATIONS, make_cores, record

L0_FRACS = (0.0, 0.0625, 0.3)
LABEL = {("S", "B"): "K_S", ("B", "S"): "V_S", ("S", "S"): "KV_S", ("X", "B"): "K_X", ("B", "X"): "V_X",
         ("X", "X"): "KV_X", ("B", "B"): "ID"}


def row_specs(nL: int) -> list[tuple[str, str, int]]:
    l0s = sorted({round(f * nL) for f in L0_FRACS})
    rows = [("B", "B", 0)]  # identity row: batched baseline for every d
    for l0 in l0s:
        rows += [("S", "B", l0), ("B", "S", l0), ("S", "S", l0)]
    rows += [("X", "B", 0), ("B", "X", 0), ("X", "X", 0)]
    return rows


def pick_x(core: dict) -> str:
    used = {core["base"], core["source"], core["initial"], core["distractor_location"]}
    pool = [l for l in LOCATIONS if l not in used]
    seed = int(hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8], 16)
    return random.Random(seed).choice(pool)


def last_logprobs(model, ids):
    out = model(ids, use_cache=False, logits_to_keep=1)
    return torch.log_softmax(out.logits[:, -1].float(), -1)


@torch.no_grad()
def run_item(model, tok, core, arm, view, device):
    X = pick_x(core)
    locs = {"B": core["base"], "S": core["source"], "X": X}
    ids = {}
    for name, loc in locs.items():
        r = record(core, view, loc)
        ids[name] = encode(tok, raw_prompt(arm, r["story"], r["query"])).to(device)
    if len({tuple(v.shape) for v in ids.values()}) > 1:
        return None
    diff = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()
    if len(diff) != 1 or (ids["B"][0] != ids["X"][0]).nonzero().flatten().tolist() != diff:
        return None
    pos = diff[0]
    nL = len(blocks(model))
    cid = candidate_ids(tok, arm)
    track = {"S": core["source"], "B": core["base"], "X": X, "init": core["initial"]}
    tid = {k: cid[LOCATIONS.index(v)] for k, v in track.items()}
    kv, clean = {}, {}
    for name in ("B", "S", "X"):
        with capture(model, range(nL), "k") as K, capture(model, range(nL), "v") as V:
            lp = last_logprobs(model, ids[name])[0]
        kv[name] = ({l: K[l][0, pos] for l in range(nL)}, {l: V[l][0, pos] for l in range(nL)})
        clean[name] = {"lp": {k: lp[i].item() for k, i in tid.items()}, "m": (lp[tid["S"]] - lp[tid["B"]]).item(),
                       "mass": lp[cid].exp().sum().item(), "argmax_cand": LOCATIONS[int(lp[cid].argmax())]}
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
    return {"core": core, "X": X, "arm": arm, "view": view, "pos": pos, "len": ids["B"].shape[1],
            "n_layers": nL, "clean": clean, "m": out}


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def boot_ratio(a, b, n=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def fmt(t):
    return f"{t[0]:+7.2f} [{t[1]:+6.2f},{t[2]:+6.2f}]"


def correct(r, run):
    target = r["core"]["initial"] if r["view"] == "other_agent" else r["core"]["base" if run == "B" else "source"]
    return r["clean"][run]["argmax_cand"] == target


def summarize(res):
    lines = []
    for arm in ARMS:
        for view in ("direct", "world", "other_agent"):
            R_all = [r for r in res if r["arm"] == arm and r["view"] == view]
            if not R_all:
                continue
            for subset, R in (("all", R_all), ("competent", [r for r in R_all if correct(r, "B") and correct(r, "S")])):
                if len(R) < 5:
                    lines.append(f"{arm:7s} {view:11s} [{subset}] n={len(R)} (too few)")
                    continue
                base = np.array([r["m"]["ID@0"]["m"] for r in R])
                span = np.array([r["clean"]["S"]["m"] for r in R]) - base
                d = {k: np.array([r["m"][k]["m"] for r in R]) - base for k in R[0]["m"] if k != "ID@0"}
                dl = {k: {t: np.array([r["m"][k]["lp"][t] - r["m"]["ID@0"]["lp"][t] for r in R]) for t in ("S", "B", "X")}
                      for k in R[0]["m"]}
                mass = np.mean([r["clean"]["B"]["mass"] for r in R])
                floor = np.mean([abs(r["m"]["ID@0"]["m"] - r["clean"]["B"]["m"]) for r in R])
                lines.append(f"{arm:7s} {view:11s} [{subset}] n={len(R):3d} cand-mass={mass:.2f} "
                             f"span={span.mean():+6.2f} batch-noise-floor={floor:.3f}")
                l0s = sorted({int(k.split("@")[1]) for k in d if k.startswith("KV_S")})
                for l0 in l0s:
                    dK, dV, dKV = d[f"K_S@{l0}"], d[f"V_S@{l0}"], d[f"KV_S@{l0}"]
                    lines.append(f"   l0={l0:2d}  dK {fmt(boot(dK))}  dV {fmt(boot(dV))}  dKV {fmt(boot(dKV))}")
                    lines.append(f"          key share {fmt(boot_ratio(dK, dK + dV))}  interaction {fmt(boot(dKV - dK - dV))}"
                                 f"  interaction/dKV {fmt(boot_ratio(dKV - dK - dV, dKV))}")
                idK = 0.5 * ((dl["K_S@0"]["S"] - dl["K_X@0"]["S"]) + (dl["K_X@0"]["X"] - dl["K_S@0"]["X"]))
                idV = 0.5 * ((dl["V_S@0"]["S"] - dl["V_X@0"]["S"]) + (dl["V_X@0"]["X"] - dl["V_S@0"]["X"]))
                lines.append(f"   identity(K) {fmt(boot(idK))}   identity(V) {fmt(boot(idV))}   "
                             f"dlogp(B) under K_S {dl['K_S@0']['B'].mean():+.2f}, K_X {dl['K_X@0']['B'].mean():+.2f}")
    return "\n".join(lines)


def provenance(a):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    return {"args": vars(a), "git_commit": commit, "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "wrapper": dict(WRAPPER_USED), "l0_fracs": L0_FRACS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--views", default="direct")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/format_factorial")
    ap.add_argument("--device-map", default=None, help="'auto' to shard a large model across GPUs")
    ap.add_argument("--revision", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    kw = {"dtype": getattr(torch, a.dtype), "revision": a.revision}
    if "gemma-2" in a.model.lower():
        kw["attn_implementation"] = "eager"  # SDPA drops Gemma-2 attention-logit softcapping
    if a.device_map:
        kw["device_map"] = a.device_map
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if not a.device_map:
        model = model.to(device)
    device = next(model.parameters()).device  # inputs go to the first shard
    cores = make_cores(a.n, random.Random(a.seed))
    res, skipped, t0 = [], 0, time.time()
    for arm in a.arms.split(","):
        for view in a.views.split(","):
            for core in cores:
                r = run_item(model, tok, core, arm, view, device)
                if r is None:
                    skipped += 1
                else:
                    res.append(r)
            print(f"  {arm}/{view} done ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_s{a.seed}"
    prov = provenance(a) | {"skipped_items": skipped}
    json.dump({"provenance": prov, "results": res}, open(f"{a.out}/{tag}.json", "w"))
    s = json.dumps(prov) + "\n" + summarize(res)
    open(f"{a.out}/{tag}_summary.txt", "w").write(s + "\n")
    print(s)
    if not res:
        raise SystemExit("no valid items")


if __name__ == "__main__":
    main()
