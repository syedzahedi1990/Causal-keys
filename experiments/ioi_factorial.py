"""Key/value factorial at the IO mention of IOI prompts across format arms (part (e) of preregistration G).

For each arm and core the key and/or value that later tokens read at the IO mention p is set to the IO_S or IO_X
run's (the three runs differ only at p) for all layers >= l0 in the IO_B prompt, through ckeys.clamp (k_proj / v_proj
for Qwen and Mistral, the key and value slices of the fused c_attn for GPT-2). One batch of ten rows per item: the
self-clamp row ID (reference), K_S, V_S, KV_S, K_X, V_X, KV_X at l0 = 0 and K_S, V_S, KV_S at l0 = round(0.3 L).
As in format_factorial, a single-channel row holds the other channel at p at the base run's values (p attends to
itself, so a key-only hook would let p's own later-layer values drift).
Three clean single-sequence passes (B, S, X) capture the tables and the clean log-probs of the four tracked names
(' IO_S', ' IO_B', ' IO_X', ' S'); LD = logit(IO_B) - logit(S), two-way = LD > 0, four-way = argmax over the four.
FP32 runs check per item floor_B = |m(ID) - m(clean B)| <= 1e-3 and floor_S = max |lp(KV_S) - lp(clean S)| <= 1e-3
(the c_attn split order and the clamp site): every item is kept with its floors, the JSON is written with the count of
violations in its provenance (exact_violations), and the run then fails if the count is not 0; BF16 runs record the
floors and warn when an arm mean exceeds 1 nat.
Chat path (chat_text, 'Answer:' prefill) for AFTER / BEFORE / QUESTION on instruct models, raw otherwise; PLAIN,
INLINE and INLINE_BEFORE are always raw; BOS iff the tokenizer has one (GPT-2, Mistral) unless --bos is given.
Output: <out>/<model>_s<seed>.json (format_factorial's shape: clean / m rows with m and lp) and a summary.
TEST_MODE (--test, or TEST_MODE=1 which keeps --n): Qwen2.5-0.5B-Instruct unless --model is given, FP32, CPU, n = 3.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import capture_kv, clamp_kv, stack_rows
from ckeys.encoding import WRAPPER_USED
from ckeys.interventions import blocks
from ckeys.ioi import ARMS, SEED, arm_chat, check_occurrences, encode_runs, identity_measures, make_cores, name_ids
from experiments.format_factorial import LABEL, boot, boot_ratio, fmt, last_logprobs

L0_FRACS = (0.0, 0.3)
TEST_MODEL, TEST_N = "Qwen/Qwen2.5-0.5B-Instruct", 3
TRACK = ("S", "B", "X", "Subj")


def row_specs(nL: int) -> list[tuple[str, str, int]]:
    l0 = round(0.3 * nL)
    return [("B", "B", 0), ("S", "B", 0), ("B", "S", 0), ("S", "S", 0), ("X", "B", 0), ("B", "X", 0), ("X", "X", 0),
            ("S", "B", l0), ("B", "S", l0), ("S", "S", l0)]


def auto_chat(model: str) -> bool:
    m = model.lower()
    return "gpt2" not in m and "instruct" in m


def stats(lp, tid) -> dict:
    four = {k: lp[i].item() for k, i in tid.items()}
    return {"lp": four, "m": four["S"] - four["B"], "LD": four["B"] - four["Subj"], "argmax4": max(four, key=four.get),
            "mass": sum(math.exp(v) for v in four.values())}


@torch.no_grad()
def run_item(model, tok, core, arm, chat, bos, device):
    ids = encode_runs(tok, core, arm, chat, bos)
    if ids is None:
        return None
    check_occurrences(tok, core, arm, ids["B"])
    ids = {k: v.to(device) for k, v in ids.items()}
    pos = (ids["B"][0] != ids["S"][0]).nonzero().item()
    nL, tid = len(blocks(model)), name_ids(tok, core)
    kv, clean = {}, {}
    for name in ("B", "S", "X"):
        with capture_kv(model, [pos], range(nL)) as t:
            lp = last_logprobs(model, ids[name])[0]
        kv[name] = {k: v[0] for k, v in t.items()}
        clean[name] = stats(lp, tid)
    rows = row_specs(nL)
    tabs = stack_rows(kv, [lambda l, ch, r=r: (r[0] if ch == "k" else r[1]) if l >= r[2] else "B" for r in rows], range(nL))
    with clamp_kv(model, [pos], tabs, range(nL)):
        lp = last_logprobs(model, ids["B"].expand(len(rows), -1))
    out = {f"{LABEL[(k, v)]}@{l0}": {"m": (lp[i, tid["S"]] - lp[i, tid["B"]]).item(), "lp": {t: lp[i, j].item() for t, j in tid.items()}}
           for i, (k, v, l0) in enumerate(rows)}
    floor_B = abs(out["ID@0"]["m"] - clean["B"]["m"])
    floor_S = max(abs(out["KV_S@0"]["lp"][t] - clean["S"]["lp"][t]) for t in TRACK)
    return {"core": core, "arm": arm, "chat": arm_chat(arm, chat), "pos": pos, "len": ids["B"].shape[1], "n_layers": nL,
            "pattern": core["pattern"], "template": core["template"], "clean": clean, "m": out, "floor_B": floor_B, "floor_S": floor_S}


def summarize(res, arms=ARMS):
    lines = []
    for arm in arms:
        for pat in ("all", "ABBA", "BABA"):
            R = [r for r in res if r["arm"] == arm and (pat == "all" or r["pattern"] == pat)]
            if not R:
                continue
            M = [identity_measures(r) for r in R]
            if len(R) < 3:
                lines.append(f"{arm:14s} [{pat}] n={len(R)} (too few)")
                continue
            g = lambda k: np.array([x[k] for x in M], float)
            idK, idV, idKV = g("idK"), g("idV"), g("idKV")
            fl = f"floor_B={g('floor_B').mean():.2e} floor_S={g('floor_S').mean():.2e}" + (" NOISY (> 1 nat)" if max(g("floor_B").mean(), g("floor_S").mean()) > 1 else "")
            lines.append(f"{arm:14s} [{pat:4s}] n={len(R):3d} two-way={g('two_B').mean():.2f} four-way={g('four_B').mean():.2f} LD={fmt(boot(g('LD_B')))} "
                         f"cand-mass={g('mass_B').mean():.2f} {fl}")
            lines.append("   clean-B lp " + " ".join(f"{t} {np.mean([x['lpB'][t] for x in M]):+6.2f}" for t in TRACK) + f"   p {min(r['pos'] for r in R)}..{max(r['pos'] for r in R)} len {min(r['len'] for r in R)}-{max(r['len'] for r in R)}")
            lines.append(f"   ID_K {fmt(boot(idK))}  ID_V {fmt(boot(idV))}  ID_KV {fmt(boot(idKV))}  f_K {fmt(boot_ratio(idK, idKV))}  f_V {fmt(boot_ratio(idV, idKV))}"
                         f"  s_ID {fmt(boot_ratio(idK, idK + idV))}")
            for l0 in sorted({int(k.split('@')[1]) for k in M[0]['d'] if k.startswith('KV_S')}):
                dK, dV, dKV = (np.array([x["d"][f"{c}_S@{l0}"] for x in M]) for c in ("K", "V", "KV"))
                lines.append(f"   l0={l0:2d}  dK {fmt(boot(dK))}  dV {fmt(boot(dV))}  dKV {fmt(boot(dKV))}  key share {fmt(boot_ratio(dK, dK + dV))}  interaction {fmt(boot(dKV - dK - dV))}")
    return "\n".join(lines)


def provenance(a, label=None, device=None):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    return {"args": vars(a), "git_commit": commit, "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "device": device_name(device), "wrapper": dict(WRAPPER_USED), "l0_fracs": L0_FRACS, "label": label}


def device_name(device):
    """The device the model actually ran on (the GPU's name when it is CUDA), not merely the device visible."""
    d = torch.device(device) if device is not None else None
    return torch.cuda.get_device_name(d) if d is not None and d.type == "cuda" else str(d or "unknown")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help=f"default gpt2 ({TEST_MODEL} in TEST_MODE)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--bos", default="auto", help="auto (iff the tokenizer has a BOS), yes, no")
    ap.add_argument("--chat", default="auto", help="auto (off for gpt2 and names without 'Instruct'), yes, no; AFTER/BEFORE/QUESTION only")
    ap.add_argument("--assert-exact", default="auto", help="auto (float32 only), yes, no")
    ap.add_argument("--out", default="results/gpu_stage5/ioi")
    ap.add_argument("--label", default=None, help="free text stored in the provenance (e.g. 'CPU pilot')")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--test", action="store_true", help=f"TEST_MODE: {TEST_MODEL} (unless --model is given), FP32, CPU, n = {TEST_N}")
    a = ap.parse_args(argv)
    test = a.test or bool(os.environ.get("TEST_MODE"))
    if test:
        a.model = a.model or TEST_MODEL
        a.dtype, a.device_map = "float32", None
        a.n = TEST_N if a.test else a.n
        a.out = a.out if a.out != ap.get_default("out") else tempfile.mkdtemp(prefix="stage5_ioi_test_")
    a.model = a.model or "gpt2"
    chat = auto_chat(a.model) if a.chat == "auto" else a.chat == "yes"
    bos = None if a.bos == "auto" else a.bos == "yes"
    exact = a.dtype == "float32" if a.assert_exact == "auto" else a.assert_exact == "yes"
    device = "cuda" if torch.cuda.is_available() and not test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    kw = {"dtype": getattr(torch, a.dtype), "revision": a.revision}
    if a.device_map:
        kw["device_map"] = a.device_map
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if not a.device_map:
        model = model.to(device)
    device = next(model.parameters()).device
    cores = make_cores(a.n, random.Random(a.seed))
    arms = a.arms.split(",")
    res, skipped, t0 = [], 0, time.time()
    for arm in arms:
        for core in cores:
            r = run_item(model, tok, core, arm, chat, bos, device)
            if r is None:
                skipped += 1
            else:
                res.append(r)
        R = [r for r in res if r["arm"] == arm]
        fB, fS = (np.mean([r[k] for r in R]) if R else float("nan") for k in ("floor_B", "floor_S"))
        print(f"  {arm} done n={len(R)} chat={arm_chat(arm, chat)} floor_B={fB:.2e} floor_S={fS:.2e} ({time.time() - t0:.0f}s)"
              + ("  WARNING: batch-noise floor > 1 nat, cell is NOISY" if max(fB, fS) > 1 else ""), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_s{a.seed}"
    bad = [r for r in res if max(r["floor_B"], r["floor_S"]) > 1e-3]  # counted whether or not asserted (FP32 only: BF16 floors exceed 1e-3 by construction)
    prov = provenance(a, a.label, device) | {"skipped_items": skipped, "chat": chat, "bos": bos, "assert_exact": exact,
                                     "exact_violations": len(bad) if a.dtype == "float32" else None, "exact_threshold": 1e-3,
                                     "max_floor": max((max(r["floor_B"], r["floor_S"]) for r in res), default=None),
                                     "attn_implementation": model.config._attn_implementation, "test_mode": test, "n_items": len(res)}
    json.dump({"provenance": prov, "results": res}, open(f"{a.out}/{tag}.json", "w"))
    s = json.dumps(prov) + "\n" + summarize(res, arms)
    open(f"{a.out}/{tag}_summary.txt", "w").write(s + "\n")
    print(s)
    print(f"wrote {a.out}/{tag}.json")
    if not res:
        raise SystemExit("no valid items")
    if exact and bad:
        raise SystemExit(f"clamp not exact in {len(bad)} items (floor > 1e-3; the file is written): first "
                         f"{bad[0]['arm']} floor_B {bad[0]['floor_B']:.2e} floor_S {bad[0]['floor_S']:.2e}")


if __name__ == "__main__":
    main()
