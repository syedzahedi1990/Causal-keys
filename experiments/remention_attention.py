"""Do the re-mentioned words attend to the writing token p, and does the answer row read them? (prereg G, part a)

One eager-attention instrument. Per story core and arm, five single-sequence passes: clean B, S and X (the pre-RoPE key
at p captured from S and X, the value at p from B), then the base prompt with the S key and with the X key clamped at p
in every layer while the value at p is held at the base run's (exactly the K_S@0 / K_X@0 rows of the factorial: p
attends to itself, so a key-only hook would let p's own later-layer values drift). ``probe`` returns the attention block [L, H, rows, cols] of every
pass plus the answer-position log-probs and serves the other parts (part d's span probe, part b's descriptive pass);
``item_2a`` derives the per-head arrays of part (a): the hop-1 match excess D = A^B[r_b -> p] - mean_N A^B[r_j -> p]
(N = candidates with no mention before the span), the follow-the-key gains under K_S, the hop-2 analogue at the answer
row, column baselines, the in-story duplicate columns q_obj / q_dist, the full answer row and the r_b row entropy.
Log-probs of all 12 candidate ids (lowercase and capitalised) are recorded under every pass, with the full-vocabulary
argmax and both candidate masses, so that every key effect can be computed on the casing the model emits.

Output: <out>/<model>.json (provenance, positions, N_i, log-probs) and <out>/<model>.npz ("<arm>/<name>" = [n, L, H]
fp16 arrays in the JSON's item order; "<arm>/<i>/ans_row" = [L, H, T] fp16). Scored by analysis/stage5_parts/attention.py.
TEST_MODE (--test, or TEST_MODE=1 in the environment, which keeps --n): Qwen2.5-0.5B-Instruct, FP32, CPU, n = 8 (a 4/4
parity split), both arms.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import ARM_BUILDERS, arm_span, build_prompt, candidate_ids, chat_text, import_arm_modules
from ckeys.interventions import blocks
from ckeys.story import LOCATIONS, make_cores, pick_x, record
from experiments.format_factorial import provenance
from experiments.row_restricted_keys import rows_in

CONDS = ("B", "S", "X", "K_S", "K_X")
TEST_MODEL, TEST_N = "Qwen/Qwen2.5-0.5B-Instruct", 8
# per-head [L, H] arrays saved for part (a): name -> (condition, row, column); "N" = mean over the no-prior candidates
ARRAYS = {"A_B_rb_p": ("B", "r_b", "p"), "A_B_rs_p": ("B", "r_s", "p"), "A_B_rx_p": ("B", "r_x", "p"),
          "A_B_N_p": ("B", "N", "p"), "A_KS_rs_p": ("K_S", "r_s", "p"), "A_KS_rb_p": ("K_S", "r_b", "p"),
          "A_KX_rx_p": ("K_X", "r_x", "p"), "A_KX_rb_p": ("K_X", "r_b", "p"), "A_KX_rs_p": ("K_X", "r_s", "p"),
          "A_B_rb_pm1": ("B", "r_b", "p-1"), "A_B_rb_pp1": ("B", "r_b", "p+1"), "A_B_rb_0": ("B", "r_b", "0"),
          "A_B_rinit_qobj": ("B", "r_init", "q_obj"), "A_B_N_qobj": ("B", "N", "q_obj"),
          "A_B_rdist_qdist": ("B", "r_dist", "q_dist"), "A_B_N_qdist": ("B", "N", "q_dist"),
          "A_B_ans_rb": ("B", "ans", "r_b"), "A_B_ans_N": ("B", "ans", "N"), "A_B_ans_rs": ("B", "ans", "r_s"),
          "A_KS_ans_rs": ("K_S", "ans", "r_s"), "A_KS_ans_rb": ("K_S", "ans", "r_b"), "A_B_ans_p": ("B", "ans", "p")}


def load_model(name, dtype="float32", revision=None, device_map=None, cpu=False):
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    kw = {"dtype": getattr(torch, dtype), "revision": revision, "attn_implementation": "eager"}
    if device_map:
        kw["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    if not device_map and torch.cuda.is_available() and not cpu:
        model = model.to("cuda")
    assert model.config._attn_implementation == "eager", model.config._attn_implementation
    return model, tok


def encode_item(tok, core, arm, view="direct", X=None):
    """Chat-wrapped B / S / X prompts of one core: ``ids`` per run, the base text and token offsets, p and T.
    None when the three runs differ in length or in more than the one position p."""
    X = X or pick_x(core)
    ids, text, off = {}, None, None
    for name, loc in (("B", core["base"]), ("S", core["source"]), ("X", X)):
        r = record(core, view, loc)
        t = chat_text(tok, build_prompt(arm, r["story"], r["query"], core, X))
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        ids[name] = enc.input_ids
        if name == "B":
            text, off, rec = t, enc.offset_mapping[0].tolist(), r
    if len({tuple(v.shape) for v in ids.values()}) > 1:
        return None
    diff = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()
    if len(diff) != 1 or (ids["B"][0] != ids["X"][0]).nonzero().flatten().tolist() != diff:
        return None
    return {"ids": ids, "text": text, "off": off, "p": diff[0], "T": ids["B"].shape[1], "X": X, "rec": rec}


def _pass(model, ids, rows, cols, full_rows, nL):
    out = model(ids, use_cache=False, output_attentions=True, logits_to_keep=1)
    assert len(out.attentions) == nL, (len(out.attentions), nL)
    att = torch.stack([a[0, :, rows][:, :, cols].float().cpu() for a in out.attentions])          # [L, H, R, C]
    full = {r: torch.stack([a[0, :, r].float().cpu() for a in out.attentions]) for r in full_rows}  # [L, H, T]
    return att, full, torch.log_softmax(out.logits[0, -1].float(), -1).cpu()


@torch.no_grad()
def probe(model, ids, p, rows, cols=None, conds=CONDS, full_rows=()):
    """Eager passes of one item: ``ids`` = {"B", "S", "X"} as from ``encode_item``; the key at ``p`` captured from S and X
    and clamped into the base run (K_S, K_X) in every layer with the value at ``p`` held at B's (the factorial's K_S@0 /
    K_X@0 rows). Returns ``att[cond]`` = [L, H, len(rows), len(cols)], ``full[cond][row]`` = [L, H, T] for ``full_rows``,
    ``lp[cond]`` = full-vocabulary log-probs at T-1 (all float32, CPU). ``cols`` defaults to ``[p]``."""
    dev, nL = next(model.parameters()).device, len(blocks(model))
    rows, cols = list(rows), [p] if cols is None else list(cols)
    out = {"att": {}, "full": {}, "lp": {}}
    kv = {}
    for c in conds:
        if c in ("B", "S", "X"):
            with capture_kv(model, [p], range(nL)) as t:
                res = _pass(model, ids[c].to(dev), rows, cols, full_rows, nL)
            kv[c] = {k: v[0] for k, v in t.items()}
        else:
            src = c[2:]
            for name in (src, "B"):
                if name not in kv:
                    with capture_kv(model, [p], range(nL)) as t:
                        model(ids[name].to(dev), use_cache=False, logits_to_keep=1)
                    kv[name] = {k: v[0] for k, v in t.items()}
            tabs = {(l, "k"): kv[src][(l, "k")] for l in range(nL)} | {(l, "v"): kv["B"][(l, "v")] for l in range(nL)}
            with clamp_kv(model, [p], tabs, range(nL)):
                res = _pass(model, ids["B"].to(dev), rows, cols, full_rows, nL)
        out["att"][c], out["full"][c], out["lp"][c] = res
    return out


def span_sum(out, rows_idx, col=0):
    """Head-averaged attention summed over a span (indices into ``probe``'s ``rows``) to column ``col``: {cond: [L]}.
    The per-layer statistic of part (d)'s form probe (span tokens -> p under K_S / K_X)."""
    return {c: A[:, :, rows_idx, col].mean(1).sum(-1) for c, A in out["att"].items()}


def cased_ids(tok):
    """The 12 candidate ids: lowercase (single tokens, asserted) and capitalised (first piece when multi-token, flagged)."""
    cap = [tok.encode(" " + w.capitalize(), add_special_tokens=False) for w in LOCATIONS]
    return {"lower": candidate_ids(tok, "P1"), "cap": [c[0] for c in cap], "cap_single": all(len(c) == 1 for c in cap)}


def positions(tok, core, enc, arm, cid):
    """Rows and columns of part (a): the six re-mention rows (canonical order), ans, p-1, p+1, 0, q_obj, q_dist and
    the no-prior set N; the asserts of the entry (a violation aborts)."""
    text, off, p, T, toks = enc["text"], enc["off"], enc["p"], enc["T"], enc["ids"]["B"][0].tolist()
    span_text = arm_span(arm, enc["rec"]["story"], enc["rec"]["query"], core, enc["X"])
    assert span_text and text.count(span_text) == 1, (arm, span_text)
    s0 = text.index(span_text)
    span = rows_in(off, s0, s0 + len(span_text))
    r = []
    for c in cid:
        hits = [i for i in span if toks[i] == c]
        assert len(hits) == 1, (arm, hits)
        r.append(hits[0])
    b, s, x = (LOCATIONS.index(core[k]) if k != "X" else LOCATIONS.index(enc["X"]) for k in ("base", "source", "X"))
    assert toks[p] == cid[b] and cid[b] not in toks[:p] and toks.count(cid[s]) == 1 and min(span) > p, (arm, p, span)
    N = [j for j, c in enumerate(cid) if c not in toks[:span[0]]]
    assert s in N and x in N and 3 <= len(N) <= 4, N
    ini, dist = LOCATIONS.index(core["initial"]), LOCATIONS.index(core["distractor_location"])

    def col(sentence, word_id):
        o = text.index(sentence)
        hits = [i for i in rows_in(off, o, o + len(sentence)) if toks[i] == word_id]
        assert len(hits) == 1 and hits[0] < p, (sentence, hits)
        return hits[0]

    q_obj = col(f"the {core['object']} is in the {core['initial']}.", cid[ini])
    q_dist = col(f"the {core['distractor']} is in the {core['distractor_location']}.", cid[dist])
    return {"p": p, "T": T, "ans": T - 1, "r": r, "b": b, "s": s, "x": x, "init": ini, "dist": dist, "N": N,
            "q_obj": q_obj, "q_dist": q_dist, "span": [span[0], span[-1]]}


@torch.no_grad()
def item_2a(model, tok, core, arm, cid, view="direct"):
    """One core x arm of part (a): the ARRAYS [L, H] (fp16), the r_b row entropy, the ans row under clean B, the
    12-id log-probs / argmax / masses under every condition."""
    enc = encode_item(tok, core, arm, view)
    assert enc is not None, (arm, core)
    pos = positions(tok, core, enc, arm, cid["lower"])
    p, ans, r = pos["p"], pos["ans"], pos["r"]
    rows = r + [ans]                                           # 0..5 = r_j, 6 = ans
    cols = [p, p - 1, p + 1, 0, pos["q_obj"], pos["q_dist"]] + r  # 0 = p, 1 = p-1, 2 = p+1, 3 = 0, 4 = q_obj, 5 = q_dist, 6.. = r_j
    ri = {"r_b": pos["b"], "r_s": pos["s"], "r_x": pos["x"], "r_init": pos["init"], "r_dist": pos["dist"], "ans": 6}
    ci_ = {"p": 0, "p-1": 1, "p+1": 2, "0": 3, "q_obj": 4, "q_dist": 5, "r_b": 6 + pos["b"], "r_s": 6 + pos["s"]}
    out = probe(model, enc["ids"], p, rows, cols, CONDS, full_rows=[r[pos["b"]], ans])
    same = core["initial"] == core["distractor_location"]

    def pick(cond, row, colname):
        A = out["att"][cond]
        if row == "N":
            v = A[:, :, pos["N"], ci_[colname]].mean(-1)
        elif colname == "N":
            v = A[:, :, ri[row], [6 + j for j in pos["N"]]].mean(-1)
        else:
            v = A[:, :, ri[row], ci_[colname]]
        if colname == "q_dist" and same:
            v = torch.full_like(v, float("nan"))
        return v

    arrays = {k: pick(*v) for k, v in ARRAYS.items()}
    arrays["A_B_r_p"] = out["att"]["B"][:, :, :6, 0]        # [L, H, 6]: every r_j -> p (permutation null)
    arrays["A_B_ans_r"] = out["att"]["B"][:, :, 6, 6:12]     # [L, H, 6]: ans -> every r_j
    arrays["D"] = arrays["A_B_rb_p"] - arrays["A_B_N_p"]
    arrays["D2"] = arrays["A_B_ans_rb"] - arrays["A_B_ans_N"]
    rb_row = out["full"]["B"][r[pos["b"]]].clamp_min(1e-30)
    arrays["H_B_rb"] = -(rb_row * rb_row.log()).sum(-1)
    arrays = {k: v.half().numpy() for k, v in arrays.items()}
    lp, am, mass = {}, {}, {}
    for c in CONDS:
        v = out["lp"][c]
        lp[c] = {cs: [v[i].item() for i in cid[cs]] for cs in ("lower", "cap")}
        am[c] = int(v.argmax())
        mass[c] = {cs: v[cid[cs]].exp().sum().item() for cs in ("lower", "cap")}
    item = {"core": core, "X": enc["X"], "arm": arm, "view": view, "same_init_dist": same, "lp": lp, "argmax": am,
            "argmax_tok": {c: tok.decode([am[c]]) for c in CONDS}, "mass": mass} | {k: v for k, v in pos.items()}
    return item, arrays, out["full"]["B"][ans].half().numpy()


def run(model, tok, cores, arms, view="direct", log=print, ans_rows=True):
    cid = cased_ids(tok)
    items, arrays, rows_, t0 = [], {}, {}, time.time()
    for arm in arms:
        acc = {}
        for i, core in enumerate(cores):
            it, ar, row = item_2a(model, tok, core, arm, cid, view)
            it["i"] = i
            items.append(it)
            for k, v in ar.items():
                acc.setdefault(k, []).append(v)
            if ans_rows:
                rows_[f"{arm}/{i}/ans_row"] = row
            if i == 0:
                log(f"  {arm}: first core {time.time() - t0:.1f}s, T={it['T']}, p={it['p']}, r={it['r']}, ans={it['ans']}")
        arrays |= {f"{arm}/{k}": np.stack(v) for k, v in acc.items()}
        log(f"  {arm} done ({time.time() - t0:.0f}s)")
    return items, arrays | rows_, cid


def write(out, model_name, a, items, arrays, cid, model):
    Path(out).mkdir(parents=True, exist_ok=True)
    tag = model_name.split("/")[-1]
    cfg = model.config
    prov = provenance(a) | {"attn_implementation": cfg._attn_implementation, "n_layers": len(blocks(model)),
                            "n_heads": cfg.num_attention_heads, "n_kv": getattr(cfg, "num_key_value_heads", None),
                            "casing_check": "ok" if cid["cap_single"] else "skipped", "ids": cid, "arrays": ARRAYS}
    np.savez_compressed(f"{out}/{tag}.npz", **arrays)   # the npz first: a readable JSON (what the pipeline's keep() checks) then certifies both
    json.dump({"provenance": prov, "items": items}, open(f"{out}/{tag}.json", "w"))
    return f"{out}/{tag}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--arms", default="POST,P1")
    ap.add_argument("--view", default="direct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gpu_stage5/attention")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--arm-modules", default="", help="comma-separated modules registering further arms")
    ap.add_argument("--no-ans-rows", action="store_true", help="skip the full answer rows ([L, H, T] per story, ~420 KB at 14B)")
    ap.add_argument("--test", action="store_true", help=f"TEST_MODE: {TEST_MODEL}, FP32, n = {TEST_N}")
    a = ap.parse_args()
    if a.test or os.environ.get("TEST_MODE"):  # --test also fixes n; TEST_MODE=1 keeps the caller's n and arms
        a.model, a.dtype, a.device_map = TEST_MODEL, "float32", None
        a.n = TEST_N if a.test else a.n
    import_arm_modules(a.arm_modules)
    arms = a.arms.split(",")
    assert all(x in ARM_BUILDERS for x in arms), arms
    torch.set_grad_enabled(False)
    model, tok = load_model(a.model, a.dtype, a.revision, a.device_map, cpu=bool(a.test or os.environ.get("TEST_MODE")))
    cores = make_cores(a.n, random.Random(a.seed))
    items, arrays, cid = run(model, tok, cores, arms, a.view, lambda s: print(s, flush=True), not a.no_ans_rows)
    f = write(a.out, a.model, a, items, arrays, cid, model)
    print(f"wrote {f} ({len(items)} items, casing_check {'ok' if cid['cap_single'] else 'skipped'})")


if __name__ == "__main__":
    main()
