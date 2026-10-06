"""Which query rows read the critical token's key? Exact row-restricted key swap.

At every layer the attention block runs twice on the same hidden states: once with the base key at the
critical position p and once with the source key K_S there. Output rows in a chosen group take the K_S
version; all other rows take the base version. Only that group "sees" the swapped key, at every layer.
Validated by: all rows == full K_S swap, no rows == clean base.

Groups: story tail after p, question, choices line, the six location words in the choices line, and the
remaining tail (instruction, chat tokens, prefill, final position). Arms registered in ckeys.encoding with a
``span`` get the groups ``mention`` / ``mention_words`` (the span's rows and its candidate-word rows).

Tasks: ``--task belief`` (default) builds the belief-story prompts; ``--task NAME`` for any other task takes
``RowTask`` from ``ckeys.NAME`` (or a class passed to ``register_task``), an object with ``chat``, ``bos``,
``cores(n, seed)``, ``prompts(core, arm) -> (raw_B, raw_S)``, ``targets(tok, core, arm) -> (id_S, id_B)`` and
``groups(tok, text, ids, offsets, p, core, arm) -> {name: rows}`` (self/rest_after_p/all/none are added here).
Output: <out>/<model>_<view|task>.json = {"provenance": (commit, args, versions, device, dtype, attention
implementation, skipped_items), "results": [items]} and a summary; items whose B and S encodings differ in length are
skipped and counted. TEST_MODE (--test or TEST_MODE=1): CPU.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import capture_kv, site
from ckeys.encoding import (ARMS, LETTER_LISTING, LISTING, ROOM, arm_span, build_prompt, candidate_ids, chat_text,
                            import_arm_modules)
from ckeys.interventions import blocks
from ckeys.story import LOCATIONS, make_cores, pick_x, record
from experiments.format_factorial import provenance

SYSTEM = "You are a helpful assistant."
GROUP_ORDER = ["self", "story_tail", "question", "choices", "choice_words", "remention", "remention_words",
               "mention", "mention_words", "rest_after_p"]


def encode_with_offsets(tok, raw, chat=True, bos=None):
    text = chat_text(tok, raw, SYSTEM) if chat else raw
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    ids, off = enc.input_ids, enc.offset_mapping[0].tolist()
    if not chat and (bos or (bos is None and tok.bos_token_id is not None)):
        ids, off = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], 1), [(0, 0)] + off
    return text, ids, off


def rows_in(offsets, a, b):
    return [i for i, (s, e) in enumerate(offsets) if e > a and s < b]


class BeliefTask:
    chat, bos = True, None

    def __init__(self, view="direct"):
        self.view = view

    def cores(self, n, seed=0):
        return make_cores(n, random.Random(seed))

    def prompts(self, core, arm):
        X = pick_x(core)
        return tuple(build_prompt(arm, r["story"], r["query"], core, X)
                     for r in (record(core, self.view, core[k]) for k in ("base", "source")))

    def targets(self, tok, core, arm):
        cid = candidate_ids(tok, arm)
        return cid[LOCATIONS.index(core["source"])], cid[LOCATIONS.index(core["base"])]

    def groups(self, tok, tb, ib, off, p, core, arm):
        rb = record(core, self.view, core["base"])
        story_end = tb.index(rb["story"]) + len(rb["story"])
        q0 = tb.index("\nQuestion: ")
        q1 = q0 + len("\nQuestion: " + rb["query"])
        loc_ids = set(candidate_ids(tok, "P1"))
        groups = {"story_tail": [i for i in rows_in(off, 0, story_end) if i > p], "question": rows_in(off, q0 + 1, q1)}
        if arm in ARMS:  # legacy group names of the standard arms
            listing = LETTER_LISTING if arm == "LETTER" else LISTING
            c0 = tb.find(listing)
            c1 = c0 + len(listing) if c0 >= 0 else -1
            r0 = tb.find(ROOM) if arm == "POST" else -1
            r1 = r0 + len(ROOM) if r0 >= 0 else -1
            groups |= {
                "choices": rows_in(off, c0, c1) if c0 >= 0 else [],
                "choice_words": [i for i in rows_in(off, c0, c1) if ib[0, i].item() in loc_ids] if c0 >= 0 else [],
                "remention": rows_in(off, r0, r1) if r0 >= 0 else [],
                "remention_words": [i for i in rows_in(off, r0, r1) if ib[0, i].item() in loc_ids] if r0 >= 0 else [],
            }
        else:
            span = arm_span(arm, rb["story"], rb["query"], core, pick_x(core))
            s0 = tb.find(span) if span else -1
            assert span is None or tb.count(span) == 1, (arm, span)
            s1 = s0 + len(span) if s0 >= 0 else -1
            groups |= {"mention": rows_in(off, s0, s1) if s0 >= 0 else [],
                       "mention_words": [i for i in rows_in(off, s0, s1) if ib[0, i].item() in loc_ids] if s0 >= 0 else []}
        return groups


TASKS = {"belief": BeliefTask}


def register_task(name, cls):
    TASKS[name] = cls


def get_task(name, **kw):
    if name not in TASKS:
        register_task(name, importlib.import_module(f"ckeys.{name}").RowTask)
    return TASKS[name](**kw)


class RowSplice:
    """Patches every layer's attention so selected rows see K_S at position p; others see K_B.
    ``ks[l]`` is the key at p (the k_proj output; for GPT-2 the key slice of c_attn)."""

    def __init__(self, model):
        self.model, self.active, self.mask, self.pos, self.ks = model, False, None, None, None
        self.use_src = False
        self.layers = None   # optional set of layers where K_S is visible
        self.group = None    # optional KV-group index; only that group's slice of the key is swapped
        self.head_dim = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
        self.orig = []
        for l, blk in enumerate(blocks(model)):
            mod, sl = site(model, l, "k")
            mod.register_forward_hook(self._khook(l, sl))
            at = blk.attn if hasattr(blk, "attn") else blk.self_attn
            f = at.forward
            self.orig.append(f)
            at.forward = self._wrap(f)

    def _khook(self, l, ksl):
        def hk(_m, _i, out):
            if self.active and self.use_src and (self.layers is None or l in self.layers):
                out = out.clone()
                if self.group is None:
                    out[:, self.pos, ksl] = self.ks[l].to(out.device, out.dtype)
                else:
                    sl = slice(self.group * self.head_dim, (self.group + 1) * self.head_dim)
                    off = ksl.start or 0
                    out[:, self.pos, off + sl.start:off + sl.stop] = self.ks[l][sl].to(out.device, out.dtype)
            return out
        return hk

    def _wrap(self, f):
        def fwd(*args, **kw):
            if not self.active:
                return f(*args, **kw)
            self.use_src = False
            base = f(*args, **kw)
            self.use_src = True
            src = f(*args, **kw)
            self.use_src = False
            m = self.mask.to(base[0].device).view(1, -1, 1)
            return (torch.where(m, src[0], base[0]),) + tuple(base[1:])
        return fwd


@torch.no_grad()
def run(model, tok, task, arms, n, seed=0, windows=0, log=print):
    dev = next(model.parameters()).device
    nL = len(blocks(model))
    rs = RowSplice(model)
    cores = task.cores(n, seed)
    res, t0 = [], time.time()
    for arm in arms:
        for core in cores:
            raw_b, raw_s = task.prompts(core, arm)
            tb, ib, off = encode_with_offsets(tok, raw_b, task.chat, task.bos)
            _, is_, _ = encode_with_offsets(tok, raw_s, task.chat, task.bos)
            if ib.shape != is_.shape:
                log(f"  skipped {arm} core {core}: B and S encodings differ in length")
                continue
            ib, is_ = ib.to(dev), is_.to(dev)
            diff = (ib[0] != is_[0]).nonzero().flatten().tolist()
            assert len(diff) == 1
            p = diff[0]
            T = ib.shape[1]
            groups = {"self": [p]} | task.groups(tok, tb, ib, off, p, core, arm)
            covered = set().union(*groups.values())
            groups["rest_after_p"] = [i for i in range(p + 1, T) if i not in covered]
            groups["all"] = list(range(T))
            iS, iB = task.targets(tok, core, arm)
            with capture_kv(model, [p], range(nL), "k") as K:
                lpS = torch.log_softmax(model(is_, use_cache=False).logits[0, -1].float(), -1)
            rs.ks = {l: K[(l, "k")][0, 0] for l in range(nL)}
            rs.pos = p
            lpB = torch.log_softmax(model(ib, use_cache=False).logits[0, -1].float(), -1)
            out = {"core": core, "arm": arm, "p": p, "T": T, "sizes": {g: len(v) for g, v in groups.items()},
                   "m_B": (lpB[iS] - lpB[iB]).item(), "m_S": (lpS[iS] - lpS[iB]).item(), "m": {}}
            rs.active = True
            for g, idx in list(groups.items()) + [("none", [])]:
                mask = torch.zeros(T, dtype=torch.bool)
                mask[idx] = True
                rs.mask = mask
                lp = torch.log_softmax(model(ib, use_cache=False).logits[0, -1].float(), -1)
                out["m"][g] = (lp[iS] - lp[iB]).item()
            if windows:
                cw = torch.zeros(T, dtype=torch.bool)
                cw[groups["choice_words"]] = True
                rs.mask = cw
                for w0 in range(0, nL, windows):
                    rs.layers = set(range(w0, min(nL, w0 + windows)))
                    lp = torch.log_softmax(model(ib, use_cache=False).logits[0, -1].float(), -1)
                    out["m"][f"win{w0:02d}"] = (lp[iS] - lp[iB]).item()
                rs.layers = None
                for g in range(model.config.num_key_value_heads):
                    rs.group = g
                    lp = torch.log_softmax(model(ib, use_cache=False).logits[0, -1].float(), -1)
                    out["m"][f"kvgroup{g}"] = (lp[iS] - lp[iB]).item()
                rs.group = None
            rs.active = False
            res.append(out)
        log(f"  {arm} done ({time.time() - t0:.0f}s)")
    return res


def summarize(res, arms):
    lines = []
    for arm in arms:
        R = [r for r in res if r["arm"] == arm]
        full = np.array([r["m"]["all"] - r["m_B"] for r in R])
        lines.append(f"{arm}: n={len(R)}  full K_S effect {full.mean():+.2f} nats;  none-check {np.mean([r['m']['none'] - r['m_B'] for r in R]):+.3f}")
        extra = sorted(k for k in R[0]["m"] if k.startswith(("win", "kvgroup")))
        task_groups = [g for g in R[0]["sizes"] if g not in GROUP_ORDER and g != "all"]  # other tasks' groups
        for g in GROUP_ORDER + task_groups + extra:
            if g not in R[0]["m"] or not R[0]["sizes"].get(g, 1):
                continue
            d = np.array([r["m"][g] - r["m_B"] for r in R])
            rng = np.random.default_rng(0)
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            size = np.mean([r['sizes'][g] for r in R]) if g in R[0]["sizes"] else float("nan")
            lines.append(f"   rows={g:13s} (avg {size:5.1f} tok)  d={d.mean():+6.2f} "
                         f"[{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]  frac of full {d.mean() / full.mean():+.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--arms", default="P1,LETTER")
    ap.add_argument("--view", default="direct")
    ap.add_argument("--task", default="belief")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm-modules", default="", help="comma-separated modules registering further arms")
    ap.add_argument("--out", default="results/row_restricted")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--windows", type=int, default=0, help="if >0, also sweep layer windows of this size and KV groups (choice-word rows only)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--test", action="store_true", help="TEST_MODE: CPU")
    a = ap.parse_args()
    test = a.test or bool(os.environ.get("TEST_MODE"))
    import_arm_modules(a.arm_modules)
    task = get_task(a.task, view=a.view) if a.task == "belief" else get_task(a.task)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    kw = {"dtype": getattr(torch, a.dtype), "revision": a.revision}
    if a.device_map:
        kw["device_map"] = a.device_map
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if not a.device_map and torch.cuda.is_available() and not test:
        model = model.to("cuda")
    arms = a.arms.split(",")
    res = run(model, tok, task, arms, a.n, a.seed, a.windows, lambda s: print(s, flush=True))
    skipped = len(arms) * len(task.cores(a.n, a.seed)) - len(res)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_{a.view}" if a.task == "belief" else f"{a.model.split('/')[-1]}_{a.task}"
    prov = provenance(a) | {"skipped_items": skipped, "task": a.task, "dtype": str(next(model.parameters()).dtype),
                            "attn_implementation": model.config._attn_implementation, "test_mode": test}
    json.dump({"provenance": prov, "results": res}, open(f"{a.out}/{tag}.json", "w"))
    print(f"skipped_items {skipped}")
    s = summarize(res, arms)
    open(f"{a.out}/{tag}_summary.txt", "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
