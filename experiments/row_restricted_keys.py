"""Which query rows read the critical token's key? Exact row-restricted key swap.

At every layer the attention block runs twice on the same hidden states: once with the base key at the
critical position p and once with the source key K_S there. Output rows in a chosen group take the K_S
version; all other rows take the base version. Only that group "sees" the swapped key, at every layer.
Validated by: all rows == full K_S swap, no rows == clean base.

Groups: story tail after p, question, choices line, the six location words in the choices line, and the
remaining tail (instruction, chat tokens, prefill, final position).
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

from ckeys.encoding import LISTING, LETTER_LISTING, candidate_ids, chat_text, raw_prompt
from ckeys.interventions import blocks, capture
from ckeys.story import LOCATIONS, make_cores, record

SYSTEM = "You are a helpful assistant."


def encode_with_offsets(tok, raw):
    text = chat_text(tok, raw, SYSTEM)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    return text, enc.input_ids, enc.offset_mapping[0].tolist()


def rows_in(offsets, a, b):
    return [i for i, (s, e) in enumerate(offsets) if e > a and s < b]


class RowSplice:
    """Patches every layer's self_attn so selected rows see K_S at position p; others see K_B."""

    def __init__(self, model):
        self.model, self.active, self.mask, self.pos, self.ks = model, False, None, None, None
        self.use_src = False
        self.layers = None   # optional set of layers where K_S is visible
        self.group = None    # optional KV-group index; only that group's slice of the key is swapped
        self.head_dim = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
        self.orig = []
        for l, blk in enumerate(blocks(model)):
            at = blk.self_attn
            at.k_proj.register_forward_hook(self._khook(l))
            f = at.forward
            self.orig.append(f)
            at.forward = self._wrap(f)

    def _khook(self, l):
        def hk(_m, _i, out):
            if self.active and self.use_src and (self.layers is None or l in self.layers):
                out = out.clone()
                if self.group is None:
                    out[:, self.pos] = self.ks[l].to(out.dtype)
                else:
                    sl = slice(self.group * self.head_dim, (self.group + 1) * self.head_dim)
                    out[:, self.pos, sl] = self.ks[l][sl].to(out.dtype)
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--arms", default="P1,LETTER")
    ap.add_argument("--view", default="direct")
    ap.add_argument("--out", default="results/row_restricted")
    ap.add_argument("--windows", type=int, default=0, help="if >0, also sweep layer windows of this size and KV groups (choice-word rows only)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    nL = len(blocks(model))
    rs = RowSplice(model)
    cores = make_cores(a.n, random.Random(0))
    res, t0 = [], time.time()
    for arm in a.arms.split(","):
        cid = candidate_ids(tok, arm)
        listing = LETTER_LISTING if arm == "LETTER" else LISTING
        for core in cores:
            rb, rsrc = record(core, a.view, core["base"]), record(core, a.view, core["source"])
            tb, ib, off = encode_with_offsets(tok, raw_prompt(arm, rb["story"], rb["query"]))
            _, is_, _ = encode_with_offsets(tok, raw_prompt(arm, rsrc["story"], rsrc["query"]))
            if ib.shape != is_.shape:
                continue
            diff = (ib[0] != is_[0]).nonzero().flatten().tolist()
            assert len(diff) == 1
            p = diff[0]
            T = ib.shape[1]
            story_end = tb.index(rb["story"]) + len(rb["story"])
            q0 = tb.index("\nQuestion: ")
            q1 = q0 + len("\nQuestion: " + rb["query"])
            c0 = tb.index(listing)
            c1 = c0 + len(listing)
            groups = {
                "self": [p],
                "story_tail": [i for i in rows_in(off, 0, story_end) if i > p],
                "question": rows_in(off, q0 + 1, q1),
                "choices": rows_in(off, c0, c1),
                "choice_words": [i for i in rows_in(off, c0, c1)
                                 if ib[0, i].item() in set(candidate_ids(tok, "P1"))],
            }
            covered = set().union(*groups.values())
            groups["rest_after_p"] = [i for i in range(p + 1, T) if i not in covered]
            groups["all"] = list(range(T))
            iS, iB = cid[LOCATIONS.index(core["source"])], cid[LOCATIONS.index(core["base"])]
            with capture(model, range(nL), "k") as K:
                lpS = torch.log_softmax(model(is_, use_cache=False).logits[0, -1].float(), -1)
            rs.ks = {l: K[l][0, p] for l in range(nL)}
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
            if a.windows:
                cw = torch.zeros(T, dtype=torch.bool)
                cw[groups["choice_words"]] = True
                rs.mask = cw
                for w0 in range(0, nL, a.windows):
                    rs.layers = set(range(w0, min(nL, w0 + a.windows)))
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
        print(f"  {arm} done ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_{a.view}"
    json.dump(res, open(f"{a.out}/{tag}.json", "w"))
    lines = []
    for arm in a.arms.split(","):
        R = [r for r in res if r["arm"] == arm]
        full = np.array([r["m"]["all"] - r["m_B"] for r in R])
        lines.append(f"{arm}: n={len(R)}  full K_S effect {full.mean():+.2f} nats;  none-check {np.mean([r['m']['none'] - r['m_B'] for r in R]):+.3f}")
        extra = sorted(k for k in R[0]["m"] if k.startswith(("win", "kvgroup")))
        for g in ["self", "story_tail", "question", "choices", "choice_words", "rest_after_p"] + extra:
            d = np.array([r["m"][g] - r["m_B"] for r in R])
            rng = np.random.default_rng(0)
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            size = np.mean([r['sizes'][g] for r in R]) if g in R[0]["sizes"] else float("nan")
            lines.append(f"   rows={g:13s} (avg {size:5.1f} tok)  d={d.mean():+6.2f} "
                         f"[{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]  frac of full {d.mean() / full.mean():+.2f}")
    s = "\n".join(lines)
    open(f"{a.out}/{tag}_summary.txt", "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
