"""Critic pilot (disclosed; CPU, FP32): parametric-memory pull on the Part A items.

Closed-book MCQ: the OPT-A prompt with the passage removed (question + options + MC instruction), chat-wrapped with
the 'Answer:' prefill. Scores the decision token (first content token of ' ' + e) of the four options in one forward
pass per item and records which option wins. If B wins far above 1/4, the model knows e_B without the passage, so
'MCQ answers survive value corruption' (A7) and accuracy-type outcomes can be carried by memory, not by the cache.
Also reports the same under the NOM free-form wording without passage (first-token argmax among the four).
Usage: closed_book.py MODEL N OUT
"""
from __future__ import annotations

import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/user/Causal-keys")
from ckeys.encoding import chat_text  # noqa: E402

A = "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partA/"
PRE = "Answer the question.\n\n"
MC = "Answer with exactly one of the options."
FREE = "Answer with a short phrase."


def dec_tok(tok, text, ent):
    ids = tok(text, add_special_tokens=False).input_ids
    full = tok(text + " " + ent, add_special_tokens=False).input_ids
    assert full[:len(ids)] == ids
    c = full[len(ids):]
    k = next(i for i, t in enumerate(c) if tok.decode([t]).strip())
    return ids, c[:k], c[k]


@torch.no_grad()
def main():
    mname, n, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.float32).eval()
    items = json.load(open(A + "items_ft5.json"))
    sp = json.load(open(A + "split_v5.json"))
    E = set(sp["E"])
    items = [i for i in items if i["id"] in E][:n]
    res, t0 = [], time.time()
    for it in items:
        row = {"id": it["id"], "sub": it["sub"], "D_in": it["D_in"], "art": it["art"]}
        for fmt in ("MC", "FREE"):
            if fmt == "MC":
                raw = PRE + "Question: " + it["question"].strip() + "\nOptions: " + "; ".join(it["options"]) + "\n" + MC
            else:
                raw = PRE + "Question: " + it["question"].strip() + "\n" + FREE
            text = chat_text(tok, raw)
            cands = {"B": it["answer"], "S": it["S"], "X": it["X"], "D": it["D"]}
            dec, w = {}, None
            for Y, e in cands.items():
                ids, ww, d = dec_tok(tok, text, e)
                w = ww if w is None else w
                if ww != w:
                    dec = None
                    break
                dec[Y] = d
            if dec is None or len(set(dec.values())) < 4:
                row[fmt] = None
                continue
            lp = model(torch.tensor([ids + w]), use_cache=False).logits[0, -1].float().log_softmax(-1)
            sc = {Y: float(lp[d]) for Y, d in dec.items()}
            row[fmt] = {"lp": sc, "win": max(sc, key=sc.get), "mass": float(sum(lp[d].exp() for d in dec.values()))}
        res.append(row)
        if len(res) % 20 == 0:
            print(len(res), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(outp, "w"))
    for fmt in ("MC", "FREE"):
        R = [r for r in res if r.get(fmt)]
        import collections
        c = collections.Counter(r[fmt]["win"] for r in R)
        print(fmt, "n", len(R), "win counts", dict(c), "P(B wins)", round(c["B"] / len(R), 3))
        for sub in ("PERSON", "PLACE", "NUMBER"):
            Rs = [r for r in R if r["sub"] == sub]
            if Rs:
                print("   ", sub, len(Rs), round(sum(r[fmt]["win"] == "B" for r in Rs) / len(Rs), 3))
        Rd = [r for r in R if r["D_in"]]
        print("    D_in", len(Rd), "P(B)", round(sum(r[fmt]["win"] == "B" for r in Rd) / max(1, len(Rd)), 3),
              "P(D)", round(sum(r[fmt]["win"] == "D" for r in Rd) / max(1, len(Rd)), 3))


if __name__ == "__main__":
    main()
