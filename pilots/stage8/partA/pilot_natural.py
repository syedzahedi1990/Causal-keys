"""Part A CPU pilot (disclosed): span K/V clamps on counterfactual SQuAD items, FP32, Qwen2.5-0.5B/1.5B-Instruct.

Per item and format: B/S/X prompts (chat + 'Answer:' prefill), entity span P, K/V captured at P in all layers;
rows ID (KV_B), K_S, V_S, KV_S, K_X, V_X, KV_X from layer 0; each scored teacher-forced on the continuation of
' ' + e_Y for Y in {B, S, X, D}: full-string log-prob and decision-token (first content token) log-prob;
greedy generation (12 tokens) for rows ID, KV_S, KV_X, K_S, V_S. Exactness: KV_S row == unclamped S run.
"""
from __future__ import annotations

import json
import re
import string
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/user/Causal-keys")
from ckeys.clamp import capture_kv, clamp_kv, stack_rows  # noqa: E402
from ckeys.encoding import chat_text  # noqa: E402
from ckeys.interventions import blocks  # noqa: E402

PREFIX = "Read the passage and answer the question.\n\n"
FREE = "Answer with the exact words from the passage."
MC = "Answer with exactly one of the options."


def and_list(xs):
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def parts(fmt, it):
    """(pre, post): raw prompt = pre + passage + post."""
    q, opts = it["question"].strip(), it["options"]
    ol = "Options: " + "; ".join(opts)
    if fmt == "NOM":
        return PREFIX + "Passage: ", "\nQuestion: " + q + "\n" + FREE
    if fmt == "OPTA":
        return PREFIX + "Passage: ", "\nQuestion: " + q + "\n" + ol + "\n" + MC
    if fmt == "OPTB":
        return PREFIX + ol + "\n\nPassage: ", "\nQuestion: " + q + "\n" + MC
    if fmt == "MENA":
        return PREFIX + "Passage: ", " Related articles mention " + and_list(opts) + ".\nQuestion: " + q + "\n" + FREE
    if fmt == "MENB":
        return PREFIX + "Passage: Related articles mention " + and_list(opts) + ". ", "\nQuestion: " + q + "\n" + FREE
    if fmt == "LETA":
        lo = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", opts))
        return PREFIX + "Passage: ", "\nQuestion: " + q + "\nOptions:\n" + lo + "\nAnswer with the letter of the correct option."
    raise ValueError(fmt)


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def matches(gen, ent):
    g, e = norm(gen.split("\n")[0]), norm(ent)
    return g == e or g.startswith(e + " ")


def encode(tok, it, fmt, ent):
    pre, post = parts(fmt, it)
    ctx = it["context"][:it["start"]] + ent + it["context"][it["start"] + len(it["answer"]):]
    raw = pre + ctx + post
    text = chat_text(tok, raw)
    off0 = text.index(raw) + len(pre) + it["start"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, enc.input_ids, enc.offset_mapping, (off0, off0 + len(ent))


def cont_ids(tok, text, ids, ent):
    full = tok(text + " " + ent, add_special_tokens=False).input_ids
    assert full[:len(ids)] == ids, "prefix not stable"
    c = full[len(ids):]
    k = next(i for i, t in enumerate(c) if tok.decode([t]).strip())
    return c, k


ROWS = [("B", "B"), ("S", "B"), ("B", "S"), ("S", "S"), ("X", "B"), ("B", "X"), ("X", "X")]
NAMES = ["ID", "K_S", "V_S", "KV_S", "K_X", "V_X", "KV_X"]
GEN_ROWS = ["ID", "KV_S", "KV_X", "K_S", "V_S", "K_X", "V_X"]


@torch.no_grad()
def run_item(model, tok, it, fmt, letters=False):
    nL = len(blocks(model))
    enc = {}
    for Y, ent in (("B", it["answer"]), ("S", it["S"]), ("X", it["X"])):
        enc[Y] = encode(tok, it, fmt, ent)
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
    out = {"P": P, "T": len(ib), "score": {}, "gen": {}}
    if letters:
        targets = {Y: "ABCD"[it["options"].index(e)] for Y, e in (("B", it["answer"]), ("S", it["S"]), ("X", it["X"]), ("D", it["D"]))}
    else:
        targets = {"B": it["answer"], "S": it["S"], "X": it["X"], "D": it["D"]}
    for Y, ent in targets.items():
        c, k = cont_ids(tok, tb, ib, ent)
        x = torch.tensor([ib + c]).expand(len(ROWS), -1)
        with clamp_kv(model, P, tabs, range(nL)):
            lg = model(x, use_cache=False).logits[:, len(ib) - 1:-1].float().log_softmax(-1)
        tl = lg[:, torch.arange(len(c)), torch.tensor(c)]  # [rows, len(c)]
        out["score"][Y] = {"full": tl.sum(1).tolist(), "dec": tl[:, k].tolist(), "k": k, "n": len(c)}
        if Y == "B":  # candidate mass at the decision position, ID row
            pre = c[:k]
            decs = [cont_ids(tok, tb, ib, e)[0][cont_ids(tok, tb, ib, e)[1]] for e in targets.values()]
            out["mass"] = float(lg[0, k].exp()[decs].sum())
    # greedy generation for selected rows (KV cache, clamp acts in prefill only)
    gi = [NAMES.index(n) for n in GEN_ROWS]
    sub = {key: v[gi] for key, v in tabs.items()}
    x = torch.tensor([ib]).expand(len(gi), -1)
    with clamp_kv(model, P, sub, range(nL)):
        g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=12, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    for n, row in zip(GEN_ROWS, g[:, len(ib):]):
        out["gen"][n] = tok.decode(row, skip_special_tokens=True)
    # exactness: KV_S row vs unclamped S run, decision token of S
    c, k = cont_ids(tok, enc["S"][0], enc["S"][1], targets["S"])
    lgS = model(torch.tensor([enc["S"][1] + c]), use_cache=False).logits[0, len(ib) - 1 + k].float().log_softmax(-1)
    out["exact"] = abs(float(lgS[c[k]]) - out["score"]["S"]["dec"][NAMES.index("KV_S")])
    return out


def main():
    model_name, n, fmts, path, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4], sys.argv[5]
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    items = [i for i in json.load(open(path)) if i["stratum"] == "FT"]
    import random
    random.Random(7).shuffle(items)
    items = [i for i in items if len(i["context"]) < 900][:n]
    res, t0 = [], time.time()
    for it in items:
        for f in fmts:
            r = run_item(model, tok, it, f, letters=(f == "LETA"))
            if r is None:
                print("skip", it["id"], f)
                continue
            r.update(id=it["id"], fmt=f, sub=it["sub"], art=it["art"], ans=it["answer"], S=it["S"], X=it["X"], D=it["D"],
                     options=it["options"])
            res.append(r)
        print(f"{len(res)} rows {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(outp, "w"))


if __name__ == "__main__":
    main()
