"""Pilot (0.5B, CPU FP32): do the belief-task readers H* (pilot ranking on belief P1) carry the NEGATIVE key read of IOI
INLINE at the listed names? HeadSplice rows with K_S or K_X per batch row; identity contrast c = lp(IO_S) - lp(IO_X)."""
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flagkit  # noqa: E402,F401  (sys.path to the repo)
from ckeys.headsplice import HeadSplice, cells_dense, head_masks  # noqa: E402
from ckeys.interventions import blocks, capture  # noqa: E402
from ckeys import ioi  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "4")))
MODEL, N, ARM = "Qwen/Qwen2.5-0.5B-Instruct", int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "INLINE"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
nL, H = model.config.num_hidden_layers, model.config.num_attention_heads
hs = HeadSplice(model)
Hs = [tuple(c) for c in json.load(open("pilot05.json"))["Hs"]]
rng = np.random.default_rng(2)
allc = [(l, h) for l in range(nL) for h in range(H)]
RAND = [[allc[i] for i in rng.permutation(len(allc))[:len(Hs)]] for _ in range(2)]


def keys(ids, p):
    with capture(model, range(nL), "k") as K:
        model(ids, use_cache=False, logits_to_keep=1)
    return {l: K[l][0, p].clone() for l in range(nL)}


cores = ioi.make_cores(N, random.Random(5))
rows_out = []
for core in cores:
    ids = ioi.encode_runs(tok, core, ARM, chat=False)
    if ids is None:
        continue
    ib = ids["B"]
    p = (ib[0] != ids["S"][0]).nonzero().flatten().tolist()[0]
    tid = ioi.name_ids(tok, core)
    text = ioi.raw_prompt(ARM, core, core["io_b"], False)
    par = ioi.parenthetical(core) if ARM.startswith("INLINE") else ioi.listing(core)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc.input_ids == ib[0].tolist()
    c0 = text.index(par)
    G = [i for i, (s, e) in enumerate(enc.offset_mapping) if e > c0 and s < c0 + len(par) and ib[0, i].item() in tid.values()]
    assert len(G) == 4 and all(g > p for g in G), (G, p)
    ks, kx = keys(ids["S"], p), keys(ids["X"], p)
    sets = [("none", []), ("H", Hs), ("r0", RAND[0]), ("r1", RAND[1]), ("allG", allc), ("koH", sorted(set(allc) - set(Hs)))]
    names = [(n, s, k) for k in ("S", "X") for n, s in sets] + [("allT", None, "S"), ("allT", None, "X")]
    B, T = len(names), ib.shape[1]
    dense = cells_dense([s if s is not None else [] for _, s, _ in names], nL, H)
    masks = head_masks(dense, G, T)
    for b, (n, s, k) in enumerate(names):
        if n == "allT":
            for l in range(nL):
                masks.setdefault(l, torch.zeros(B, T, H, dtype=torch.bool))[b] = True
    hs.ks = {l: torch.stack([(ks if k == "S" else kx)[l] for _, _, k in names]) for l in range(nL)}
    hs.pos, hs.mode, hs.masks, hs.active = p, "splice", masks, True
    try:
        with torch.no_grad():
            lp = torch.log_softmax(model(ib.expand(B, -1), use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1)
    finally:
        hs.active, hs.masks = False, None
    c = (lp[:, tid["S"]] - lp[:, tid["X"]]).numpy()
    cc = {(n, k): c[i] for i, (n, _, k) in enumerate(names)}
    idk = {n: 0.5 * ((cc[(n, "S")] - cc[("none", "S")]) - (cc[(n, "X")] - cc[("none", "X")])) for n, _ in sets + [("allT", None)]}
    two = float(lp[0, tid["B"]] - lp[0, tid["Subj"]])
    rows_out.append(dict(idk=idk, LD=two))
print(f"[{time.time() - t0:.0f}s] {ARM}: n={len(rows_out)}; clean LD(IO_B - S) mean {np.mean([r['LD'] for r in rows_out]):+.2f}, frac>0 {np.mean([r['LD'] > 0 for r in rows_out]):.2f}")
m = {n: np.array([r["idk"][n] for r in rows_out]) for n in rows_out[0]["idk"]}
for n in m:
    print(f"   ID_K[{n:5s}] {m[n].mean():+.3f} (se {m[n].std() / np.sqrt(len(m[n])):.3f})")
dG = m["allG"].mean()
print(f"   R(H*) = {m['H'].mean() / dG:+.2f}; KO(H*) = {1 - m['koH'].mean() / dG:+.2f}; R(rand) = {m['r0'].mean() / dG:+.2f}, {m['r1'].mean() / dG:+.2f}; d_G/d_full = {dG / m['allT'].mean():+.2f}")
