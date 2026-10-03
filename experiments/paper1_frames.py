"""Stage 2: Paper 1's released DAS/PCA bases evaluated across answer formats (transfer-law test, C6).

For every native story core (Paper 1 gpu/component_data/native_story_120.json) and format arm:
  * natural runs B, S, T (critical location = base, source, target = pair-swap(source));
  * patched runs: block-4 (1-based; 0-based index 3) output over the complete event span is replaced by
    h_B + (h_S - h_B) U^T U with U one of Paper 1's saved rank-16 bases (learned remap m3, PCA, intended f_star;
    seeds 101-103), computed in BF16 exactly as Paper 1's engine (mistral_engine.fixed);
  * key-only exchange, as in Paper 1's native study: the critical-token key (k_proj output, pre-RoPE) of
    1-based blocks 6..L (0-based 5..L-1) is replaced by the other patched run's key, with queries and values left
    to evolve: P + K_M ("addition") and M + K_P ("removal").
Readout: full-vocabulary log-probs at the answer position, scored on the six locations (letters for LETTER).

Outputs per item: candidate log-probs, global argmax token id, for every run. Paper 1 reproduction is checked
afterwards against data/mechanism/native_*.jsonl.gz (analysis/stage2_score.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.encoding import alphabet, candidate_ids, chat_text, raw_prompt
from ckeys.interventions import blocks, capture, hooks
from ckeys.story import LOCATIONS, PAIR_SWAP, record

PROFILES = {
    "qwen": ("Qwen/Qwen2.5-72B-Instruct", "495f39366efef23836d0cfae4fbe635880d2be31"),
    "mistral": ("mistralai/Mistral-Small-24B-Instruct-2501", "9527884be6e5616bdd54de542f9ae13384489724"),
}
FIT_LAYER0 = 3          # Paper 1 "block 4" (1-based) decoder output
FIRST_EXCHANGE0 = 5     # Paper 1 exchange from 1-based block 6
OBJECTIVES = ("m3", "pca", "f_star")
SEEDS = (101, 102, 103)


def encode_with_span(tok, arm, core, location):
    rec = record(core, "direct", location)
    raw = raw_prompt(arm, rec["story"], rec["query"])
    text = chat_text(tok, raw)
    off = text.index(raw)
    story_at = raw.index(rec["story"])
    initial_len = rec["story"].index(f"{core['agent']} watches as")
    ev0 = off + story_at + initial_len
    ev1 = off + story_at + len(rec["story"])
    val0 = ev0 + rec["story"][initial_len:].index("to the " + location) + len("to the ")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    offs = enc.offset_mapping[0].tolist()
    span = [i for i, (a, b) in enumerate(offs) if b > ev0 and a < ev1]
    val = [i for i, (a, b) in enumerate(offs) if b > val0 and a < val0 + len(location)]
    assert span == list(range(span[0], span[-1] + 1)) and len(val) == 1 and val[0] in span, (span, val)
    return enc.input_ids, span, val[0]


def load_bases(p1_root, model_key, cohort="original_1000"):
    out = {}
    for obj in OBJECTIVES:
        for s in SEEDS:
            z = np.load(Path(p1_root) / "gpu/component_data/bases" / f"{model_key}_{cohort}_{obj}_{s}.npz")
            U = torch.tensor(z["rank_16"], dtype=torch.float32)
            assert torch.allclose(U @ U.T, torch.eye(U.shape[0]), atol=1e-4)
            out[(obj, s)] = U
    return out


def lp_rows(model, ids, cid):
    out = model(ids, use_cache=False, logits_to_keep=1)
    lp = torch.log_softmax(out.logits[:, -1].float(), -1)
    return lp[:, cid].cpu(), lp.argmax(-1).cpu()


@torch.no_grad()
def run_core(model, tok, core, arm, bases, dev):
    ids = {}
    for name, loc in (("B", core["base"]), ("S", core["source"]), ("T", core["target"])):
        ids[name], span, vpos = encode_with_span(tok, arm, core, loc)
    if len({tuple(v.shape) for v in ids.values()}) > 1:
        return None
    ib = ids["B"].to(dev)
    cid = candidate_ids(tok, arm)
    nL = len(blocks(model))
    res = {"core": core, "arm": arm, "span": [span[0], span[-1] + 1], "vpos": vpos, "runs": {}}
    # natural runs + event-span residuals at the fit layer
    h = {}
    for name in ("B", "S", "T"):
        with capture(model, [FIT_LAYER0], "resid") as st:
            c, g = lp_rows(model, ids[name].to(dev), cid)
        h[name] = st[FIT_LAYER0][0, span[0]:span[-1] + 1]
        res["runs"][name] = {"cand": c[0].tolist(), "argmax": int(g[0])}
    # materialise patches exactly as Paper 1 (BF16 basis, base + (delta @ U^T) @ U)
    keys = list(bases)
    hb, hs = h["B"], h["S"]
    patches = []
    for k in keys:
        U = bases[k].to(hb.device, dtype=hb.dtype)
        patches.append(hb + ((hs - hb) @ U.T) @ U)
    patches = torch.stack(patches)  # [n_bases, span, d]
    lo, hi = span[0], span[-1] + 1

    def patch_hook(_m, _i, out, P=patches):
        o = out[0] if isinstance(out, tuple) else out
        o = o.clone()
        o[:, lo:hi] = P.to(o.device, o.dtype)
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o

    hdl = blocks(model)[FIT_LAYER0].register_forward_hook(patch_hook)
    try:
        with capture(model, range(FIRST_EXCHANGE0, nL), "k") as K, capture(model, range(FIRST_EXCHANGE0, nL), "v") as V:
            c, g = lp_rows(model, ib.expand(len(keys), -1), cid)
        kcrit = {l: K[l][:, vpos].clone() for l in range(FIRST_EXCHANGE0, nL)}  # [n_bases, kv_dim]
        vcrit = {l: V[l][:, vpos].clone() for l in range(FIRST_EXCHANGE0, nL)}
    finally:
        hdl.remove()
    for i, k in enumerate(keys):
        res["runs"][f"{k[0]}_{k[1]}"] = {"cand": c[i].tolist(), "argmax": int(g[i])}
    # key-only exchange between m3 (M) and pca (P) of the same seed; Q/V evolve (Paper 1 native design)
    rows, swaps = [], []
    for s in SEEDS:
        iM, iP = keys.index(("m3", s)), keys.index(("pca", s))
        rows += [iP, iM]
        swaps += [iM, iP]   # P gets K_M (addition); M gets K_P (removal)
    P2 = patches[rows]

    def patch_hook2(_m, _i, out, P=P2):
        o = out[0] if isinstance(out, tuple) else out
        o = o.clone()
        o[:, lo:hi] = P.to(o.device, o.dtype)
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o

    # channel-only exchanges: keys (Paper 1's native design) and, symmetrically, values; the other channel,
    # queries and residuals evolve endogenously
    for chan, crit, tag in (("k_proj", kcrit, ""), ("v_proj", vcrit, "_v")):
        hs_ = [blocks(model)[FIT_LAYER0].register_forward_hook(patch_hook2)]
        for l in range(FIRST_EXCHANGE0, nL):
            tab = crit[l][swaps]

            def xhook(_m, _i, out, t=tab):
                out = out.clone()
                out[:, vpos] = t.to(out.device, out.dtype)
                return out
            hs_.append(getattr(blocks(model)[l].self_attn, chan).register_forward_hook(xhook))
        with hooks(hs_):
            c, g = lp_rows(model, ib.expand(len(rows), -1), cid)
        for j, s in enumerate(SEEDS):
            res["runs"][f"addition{tag}_{s}"] = {"cand": c[2 * j].tolist(), "argmax": int(g[2 * j])}
            res["runs"][f"removal{tag}_{s}"] = {"cand": c[2 * j + 1].tolist(), "argmax": int(g[2 * j + 1])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PROFILES), required=True)
    ap.add_argument("--p1-root", required=True, help="unpacked Paper 1 reviewer repository")
    ap.add_argument("--arms", default="P1,NONE,BEFORE,POST,LETTER")
    ap.add_argument("--n", type=int, default=0, help="limit cores (0 = all 120)")
    ap.add_argument("--model-override", default=None, help="testing only: a small HF model id")
    ap.add_argument("--bases-override", default=None, help="testing only: random bases of this width")
    ap.add_argument("--out", default="results/paper1_frames")
    a = ap.parse_args()
    repo, rev = PROFILES[a.model]
    if a.model_override:
        repo, rev = a.model_override, None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    kw = dict(dtype=torch.bfloat16 if dev == "cuda" else torch.float32, revision=rev, attn_implementation="sdpa")
    if dev == "cuda":
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(repo, **kw).eval()
    if dev == "cpu":
        model = model.to(dev)
    first = next(model.parameters()).device
    cores = json.load(open(Path(a.p1_root) / "gpu/component_data/native_story_120.json"))["stories"]
    for c in cores:
        assert c["target"] == PAIR_SWAP[c["source"]]
    if a.n:
        cores = cores[: a.n]
    if a.bases_override:
        g = torch.Generator().manual_seed(0)
        d = model.config.hidden_size
        bases = {(o, s): torch.linalg.qr(torch.randn(d, 16, generator=g))[0].T.contiguous() for o in OBJECTIVES for s in SEEDS}
    else:
        bases = load_bases(a.p1_root, a.model)
    res, t0 = [], time.time()
    for arm in a.arms.split(","):
        for i, core in enumerate(cores):
            r = run_core(model, tok, core, arm, bases, first)
            if r is not None:
                r["core_index"] = i
                res.append(r)
        print(f"  {arm} done: {sum(r['arm'] == arm for r in res)} items ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    prov = {"args": vars(a), "repo": repo, "revision": rev, "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "n_gpus": torch.cuda.device_count(), "alphabet": {arm: list(alphabet(arm)) for arm in a.arms.split(",")}}
    tag = a.model if not a.model_override else "TEST_" + repo.split("/")[-1]
    json.dump({"provenance": prov, "results": res}, open(f"{a.out}/{tag}.json", "w"))
    print(json.dumps(prov))


if __name__ == "__main__":
    main()
