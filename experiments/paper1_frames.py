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

Stage 4 (P-2026-10-05-F): --refit FAM=RUN_DIR adds the bases of a run of experiments/refit_remap.py as family "FAM/"
(runs "FAM/m3_101", ..., exchanges "FAM/addition_101", ...), each m3 paired with its own run's pca of the same seed;
every family is patched and exchanged in its own batches, so the released (unprefixed) family is computed exactly as
before. --prefill sets the assistant prefill (default "Answer:").

Outputs per item: candidate log-probs, global argmax token id, for every run. Paper 1 reproduction is checked
afterwards against data/mechanism/native_*.jsonl.gz (analysis/stage2_score.py).
"""
from __future__ import annotations

import argparse
import hashlib
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


def encode_with_span(tok, arm, core, location, prefill="Answer:"):
    rec = record(core, "direct", location)
    raw = raw_prompt(arm, rec["story"], rec["query"])
    text = chat_text(tok, raw, prefill=prefill)
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_refit(run, fam, width):
    """Bases of one refit_remap.py run as keys (f"{fam}/{obj}", seed), plus their provenance; the run's FRAME.json
    must name the frame fam (none -> NONE, p1 -> P1), be COMPLETE and match the run's COMPLETE.json, and every basis
    file must have the sha256 that COMPLETE.json's inventory and FRAME.json's validation record."""
    assert fam.isalnum() and fam != "v", fam
    run, out, prov = Path(run), {}, {}
    fr = run.parent / "FRAME.json"
    info = json.loads(fr.read_text())
    meta = {"run": str(run), "complete_sha256": sha(run / "COMPLETE.json"), "frame_sha256": sha(fr),
            "frame": info.get("frame")}
    assert str(meta["frame"]).lower() == fam, f"family {fam!r} given a run fit under frame {meta['frame']!r}"
    assert info.get("status") == "COMPLETE" and info.get("complete_sha256") == meta["complete_sha256"], \
        (fr, info.get("status"))
    inventory = json.loads((run / "COMPLETE.json").read_text())["artifacts"]
    for obj in OBJECTIVES:
        for s in SEEDS:
            f = run / "bases" / f"{obj}_ts{s}.npz"
            h, recorded = sha(f), info["validation"]["bases"][f"{obj}_ts{s}"]["sha256"]
            assert h == inventory[f"bases/{f.name}"]["sha256"] == recorded, f"{f} is not the basis this run wrote"
            with np.load(f, allow_pickle=False) as z:
                assert z.files == ["rank_16"], z.files
                U = torch.tensor(z["rank_16"], dtype=torch.float32)
            assert U.shape == (16, width), (f, U.shape)
            assert torch.allclose(U @ U.T, torch.eye(U.shape[0]), atol=1e-4)
            out[(f"{fam}/{obj}", s)] = U
            prov[f"{fam}/{obj}_{s}"] = {"path": str(f), "sha256": h}
    meta["bases_match_inventory"] = True
    return out, prov, meta


def pre_tokenizer(tok):
    p = tok.backend_tokenizer.pre_tokenizer
    return b"null" if p is None else p.__getstate__()  # the pre-tokenizer's JSON


def fixed_tokenizer(repo, rev=None, strict=True, **kw):
    """AutoTokenizer with fix_mistral_regex=True. For a Hub repo id transformers applies the fix only if a Hub lookup
    succeeds (a failure is swallowed), so with strict the fixed pre-tokenizer must differ from the unfixed one; on a
    miss both are reloaded from the snapshot directory of the pinned commit, where detection reads its config.json
    (5.9.0 resolves a repo id's config.json at 'main', absent from a commit-only cache); the chat template must be the
    repo id's. strict=False (test models) loads only the fixed tokenizer and checks nothing. Returns (tokenizer, record)."""
    kw = dict(kw, use_fast=True, trust_remote_code=False, **(dict(revision=rev) if rev else {}))
    if not strict:
        return AutoTokenizer.from_pretrained(repo, **kw, fix_mistral_regex=True), {"strict": False}
    base = AutoTokenizer.from_pretrained(repo, **kw)
    plain, tok, src = pre_tokenizer(base), AutoTokenizer.from_pretrained(repo, **kw, fix_mistral_regex=True), None
    if pre_tokenizer(tok) == plain and not Path(repo).is_dir():
        from huggingface_hub import snapshot_download
        src = snapshot_download(repo, revision=rev, allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json"],
                                **{k: kw[k] for k in ("cache_dir", "token", "local_files_only") if k in kw})
        lkw = {k: v for k, v in kw.items() if k != "revision"}
        plain = pre_tokenizer(AutoTokenizer.from_pretrained(src, **lkw))
        tok = AutoTokenizer.from_pretrained(src, **lkw, fix_mistral_regex=True)
        assert tok.chat_template == base.chat_template, f"chat template of {src} differs from {repo}'s"
    fixed = pre_tokenizer(tok)
    assert fixed != plain, f"fix_mistral_regex was not applied to {repo} (pre-tokenizer unchanged)"
    return tok, {"strict": True, "fix_applied": True, "local_retry": src,
                 "pre_tokenizer_sha256": {"fixed": hashlib.sha256(fixed).hexdigest(),
                                          "unfixed": hashlib.sha256(plain).hexdigest()}}


def tokenizer_check(repo, rev, tok, arms, cores, prefills, strict=True):
    """Ids with and without fix_mistral_regex must be identical on every prompt evaluated (abort otherwise); with
    strict the fix must actually be installed (fixed_tokenizer)."""
    tok_fix, fix = fixed_tokenizer(repo, rev, strict)
    n = 0
    for arm in arms:
        for core in cores:
            for loc in (core["base"], core["source"], core["target"]):
                rec = record(core, "direct", loc)
                raw = raw_prompt(arm, rec["story"], rec["query"])
                for pf in prefills:
                    text = chat_text(tok, raw, prefill=pf)
                    a = tok(text, add_special_tokens=False).input_ids
                    b = tok_fix(text, add_special_tokens=False).input_ids
                    assert a == b, f"fix_mistral_regex changes the ids ({arm}, core {core['id']}, {loc}, prefill {pf!r})"
                    n += 1
    return {"prompts": n, "identical": True, "prefills": list(prefills), "fix_mistral_regex": fix}


def lp_rows(model, ids, cid):
    out = model(ids, use_cache=False, logits_to_keep=1)
    lp = torch.log_softmax(out.logits[:, -1].float(), -1)
    return lp[:, cid].cpu(), lp.argmax(-1).cpu()


@torch.no_grad()
def run_family(model, ib, cid, nL, h, span, vpos, bases, keys, fam, res):
    # materialise patches exactly as Paper 1 (BF16 basis, base + (delta @ U^T) @ U)
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
        iM, iP = keys.index((fam + "m3", s)), keys.index((fam + "pca", s))
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
            res["runs"][f"{fam}addition{tag}_{s}"] = {"cand": c[2 * j].tolist(), "argmax": int(g[2 * j])}
            res["runs"][f"{fam}removal{tag}_{s}"] = {"cand": c[2 * j + 1].tolist(), "argmax": int(g[2 * j + 1])}


@torch.no_grad()
def run_core(model, tok, core, arm, bases, dev, prefill="Answer:"):
    ids = {}
    for name, loc in (("B", core["base"]), ("S", core["source"]), ("T", core["target"])):
        ids[name], span, vpos = encode_with_span(tok, arm, core, loc, prefill)
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
    fams = {}
    for k in bases:
        fams.setdefault(k[0][: k[0].rfind("/") + 1], []).append(k)
    for fam, keys in fams.items():  # one family per batch: the released family is computed exactly as before
        run_family(model, ib, cid, nL, h, span, vpos, bases, keys, fam, res)
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
    ap.add_argument("--refit", action="append", default=[], metavar="FAM=DIR",
                    help="stage 4: bases of a refit_remap.py run directory (DIR/bases/{obj}_ts{seed}.npz) as family FAM/")
    ap.add_argument("--prefill", default="Answer:", help="assistant prefill (stage 4 exploratory: '')")
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
    tok_check = None
    if a.model == "mistral":
        tok_check = tokenizer_check(repo, rev, tok, a.arms.split(","), cores, sorted({a.prefill, "Answer:", ""}),
                                    strict=not a.model_override)
        print("tokenizer check (fix_mistral_regex):", json.dumps(tok_check), flush=True)
    if a.bases_override:
        g = torch.Generator().manual_seed(0)
        d = model.config.hidden_size
        bases = {(o, s): torch.linalg.qr(torch.randn(d, 16, generator=g))[0].T.contiguous() for o in OBJECTIVES for s in SEEDS}
    else:
        bases = load_bases(a.p1_root, a.model)
    basis_prov = {}
    for o in OBJECTIVES:
        for s in SEEDS:
            f = Path(a.p1_root) / "gpu/component_data/bases" / f"{a.model}_original_1000_{o}_{s}.npz"
            basis_prov[f"{o}_{s}"] = {"path": "random (--bases-override)"} if a.bases_override else {"path": str(f), "sha256": sha(f)}
    refits = {}
    for spec in a.refit:
        fam, run = spec.split("=", 1)
        assert fam not in refits, fam
        new, prov, refits[fam] = load_refit(run, fam, model.config.hidden_size)
        bases.update(new)
        basis_prov.update(prov)
    for (o, s), U in bases.items():
        basis_prov[f"{o}_{s}"]["tensor_sha256"] = hashlib.sha256(U.numpy().tobytes()).hexdigest()
    res, t0 = [], time.time()
    for arm in a.arms.split(","):
        for i, core in enumerate(cores):
            r = run_core(model, tok, core, arm, bases, first, a.prefill)
            if r is not None:
                r["core_index"] = i
                res.append(r)
        print(f"  {arm} done: {sum(r['arm'] == arm for r in res)} items ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    prov = {"args": vars(a), "repo": repo, "revision": rev, "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "n_gpus": torch.cuda.device_count(), "alphabet": {arm: list(alphabet(arm)) for arm in a.arms.split(",")},
            "prefill": a.prefill, "tokenizer_check": tok_check, "bases": basis_prov, "refits": refits}
    tag = a.model if not a.model_override else "TEST_" + repo.split("/")[-1]
    json.dump({"provenance": prov, "results": res}, open(f"{a.out}/{tag}.json", "w"))
    print(json.dumps(prov))


if __name__ == "__main__":
    main()
