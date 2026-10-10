"""Part C CPU pilot (Qwen2.5-1.5B-Instruct, FP32): non-natural identity edits at the writing token p.

Stage A (--stage A): full residual edits at the output of block l at p, every row in one batch per (story, format):
  T_t      h_t,l(p)                        natural residual of the run with t written (tautological control)
  CI_t     h_B,l(p) + mu_l(t) - mu_l(B)    in-task mean difference over held-out stories (CAA-in)
  CO_t     h_B,l(p) + mu'_l(t) - mu'_l(B)  out-of-task lexical mean difference over neutral sentences (CAA-out)
  R_S      h_B,l(p) + random dir, |.| = |mu_l(S) - mu_l(B)|
  BIND     h_C,l(p), C = the story with the moved object replaced by the distractor (binding edit, target init)
  for t in {S, X}. Reports ID_KV(edit) (symmetric S/X identity contrast), flip rates, nu (residual and K/V distance
  from the natural edit), Phi_bind.
Stage B (--stage B): key-only / value-only clamps of the edited runs' K/V at p from block l+1 (exact for edits at p only):
  kappa_ID(edit) = mean ID_K / (mean ID_K + mean ID_V) against the natural s_ID(l+1), per format and depth.
"""
import argparse
import json
import math
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/user/Causal-keys")
from ckeys.clamp import capture_kv, clamp_kv  # noqa: E402
from ckeys.encoding import LETTERS, candidate_ids, encode, raw_prompt  # noqa: E402
from ckeys.interventions import _out_tensor, _with_tensor, blocks, hooks  # noqa: E402
from ckeys.story import LOCATIONS, make_cores, pick_x, record  # noqa: E402

NEUTRAL = [
    "The old {x} in the hallway was painted green last spring.",
    "She bought a new {x} at the market on Saturday.",
    "Every {x} in the shop had a small price tag.",
    "My uncle builds a wooden {x} for every new neighbour.",
    "A dusty {x} stood quietly in the corner of the attic.",
    "The catalogue lists one {x} under the heading of home goods.",
    "He wrote the word {x} on the whiteboard during the lesson.",
    "The museum displays a carved {x} from the eighteenth century.",
    "Our neighbour sells a sturdy {x} for a fair price.",
    "The designer sketched a modern {x} for the new apartment.",
    "Children like to draw a big {x} with bright colours.",
    "The hotel room had a large {x} near the window.",
]


def story_text(core, loc, bind=False):
    rec = record(core, "direct", loc)
    s = rec["story"]
    if bind:
        a, o, d = core["agent"], core["object"], core["distractor"]
        old = f"{a} watches as the {o} is moved"
        assert s.count(old) == 1
        s = s.replace(old, f"{a} watches as the {d} is moved")
    return s, rec["query"]


def ids_for(tok, core, arm, loc, bind=False):
    s, q = story_text(core, loc, bind)
    return encode(tok, raw_prompt(arm, s, q))


@torch.no_grad()
def resid_at(model, ids, pos, layers):
    """{l: [D]} block outputs at pos (prefix through pos is enough)."""
    store, hs = {}, []
    for l in layers:
        def fn(_m, _i, out, l=l):
            store[l] = _out_tensor(out)[0, pos].detach().clone()
        hs.append(blocks(model)[l].register_forward_hook(fn))
    with hooks(hs):
        model(ids[:, : pos + 1], use_cache=False)
    return store


@torch.no_grad()
def run_rows(model, ids, pos, edits, kv_layers=None):
    """edits: list of (layer or None, vector or None) per row; residual at pos replaced at block output l.
    Returns log-probs at the last position [R, V] and, if kv_layers, {(l,ch): [R, D]} K/V at pos."""
    R = len(edits)
    byl = {}
    for r, (l, v) in enumerate(edits):
        if l is not None:
            byl.setdefault(l, []).append((r, v))
    hs = []
    for l, lst in byl.items():
        def fn(_m, _i, out, lst=lst):
            h = _out_tensor(out).clone()
            for r, v in lst:
                h[r, pos] = v.to(h.dtype)
            return _with_tensor(out, h)
        hs.append(blocks(model)[l].register_forward_hook(fn))
    x = ids.expand(R, -1)
    with hooks(hs):
        if kv_layers is not None:
            with capture_kv(model, [pos], kv_layers) as kv:
                lg = model(x, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            kv = {k: v[:, 0].clone() for k, v in kv.items()}
        else:
            lg = model(x, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            kv = None
    return torch.log_softmax(lg, -1), kv


def cand_tables(tok, arm):
    lower = candidate_ids(tok, arm)
    if arm == "LETTER":
        return {"lower": lower, "marg": [[i] for i in lower]}
    marg = []
    for c in LOCATIONS:
        forms = []
        for f in (" " + c, " " + c.capitalize(), c, c.capitalize()):
            t = tok.encode(f, add_special_tokens=False)
            if len(t) == 1:
                forms.append(t[0])
        marg.append(sorted(set(forms)))
    return {"lower": lower, "marg": marg}


def score(lp, tabs, scorer):
    """[R, 6] candidate log-scores."""
    if scorer == "lower":
        return lp[:, tabs["lower"]]
    return torch.stack([torch.logsumexp(lp[:, f], -1) for f in tabs["marg"]], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--nh", type=int, default=20)
    ap.add_argument("--depths", default="1,5,9,13")
    ap.add_argument("--arms", default="NONE,LETTER")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", "4")))
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    nL = len(blocks(model))
    depths = [int(x) for x in a.depths.split(",")]
    t0 = time.time()
    import os
    mpath = a.out + f".means_nh{a.nh}_d{a.depths.replace(',', '-')}.pt"
    if os.path.exists(mpath):
        mu, muo = torch.load(mpath)
    else:
        # ---------------- held-out class means (CAA-in): prefix through p, NONE format (prefix is format-independent)
        held = make_cores(a.nh, random.Random(202))
        mu = {x: {l: torch.zeros(model.config.hidden_size) for l in depths} for x in LOCATIONS}
        for core in held:
            for x in LOCATIONS:
                ids = ids_for(tok, core, "NONE", x)
                ref = ids_for(tok, core, "NONE", LOCATIONS[0] if x != LOCATIONS[0] else LOCATIONS[1])
                p = (ids[0] != ref[0]).nonzero().flatten().tolist()
                assert len(p) == 1
                st = resid_at(model, ids, p[0], depths)
                for l in depths:
                    mu[x][l] += st[l] / len(held)
        # ---------------- out-of-task lexical means (CAA-out)
        muo = {x: {l: torch.zeros(model.config.hidden_size) for l in depths} for x in LOCATIONS}
        for tmpl in NEUTRAL:
            for x in LOCATIONS:
                ids = encode(tok, tmpl.format(x=x), prefill="")
                xid = tok.encode(" " + x, add_special_tokens=False)
                assert len(xid) == 1
                pos = [i for i, t in enumerate(ids[0].tolist()) if t == xid[0]]
                assert len(pos) == 1, (tmpl, x)
                st = resid_at(model, ids, pos[0], depths)
                for l in depths:
                    muo[x][l] += st[l] / len(NEUTRAL)
        torch.save((mu, muo), mpath)
    print(f"means done {time.time() - t0:.0f}s", flush=True)
    g = torch.Generator().manual_seed(7)
    cores = [c for c in make_cores(4 * a.n, random.Random(101))]
    out = []
    used = 0
    for core in cores:
        if used >= a.n:
            break
        X = pick_x(core)
        B, S = core["base"], core["source"]
        # binding counterfactual must align token by token
        idsB = ids_for(tok, core, "NONE", B)
        idsC = ids_for(tok, core, "NONE", B, bind=True)
        if idsB.shape != idsC.shape or int((idsB != idsC).sum()) != 1:
            continue
        used += 1
        item = {"core": core, "X": X, "arms": {}}
        for arm in a.arms.split(","):
            ids = {n: ids_for(tok, core, arm, loc) for n, loc in (("B", B), ("S", S), ("X", X))}
            idc = ids_for(tok, core, arm, B, bind=True)
            p = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()
            assert len(p) == 1 and (ids["B"][0] != ids["X"][0]).nonzero().flatten().tolist() == p
            p = p[0]
            hB = resid_at(model, ids["B"], p, depths)
            hS = resid_at(model, ids["S"], p, depths)
            hX = resid_at(model, ids["X"], p, depths)
            hC = resid_at(model, idc, p, depths)
            nat = {"S": hS, "X": hX}
            tabs = cand_tables(tok, arm)
            # natural K/V at p for nu_K (all layers)
            _, kvn = run_rows(model, ids["S"], p, [(None, None)], range(nL))
            _, kvx = run_rows(model, ids["X"], p, [(None, None)], range(nL))
            names, edits = ["self"], [(None, None)]
            for l in depths:
                for t in ("S", "X"):
                    tloc = S if t == "S" else X
                    edits += [(l, nat[t][l]), (l, hB[l] + mu[tloc][l] - mu[B][l]), (l, hB[l] + muo[tloc][l] - muo[B][l])]
                    names += [f"T_{t}@{l}", f"CI_{t}@{l}", f"CO_{t}@{l}"]
                r = torch.randn(hB[l].shape, generator=g)
                r = r / r.norm() * (mu[S][l] - mu[B][l]).norm()
                edits += [(l, hB[l] + r), (l, hC[l])]
                names += [f"R_S@{l}", f"BIND@{l}"]
            if a.stage == "A":
                lp, kv = run_rows(model, ids["B"], p, edits, range(nL))
                res = {}
                for sc in ("lower", "marg"):
                    c = score(lp, tabs, sc)
                    res[sc] = {n: c[i].tolist() for i, n in enumerate(names)}
                res["mass_lower"] = lp[0, tabs["lower"]].exp().sum().item()
                res["mass_marg"] = sum(torch.logsumexp(lp[0, f], -1).exp().item() for f in tabs["marg"])
                # nu: residual and K/V distances from the natural edit
                nu = {}
                for l in depths:
                    for t in ("S", "X"):
                        tloc = S if t == "S" else X
                        dn = nat[t][l] - hB[l]
                        for fam, v in (("CI", mu[tloc][l] - mu[B][l]), ("CO", muo[tloc][l] - muo[B][l])):
                            nu[f"{fam}_{t}@{l}_resid"] = ((v - dn).norm() / dn.norm()).item()
                            nu[f"{fam}_{t}@{l}_cos"] = torch.nn.functional.cosine_similarity(v, dn, 0).item()
                            nu[f"{fam}_{t}@{l}_normratio"] = (v.norm() / dn.norm()).item()
                        kvnat = kvn if t == "S" else kvx
                        for fam in ("CI", "CO", "T"):
                            i = names.index(f"{fam}_{t}@{l}")
                            num, den = 0.0, 0.0
                            rel = []
                            for ll in range(l + 1, nL):
                                for ch in "kv":
                                    e = kv[(ll, ch)][i]
                                    n_ = kvnat[(ll, ch)][0]
                                    b_ = kv[(ll, ch)][0]
                                    rel.append(((e - n_).norm() / (n_ - b_).norm()).item())
                            nu[f"{fam}_{t}@{l}_kv"] = float(np.mean(rel))
                res["nu"] = nu
                res["names"] = names
                item["arms"][arm] = res
            else:
                # stage B: capture edited K/V at p (from the A-style full run), then K-only / V-only clamps
                lp, kv = run_rows(model, ids["B"], p, edits, range(nL))
                rows, tabk = ["self"], []
                tables = {(ll, ch): [kv[(ll, ch)][0]] for ll in range(nL) for ch in "kv"}
                for i, n in enumerate(names[1:], start=1):
                    l = int(n.split("@")[1])
                    if n.startswith("R_"):
                        continue
                    for mode in ("K", "V"):
                        rows.append(f"{n}|{mode}")
                        for ll in range(nL):
                            for ch in "kv":
                                use = ll >= l + 1 and ((ch == "k") == (mode == "K"))
                                tables[(ll, ch)].append(kv[(ll, ch)][i] if use else kv[(ll, ch)][0])
                tab = {k: torch.stack(v)[:, None] for k, v in tables.items()}
                with torch.no_grad(), clamp_kv(model, [p], tab, range(nL)):
                    lp2 = torch.log_softmax(model(ids["B"].expand(len(rows), -1), use_cache=False,
                                                  logits_to_keep=1).logits[:, -1].float(), -1)
                res = {}
                for sc in ("lower", "marg"):
                    c = score(lp, tabs, sc)
                    c2 = score(lp2, tabs, sc)
                    res[sc] = {n: c[i].tolist() for i, n in enumerate(names)} | {n: c2[i].tolist() for i, n in enumerate(rows)}
                res["exact_check"] = float((lp2[0] - lp[0]).abs().max())
                item["arms"][arm] = res
            print(f"  core {used} {arm} {time.time() - t0:.0f}s", flush=True)
        out.append(item)
        json.dump({"args": vars(a), "items": out}, open(a.out, "w"))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
