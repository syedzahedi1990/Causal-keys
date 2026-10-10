"""Stage 8 part A, reader heads on natural text (P-2026-10-10-J, lines J-A-HA1 to J-A-HA3): the stage-6 head analysis
on the counterfactual SQuAD items, at Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (BF16, eager attention,
use_cache=False in every pass). The answer frame is the model's (OUT/frames/<tag>.json, from natural_factorial).

Definitions (OPTA unless stated): P = the answer entity's token span; G = the token rows of the four option strings in
the options line, G_Y those of option Y; the decision position is the last row of B prompt + w; m = lp(dec_S) - lp(dec_B)
there; K_S = the S run's keys at every position of P (span tables, ckeys.headsplice); Q+ = every row whose first
character follows the question's last character (instruction, chat tokens, prefill, frame) plus every teacher-forced
answer row.
  rank     the first 60 R items: a3(l, h) = 1/2[(A^{K_S} - A^{ID})[G_S -> P] + (A^{ID} - A^{K_S})[G_B -> P]] with
           A[G_Y -> P] = mean over r in G_Y of the sum over p in P of A_lh[r, p] and the K_S clamp from layer 0;
           N* = the top k* = ceil(0.05 x heads) by mean a3; T* = the first k* of arms.P1.rankings.a3 of the committed
           stage-6 file results/gpu_stage6/heads/<model>.json (sha256 pinned); three random sets = the first k* of three
           numpy default_rng(2) permutations of every head (stage 6); C* (exploratory) = the top k* by direct logit
           attribution at the decision row in NOM (head h's o_proj contribution through the final RMSNorm, linearised at
           the run's own scale, on W_U[dec_B] minus the mean of W_U over the other three options' decision tokens).
           mu_f(l, h) = the mean of head (l, h)'s o_proj input over the R items and the Q+ rows of format f (NOM, OPTA)
           in the clean B run on B prompt + c_B (OUT/heads/mu_<tag>.pt, sha256 recorded).
  eval     the first 80 E items: the clean pass and the full K_S clamp (d_full = mean[m(full K_S clamp) - m(clean)]);
           the stage-6 sufficiency and knockout batches (experiments/stage6_heads.Stage6.curves, HeadSplice "splice"
           with span tables) for N*, T* and the random sets over KS = (1, 2, 5, 10, 20, k*, 2k*), each set's batches
           with their own none, all_G and all_T rows and knockout none and all_G rows (the denominators of that set's R(k)
           and KO(k)), and the layer profile.
  ablate   every E item, OPTA and NOM: one batch whose rows are the conditions none, N*, T*, rand0, rand1, rand2, C*
           (HeadSplice "ablate" of the condition's heads at Q+ with mu_f), under the KV_S clamp on B prompt + c_S: the
           argmax chain over c_S, the decision argmax = dec_S, the chain over c_S after the decision token; and the same
           conditions on the clean B prompt + c_B: the four options' decision log-probs (the option margin) and the
           chain over c_B.
  explore  the first 80 E items: the base rows ID, K_S, V_S, KV_S, K_X, V_X, KV_X with none and with N* ablated at Q+ in
           one batch (ID_K, ID_V under N*; exploratory).
The R and E items are those valid in NOM and OPTA under the frame (ckeys.natural_rows.valid_all); provenance n_valid
records how many of each there are, so rank and eval take min(60, n_valid R) and min(80, n_valid E) items.
Output OUT/heads/<tag>.json (atomic; rewritten after each part). TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B FP32
on the CPU, windowed passages, 2 items per part, k* fraction 0.012 (k* = 5), KS = (1, 2, 5, 10), T* = the first k* cells
of the Qwen2.5-7B stage-6 ranking that exist in the small model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import clamp_kv
from ckeys.headsplice import HeadSplice, HopSplice, cells_dense, head_masks, mean_table
from ckeys.interventions import blocks
from ckeys.natural_formats import option_rows
from ckeys.natural_rows import BASE, capture, prep, tables, valid_all
from experiments.natural_factorial import (ITEMS, TINY, load_items, read_frame, sha_file, split, verified)
from experiments.format_factorial import provenance as ff_provenance
from experiments.ioi_factorial import device_name
from experiments.stage6_heads import Stage6, rank_of, write_atomic

ROOT = Path(__file__).resolve().parents[1]
STAGE6 = {"qwen7": ("Qwen2.5-7B-Instruct.json", "ed828a9b701d60f552eb8dc10247d85de44364b75f6086e5abe4eedb637353be"),
          "mistral7": ("Mistral-7B-Instruct-v0.3.json", "88ababd91bdb3155ef8e4aa691eff30d9edf88a13260148c145e654ed37e69af")}
KSTAR_FRAC, TEST_KSTAR_FRAC, N_RAND, RAND_SEED = 0.05, 0.012, 3, 2
N_RANK, N_EVAL = 60, 80
CONDS = ("none", "N", "T", "rand0", "rand1", "rand2", "C")


def log(s):
    print(s, flush=True)


def canon(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def qplus(d, n_extra=0):
    """Q+ rows of an encoded OPTA/NOM prompt: every token starting after the question's last character, plus the
    ``n_extra`` rows appended after the prompt."""
    q = "\nQuestion: " + d["question"]
    q_end = d["text"].index(q) + len(q)
    T0 = d.get("T0", d["T"])
    rows = [i for i, (s, e) in enumerate(d["offsets"]) if s >= q_end]
    assert rows and rows[-1] == T0 - 1, "Q+ must reach the end of the prompt"
    return rows + list(range(T0, T0 + n_extra))


class Natural(Stage6):
    """Stage6's splice / curves on a natural item: d["ib"] = B prompt + w, d["p"] = the span P, d["G"] = the option rows,
    d["iS"] / d["iB"] = the decision tokens of S / B."""

    def __init__(self, model, tok, hs, hop, frame):
        self.model, self.tok, self.hs, self.hop, self.frame = model, tok, hs, hop, frame
        self.dev = next(model.parameters()).device
        self.nL, self.H, self.hd = hs.nL, hs.H, hs.hd
        self.allcells = [(l, h) for l in range(self.nL) for h in range(self.H)]

    def prep_item(self, it, fmt="OPTA"):
        d = prep(self.tok, it, fmt, self.frame)
        d["question"] = it["question"].strip()
        rows = option_rows(d["text"], d["offsets"], it, fmt, it["answer"]) if fmt != "NOM" else {}
        G = sorted({r for v in rows.values() for r in v})
        d |= dict(G=G, GS=rows.get(it["S"], []), GB=rows.get(it["answer"], []), p=list(d["P"]), iS=d["dec"]["S"], iB=d["dec"]["B"],
                  ib=torch.tensor([d["ids"]["B"] + d["w"]], device=self.dev), T=d["T"] + len(d["w"]), T0=d["T"])
        assert fmt == "NOM" or (d["GS"] and d["GB"] and all(g > max(d["P"]) for g in G)), "option rows after the passage"
        return d

    def ks_of(self, d):
        """S's keys at P in every layer (the capture batch of ckeys.natural_rows), {l: [|P|, D]}."""
        self.hs.active = self.hop.active = False
        kv, _ = capture(self.model, dict(d, T=d["T0"]), range(self.nL), self.dev)
        self.kv = kv
        return {l: kv["S"][(l, "k")] for l in range(self.nL)}

    def kclamp(self, d, ks):
        return clamp_kv(self.model, d["p"], {(l, "k"): ks[l] for l in range(self.nL)}, range(self.nL), "k")

    def a3_of(self, d, ks):
        """a3 [nL, H] of one ranking item (eager: attention weights of the clean and the K_S-clamped pass)."""
        P = d["p"]

        def read(o):
            return torch.stack([torch.stack([a[0][:, rows][:, :, P].float().sum(-1).mean(-1) for rows in (d["GS"], d["GB"])])
                                for a in o.attentions]).cpu()     # [nL, 2, H]
        ob = self.model(d["ib"], use_cache=False, logits_to_keep=1, output_attentions=True)
        assert ob.attentions is not None and ob.attentions[0] is not None, "attentions missing: load with attn_implementation='eager'"
        Ab = read(ob)
        with self.kclamp(d, ks):
            Af = read(self.model(d["ib"], use_cache=False, logits_to_keep=1, output_attentions=True))
        return (0.5 * ((Af[:, 0] - Ab[:, 0]) + (Ab[:, 1] - Af[:, 1]))).numpy()

    def oproj_sums(self, d, rows, seq):
        """Sum over ``rows`` of every head's o_proj input in a clean pass of ``seq``: [nL, H, hd] (FP32, CPU)."""
        store = {}
        hk = [blocks(self.model)[l].self_attn.o_proj.register_forward_pre_hook(
            lambda _m, a, l=l: store.__setitem__(l, a[0][0, rows].float().sum(0).cpu())) for l in range(self.nL)]
        try:
            self.model(torch.tensor([seq], device=self.dev), use_cache=False, logits_to_keep=1)
        finally:
            for h in hk:
                h.remove()
        return torch.stack([store[l].view(self.H, self.hd) for l in range(self.nL)])

    def dla(self, d):
        """Direct logit attribution of every head at the decision row on dec_B minus the other options' mean: [nL, H]."""
        store, fin = {}, {}
        model = self.model
        hk = [blocks(model)[l].self_attn.o_proj.register_forward_pre_hook(
            lambda _m, a, l=l: store.__setitem__(l, a[0][0, -1].float())) for l in range(self.nL)]
        hk.append(model.model.norm.register_forward_pre_hook(lambda _m, a: fin.__setitem__("x", a[0][0, -1].float())))
        try:
            model(d["ib"], use_cache=False, logits_to_keep=1)
        finally:
            for h in hk:
                h.remove()
        norm = model.model.norm
        eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
        x = fin["x"]
        scale = norm.weight.float() / torch.sqrt((x * x).mean() + eps)
        WU = model.get_output_embeddings().weight
        others = [d["dec"][Y] for Y in ("S", "X", "D")]
        u = WU[d["dec"]["B"]].float() - WU[others].float().mean(0)
        out = torch.zeros(self.nL, self.H)
        for l in range(self.nL):
            Wo = blocks(model)[l].self_attn.o_proj.weight.float()     # [d_model, H * hd]
            o = store[l].view(self.H, self.hd)
            for h in range(self.H):
                out[l, h] = ((Wo[:, h * self.hd:(h + 1) * self.hd] @ o[h]) * scale) @ u
        return out.numpy()

    def ablate_batch(self, d, seq, conds, mu, clamp_tab=None, n_keep=None):
        """Rows = conditions ({name: cells}); HeadSplice "ablate" at Q+ of ``seq`` with the means mu ({l: [H, hd]});
        ``clamp_tab``: K/V tables at P (every row, every layer). Returns log-softmax logits [R, n_keep, V] of the last
        n_keep positions."""
        names = list(conds)
        T = len(seq)
        rows = qplus(d, T - d["T0"])
        dense = cells_dense([conds[n] for n in names], self.nL, self.H)
        masks = head_masks(dense, rows, T)
        mut = {l: mean_table(mu[l][None].expand(len(rows), -1, -1).contiguous(), rows, T) for l in masks}
        x = torch.tensor([seq], device=self.dev).expand(len(names), -1)
        hs = self.hs
        hs.mode, hs.masks, hs.mu, hs.active = "ablate", masks, mut, bool(masks)
        try:
            if clamp_tab is not None:
                with clamp_kv(self.model, d["p"], clamp_tab, range(self.nL)):
                    lg = self.model(x, use_cache=False, logits_to_keep=n_keep).logits
            else:
                lg = self.model(x, use_cache=False, logits_to_keep=n_keep).logits
        finally:
            hs.active, hs.masks, hs.mu, hs.mode = False, None, None, "splice"
        return lg.float().log_softmax(-1)


def chain(lp, c, j0):
    """Argmax checks of a teacher-forced answer: lp [n_keep, V] covers the positions predicting c[0..]; returns (all of c,
    the decision token c[j0], every token after it)."""
    am = lp[:len(c)].argmax(-1).cpu().tolist()
    ok = [a == t for a, t in zip(am, c)]
    return all(ok), ok[j0], all(ok[j0 + 1:])


def load_model(a):
    dev = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    kw = dict(dtype=getattr(torch, a.dtype), revision=a.revision, attn_implementation="eager")
    if dev == "cuda":
        kw["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if dev == "cpu":
        model = model.to(dev)
    assert model.config._attn_implementation == "eager"
    return model, tok, dev


def template_set(a, nL, H, kstar):
    """The stage-6 P1 a3 ranking (sha256 pinned; T* is its first k*); TEST_MODE: the Qwen2.5-7B ranking's cells that
    exist in the small model."""
    name, pin = STAGE6["qwen7" if a.test else a.key]
    f = ROOT / "results" / "gpu_stage6" / "heads" / name
    h = sha_file(f)
    assert h == pin, f"{f}: sha256 {h} != {pin}"
    rk = json.load(open(f))["arms"]["P1"]["rankings"]["a3"]
    cells = [list(c) for c in rk if c[0] < nL and c[1] < H]
    return cells, {"file": name, "sha256": h}


@torch.no_grad()
def run(a):
    frame, fsha = read_frame(a)
    model, tok, dev = load_model(a)
    hs, hop = HeadSplice(model), HopSplice(model)
    S = Natural(model, tok, hs, hop, frame)
    nL, H, hd = S.nL, S.H, S.hd
    kstar = math.ceil(a.kstar_frac * nL * H)
    KS = sorted({k for k in a.KS if k <= nL * H} | {kstar, 2 * kstar}) if not a.test else sorted(set(a.KS) | {kstar})
    items = load_items(a)
    ok = lambda its, fmts: [i for i in its if valid_all(tok, i, fmts, frame) is None]  # noqa: E731
    R_all = ok(split(items, "R"), ("NOM", "OPTA"))
    R = R_all[:a.n_rank]
    E_all = ok(split(items, "E"), ("NOM", "OPTA"))
    E_eval, E_abl = E_all[:a.n_eval], (E_all[:a.n_ablate] if a.n_ablate else E_all)
    Trank, t6 = template_set(a, nL, H, kstar)
    Tset = Trank[:kstar]
    P = ff_provenance(a) | {"model": a.model, "model_key": a.key, "revision": a.revision, "verified": verified(a.model),
                            "dtype": str(next(model.parameters()).dtype), "attn_implementation": model.config._attn_implementation,
                            "device": device_name(dev), "test_mode": a.test, "n_layers": nL, "heads_per_layer": H,
                            "n_heads": nL * H, "head_dim": hd, "kstar": kstar, "KS": KS, "frame": frame, "frames_sha256": fsha,
                            "stage6": t6, "items_sha256": sha_file(a.items), "n_valid": {"R": len(R_all), "E": len(E_all)},
                            "n_rank": len(R), "n_eval": len(E_eval), "n_ablate": len(E_abl), "skipped_parts": [], "timings": {}}
    path = Path(a.out) / "heads" / f"{a.tag}.json"
    out, t0 = {"provenance": P}, time.time()

    # ---- rank: a3 on R (OPTA), the means mu over Q+ (NOM, OPTA), DLA (NOM)
    A3, DLA, rk_rows = [], [], []
    mus = {f: torch.zeros(nL, H, hd) for f in ("NOM", "OPTA")}
    cnt = {f: 0 for f in mus}
    for it in R:
        d = S.prep_item(it, "OPTA")
        ks = S.ks_of(d)
        a3 = S.a3_of(d, ks)
        A3.append(a3)
        rk_rows.append({"id": it["id"], "P": d["p"], "G": d["G"], "T": d["T"]})
        for f in ("NOM", "OPTA"):
            df = d if f == "OPTA" else S.prep_item(it, "NOM")
            seq = df["ids"]["B"] + df["c"]["B"]
            rows = qplus(df, len(df["c"]["B"]))
            mus[f] += S.oproj_sums(df, rows, seq)
            cnt[f] += len(rows)
            if f == "NOM":
                DLA.append(S.dla(df))
    MU = {f: {l: (mus[f][l] / cnt[f]).contiguous() for l in range(nL)} for f in mus}
    mu_f = Path(a.out) / "heads" / f"mu_{a.tag}.pt"
    mu_f.parent.mkdir(parents=True, exist_ok=True)
    torch.save(MU, mu_f)
    A3m, DLAm = np.mean(A3, 0), np.mean(DLA, 0)
    rank_a3, rank_dla = rank_of(A3m), rank_of(DLAm)
    rng = np.random.default_rng(RAND_SEED)
    allc = S.allcells
    perms = [[list(allc[i]) for i in rng.permutation(len(allc))] for _ in range(N_RAND)]
    rand = [p[:kstar] for p in perms]
    sets = {"N": rank_a3[:kstar], "T": Tset, "rand": rand, "C": rank_dla[:kstar], "kstar": kstar}
    out["rank"] = {"items": rk_rows, "a3_mean": A3m.tolist(), "dla_mean": DLAm.tolist(), "ranking_a3": rank_a3, "ranking_dla": rank_dla,
                   "mu_file": mu_f.name, "mu_sha256": sha_file(mu_f), "mu_rows": cnt}
    out["sets"], out["sets_sha256"] = sets, canon(sets)
    P["timings"]["rank"] = round(time.time() - t0, 1)
    write_atomic(out, path)
    log(f"[{a.tag}] ranked on {len(R)} R items; k* {kstar}; |T* & N*| = {len({tuple(c) for c in sets['N']} & {tuple(c) for c in Tset})} ({time.time() - t0:.0f}s)")
    cells = lambda x: [tuple(c) for c in x]  # noqa: E731
    named = {"N": cells(sets["N"]), "T": cells(Tset)} | {f"rand{i}": cells(r) for i, r in enumerate(rand)}
    full_rank = {"N": cells(rank_a3), "T": cells(Trank)} | {f"rand{i}": cells(p) for i, p in enumerate(perms)}

    # ---- eval: curves for N*, T*, random sets on E_eval (OPTA)
    ev = []
    for it in E_eval:
        d = S.prep_item(it, "OPTA")
        ks = S.ks_of(d)
        mB = float(S.mb(d, 1)[0])
        with S.kclamp(d, ks):
            mF = float(S.mb(d, 1)[0])
        cv, layer = S.curves(d, ks, full_rank, KS)
        ev.append({"id": it["id"], "art": it["art"], "title": it["title"], "sub": it["sub"], "P": d["p"], "G": d["G"], "T": d["T"],
                   "mB": mB, "mF": mF, "curves": cv, "layer": layer})
    out["eval"] = ev
    P["timings"]["eval"] = round(time.time() - t0, 1)
    P["n_double"] = hs.n_double
    write_atomic(out, path)
    log(f"[{a.tag}] eval curves on {len(ev)} E items ({time.time() - t0:.0f}s)")

    # ---- ablate: the conditions at Q+ under KV_S (faithful S) and on the clean B prompt, OPTA and NOM
    conds = {"none": [], "N": named["N"], "T": named["T"], "rand0": named["rand0"], "rand1": named["rand1"],
             "rand2": named["rand2"], "C": cells(sets["C"])}
    abl = {}
    for f in ("OPTA", "NOM"):
        recs = []
        for it in E_abl:
            d = S.prep_item(it, f)
            S.ks_of(d)
            kv = S.kv
            cS, cB, j = d["c"]["S"], d["c"]["B"], d["j"]
            tab = tables(kv, [BASE["KV_S"]] * len(conds), nL, len(d["p"]))
            lpS = S.ablate_batch(d, d["ids"]["B"] + cS, conds, MU[f], tab, len(cS) + 1)
            lpB = S.ablate_batch(d, d["ids"]["B"] + cB, conds, MU[f], None, len(cB) + 1)
            rec = {"id": it["id"], "art": it["art"], "title": it["title"], "sub": it["sub"], "conds": {}, "clean": {}}
            for r, n in enumerate(conds):
                ca, cd, cc = chain(lpS[r], cS, j)
                rec["conds"][n] = {"chain": ca, "dec": cd, "cont": cc}
                ba, _, _ = chain(lpB[r], cB, j)
                rec["clean"][n] = {"chain": ba, "lp": {Y: float(lpB[r, j, t]) for Y, t in d["dec"].items()}}
            recs.append(rec)
        abl[f] = recs
        out["ablate"] = abl
        P["timings"][f"ablate_{f}"] = round(time.time() - t0, 1)
        write_atomic(out, path)
        log(f"[{a.tag}] ablate {f} on {len(recs)} E items ({time.time() - t0:.0f}s)")

    # ---- explore: ID_K, ID_V with N* ablated at Q+ (deadline-gated)
    d_ = os.environ.get("STAGE8_DEADLINE")
    if d_ and time.time() > float(d_):
        P["skipped_parts"].append("explore")
        log(f"[{a.tag}] explore skipped: the deadline has passed")
    else:
        exr = []
        names = list(BASE)
        for it in E_eval:
            d = S.prep_item(it, "OPTA")
            S.ks_of(d)
            tab = tables(S.kv, [BASE[n] for n in names] * 2, nL, len(d["p"]))
            cond = {f"{n}|{c}": (named["N"] if c == "N" else []) for c in ("none", "N") for n in names}
            lp = S.ablate_batch(d, d["ids"]["B"] + d["w"], cond, MU["OPTA"], tab, 1)
            exr.append({"id": it["id"], "art": it["art"], "title": it["title"],
                        "lp": {k: {Y: float(lp[r, 0, t]) for Y, t in d["dec"].items()} for r, k in enumerate(cond)}})
        out["explore"] = exr
        P["timings"]["explore"] = round(time.time() - t0, 1)
    P["complete"] = True
    write_atomic(out, path)
    log(f"wrote {path} ({time.time() - t0:.0f}s)")
    return 3 if P["skipped_parts"] else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="a Hub name or a local directory (s8_fetch)")
    ap.add_argument("--key", required=True, choices=sorted(STAGE6), help="qwen7 or mistral7 (the stage-6 template file)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--squad", required=True)
    ap.add_argument("--items", default=str(ITEMS))
    ap.add_argument("--out", default="results/gpu_stage8a")
    ap.add_argument("--n-rank", type=int, default=N_RANK)
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--n-ablate", type=int, default=0, help="0 = every valid E item")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    a.KS, a.kstar_frac = (1, 2, 5, 10, 20), KSTAR_FRAC
    if a.test:
        a.model, a.revision, a.dtype, a.n_rank, a.n_eval, a.n_ablate = TINY, None, "float32", 2, 2, 2
        a.KS, a.kstar_frac = (1, 2, 5, 10), TEST_KSTAR_FRAC
    assert a.model, "--model is required"
    a.tag = ("TEST_" if a.test else "") + a.key
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())   # 3: the exploratory part was skipped at the deadline (scripts/stage8_common.sh "partial")
