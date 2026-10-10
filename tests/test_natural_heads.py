"""Gate J-A-HA-G0 of P-2026-10-10-J part A: FP32 exactness of the natural-text head hooks (experiments/natural_heads.py:
HeadSplice with span tables, region mean-ablation at Q+, the argmax chain, a3, DLA) at Qwen2.5-0.5B-Instruct on the CPU,
eager attention, 1e-4 in the log-probabilities. Items as in tests/test_natural_clamp.py (committed items, a short passage
written for the test).

Checks: every head in every row seeing K_S at the whole span equals the full K_S clamp, no head equals clean, every head
in the rows G equals RowSplice(G) with the span table; a batch whose size equals |P| equals its rows run singly (the
span axis is never taken for the batch axis); ablation with each head's own o_proj input as its mean equals the clean
run, and ablation with a fixed mean equals an independent o_proj pre-hook that writes it; the argmax chain over the greedy
tokens holds and fails for a changed token; a3 with S's keys replaced by B's own keys is 0; the DLA of a head equals
o_proj applied to that head's slice alone; Q+ starts after the question and reaches the last answer row."""
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.generate import greedy
from ckeys.headsplice import HeadSplice, HopSplice, cells_dense, head_masks, mean_table
from ckeys.interventions import blocks
from ckeys.natural_rows import BASE, tables
from experiments.natural_heads import Natural, chain, qplus
from experiments.row_restricted_keys import RowSplice

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 1e-4
ROOT = Path(__file__).resolve().parents[1]
ITEMS = {i["id"]: i for i in json.load(open(ROOT / "data" / "stage8a_items.json"))}
CTX = {"57268739708984140094c8f0": "By 1947 the network had signed Bing Crosby, who taped his radio show on Magnetophon recorders.",
       "56f88025aef2371900626121": "For the first choral hymnal Luther wrote 24 hymns, which Johann Walter set for several voices."}
torch.set_grad_enabled(False)


def item(i):
    it = dict(ITEMS[i])
    it["context"] = CTX[i]
    it["start"] = CTX[i].index(it["answer"])
    return it


@pytest.fixture(scope="module")
def ctx():
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="eager").eval()
    rsp, hs, hop = RowSplice(model), HeadSplice(model), HopSplice(model)
    S = Natural(model, tok, hs, hop, " ")
    d = S.prep_item(item(list(CTX)[0]), "OPTA")
    ks = S.ks_of(d)
    assert len(d["p"]) >= 2 and d["G"]
    lp = lambda ids: model(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)  # noqa: E731
    clean = lp(d["ib"])
    with S.kclamp(d, ks):
        full = lp(d["ib"])
    assert (full - clean).abs().max() > 1e-2
    return dict(model=model, tok=tok, rsp=rsp, hs=hs, S=S, d=d, ks=ks, lp=lp, clean=clean, full=full, nL=S.nL, H=S.H)


def splice(c, masks, n=1, d=None, ks=None):
    hs, d, ks = c["hs"], d or c["d"], ks or c["ks"]
    hs.ks, hs.pos, hs.mode, hs.masks, hs.mu, hs.active = ks, d["p"], "splice", masks, None, True
    try:
        return c["lp"](d["ib"].expand(n, -1))
    finally:
        hs.active, hs.masks = False, None


def close(a, b):
    return torch.allclose(a, b, atol=TOL, rtol=0)


def test_span_splice_all_none_and_rows_G(ctx):
    nL, H, d = ctx["nL"], ctx["H"], ctx["d"]
    allT = head_masks(torch.ones(1, nL, H, dtype=torch.bool), slice(None), d["T"])
    assert close(splice(ctx, allT), ctx["full"])
    assert close(splice(ctx, {}), ctx["clean"])
    rows = torch.zeros(d["T"], dtype=torch.bool)
    rows[d["G"]] = True
    rsp = ctx["rsp"]
    rsp.ks, rsp.pos, rsp.mask, rsp.group, rsp.active = ctx["ks"], d["p"], rows, None, True
    try:
        ref = ctx["lp"](d["ib"])
    finally:
        rsp.active = False
    assert close(splice(ctx, head_masks(torch.ones(1, nL, H, dtype=torch.bool), d["G"], d["T"])), ref)
    assert (ref - ctx["clean"]).abs().max() > 1e-3
    m = ctx["S"].m(ref, d)[0] - ctx["S"].m(ctx["clean"], d)[0]
    assert abs(m) > 1e-3


def test_batch_size_equal_to_span_length(ctx):
    nL, H, d = ctx["nL"], ctx["H"], ctx["d"]
    nP = len(d["p"])
    sets = [ctx["S"].allcells[:7 * (b + 1)] for b in range(nP)]
    got = splice(ctx, head_masks(cells_dense(sets, nL, H), d["G"], d["T"]), n=nP)
    for b in range(nP):
        one = splice(ctx, head_masks(cells_dense([sets[b]], nL, H), d["G"], d["T"]))
        assert close(got[b:b + 1], one), b
    per_row = {l: torch.stack([ctx["ks"][l]] * nP) for l in range(nL)}   # [B, |P|, D] per-row span tables
    assert close(splice(ctx, head_masks(cells_dense(sets, nL, H), d["G"], d["T"]), n=nP, ks=per_row), got)


def own_oproj_inputs(c, d, seq, rows):
    store = {}
    hk = [blocks(c["model"])[l].self_attn.o_proj.register_forward_pre_hook(
        lambda _m, a, l=l: store.__setitem__(l, a[0][0, rows].clone())) for l in range(c["nL"])]
    try:
        c["model"](torch.tensor([seq]), use_cache=False, logits_to_keep=1)
    finally:
        for h in hk:
            h.remove()
    return {l: store[l].view(len(rows), c["H"], -1) for l in store}


def test_region_ablation_with_own_outputs_is_clean_and_fixed_means_match_a_hook(ctx):
    S, d, nL, H = ctx["S"], ctx["d"], ctx["nL"], ctx["H"]
    seq = d["ids"]["B"] + d["c"]["S"]
    rows = qplus(d, len(d["c"]["S"]))
    T = len(seq)
    clean = ctx["model"](torch.tensor([seq]), use_cache=False).logits[0, -len(d["c"]["S"]) - 1:].float().log_softmax(-1)
    own = own_oproj_inputs(ctx, d, seq, rows)
    hs = ctx["hs"]
    hs.mode, hs.masks = "ablate", head_masks(torch.ones(1, nL, H, dtype=torch.bool), rows, T)
    hs.mu, hs.active = {l: mean_table(own[l], rows, T) for l in range(nL)}, True
    try:
        got = ctx["model"](torch.tensor([seq]), use_cache=False).logits[0, -len(d["c"]["S"]) - 1:].float().log_softmax(-1)
    finally:
        hs.active, hs.masks, hs.mu, hs.mode = False, None, None, "splice"
    assert close(got, clean)
    # fixed means: ablate_batch (rows = conditions) against an independent pre-hook writing the mean at Q+
    torch.manual_seed(3)
    mu = {l: torch.randn(H, S.hd) * 0.05 for l in range(nL)}
    cells = [(l, h) for l in (2, 9, 17) for h in (0, 5, 11)]
    lp = S.ablate_batch(d, seq, {"none": [], "C": cells}, mu, None, len(d["c"]["S"]) + 1)
    hk = []
    for l in {c[0] for c in cells}:
        def pre(_m, a, l=l):
            x = a[0].clone().view(1, T, H, S.hd)
            for (ll, h) in cells:
                if ll == l:
                    x[0, rows, h] = mu[l][h]
            return (x.view(1, T, -1),) + tuple(a[1:])
        hk.append(blocks(ctx["model"])[l].self_attn.o_proj.register_forward_pre_hook(pre))
    try:
        ref = ctx["model"](torch.tensor([seq]), use_cache=False).logits[0, -len(d["c"]["S"]) - 1:].float().log_softmax(-1)
    finally:
        for h in hk:
            h.remove()
    assert close(lp[0], clean) and close(lp[1], ref)
    assert (ref - clean).abs().max() > 1e-3
    # with the KV_S clamp tables of the run, the none row equals the plain S run (the faithful-answer batch)
    S.ks_of(d)
    tab = tables(S.kv, [BASE["KV_S"]] * 2, nL, len(d["p"]))
    lpS = S.ablate_batch(d, seq, {"none": [], "N": cells}, mu, tab, len(d["c"]["S"]) + 1)
    refS = ctx["model"](torch.tensor([d["ids"]["S"] + d["c"]["S"]]), use_cache=False).logits[0, -len(d["c"]["S"]) - 1:].float().log_softmax(-1)
    assert close(lpS[0], refS)


def test_argmax_chain_is_greedy_and_qplus(ctx):
    S, d, model, tok = ctx["S"], ctx["d"], ctx["model"], ctx["tok"]
    g = greedy(model, tok, torch.tensor([d["ids"]["B"]]), max_new=8)[0]
    assert len(g) >= 2
    lp = S.ablate_batch(d, d["ids"]["B"] + g, {"none": []}, {}, None, len(g) + 1)
    assert chain(lp[0], g, 0) == (True, True, True)
    bad = g[:-1] + [(g[-1] + 1) % 1000]
    lpb = S.ablate_batch(d, d["ids"]["B"] + bad, {"none": []}, {}, None, len(bad) + 1)
    assert chain(lpb[0], bad, 0)[0] is False and chain(lpb[0], bad, 0)[2] is False
    rows = qplus(d, 3)
    q_end = d["text"].index("\nQuestion: " + d["question"]) + len("\nQuestion: " + d["question"])
    assert d["offsets"][rows[0]][0] >= q_end and d["offsets"][rows[0] - 1][0] < q_end
    assert rows[-4:] == [d["T0"] - 1, d["T0"], d["T0"] + 1, d["T0"] + 2]
    assert not set(rows) & set(d["p"]) and set(d["G"]) <= set(rows)


def test_a3_self_clamp_is_zero_and_dla_slices(ctx):
    S, d = ctx["S"], ctx["d"]
    S.ks_of(d)
    own = {l: S.kv["B"][(l, "k")] for l in range(ctx["nL"])}
    assert abs(S.a3_of(d, own)).max() < 1e-6
    a3 = S.a3_of(d, ctx["ks"])
    assert a3.shape == (ctx["nL"], ctx["H"]) and abs(a3).max() > 1e-4
    dn = S.prep_item(item(list(CTX)[0]), "NOM")
    dla = S.dla(dn)
    model, l, h, hd = ctx["model"], 5, 3, S.hd
    o = {}
    hk = blocks(model)[l].self_attn.o_proj.register_forward_pre_hook(lambda _m, a: o.__setitem__("x", a[0][0, -1].clone()))
    fin = {}
    hk2 = model.model.norm.register_forward_pre_hook(lambda _m, a: fin.__setitem__("x", a[0][0, -1].clone()))
    try:
        model(dn["ib"], use_cache=False, logits_to_keep=1)
    finally:
        hk.remove()
        hk2.remove()
    x = torch.zeros_like(o["x"])
    x[h * hd:(h + 1) * hd] = o["x"][h * hd:(h + 1) * hd]
    op = blocks(model)[l].self_attn.o_proj
    contrib = op(x[None])[0] - (op.bias if op.bias is not None else 0)
    n = model.model.norm
    y = contrib * n.weight / torch.sqrt((fin["x"] ** 2).mean() + n.variance_epsilon)
    W = model.get_output_embeddings().weight
    u = W[dn["dec"]["B"]] - W[[dn["dec"][Y] for Y in ("S", "X", "D")]].mean(0)
    assert abs(float(y @ u) - float(dla[l, h])) < 1e-4
