"""Gate a1 of P-2026-10-05-H (docs/PREREGISTRATION.md): FP32 exactness of the stage-6 hooks (ckeys/headsplice.py) at
Qwen2.5-0.5B-Instruct on the CPU, eager attention, 1e-4 in the logits; batch rows are compared with each other and with
single runs only at that tolerance (no bitwise equality across batch positions or thread counts)."""
import random

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import capture_kv, clamp_kv
from ckeys.headsplice import HeadSplice, HopSplice, cells_dense, head_masks, mean_table
from ckeys.interventions import blocks, edits
from ckeys.story import make_cores
from experiments.row_restricted_keys import RowSplice
from experiments.stage6_heads import HOP_ROWS, Stage6, configure_hop

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 1e-4


def close(a, b):
    return torch.allclose(a, b, atol=TOL, rtol=0)


@pytest.fixture(scope="module")
def ctx():
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="eager").eval()
    rsp, hs, hop = RowSplice(model), HeadSplice(model), HopSplice(model)
    S = Stage6(model, tok, "P1", hs, hop)
    d = S.prep(make_cores(1, random.Random(1))[0])
    ks = S.ks_of(d)
    nL = len(blocks(model))
    lg = lambda ids: model(ids, use_cache=False).logits[:, -1]  # noqa: E731
    clean = lg(d["ib"])
    kS = [(l, "k", [d["p"]], (lambda h, l=l: ks[l].expand_as(h))) for l in range(nL)]
    with edits(model, kS):
        full = lg(d["ib"])
    assert (full - clean).abs().max() > 1e-2
    yield dict(model=model, tok=tok, rsp=rsp, hs=hs, hop=hop, S=S, d=d, ks=ks, nL=nL, H=hs.H, T=d["T"], G=d["G"], lg=lg,
               clean=clean, full=full, kS=kS)
    torch.set_grad_enabled(True)


def run_hs(c, masks, mode="splice", mu=None, n=1):
    hs = c["hs"]
    hs.ks, hs.pos, hs.mode, hs.masks, hs.mu, hs.active = c["ks"], c["d"]["p"], mode, masks, mu, True
    try:
        return c["lg"](c["d"]["ib"].expand(n, -1))
    finally:
        hs.active, hs.masks, hs.mu = False, None, None


def test_headsplice_all_heads_is_full_clamp_and_empty_is_clean(ctx):
    nL, T, H = ctx["nL"], ctx["T"], ctx["H"]
    assert close(run_hs(ctx, head_masks(torch.ones(1, nL, H, dtype=torch.bool), slice(None), T)), ctx["full"])
    assert close(run_hs(ctx, {}), ctx["clean"])
    assert close(run_hs(ctx, {l: torch.zeros(1, T, H, dtype=torch.bool) for l in range(nL)}), ctx["clean"])


def test_headsplice_rows_G_and_group_slices_equal_rowsplice(ctx):
    rsp, nL, T, H, G = ctx["rsp"], ctx["nL"], ctx["T"], ctx["H"], ctx["G"]
    nkv = ctx["model"].config.num_key_value_heads
    rows = torch.zeros(T, dtype=torch.bool)
    rows[G] = True
    for g in [None] + list(range(nkv)):
        rsp.ks, rsp.pos, rsp.mask, rsp.group, rsp.active = ctx["ks"], ctx["d"]["p"], rows, g, True
        try:
            ref = ctx["lg"](ctx["d"]["ib"])
        finally:
            rsp.active, rsp.group = False, None
        dense = torch.zeros(1, nL, H, dtype=torch.bool)
        dense[0, :, slice(None) if g is None else slice(g * H // nkv, (g + 1) * H // nkv)] = True
        assert close(run_hs(ctx, head_masks(dense, G, T)), ref), g
        assert g is not None or (ref - ctx["clean"]).abs().max() > 1e-3


def test_headsplice_batched_grid_equals_single_runs(ctx):
    nL, T, H, G = ctx["nL"], ctx["T"], ctx["H"], ctx["G"]
    l, ar = nL // 2, torch.arange(H)
    add = torch.zeros(H + 1, nL, H, dtype=torch.bool)
    add[ar, l, ar] = True
    loo = torch.ones(H + 1, nL, H, dtype=torch.bool)
    loo[ar, l, ar] = False
    for dense, ref_last in ((add, ctx["clean"]), (loo, None)):
        grid = run_hs(ctx, head_masks(dense, G, T), n=H + 1)
        for b in (0, H - 1, H):
            assert close(grid[b:b + 1], run_hs(ctx, head_masks(dense[b:b + 1], G, T))), b
        assert ref_last is None or close(grid[H:], ref_last)
    sets = [[(0, 1), (l, 3)], [], [(c // H, c % H) for c in range(0, nL * H, 7)]]
    grid = run_hs(ctx, head_masks(cells_dense(sets, nL, H), G, T), n=3)
    for b, cells in enumerate(sets):
        assert close(grid[b:b + 1], run_hs(ctx, head_masks(cells_dense([cells], nL, H), G, T))), b
    with pytest.raises(AssertionError):
        run_hs(ctx, head_masks(cells_dense(sets[:2], nL, H), G, T), n=3)


def test_self_ablation_is_clean(ctx):
    model, nL, T, H, G, hd = ctx["model"], ctx["nL"], ctx["T"], ctx["H"], ctx["G"], ctx["hs"].hd
    store = {}
    hk = [blocks(model)[l].self_attn.o_proj.register_forward_pre_hook(lambda _m, a, l=l: store.__setitem__(l, a[0][0].clone()))
          for l in range(nL)]
    try:
        ctx["lg"](ctx["d"]["ib"])
    finally:
        for h in hk:
            h.remove()
    masks = head_masks(torch.ones(1, nL, H, dtype=torch.bool), G, T)
    mu = {l: mean_table(store[l].view(T, H, hd)[G], G, T) for l in range(nL)}
    assert close(run_hs(ctx, masks, "ablate", mu), ctx["clean"])
    assert close(run_hs(ctx, masks, "ablate", mu, n=3), ctx["clean"].expand(3, -1))
    assert (run_hs(ctx, masks, "ablate", None) - ctx["clean"]).abs().max() > 1e-3


def run_hop(c, names, kv, rows, kp=None):
    kpr = configure_hop(c["hop"], names, kv, rows, c["T"])
    c["hop"].active = True
    try:
        with clamp_kv(c["model"], [c["d"]["p"]], {(l, "k"): c["ks"][l][None] for l in range(c["nL"])}, range(c["nL"]), "k",
                      per_row=kpr if kp is None else kp):
            return c["lg"](c["d"]["ib"].expand(len(names), -1))
    finally:
        c["hop"].active = False


def test_hopsplice_all_rows_is_plain_clamp(ctx):
    model, nL, G, hop = ctx["model"], ctx["nL"], ctx["G"], ctx["hop"]
    with capture_kv(model, G, range(nL)) as src:
        ctx["lg"](ctx["d"]["is_"])
    tab = {k: v[0] for k, v in src.items()}
    with clamp_kv(model, G, tab, range(nL)):
        ref = ctx["lg"](ctx["d"]["ib"])
    assert (ref - ctx["clean"]).abs().max() > 1e-3
    kv = {"base": tab, "ksrun": tab}
    assert close(run_hop(ctx, ["all_KV"], kv, G, kp=torch.tensor([False])), ref)
    hop.rows, hop.tabs, hop.which, hop.mask, hop.active = G, {"A": {k: v[None] for k, v in tab.items()}}, {"A": {"k": [True], "v": [True]}}, None, True
    try:
        assert close(ctx["lg"](ctx["d"]["ib"]), ref)
    finally:
        hop.active = False


def test_hopsplice_no_overwrite_and_answer_only_cache(ctx):
    model, nL, G, T = ctx["model"], ctx["nL"], ctx["G"], ctx["T"]
    with capture_kv(model, G, range(nL)) as base:
        ctx["lg"](ctx["d"]["ib"])
    with edits(model, ctx["kS"]), capture_kv(model, G, range(nL)) as ksr:
        ctx["lg"](ctx["d"]["ib"])
    kv = {"base": {k: v[0] for k, v in base.items()}, "ksrun": {k: v[0] for k, v in ksr.items()}}
    hop = ctx["hop"]
    for kp, ref in ((False, ctx["clean"]), (True, ctx["full"])):
        configure_hop(hop, ["ans_KV"], kv, G, T)
        hop.which = {P: {"k": [False], "v": [False]} for P in "AB"}
        hop.mask[0] = True
        hop.active = True
        try:
            with clamp_kv(model, [ctx["d"]["p"]], {(l, "k"): ctx["ks"][l][None] for l in range(nL)}, range(nL), "k", per_row=torch.tensor([kp])):
                assert close(ctx["lg"](ctx["d"]["ib"]), ref), kp
        finally:
            hop.active = False
    assert close(run_hop(ctx, ["exact"], kv, G), ctx["full"])
    assert close(run_hop(ctx, ["ID"], kv, G), ctx["clean"]) and close(run_hop(ctx, ["K_S"], kv, G), ctx["full"])
    assert (run_hop(ctx, ["ans_KV"], kv, G) - ctx["full"]).abs().max() > 1e-3


def test_hop_batch_equals_single_rows(ctx):
    S, d, ks = ctx["S"], ctx["d"], ctx["ks"]
    names = list(HOP_ROWS)
    kv = S.hop_cache(d, ks, d["G"])
    batch = run_hop(ctx, names, kv, d["G"])
    for i, n in enumerate(names):
        assert close(batch[i:i + 1], run_hop(ctx, [n], kv, d["G"])), n
    assert close(batch[0:1], ctx["clean"]) and close(batch[1:2], ctx["full"]) and close(batch[7:8], batch[1:2])
    m = S.hop_batch(d, ks, names, kv, d["G"])
    lp = torch.log_softmax(batch.double(), -1)
    assert abs(m - (lp[:, d["iS"]] - lp[:, d["iB"]]).numpy()).max() < TOL
