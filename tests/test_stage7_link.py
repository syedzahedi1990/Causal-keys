"""Gate I-G0 of P-2026-10-08-I (docs/PREREGISTRATION.md): FP32 exactness of the stage-7 hooks (ckeys/readerblind.py,
experiments/stage7_link.py) at Qwen2.5-0.5B-Instruct on the CPU, sdpa (the ranking test eager), random rank-16 bases of
width 896 (generator seed 0, as paper1_frames --bases-override) and native cores 0-1 of the predecessor's release ($P1R).
Tolerance 1e-4 in the logits unless stated (host-independent: batch rows are compared with each other and with single runs
only at that tolerance); the LIST-BEFORE check is exact (0.0). The run's mean-ablation path (rank_core -> MU -> A) is
checked by value, and the hooks at the 24B head geometry (heads x head_dim != hidden size) on a tiny random-weight
Mistral. The pipeline treats a skipped test as failing the gate."""
import gc
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.headsplice import mean_table
from ckeys.interventions import blocks, capture
from ckeys.knockout import knockout_mask
from ckeys.readerblind import blind_masks, cell_masks, layer_cells
from experiments import stage7_link as s7
from experiments.paper1_frames import run_core
from experiments.refit_remap import PINNED
from experiments.row_restricted_keys import RowSplice

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 1e-4
ROOT = Path(__file__).resolve().parents[1]
P1R = os.environ.get("P1R", "")
need_p1r = pytest.mark.skipif(not (P1R and Path(P1R, s7.NATIVE).is_file()), reason="P1R (the predecessor's release) not set")


def close(a, b, tol=TOL):
    return torch.allclose(a, b, atol=tol, rtol=0)


def maxdiff(a, b):
    return float((a - b).abs().max())


@pytest.fixture(scope="module")
def ctx():
    if not (P1R and Path(P1R, s7.NATIVE).is_file()):
        pytest.skip("P1R (the predecessor's release) not set")
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval()
    L = s7.Link(model, tok)
    rsp = RowSplice(model)
    bases = s7.random_bases(model.config.hidden_size)
    E = s7.native_cores(P1R)[:2]
    d = s7.prep(tok, E[0], "P1", True)
    res = {}
    Pt = L.natural(d, bases, res)
    X = L.tables(d, Pt, res)
    S = [(s7.ONSET, 0), (s7.ONSET + 2, 3), (L.nL - 1, 5), (L.nL // 2, 7), (L.nL // 2, 8)]
    c = dict(model=model, tok=tok, L=L, rsp=rsp, bases=bases, E=E, d=d, res=res, Pt=Pt, X=X, S=S, nL=L.nL, H=L.H, hd=L.hd)
    yield c
    c.clear()           # the hooks hold reference cycles to the model: free it before the next test module loads its own
    del model, L, rsp
    gc.collect()
    torch.set_grad_enabled(True)


def fam(c, cond=None, sel=None, am=None, d=None, Pt=None, X=None):
    c["L"].family(d or c["d"], Pt or c["Pt"], X or c["X"], cond, sel=sel, am=am)
    return c["L"].last.clone()


def bx(c, rows, d=None, Pt=None, X=None, **kw):
    c["L"].bx(d or c["d"], Pt or c["Pt"], X or c["X"], rows, **kw)
    return c["L"].last.clone()


def plain(c, host, s, d=None, Pt=None):
    """One patched row without any other hook (explicit plain causal mask)."""
    d, Pt, L = d or c["d"], Pt or c["Pt"], c["L"]
    with L.cfg(patch=(Pt[(host, s)][None], torch.ones(1, dtype=torch.bool), d["span"][0], d["span"][-1] + 1)):
        L.lp(d["ids"]["B"], d["cid"])
    return L.last.clone()


def comp(c, C):
    return sorted(set(c["L"].elig) - set(C))


def test_empty_sets_equal_unblocked(ctx):
    c = ctx
    ref = fam(c)
    assert close(fam(c, ("A", [], {})), ref) and close(fam(c, ("N", [], False, False)), ref)
    for s in s7.SEEDS:
        assert close(bx(c, [("P", s, "P", [], False), ("M", s, "M", [], False)]), torch.cat([plain(c, "P", s), plain(c, "M", s)]))
    T, nL, H, L = c["d"]["T"], c["nL"], c["H"], c["L"]
    n0 = L.hs.n_double
    with L.cfg(hs=dict(mode="splice", ks={l: c["X"][("M", 101)]["k"][l] for l in range(s7.ONSET, nL)}, pos=c["d"]["p"], mu=None,
                       masks={l: torch.zeros(1, T, H, dtype=torch.bool) for l in range(nL)})):
        L.lp(c["d"]["ids"]["B"], c["d"]["cid"])
    a = L.last.clone()
    L.lp(c["d"]["ids"]["B"], c["d"]["cid"])
    assert L.hs.n_double == n0 and close(a, L.last)


def test_bx_all_equals_frames_addition(ctx):
    c = ctx
    rc = run_core(c["model"], c["tok"], c["d"]["core"], "P1", c["bases"], c["L"].dev)
    keys = [f"{n}_{s}" for s in s7.SEEDS for n in s7.FAM_NAMES]
    c["L"].family(c["d"], c["Pt"], c["X"], None)
    ref = c["L"].last.clone()
    out = c["L"].family(c["d"], c["Pt"], c["X"], None)
    for s in s7.SEEDS:
        x = bx(c, [("x_all", s, "P", c["L"].elig, True)])
        assert close(x, ref[keys.index(f"PK_{s}")][None]), (s, maxdiff(x, ref[keys.index(f"PK_{s}")][None]))
        assert max(abs(a - b) for a, b in zip(out[f"PK_{s}"]["cand"], rc["runs"][f"addition_{s}"]["cand"])) < TOL
        assert maxdiff(x, plain(c, "P", s)) > 1e-3


def test_rows_G_only_equals_rowsplice(ctx):
    c, d = ctx, ctx["d"]
    rsp, T = c["rsp"], d["T"]
    for s in (101, 103):
        x = bx(c, [("s_G", s, "P", c["L"].elig, False)])
        G = torch.zeros(T, dtype=torch.bool)
        G[d["G"]] = True
        rsp.ks, rsp.pos, rsp.mask, rsp.layers = c["X"][("M", s)]["k"], d["p"], G, set(range(s7.ONSET, c["nL"]))
        rsp.active = True
        try:
            ref = plain(c, "P", s)
        finally:
            rsp.active, rsp.layers = False, None
        assert close(x, ref), (s, maxdiff(x, ref))
        assert maxdiff(x, plain(c, "P", s)) > 1e-3


def test_bx_notG_with_pin_equals_exchange_plus_rowsplice(ctx):
    c, d, L = ctx, ctx["d"], ctx["L"]
    rsp, T, p, s = c["rsp"], d["T"], d["p"], 102
    x = bx(c, [("x_notG", s, "P", [], True)])
    G = torch.zeros(T, dtype=torch.bool)
    G[d["G"]] = True
    rsp.ks, rsp.pos, rsp.mask, rsp.layers = c["X"][("P", s)]["k"], p, G, set(range(s7.ONSET, c["nL"]))
    tab = {l: c["X"][("M", s)]["k"][l][None] for l in range(s7.ONSET, c["nL"])}
    rsp.active = True
    try:
        with L.cfg(patch=(c["Pt"][("P", s)][None], torch.ones(1, dtype=torch.bool), d["span"][0], d["span"][-1] + 1),
                   k=(tab, torch.ones(1, dtype=torch.bool), torch.tensor([p]))):
            L.lp(d["ids"]["B"], d["cid"])
        ref = L.last.clone()
    finally:
        rsp.active, rsp.layers = False, None
    assert close(x, ref), maxdiff(x, ref)
    nopin = bx(c, [("x_notG", s, "P", [], True)], pin=False)
    print(f"x_notG without the key pin differs from the reference by {maxdiff(nopin, ref):.3g} (the leak the pin removes)")


def test_pin_runs_before_headsplice_khook(ctx):
    c, d, L = ctx, ctx["d"], ctx["L"]
    l, s, p = s7.ONSET + 1, 101, d["p"]
    seen = []
    h = blocks(c["model"])[l].self_attn.k_proj.register_forward_hook(lambda _m, _i, o: seen.append(o[0, p].clone()))
    try:
        bx(c, [("x_all", s, "P", L.elig, True)])
    finally:
        h.remove()
    assert len(seen) == 2, len(seen)
    assert torch.equal(seen[0], c["X"][("P", s)]["k"][l].to(seen[0].dtype)), "first pass: the pinned host key K_P"
    assert torch.equal(seen[1], c["X"][("M", s)]["k"][l].to(seen[1].dtype)), "second pass: HeadSplice's ks (K_M)"


def test_self_ablation_is_unablated(ctx):
    c, d, L = ctx, ctx["d"], ctx["L"]
    nL, H, hd, T = c["nL"], c["H"], c["hd"], d["T"]
    store = {}
    hk = [blocks(c["model"])[l].self_attn.o_proj.register_forward_pre_hook(lambda _m, a, l=l: store.__setitem__(l, a[0].clone()))
          for l in range(s7.L4, nL)]
    try:
        ref = fam(c)
    finally:
        for h in hk:
            h.remove()
    cells = [(l, h) for l in range(s7.L4, nL) for h in range(H)]
    mu = {l: store[l].view(store[l].shape[0], T, H, hd) for l in store}
    out = fam(c, ("A", cells, mu))
    assert close(out, ref), maxdiff(out, ref)
    zero = {l: torch.zeros(1, T, H, hd) for l in store}
    assert maxdiff(fam(c, ("A", cells, zero)), ref) > 1e-3


def test_aplus_masks_exactly_S_and_L4_at_G(ctx):
    c, d, L = ctx, ctx["d"], ctx["L"]
    nL, H, hd, T, G = c["nL"], c["H"], c["hd"], d["T"], d["G"]
    S = c["S"]
    want = set(S) | set(layer_cells(s7.L4, H))
    m = cell_masks([sorted(want)], G, T, nL, H)
    assert min(m) == s7.L4
    for l, M in m.items():
        for t in range(T):
            for h in range(H):
                assert bool(M[0, t, h]) == (t in G and (l, h) in want), (l, t, h)
    store = {}
    hk = [blocks(c["model"])[l].self_attn.o_proj.register_forward_pre_hook(lambda _m, a, l=l: store.__setitem__(l, a[0].clone()))
          for l in range(nL)]
    try:
        fam(c, ("A", sorted(want), {l: torch.full((1, T, H, hd), 7.0) for l in range(s7.L4, nL)}))
    finally:
        for h in hk:
            h.remove()
    for l in range(nL):
        sev = (store[l].view(-1, T, H, hd) == 7.0).all(-1)        # [B, T, H]
        for t in range(T):
            for h in range(H):
                assert bool(sev[:, t, h].all()) == (t in G and (l, h) in want) and (bool(sev[:, t, h].any()) == bool(sev[:, t, h].all())), (l, t, h)


def test_readerko_all_heads_equals_4d_knockout(ctx):
    c, d, L = ctx, ctx["d"], ctx["L"]
    B = 21
    out = fam(c, ("N", L.allcells, False, False))
    ref = fam(c, None, am=knockout_mask(d["T"], [(g, d["p"]) for g in d["G"]], B))
    assert close(out, ref), maxdiff(out, ref)
    assert maxdiff(ref, fam(c)) > 1e-3


def test_null_knockout_and_explicit_mask(ctx):
    c, d, L = ctx, ctx["d"], ctx["L"]
    ref = fam(c)
    assert close(fam(c, ("N", c["S"], True, False)), ref)
    with pytest.raises(AssertionError):
        cm = cell_masks([c["S"]], d["G"], d["T"], c["nL"], c["H"])
        with L.cfg(ko=dict(masks=cm, cols=torch.tensor([d["p"]]), null=False, allow_before=False)):
            L.lp(d["ids"]["B"], d["cid"], explicit=False)


def lpc(logits, d):
    """Candidate log-probs (nats) of the last-position logits."""
    return torch.log_softmax(logits.double(), -1)[:, d["cid"]]


def test_mixed_batches_equal_single_rows(ctx):
    """Each row of a mixed batch equals the same row run alone (candidate log-probs, 1e-4 nats) and, in the logits, the
    same row in a batch of its own copies of the same size (the GEMM blocking then matches: no cross-row leakage)."""
    c, L = ctx, ctx["L"]
    tok, core = c["tok"], c["E"][1]
    d = s7.prep(tok, core, "LETTER", True)
    Pt = L.natural(d, c["bases"])
    X = L.tables(d, Pt)
    S = c["S"]
    rows = [("P", 101, "P", [], False), ("x_H", 102, "P", comp(c, S), True), ("s_G", 103, "P", L.elig, False),
            ("rem_H", 101, "M", S, False), ("x_notG", 103, "P", [], True)]
    batch = bx(c, rows, d, Pt, X)
    for i, r in enumerate(rows):
        one = bx(c, [r], d, Pt, X)
        assert close(lpc(batch[i:i + 1], d), lpc(one, d)), (r[0], maxdiff(lpc(batch[i:i + 1], d), lpc(one, d)))
        same = bx(c, [r] * len(rows), d, Pt, X)
        assert close(batch[i:i + 1], same[i:i + 1]), (r[0], maxdiff(batch[i:i + 1], same[i:i + 1]))
    T, H, hd, nL = d["T"], c["H"], c["hd"], c["nL"]
    mu = {l: torch.randn(1, T, H, hd, generator=torch.Generator().manual_seed(l)) for l in range(s7.L4, nL)}
    keys = [f"{n}_{s}" for s in s7.SEEDS for n in s7.FAM_NAMES] + ["B", "S", "T"]
    sel = ["P_101", "PK_102", "MV_103", "S", "T"]
    for cond in (("A", S, mu), ("A", sorted(set(S) | set(layer_cells(s7.L4, H))), mu), ("N", S, False, False), ("N", L.allcells, False, False)):
        full = fam(c, cond, d=d, Pt=Pt, X=X)
        for k in sel:
            i = keys.index(k)
            one = fam(c, cond, sel=[k], d=d, Pt=Pt, X=X)
            assert close(lpc(full[i:i + 1], d), lpc(one, d)), (cond[0], k, maxdiff(lpc(full[i:i + 1], d), lpc(one, d)))
            same = fam(c, cond, sel=[k] * len(keys), d=d, Pt=Pt, X=X)
            assert close(full[i:i + 1], same[i:i + 1]), (cond[0], k, maxdiff(full[i:i + 1], same[i:i + 1]))


def test_natural_passes_and_tables_equal_run_core(ctx):
    c, d, L = ctx, ctx["d"], ctx["L"]
    rc = run_core(c["model"], c["tok"], d["core"], "P1", c["bases"], L.dev)
    for n in ("B", "S", "T"):
        assert c["res"][f"nat/{n}"] == rc["runs"][n], n          # bitwise: the same single passes
    for s in s7.SEEDS:
        for obj in ("m3", "pca"):
            a, b = c["res"][f"cap/{obj}_{s}"], rc["runs"][f"{obj}_{s}"]
            assert max(abs(x - y) for x, y in zip(a["cand"], b["cand"])) < TOL and a["argmax"] == b["argmax"]
    # run_family's capture batch (all nine bases, in its order) gives the same K/V tables at p
    span, p = d["span"], d["p"]
    keys = list(c["bases"])
    hb = None
    with capture(c["model"], [s7.FIT_LAYER0], "resid") as st:
        L.lp(d["ids"]["B"], d["cid"], explicit=False)
    hb = st[s7.FIT_LAYER0][0, span[0]:span[-1] + 1]
    with capture(c["model"], [s7.FIT_LAYER0], "resid") as st:
        L.lp(d["ids"]["S"], d["cid"], explicit=False)
    hs = st[s7.FIT_LAYER0][0, span[0]:span[-1] + 1]
    pat = torch.stack([hb + ((hs - hb) @ c["bases"][k].T) @ c["bases"][k] for k in keys])
    for (tag, s) in c["Pt"]:
        assert torch.equal(c["Pt"][(tag, s)], pat[keys.index(("m3" if tag == "M" else "pca", s))]), (tag, s)
    lay = range(s7.ONSET, c["nL"])
    with L.cfg(patch=(pat, torch.ones(len(keys), dtype=torch.bool), span[0], span[-1] + 1)), \
            capture(c["model"], lay, "k") as K, capture(c["model"], lay, "v") as V:
        L.lp(d["ids"]["B"].expand(len(keys), -1), d["cid"], explicit=False)
    for (tag, s), t in c["X"].items():
        i = keys.index(("m3" if tag == "M" else "pca", s))
        for l in lay:
            assert close(t["k"][l], K[l][i, p]) and close(t["v"][l], V[l][i, p]), (tag, s, l)


def test_family_rows_equal_run_family(ctx):
    c, d = ctx, ctx["d"]
    rc = run_core(c["model"], c["tok"], d["core"], "P1", c["bases"], c["L"].dev)
    out = c["L"].family(d, c["Pt"], c["X"], None)
    pairs = {"P": "pca", "M": "m3", "PK": "addition", "MK": "removal", "PV": "addition_v", "MV": "removal_v"}
    for s in s7.SEEDS:
        for n, r in pairs.items():
            a, b = out[f"{n}_{s}"]["cand"], rc["runs"][f"{r}_{s}"]["cand"]
            assert max(abs(x - y) for x, y in zip(a, b)) < TOL, (n, s)
    for n in ("B", "S", "T"):
        assert max(abs(x - y) for x, y in zip(out[n]["cand"], rc["runs"][n]["cand"])) < TOL, n


def test_positions_up_to_p(ctx):
    c, d, L = ctx, ctx["d"], ctx["L"]
    p, nL, S = d["p"], c["nL"], c["S"]
    T, H, hd = d["T"], c["H"], c["hd"]

    def resid(fn):
        with capture(c["model"], range(nL), "resid") as st:
            fn()
        return torch.stack([st[l] for l in range(nL)], 1)          # [B, nL, T, D]
    mu = {l: torch.randn(1, T, H, hd, generator=torch.Generator().manual_seed(l)) for l in range(s7.L4, nL)}
    ref = resid(lambda: L.family(d, c["Pt"], c["X"], None))
    for cond in (("A", S, mu), ("A", sorted(set(S) | set(layer_cells(s7.L4, H))), mu), ("N", S, False, False), ("N", L.allcells, False, False)):
        r = resid(lambda: L.family(d, c["Pt"], c["X"], cond))
        assert close(r[:, :, :p + 1], ref[:, :, :p + 1], 1e-5), cond[0]
        assert maxdiff(r[:, :, p + 1:], ref[:, :, p + 1:]) > 1e-4
    s = 101
    names = ["P", "x_all", "x_H", "x_notG", "s_H", "s_G"]
    spec = {"P": ([], False), "x_all": (L.elig, True), "x_H": (comp(c, S), True), "x_notG": ([], True), "s_H": (S, False), "s_G": (L.elig, False)}
    r = resid(lambda: L.bx(d, c["Pt"], c["X"], [(n, s, "P", *spec[n]) for n in names]))
    for i, n in enumerate(names):
        assert close(r[i, :, :p], r[0, :, :p], 1e-5), n                               # positions < p: the host's own
        j = 1 if n.startswith("x_") else 0                                            # position p: K_M seen by p itself or not
        assert close(r[i, :, p], r[j, :, p], 1e-5), n


def test_list_before_structural_zero(ctx):
    c, L = ctx, ctx["L"]
    d = s7.prep(c["tok"], c["E"][0], "BEFORE", True)
    assert all(g < d["p"] for g in d["G"])
    Pt = L.natural(d, c["bases"])
    X = L.tables(d, Pt)
    S = c["S"]
    for s in s7.SEEDS:
        out = bx(c, [("x_all", s, "P", L.elig, True), ("x_HP1", s, "P", comp(c, S), True)], d, Pt, X)
        assert maxdiff(out[0], out[1]) == 0.0, s
    ref = fam(c, None, d=d, Pt=Pt, X=X)
    assert maxdiff(fam(c, ("N", S, False, True), d=d, Pt=Pt, X=X), ref) == 0.0
    with pytest.raises(AssertionError):
        fam(c, ("N", S, False, False), d=d, Pt=Pt, X=X)       # rows before p need allow_before


def test_pins():
    J = json.load(open(ROOT / "results/gpu_stage4/frames/mistral.json"))["provenance"]["bases"]
    assert {k: v["sha256"] for k, v in J.items() if "/" not in k} == s7.BASES_SHA
    if not (P1R and Path(P1R, "RELEASE.json").is_file()):
        pytest.skip("P1R (the predecessor's release) not set")
    rel = Path(P1R, "RELEASE.json")
    assert json.load(open(rel))["files"][s7.NATIVE]["sha256"] == s7.NATIVE_SHA
    assert hashlib.sha256(rel.read_bytes()).hexdigest() == PINNED["RELEASE.json"]
    r = s7.release_check(P1R, False)
    assert r["native_sha256"] == s7.NATIVE_SHA and r["bases_sha256"] == s7.BASES_SHA


@need_p1r
def test_rank_a3_zero_below_onset_and_remap_rows():
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="eager").eval()
    L = s7.Link(model, tok)
    import random
    from ckeys.story import make_cores
    d = s7.prep(tok, make_cores(1, random.Random(0))[0], "P1", False)
    r, oin = L.rank_core(d)
    a3 = torch.tensor(r["a3"])
    assert float(a3[:s7.ONSET].abs().max()) <= 1e-6 and float(a3[s7.ONSET:].abs().max()) > 1e-4
    assert oin.shape == (L.nL - s7.L4, 6, L.H, L.hd)
    de = s7.prep(tok, s7.native_cores(P1R)[0], "P1", True)
    bases = s7.random_bases(model.config.hidden_size)
    Pt = L.natural(de, bases)
    X = L.tables(de, Pt)
    mu_path(L, de, Pt, X)
    o = L.pk_rows(de, Pt, X)
    pk = o.logits[:, -1].float()
    L.family(de, Pt, X, None, sel=[f"{n}_{s}" for s in s7.SEEDS for n in ("P", "PK")])
    assert close(pk, L.last), maxdiff(pk, L.last)
    assert len(o.attentions) == L.nL and o.attentions[0].shape[0] == 2 * len(s7.SEEDS)
    del o, L, model
    gc.collect()
    torch.set_grad_enabled(True)


def mu_path(L, d, Pt, X):
    """The run's mean-ablation path by value: rank_core's capture of the o_proj inputs at G in the natural B run of story
    d, as MU (stage_rank with one story), mean_table, then A over every head of layers >= L4 at G in the family batch:
    the natural B row equals the unablated B row (each head is replaced by its own value); zero means change it."""
    _, oin = L.rank_core(d)
    MU = {l: oin[l - s7.L4].contiguous() for l in range(s7.L4, L.nL)}
    mut = {l: mean_table(MU[l], d["G"], d["T"]) for l in MU}
    cells = [(l, h) for l in range(s7.L4, L.nL) for h in range(L.H)]
    L.family(d, Pt, X, None, sel=["B"])
    ref = L.last.clone()
    L.family(d, Pt, X, ("A", cells, mut), sel=["B"])
    assert close(L.last, ref), maxdiff(L.last, ref)
    L.family(d, Pt, X, ("A", cells, {l: torch.zeros_like(t) for l, t in mut.items()}), sel=["B"])
    assert maxdiff(L.last, ref) > 1e-6


@need_p1r
def test_hooks_at_the_24b_head_geometry():
    """A tiny random-weight Mistral whose heads x head_dim differs from the hidden size (as at Mistral-Small-24B: 32 x 128 =
    4096 != 5120; here 4 x 32 = 128 != 96, 2 KV heads, 7 layers), with the Qwen2.5-0.5B tokenizer: rank_core and the MU
    path (eager); B_x x_all = the P + K_M row, A with each head's own o_proj input = unablated, N(all heads) = the 4D
    knockout, positions <= p unchanged under A and N (sdpa)."""
    from transformers import MistralConfig, MistralForCausalLM
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(NAME)

    def build(attn):
        cfg = MistralConfig(vocab_size=len(tok), hidden_size=96, intermediate_size=128, num_hidden_layers=7, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=32, sliding_window=None, max_position_embeddings=4096,
                            attn_implementation=attn)
        torch.manual_seed(0)
        m = MistralForCausalLM(cfg).float().eval()
        for p in m.parameters():            # larger weights than the default init, so that every check sees a visible effect
            p.mul_(8.0)
        assert m.config._attn_implementation == attn
        return m
    for attn in ("eager", "sdpa"):
        model = build(attn)
        L = s7.Link(model, tok)
        assert L.H * L.hd != model.config.hidden_size and L.hd == 32
        d = s7.prep(tok, s7.native_cores(P1R)[0], "P1", True)
        bases = s7.random_bases(model.config.hidden_size)
        Pt = L.natural(d, bases)
        X = L.tables(d, Pt)
        if attn == "eager":
            mu_path(L, d, Pt, X)
        else:
            L.family(d, Pt, X, None)
            ref = L.last.clone()
            for s in s7.SEEDS:
                L.bx(d, Pt, X, [("x_all", s, "P", L.elig, True)])
                x = L.last.clone()
                keys = [f"{n}_{t}" for t in s7.SEEDS for n in s7.FAM_NAMES]
                assert close(x, ref[keys.index(f"PK_{s}")][None]), (s, maxdiff(x, ref[keys.index(f"PK_{s}")][None]))
            store = {}
            hk = [blocks(model)[l].self_attn.o_proj.register_forward_pre_hook(lambda _m, a, l=l: store.__setitem__(l, a[0].clone()))
                  for l in range(s7.L4, L.nL)]
            try:
                L.family(d, Pt, X, None)
            finally:
                for h in hk:
                    h.remove()
            T = d["T"]
            cells = [(l, h) for l in range(s7.L4, L.nL) for h in range(L.H)]
            L.family(d, Pt, X, ("A", cells, {l: store[l].view(store[l].shape[0], T, L.H, L.hd) for l in store}))
            assert close(L.last, ref), maxdiff(L.last, ref)
            L.family(d, Pt, X, ("N", L.allcells, False, False))
            out = L.last.clone()
            L.family(d, Pt, X, None, am=knockout_mask(T, [(g, d["p"]) for g in d["G"]], 21))
            assert close(out, L.last), maxdiff(out, L.last)
            assert maxdiff(out, ref) > 1e-6
            p = d["p"]
            mu = {l: torch.randn(1, T, L.H, L.hd, generator=torch.Generator().manual_seed(l)) for l in range(s7.L4, L.nL)}
            for cond in (("A", [(s7.ONSET, 1), (L.nL - 1, 3)], mu), ("N", L.allcells, False, False)):
                with capture(model, range(L.nL), "resid") as st0:
                    L.family(d, Pt, X, None)
                with capture(model, range(L.nL), "resid") as st1:
                    L.family(d, Pt, X, cond)
                for l in range(L.nL):
                    assert close(st1[l][:, :p + 1], st0[l][:, :p + 1], 1e-5), (cond[0], l)
        del L, model
        gc.collect()
    torch.set_grad_enabled(True)
