"""Gate J-D-G0 of P-2026-10-10-J, part D (docs/PREREGISTRATION.md): FP32 exactness of the part-D hooks (ckeys/flag.py,
the batch helpers of experiments/stage8_flag.py, ckeys/questions.py) at Qwen2.5-0.5B-Instruct on the CPU, sdpa, before
any GPU model loads. Tolerance 1e-4 in the logits (1e-5 where a quantity is computed exactly), each hooked row compared
with an independently computed reference: a plain forward, a separately constructed clamp, or the shared stage-5/6 code.
  1 a zero injection equals the clean run; 2 an injection at layer l changes that layer's attention output by exactly the
  added vector at the row and nothing else, and positions before the row not at all; 3 batched rows equal single runs;
  4 directional ablation with mu = the row's own projection equals clean (1e-4), and after ablation the projection
  equals mu (1e-4);
  5 sum_h head_out(z_h) equals the o_proj output (OCap rows, head_out); 6 NONZERO-add composition: Inject(v) with a no-op
  HopSplice, with an empty-set HeadSplice, with HeadSplice "ablate" whose mu is its own o_proj input, and with a no-op
  all-heads HeadSplice splice, each equals Inject(v) alone (an add applied twice would differ); 7 HeadSplice with per-row
  K_S / K_X: the all-heads all-rows row equals the full key clamp of that row (the transfer batch of the run);
  8 the run's K/V clamp rows equal format_factorial.run_item's K_S, K_X, V_S rows; 9 the INLINE_CHAT item code equals
  ioi_factorial.run_item on INLINE; 10 the case-marginalised trie scoring with a HopSplice whose tables are the run's own
  K/V equals the plain trie scoring (no-op), and the trie equals the plain sum over one-token forms; 11 the flag of a story from write_pass
  equals the flag from an independent o_proj pre-hook capture; 12 the per-head hop-2 key splice with every head equals
  HopSplice ans_K; 13 trie scoring under a non-trivial HopSplice equals plain per-form forwards. The pipeline treats a
  skipped test as failing the gate."""
import types

import pytest
import torch

import experiments.format_factorial as ff
from ckeys import ioi
from ckeys import questions as Q
from ckeys.clamp import clamp_kv
from ckeys.flag import add_map, by_layer, flag_delta, head_out, row_masks, unit
from ckeys.interventions import blocks
from ckeys.story import LOCATIONS
from experiments import stage8_flag as s8
from experiments.ioi_factorial import run_item as ioi_run_item
from experiments.stage6_heads import configure_hop

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 1e-4


def close(a, b, tol=TOL):
    return torch.allclose(a, b, atol=tol, rtol=0)


def md(a, b):
    return float((a - b).abs().max())


@pytest.fixture(scope="module")
def ctx():
    torch.set_grad_enabled(False)
    a = types.SimpleNamespace(model=NAME, revision=None, dtype="float32", test=True)
    m = s8.M(a, "sdpa")
    P = s8.populations(check=True)["P"]
    core = P["E8"][0]
    d = s8.prep(m.tok, core, "P1")
    D = m.D
    g = torch.Generator().manual_seed(0)
    vec = {l: torch.randn(D, generator=g) for l in range(m.nL)}
    c = dict(m=m, P=P, core=core, d=d, vec=vec)
    yield c
    c.clear()


def plain(m, ids):
    return torch.log_softmax(m.model(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1)


def attn_out(m, layer):
    """A forward hook on self_attn of ``layer`` (registered after Inject's o_proj hook fires) capturing its output."""
    store = {}
    h = blocks(m.model)[layer].self_attn.register_forward_hook(lambda _m, _i, o: store.__setitem__("y", o[0].detach().clone()))
    return store, h


def test_zero_injection_equals_clean(ctx):
    m, d = ctx["m"], ctx["d"]
    plain(m, d["ids"]["B"])   # warm-up: the session's first CPU forward once differed by 1.4e-3 (not reproducible)
    ref = plain(m, d["ids"]["B"])
    z = {l: torch.zeros(m.D) for l in range(m.nL)}
    out, _, _ = s8.fwd(m, d["ids"]["B"], 1, add=add_map([[(d["row"]["X"], z, 1.0)]], range(m.nL)))
    assert close(out["lp"], ref, 1e-5), md(out["lp"], ref)


def test_injection_is_exactly_the_added_vector(ctx):
    m, d, vec = ctx["m"], ctx["d"], ctx["vec"]
    l, r = 5, d["row"]["X"]
    store, h = attn_out(m, l)
    hs = {}
    hh = blocks(m.model)[-1].register_forward_hook(lambda _m, _i, o: hs.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().clone()))
    try:
        s8.fwd(m, d["ids"]["B"], 1)
        y0, x0 = store["y"], hs["x"]
        s8.fwd(m, d["ids"]["B"], 1, add=add_map([[(r, {l: vec[l]}, 1.0)]], [l]))
        y1, x1 = store["y"], hs["x"]
    finally:
        h.remove()
        hh.remove()
    assert close(y1[0, r] - y0[0, r], vec[l], 1e-5), md(y1[0, r] - y0[0, r], vec[l])
    other = [t for t in range(d["T"]) if t != r]
    assert close(y1[0, other], y0[0, other], 1e-5)
    assert close(x1[0, :r], x0[0, :r], 1e-5)                   # positions before the row: unchanged (causal)
    assert md(x1[0, r], x0[0, r]) > 1e-3


def test_batched_equals_single(ctx):
    m, d, vec = ctx["m"], ctx["d"], ctx["vec"]
    caps = s8.runs_kv(m, d["ids"], ["B", "S", "X"], d["p"])
    Ls = [3, 7, 11]
    v = {l: vec[l] for l in Ls}
    rows = [("none", [], None), ("add", [(d["row"]["X"], v, 1.0)], None), ("move", s8.mv(v, d["row"]["S"], d["row"]["B"]), None),
            ("K_S", [], s8.ROWSRC["K_S"]), ("V_X+add", [(d["row"]["X"], v, -0.5)], s8.ROWSRC["V_X"])]
    kk, vv = s8.kv_clamp(d["p"], [r[2] for r in rows], caps, m.nL)
    out, _, _ = s8.fwd(m, d["ids"]["B"], len(rows), k=kk, v=vv, add=add_map([r[1] for r in rows], Ls))
    for i, (_, spec, src) in enumerate(rows):
        k1, v1 = s8.kv_clamp(d["p"], [src], caps, m.nL)
        one, _, _ = s8.fwd(m, d["ids"]["B"], 1, k=k1, v=v1, add=add_map([spec], Ls))
        assert close(out["lp"][i], one["lp"][0]), (rows[i][0], md(out["lp"][i], one["lp"][0]))


def test_ablation_mu_own_is_clean_and_sets_projection(ctx):
    m, d, vec = ctx["m"], ctx["d"], ctx["vec"]
    Ls, r = [4, 9], d["row"]["B"]
    u = {l: unit(vec[l]) for l in Ls}
    own = {}
    hk = [blocks(m.model)[l].self_attn.o_proj.register_forward_hook(lambda _m, _i, o, l=l: own.__setitem__(l, o[0, r].float().clone()))
          for l in Ls]
    try:
        ref = plain(m, d["ids"]["B"])
    finally:
        for h in hk:
            h.remove()
    rm = torch.zeros(1, d["T"], dtype=torch.bool)
    rm[0, r] = True
    proj = {l: (u[l], float(own[l] @ u[l]), rm) for l in Ls}
    out, _, _ = s8.fwd(m, d["ids"]["B"], 1, proj=proj)
    assert close(out["lp"], ref), md(out["lp"], ref)             # FP32 rounding of y - (c - mu) u: ~1e-5
    mu = {4: 0.37, 9: -1.25}
    store, h = attn_out(m, 9)
    try:
        s8.fwd(m, d["ids"]["B"], 1, proj={l: (u[l], mu[l], rm) for l in Ls})
    finally:
        h.remove()
    assert abs(float(store["y"][0, r] @ u[9]) - mu[9]) < 1e-4
    rows = [t for t in d["G"] if t != r]
    s2, h2 = attn_out(m, 9)
    try:
        s8.fwd(m, d["ids"]["B"], 1)
    finally:
        h2.remove()
    s3, h3 = attn_out(m, 9)
    try:
        s8.fwd(m, d["ids"]["B"], 1, proj={9: (u[9], mu[9], rm)})
    finally:
        h3.remove()
    assert close(s3["y"][0, rows], s2["y"][0, rows], 1e-5)       # unmasked rows untouched at the layer


def test_head_outputs_sum_to_o_proj(ctx):
    m, d = ctx["m"], ctx["d"]
    Ls = [2, 13]
    outs = {}
    hk = [blocks(m.model)[l].self_attn.o_proj.register_forward_hook(lambda _m, _i, o, l=l: outs.__setitem__(l, o[0].float().clone()))
          for l in Ls]
    try:
        _, _, z = s8.fwd(m, d["ids"]["B"], 1, z=(d["G"], Ls))
    finally:
        for h in hk:
            h.remove()
    for l in Ls:
        tot = head_out(m.W(l), z[l][0], list(range(m.H)), m.hd)
        assert close(tot, outs[l][d["G"]]), md(tot, outs[l][d["G"]])


def test_composition_nonzero_add(ctx):
    m, d, vec = ctx["m"], ctx["d"], ctx["vec"]
    Ls = [3, 7, 13]
    v = {l: 3.0 * unit(vec[l]) for l in Ls}
    spec = [[(d["row"]["X"], v, 1.0), (d["row"]["B"], v, -1.0)]]
    add = add_map(spec, Ls)
    ref, cap, zin = s8.fwd(m, d["ids"]["B"], 1, add=add, kv=(d["G"], range(m.nL), "kv"), z=(list(range(d["T"])), range(m.nL)))
    base, _, _ = s8.fwd(m, d["ids"]["B"], 1)
    assert md(ref["lp"], base["lp"]) > 1e-2                     # the add does something
    # (a) a no-op HopSplice: both passes see the injected run's own K/V at G; the answer row reads pass B
    kvt = {"base": {k: t[0] for k, t in cap.items()}, "ksrun": {k: t[0] for k, t in cap.items()}}
    configure_hop(m.hop, ["none", "ans_KV"], kvt, d["G"], d["T"], s8.ROUTE_SPEC)
    out, _, _ = s8.fwd(m, d["ids"]["B"], 2, add=add_map(spec * 2, Ls), hop=True)
    assert close(out["lp"][0], ref["lp"][0]) and close(out["lp"][1], ref["lp"][0]), md(out["lp"][1], ref["lp"][0])
    # (b) an empty-set HeadSplice
    out, _, _ = s8.fwd(m, d["ids"]["B"], 1, add=add, hs=dict(mode="splice", pos=d["p"], ks={l: torch.zeros(m.D) for l in range(m.nL)},
                                                              masks={}, mu=None))
    assert close(out["lp"], ref["lp"])
    # (c) HeadSplice "ablate" with mu = its own o_proj input (three o_proj calls per masked layer)
    cells = [(l, h) for l in (3, 7, 13, 20) for h in range(0, m.H, 3)]
    masks = row_masks([(cells, d["G"] + [d["T"] - 1])], d["T"], m.nL, m.H)
    mu = {l: zin[l][:, :].view(1, d["T"], m.H, m.hd) for l in masks}
    out, _, _ = s8.fwd(m, d["ids"]["B"], 1, add=add, hs=dict(mode="ablate", pos=d["p"], ks=None, masks=masks, mu=mu))
    assert close(out["lp"], ref["lp"]), md(out["lp"], ref["lp"])
    # (d) a no-op all-heads HeadSplice splice: the second pass sees the run's own key at p
    owk = s8.runs_kv(m, d["ids"], ["B"], d["p"])["B"]["k"]
    masks = row_masks([("all", list(range(d["T"])))], d["T"], m.nL, m.H)
    out, _, _ = s8.fwd(m, d["ids"]["B"], 1, add=add, hs=dict(mode="splice", pos=d["p"], ks=owk, masks=masks, mu=None))
    assert close(out["lp"], ref["lp"]), md(out["lp"], ref["lp"])


def test_transfer_allT_equals_full_key_clamp(ctx):
    m, d = ctx["m"], ctx["d"]
    caps = s8.runs_kv(m, d["ids"], ["B", "S", "X"], d["p"])
    keys = {"S": caps["S"]["k"], "X": caps["X"]["k"]}
    S = {"H": [[3, 0], [7, 7], [13, 2]], "rand": [[[1, 1]], [[2, 2]], [[5, 5]]]}
    tid = {"S": d["cid"][d["ix"]["S"]], "X": d["cid"][d["ix"]["X"]]}
    tr = s8.transfer(m, d["ids"]["B"], d["p"], keys, d["G"], [d["row"][k] for k in "BSX"], d["T"], S, tid)
    for r in ("S", "X"):
        tabs = {(l, "k"): keys[r][l][None, None] for l in range(m.nL)}
        with clamp_kv(m.model, [d["p"]], tabs, range(m.nL), "k"):
            lp = plain(m, d["ids"]["B"])[0]
        assert abs(tr[r]["allT"] - float(lp[tid["S"]] - lp[tid["X"]])) < TOL, (r, tr[r]["allT"])
        lp0 = plain(m, d["ids"]["B"])[0]
        assert abs(tr[r]["none"] - float(lp0[tid["S"]] - lp0[tid["X"]])) < TOL


def test_kv_clamp_rows_equal_run_item(ctx):
    m, d, core = ctx["m"], ctx["d"], ctx["core"]
    it = ff.run_item(m.model, m.tok, core, "P1", "direct", "cpu")
    caps = s8.runs_kv(m, d["ids"], ["B", "S", "X"], d["p"])
    names = ["none", "K_S", "K_X", "V_S"]
    kk, vv = s8.kv_clamp(d["p"], [s8.ROWSRC.get(n) for n in names], caps, m.nL)
    out, _, _ = s8.fwd(m, d["ids"]["B"], len(names), k=kk, v=vv)
    tid = {"S": d["cid"][d["ix"]["S"]], "B": d["cid"][d["ix"]["B"]], "X": d["cid"][d["ix"]["X"]]}
    for i, n in enumerate(names):
        row = it["m"][{"none": "ID", "K_S": "K_S", "K_X": "K_X", "V_S": "V_S"}[n] + "@0"]
        for t, j in tid.items():
            assert abs(float(out["lp"][i, j]) - row["lp"][t]) < TOL, (n, t)


def test_inline_chat_item_code_equals_ioi_factorial(ctx):
    m = ctx["m"]
    core = ctx["P"]["E_IOI_CAND"][0]
    ref = ioi_run_item(m.model, m.tok, core, "INLINE", True, None, "cpu")
    ids = ioi.encode_runs(m.tok, core, "INLINE", True)
    mine = Q.ioi_item_from_ids(m.model, m.tok, core, "INLINE", False, ids, "cpu")
    for k in ref["m"]:
        for t in ref["m"][k]["lp"]:
            assert abs(ref["m"][k]["lp"][t] - mine["m"][k]["lp"][t]) < 1e-5, (k, t)
    assert ioi.identity_measures(ref)["idK"] == pytest.approx(ioi.identity_measures(mine)["idK"], abs=1e-5)
    it = Q.ioi_run_item(m.model, m.tok, core, "INLINE_CHAT", True, None, "cpu")
    assert it is not None and it["floor_B"] < 1e-3 and it["floor_S"] < 1e-3


def test_trie_scoring_with_noop_hop(ctx):
    m, vec = ctx["m"], ctx["vec"]
    dp = s8.prep(m.tok, ctx["core"], "POST")
    fs = s8.fs_of(m.tok)
    assert len(fs) > 0, "this test needs multi-token forms"
    Ls = [3, 7]
    v = {l: 2.0 * unit(vec[l]) for l in Ls}
    spec = [[], [(dp["row"]["X"], v, 1.0)]]
    ref, cap, _ = s8.fwd(m, dp["ids"]["B"], 2, add=add_map(spec, Ls), kv=(dp["G"], range(m.nL), "kv"), fs=fs)
    kvt = {"base": {k: t[1] for k, t in cap.items()}, "ksrun": {k: t[1] for k, t in cap.items()}}
    configure_hop(m.hop, ["none", "ans_KV"], kvt, dp["G"], dp["T"], s8.ROUTE_SPEC)
    out, _, _ = s8.fwd(m, dp["ids"]["B"], 2, add=add_map([spec[1]] * 2, Ls), hop=True, fs=fs)
    assert close(out["E"][0], ref["E"][1]) and close(out["E"][1], ref["E"][1]), md(out["E"][1], ref["E"][1])
    # the trie score of the clean row equals a plain sum over single-token forms where every form is one token
    plain_lp = plain(m, dp["ids"]["B"])[0]
    for j, w in enumerate(LOCATIONS):
        seqs = list(fs.sets["E"][w])
        if all(len(s) == 1 for s in seqs):
            want = torch.logsumexp(torch.stack([plain_lp[s[0]] for s in seqs]), 0)
            assert abs(float(ref["E"][0, j]) - float(want)) < TOL, w


def test_flag_from_write_pass_matches_independent_capture(ctx):
    m, d = ctx["m"], ctx["d"]
    cells = [(3, 0), (3, 4), (7, 7), (13, 2), (13, 6)]
    byH = by_layer(cells)
    Ls = sorted(byH)
    ks = s8.runs_kv(m, d["ids"], ["S"], d["p"])["S"]["k"]
    rows = [d["row"]["S"], d["row"]["B"]]
    zb, zk = s8.write_pass(m, d, [("B", None, None), ("B", ks, d["p"])], rows, Ls)
    mine = flag_delta(m.Ws(Ls), zk, zb, 0, 1, byH, m.hd)
    store = {}
    hk = [blocks(m.model)[l].self_attn.o_proj.register_forward_pre_hook(lambda _m, x, l=l: store.setdefault(l, []).append(x[0][0].clone()))
          for l in Ls]
    try:
        plain(m, d["ids"]["B"])
        with clamp_kv(m.model, [d["p"]], {(l, "k"): ks[l][None, None] for l in range(m.nL)}, range(m.nL), "k"):
            plain(m, d["ids"]["B"])
    finally:
        for h in hk:
            h.remove()
    for l in Ls:
        Wl = m.W(l)
        ref = torch.zeros(m.D)
        for h in byH[l]:
            sl = slice(h * m.hd, (h + 1) * m.hd)
            zB, zK = store[l][0], store[l][1]
            ref += 0.5 * (Wl[:, sl] @ (zK[rows[0], sl] - zB[rows[0], sl]) + Wl[:, sl] @ (zB[rows[1], sl] - zK[rows[1], sl]))
        assert close(mine[l], ref, 1e-4), (l, md(mine[l], ref))


def test_hop2_key_splice_equals_hopsplice_answer_key(ctx):
    """The per-head hop-2 batch (J-D-HOP2): with every head listed, the answer row reading the clean run's keys at G
    through HeadSplice equals HopSplice's ans_K row (the answer row reads the clean K at G, its own V) under the same
    injection; the inj row equals the injection alone and the none row the clean run."""
    m, d, vec = ctx["m"], ctx["d"], ctx["vec"]
    Ls = [3, 7, 11, 13]
    v = {l: 3.0 * unit(vec[l]) for l in Ls}
    inj = [(d["row"]["X"], v, 1.0)]
    S = {"hop_top": [[3, 1], [7, 2], [13, 0]], "hop_rand": [[5, 5]]}
    cand = d["cid"]
    h = s8.hop2(m, d["ids"]["B"], d["T"], d["G"], inj, S, Ls, cand)
    _, cap, _ = s8.fwd(m, d["ids"]["B"], 2, add=add_map([[], inj], Ls), kv=(d["G"], range(m.nL), "kv"))
    kvt = {"base": {k: t[0] for k, t in cap.items()}, "ksrun": {k: t[1] for k, t in cap.items()}}
    configure_hop(m.hop, ["move", "ans_K"], kvt, d["G"], d["T"], s8.ROUTE_SPEC)
    ref, _, _ = s8.fwd(m, d["ids"]["B"], 2, add=add_map([inj, inj], Ls), hop=True)
    clean = plain(m, d["ids"]["B"])[0, cand]
    got = {k: torch.tensor(x) for k, x in h.items()}
    assert close(got["all"], ref["lp"][1, cand]), md(got["all"], ref["lp"][1, cand])
    assert close(got["inj"], ref["lp"][0, cand]) and close(got["none"], clean)
    assert md(got["all"], got["inj"]) > 1e-3 and md(got["top10"], got["inj"]) > 1e-6     # the splices act


def test_trie_scoring_hop_mask_extends_over_the_trie(ctx):
    """J-D6-ROUTE's batch: HopSplice with the clean run's K/V at G read by the answer row of an injected run, scored
    through the trie, equals plain causal forwards of prompt + each form in which the answer row and every form
    position read pass B (an independent reference for the trie + HopSplice + Inject path; at 0.5B the second tokens of
    the multi-token forms are near-certain, so the trie nodes' own reading moves the scores by < 1e-5)."""
    m, vec = ctx["m"], ctx["vec"]
    dp = s8.prep(m.tok, ctx["core"], "POST")
    fs = s8.fs_of(m.tok)
    assert any(len(s) > 1 for s in fs.seqs), "this test needs multi-token forms"
    Ls = [3, 7, 11, 13]
    v = {l: 4.0 * unit(vec[l]) for l in Ls}
    inj = [(dp["row"]["X"], v, 1.0)]
    ids, T, G = dp["ids"]["B"], dp["T"], dp["G"]
    _, cap, _ = s8.fwd(m, ids, 2, add=add_map([[], inj], Ls), kv=(G, range(m.nL), "kv"))
    kvt = {"base": {k: t[0] for k, t in cap.items()}, "ksrun": {k: t[1] for k, t in cap.items()}}
    spec = s8.ROUTE_SPEC | {"inj": s8.ROUTE_SPEC["move"]}
    configure_hop(m.hop, ["inj", "ans_KV"], kvt, G, T, spec)
    out, _, _ = s8.fwd(m, ids, 2, add=add_map([inj, inj], Ls), hop=True, fs=fs)
    seqlp = {}
    for s in fs.seqs:
        x = torch.cat([ids, torch.tensor([list(s[:-1])], dtype=ids.dtype)], 1) if len(s) > 1 else ids
        configure_hop(m.hop, ["ans_KV"], kvt, G, T, spec)
        mk = torch.zeros(1, x.shape[1], dtype=torch.bool)
        mk[0, T - 1:] = True                       # the answer row and every form position read pass B
        m.hop.mask, m.hop.active = mk, True
        m.inj.add, m.inj.active = add_map([inj], Ls), True
        try:
            lp = torch.log_softmax(m.model(x, use_cache=False).logits[0].float(), -1)
        finally:
            s8.reset(m)
        seqlp[s] = sum(float(lp[T - 1 + t, s[t]]) for t in range(len(s)))
    for j, w in enumerate(LOCATIONS):
        want = float(torch.logsumexp(torch.tensor([seqlp[s] for s in fs.sets["E"][w]]), 0))
        assert abs(float(out["E"][1, j]) - want) < TOL, (w, float(out["E"][1, j]), want)
    assert float((out["E"][1] - out["E"][0]).abs().max()) > 1e-3     # the route splice acts
