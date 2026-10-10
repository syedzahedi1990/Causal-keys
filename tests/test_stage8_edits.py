"""Gate J-C-G0 of P-2026-10-10-J part C (docs/PREREGISTRATION.md): FP32 exactness of the edit pipeline at
Qwen2.5-0.5B-Instruct on the CPU (sdpa), 1e-4 in the logits unless stated, each against an independently computed
reference. The lemma: (i) writing h_S,l(p) at (p, l) equals the natural clamp KV(S) from l + 1 (l in 0, 3, 10); (ii) for
random vectors, B with the edit's captured K/V from l + 1 equals the edited run; (iii) prefix-captured tables equal
full-run tables (1e-5); (iv) the tables are identical in the four formats (exact). Then: the scored clamp rows against
ckeys.surface.score_reference under a separately built clamp; the T rows against the natural rows through the pipeline
(kappa_T = sigma); the row layout of every cell; the PAR / PERP and LEX / NONLEX algebra and kv_stats on synthetic tables
(known answers); the HeadSplice reader rows (the "all" row equals the plain K-clamp row, the empty row clean B); the
dictionary's io = "out" site (a block's forward-hook output equals that block's output and hidden_states[l + 1]) on a tiny
random Qwen2; the DAS-at-p code; the populations and the neutral sentences; the no_grad guard; and, on Prakash et al.'s
release ($PRAKASH_REPO, default /home/user/nix07/mind), the CAA arm with v = h_S - h_B at both positions equals ID.
The pipeline treats a skipped test as failing the gate."""
import gc

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ckeys import das_at, edits, neutral
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.interventions import blocks
from ckeys.story import LOCATIONS, PAIR_SWAP, pick_x
from ckeys.surface import FormSet, score, score_reference
from experiments import stage8_edits as s8

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 1e-4


def close(a, b, tol=TOL):
    return float((a - b).abs().max()) <= tol


@pytest.fixture(scope="module")
def ctx():
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval()
    P = edits.populations()
    c = P["E"][0]
    d = edits.prep(tok, c)
    c_ = dict(tok=tok, model=model, P=P, core=c, d=d, nL=len(blocks(model)))
    yield c_
    c_.clear()
    del model
    gc.collect()
    torch.set_grad_enabled(True)


def full_logits(model, ids, write=None, clamp=None):
    """Last-position log-softmax of a plain forward (optionally with a residual write or a clamp)."""
    import contextlib
    with contextlib.ExitStack() as st:
        if write is not None:
            st.enter_context(edits.write_resid(model, *write))
        if clamp is not None:
            st.enter_context(clamp_kv(model, *clamp))
        lg = model(ids, use_cache=False).logits[:, -1].float()
    return torch.log_softmax(lg, -1)


# --------------------------------------------------------------------------- populations, sentences
def test_populations_and_sentences():
    P = edits.populations()
    assert {k: len(v) for k, v in P.items()} == {"E": 80, "H": 200, "TSET": 1000, "THOLD": 100}
    U = edits.universe()
    import hashlib
    import json
    assert len(U) == 3981 and hashlib.sha256(json.dumps(sorted(list(t) for t in U)).encode()).hexdigest() == edits.U_SHA256
    keys = {k: {edits.key(c) for c in v} for k, v in P.items()}
    for k, v in keys.items():
        assert not (v & U), k
        for k2, v2 in keys.items():
            assert k == k2 or not (v & v2), (k, k2)
    assert all(edits.cond_E(c) for c in P["E"]) and all(PAIR_SWAP[c["source"]] != c["base"] for c in P["TSET"] + P["THOLD"])
    assert not {8101, 8102, 8103, 8104} & {0, 1, 7, 8, 9, 99, 101, 202, 20261011}   # seeds the pilots drew from
    assert neutral.sentences_sha256() == neutral.SENTENCES_SHA256 and len(neutral.SENTENCES) == 24
    tok = AutoTokenizer.from_pretrained(NAME)
    rep = neutral.check(tok)
    assert all(r["n"] == 24 for r in rep.values()), {f: r["n"] for f, r in rep.items()}
    for s in neutral.SENTENCES:
        assert s.count("{x}") == 1 and not any(w in s.replace("{x}", "") for w in LOCATIONS), s


def test_no_grad_guard(ctx):
    """Every forward of the edit path runs without autograd, even when the caller has it on (the pilot's memory bug)."""
    m, d, tok = ctx["model"], ctx["d"], ctx["tok"]
    seen = []
    h = blocks(m)[0].register_forward_hook(lambda *_: seen.append(torch.is_grad_enabled()))
    try:
        with torch.enable_grad():
            res, kv = edits.prefix_pass(m, d["prefix"], d["p"], resid_layers=[3], kv_layers=[5])
            lay = [5]
            rows = [("self", {5: kv[(5, "k")][0]}, {5: kv[(5, "v")][0]})] * 2
            edits.score_rows(m, d["ids"]["NONE"]["B"], FormSet(tok, LOCATIONS, {"E": (" ",)}), d["p"], rows, lay)
    finally:
        h.remove()
    assert seen and not any(seen) and not res[3].requires_grad


# --------------------------------------------------------------------------- the lemma
def test_lemma_natural_write_equals_kv_clamp(ctx):
    """(i) writing h_S,l(p) at (p, l) == the natural KV(S) clamp from l + 1 (tables from the S run), l in {0, 3, 10}."""
    m, d, nL = ctx["model"], ctx["d"], ctx["nL"]
    p = d["p"]
    for f in ("NONE", "P1"):
        idsB, idsS = d["ids"][f]["B"], d["ids"][f]["S"]
        for l in (0, 3, 10):
            res, _ = edits.prefix_pass(m, idsS[:, :p + 1], p, resid_layers=[l])
            lay = list(range(l + 1, nL))
            with capture_kv(m, [p], lay) as kvS:
                m(idsS, use_cache=False, logits_to_keep=1)
            a = full_logits(m, idsB, write=(l, p, res[l]))
            b = full_logits(m, idsB, clamp=([p], kvS, lay))
            assert close(a, b), (f, l, float((a - b).abs().max()))
            assert not close(a, full_logits(m, idsB), 1e-2), "the edit must change the run"


def test_lemma_random_edit_equals_clamp_rows(ctx):
    """(ii) for random vectors: B + the captured K/V of the edited prefix from l + 1 == the edited run (two rows, per-row
    tables in one batch)."""
    m, d, nL = ctx["model"], ctx["d"], ctx["nL"]
    p, l = d["p"], 3
    res, _ = edits.prefix_pass(m, d["prefix"], p, resid_layers=[l])
    g = torch.Generator().manual_seed(5)
    v = res[l][0] + 3.0 * torch.randn(2, res[l].shape[-1], generator=g)
    lay = list(range(l + 1, nL))
    _, kv = edits.prefix_pass(m, d["prefix"], p, kv_layers=lay, write=(l, v))
    ids = d["ids"]["POST"]["B"]
    tabs = {(q, ch): kv[(q, ch)][:, None] for q in lay for ch in "kv"}
    b = full_logits(m, ids.expand(2, -1), clamp=([p], tabs, lay))
    for r in range(2):
        a = full_logits(m, ids, write=(l, p, v[r:r + 1]))
        assert close(a[0], b[r]), (r, float((a[0] - b[r]).abs().max()))


def test_prefix_tables_equal_full_run_and_formats(ctx):
    """(iii) prefix tables == full-run tables (1e-5); (iv) the B prefix and its tables are identical in the four formats."""
    m, d, nL = ctx["model"], ctx["d"], ctx["nL"]
    p = d["p"]
    lay = list(range(nL))
    for f in ("NONE", "LETTER"):
        with capture_kv(m, [p], lay) as full:
            m(d["ids"][f]["S"], use_cache=False, logits_to_keep=1)
        _, pre = edits.prefix_pass(m, d["ids"][f]["S"][:, :p + 1], p, kv_layers=lay)
        scale = max(1.0, max(float(full[q].abs().max()) for q in full))
        assert max(float((full[q][:, 0] - pre[q]).abs().max()) for q in full) <= 1e-5 * scale
    tabs = {}
    for f in edits.FORMATS:
        assert torch.equal(d["ids"][f]["B"][:, :p + 1], d["prefix"])
        _, tabs[f] = edits.prefix_pass(m, d["ids"][f]["B"][:, :p + 1], p, kv_layers=[5, 20])
    assert all(torch.equal(tabs[f][q], tabs["NONE"][q]) for f in edits.FORMATS for q in tabs["NONE"])


# --------------------------------------------------------------------------- scoring and rows
def test_score_rows_against_reference(ctx):
    """score_rows' E and L fields (row minus in-chunk self) equal score_reference under a separately built single-row
    clamp; the self row's L equals a plain forward; chunking does not change a row."""
    m, tok, d, nL = ctx["model"], ctx["tok"], ctx["d"], ctx["nL"]
    p, l = d["p"], 7
    lay = list(range(l + 1, nL))
    _, kv = edits.prefix_pass(m, torch.cat([d["ids"]["NONE"][k][:, :p + 1] for k in ("B", "S", "X")]), p, kv_layers=lay)
    T = lambda i, ch: {q: kv[(q, ch)][i] for q in lay}  # noqa: E731
    rows = [("self", T(0, "k"), T(0, "v")), ("K_S", T(1, "k"), T(0, "v")), ("V_X", T(0, "k"), T(2, "v")), ("KV_S", T(1, "k"), T(1, "v"))]
    for f, fs, letters in (("P1", FormSet(tok, LOCATIONS, {"E": (" ", " The ")}), False),
                           ("LETTER", FormSet(tok, s8.edits.LETTERS, {"E": (" ", "")}), True)):
        ids = d["ids"][f]["B"]
        out = edits.score_rows(m, ids, fs, p, rows, lay, chunk=3, letters=letters)     # chunks: self + 2, self + 1
        words = s8.edits.LETTERS if letters else LOCATIONS
        # the reference: one plain causal forward per form sequence (score_reference), the four rows in one batch under a
        # clamp whose per-row tables are built here
        tabs = {(q, "k"): torch.stack([r[1][q] for r in rows])[:, None] for q in lay} | \
               {(q, "v"): torch.stack([r[2][q] for r in rows])[:, None] for q in lay}
        with clamp_kv(m, [p], tabs, lay):
            r = score_reference(m, ids.expand(len(rows), -1), fs)
        E = torch.stack([r["E"][w] for w in words], 1)
        Lr = torch.stack([r["L"][w] for w in words], 1)
        ref = {name: (E[i], Lr[i]) for i, (name, _, _) in enumerate(rows)}
        for name in ("K_S", "V_X", "KV_S"):
            dE = torch.tensor(out[name]["dE"])
            dL = torch.tensor(out[name]["dL"])
            assert close(dE, (ref[name][0] - ref["self"][0]).double()), (f, name)
            assert close(dL, (ref[name][1] - ref["self"][1]).double()), (f, name)
        plain = full_logits(m, ids)[0]
        cid = [tok(" " + w, add_special_tokens=False).input_ids[0] for w in words]
        assert close(torch.tensor(out["self"]["L"]), plain[cid].double())


def test_T_rows_equal_natural_rows_and_layout(ctx):
    """(viii) T (h_t written at (p, l)) through the pipeline: its tables equal the natural ones (1e-5) and its scored
    rows the natural rows (1e-4), so kappa_T = sigma; the cell layout of rows_for (the synthetic rows interpolate exactly;
    the components only in P1 / NONE; PAR and LEX only in NONE; K rows keep B's value, V rows B's key)."""
    m, tok, d, nL = ctx["model"], ctx["tok"], ctx["d"], ctx["nL"]
    p, l = d["p"], 7
    lay = list(range(l + 1, nL))
    names = ("B", "S", "X")
    res, kv = edits.prefix_pass(m, torch.cat([d["ids"]["NONE"][k][:, :p + 1] for k in names]), p, resid_layers=[l], kv_layers=lay)
    _, kvT = edits.prefix_pass(m, d["prefix"], p, kv_layers=lay, write=(l, res[l][1:3]))
    scale = max(1.0, max(float(kv[q].abs().max()) for q in kv))    # 1e-5 of the tables' scale (FP32 ulps at |K| ~ 200)
    assert max(float((kvT[q] - kv[q][1:3]).abs().max()) for q in kvT) <= 1e-5 * scale
    assert edits.kv_stats({q: kvT[q][0] for q in kvT}, {q: kv[q][1] for q in kv}, {q: kv[q][0] for q in kv}, lay)["nu"] < 1e-5
    tabB = s8.split_tables(kv, 0, lay)
    tabT = {"S": s8.split_tables(kv, 1, lay), "X": s8.split_tables(kv, 2, lay)}
    tab = {"T|S": s8.split_tables(kvT, 0, lay), "T|X": s8.split_tables(kvT, 1, lay)}
    rows = s8.rows_for("POST", ["T"], [], tab, tabB, tabT, lay)
    rows = [r for r in rows if r[0] == "self" or r[0].startswith(("T|", "nat|"))]
    out = edits.score_rows(m, d["ids"]["POST"]["B"], FormSet(tok, LOCATIONS, {"E": (" ", "")}), p, rows, lay, chunk=16)
    for C in ("K", "V", "KV"):
        for t in ("S", "X"):
            assert close(torch.tensor(out[f"T|{C}|{t}"]["dE"]), torch.tensor(out[f"nat|{C}|{t}"]["dE"])), (C, t)
    ID = {C: s8.id_contrast({k.replace(f"|{C}", ""): v for k, v in out.items() if f"|{C}|" in k}, z, d["iS"], d["iX"])
          for C in ("K", "V") for z in ("T",)}
    IDn = {C: s8.id_contrast({k.replace(f"|{C}", ""): v for k, v in out.items() if f"|{C}|" in k}, "nat", d["iS"], d["iX"]) for C in ("K", "V")}
    assert abs(ID["K"] / (ID["K"] + ID["V"]) - IDn["K"] / (IDn["K"] + IDn["V"])) <= 1e-4     # kappa_T = sigma
    # the layout
    inst, decomp = ["T", "R", "E1"], ["E1"]
    fake = {f"{z}|{t}": tabT[t] for z in inst + [f"{c}:E1" for c in ("PAR", "PERP", "LEX", "NONLEX")] + [f"{c}:nat" for c in ("LEX", "NONLEX")] for t in "SX"}
    for f in edits.FORMATS:
        rr = {r[0]: r for r in s8.rows_for(f, inst, decomp, fake, tabB, tabT, lay)}
        assert ("PERP:E1|K|S" in rr) == (f in ("P1", "NONE")) and ("LEX:E1|KV|S" in rr) == (f == "NONE")
        assert "R|K|S" not in rr and "R|KV|X" in rr and "PAR:E1|K|S" not in rr
        k = rr["E1|K|S"]
        assert all(torch.equal(k[2][q], tabB[(q, "v")]) and torch.equal(k[1][q], tabT["S"][(q, "k")]) for q in lay)
        v = rr["E1|V|X"]
        assert all(torch.equal(v[1][q], tabB[(q, "k")]) and torch.equal(v[2][q], tabT["X"][(q, "v")]) for q in lay)
        sk = rr["SK50|K|S"]
        assert all(torch.allclose(sk[1][q], tabB[(q, "k")] + 0.5 * (tabT["S"][(q, "k")] - tabB[(q, "k")])) for q in lay)
        assert all(torch.equal(sk[2][q], tabB[(q, "v")]) for q in lay)
        sv = rr["SV80|KV|X"]
        assert all(torch.equal(sv[1][q], tabT["X"][(q, "k")]) for q in lay)
        assert all(torch.allclose(sv[2][q], tabB[(q, "v")] + 0.8 * (tabT["X"][(q, "v")] - tabB[(q, "v")])) for q in lay)


def test_components_and_kv_stats_known_answers():
    g = torch.Generator().manual_seed(1)
    d, dn = torch.randn(32, generator=g), torch.randn(32, generator=g)
    par, perp, c = edits.par_perp(d, dn)
    assert torch.allclose(par + perp, d, atol=1e-6) and abs(float(perp @ dn)) < 1e-4 and abs(c - float(d @ dn / (dn @ dn))) < 1e-6
    mu = torch.randn(6, 32, generator=g)
    Q = edits.lexical_span(mu)
    assert Q.shape == (32, 5) and torch.allclose(Q.T @ Q, torch.eye(5), atol=1e-5)
    lex, non = edits.lex_split(d, Q)
    assert torch.allclose(lex + non, d, atol=1e-6) and float((Q.T @ non).abs().max()) < 1e-5
    assert float((lex - Q @ (Q.T @ lex)).abs().max()) < 1e-5
    for x in range(1, 6):   # the lexical differences lie in the span
        dx = mu[x] - mu[0]
        assert float(edits.lex_split(dx, Q)[1].norm()) < 1e-4 * float(dx.norm())
    r1, r2 = edits.random_dirs(32, [3, 7]), edits.random_dirs(32, [3, 7])
    assert all(torch.equal(r1[l], r2[l]) for l in (3, 7)) and torch.allclose(r1[3].norm(dim=1), torch.ones(6))
    lay = [1, 2]
    B = {(q, ch): torch.randn(8, generator=g) for q in lay for ch in "kv"}
    t = {k: v + torch.randn(8, generator=g) for k, v in B.items()}
    st = edits.kv_stats(t, t, B, lay)
    assert st["nu"] < 1e-9 and abs(st["cos_k"] - 1) < 1e-9 and abs(st["proj_v"] - 1) < 1e-9
    e2 = {k: B[k] + 2 * (t[k] - B[k]) for k in B}
    st = edits.kv_stats(e2, t, B, lay)
    assert abs(st["nu"] - 1) < 1e-6 and abs(st["proj_k"] - 2) < 1e-6 and abs(st["ratio_v"] - 2) < 1e-6
    ortho = {}
    for k in B:
        dn_ = t[k] - B[k]
        o = torch.randn(8, generator=g)
        ortho[k] = B[k] + o - (o @ dn_) / (dn_ @ dn_) * dn_
    st = edits.kv_stats(ortho, t, B, lay, {1: torch.tensor([0, 1, 2, 3])})
    assert abs(st["cos_k"]) < 1e-5 and abs(st["cos_v"]) < 1e-5 and abs(st["proj_k"]) < 1e-5
    assert "proj_k_readers" in st and edits.kv_stats(e2, t, B, lay, {9: torch.tensor([0])})["proj_k_readers"] != edits.kv_stats(e2, t, B, lay, {9: torch.tensor([0])})["proj_k_readers"]


# --------------------------------------------------------------------------- the reader rows
def test_reader_rows_headsplice(ctx):
    """(vii) the HeadSplice construction of J-C-READ: the "all" row (every head in every row sees K_S from l + 1, B's value
    pinned) equals the plain K-only clamp row; the empty row equals clean B; a blinded row differs from both."""
    import numpy as np
    from ckeys.headsplice import HeadSplice
    from ckeys.readerblind import KeyPin
    tok = AutoTokenizer.from_pretrained(NAME)
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval()
    d = edits.prep(tok, edits.populations()["E"][1])
    nL, H = len(blocks(m)), m.config.num_attention_heads
    p, on = d["p"], 8
    G, ids = s8.g_rows(tok, d["core"])
    assert torch.equal(ids, d["ids"]["P1"]["B"])
    lay = list(range(on, nL))
    _, kv = edits.prefix_pass(m, torch.cat([d["ids"]["NONE"][k][:, :p + 1] for k in ("B", "S")]), p, kv_layers=lay)
    KB, VB, KS = ({q: kv[(q, ch)][i] for q in lay} for ch, i in (("k", 0), ("v", 0), ("k", 1)))
    hooks3 = (HeadSplice(m), KeyPin(m, lay, "k"), KeyPin(m, lay, "v"))
    elig = [(l, h) for l in lay for h in range(H)]
    some = [elig[i] for i in np.random.default_rng(0).permutation(len(elig))[:20]]
    rows = [("self", KB, ([], False)), ("all", KS, (elig, True)), ("none", KS, ([], False)),
            ("blind", KS, (sorted(set(elig) - set(some)), True))]
    cid = [tok(" " + w, add_special_tokens=False).input_ids[0] for w in LOCATIONS]
    out = s8.reader_logprobs(m, hooks3, ids, p, G, rows, KB, VB, on, cid, chunk=4)
    tabs = {(q, "k"): KS[q][None, None] for q in lay} | {(q, "v"): VB[q][None, None] for q in lay}
    ref = full_logits(m, ids, clamp=([p], tabs, lay))[0, cid].double()
    clean = full_logits(m, ids)[0, cid].double()
    selfrow = torch.tensor(out["self"])
    assert close(selfrow, clean)
    assert close(torch.tensor(out["all"]) + selfrow, ref)
    assert close(torch.tensor(out["none"]), torch.zeros(6))
    assert not close(torch.tensor(out["blind"]) + selfrow, ref, 1e-3)
    del m
    gc.collect()


def test_sae_site_is_the_block_output():
    """C-8: the dictionaries were trained on io = "out" of model.layers[l]; our capture at block l (a forward hook on that
    module) equals the module's output and HF's hidden_states[l + 1] (tiny random Qwen2, 3 layers)."""
    cfg = AutoConfig.from_pretrained(NAME)
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 3, 64, 128
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    cfg.layer_types = ["full_attention"] * 3
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg).eval()
    ids = torch.randint(0, 1000, (1, 12))
    for l in range(2):
        got = {}
        h = m.model.layers[l].register_forward_hook(lambda _m, _i, o: got.__setitem__("o", (o[0] if isinstance(o, tuple) else o).clone()))
        with edits.capture_resid(m, [l], 5) as st:
            hs = m(ids, use_cache=False, output_hidden_states=True).hidden_states
        h.remove()
        assert torch.equal(st[l][0], got["o"][0, 5]) and torch.allclose(hs[l + 1][0, 5], st[l][0], atol=1e-6)


# --------------------------------------------------------------------------- E4 (DAS at p)
def test_das_at(ctx):
    m, tok, nL = ctx["model"], ctx["tok"], ctx["nL"]
    forms = das_at.form_ids(tok)
    assert all(len(f) >= 1 for f in forms) and [f[0] for f in forms] == [tok(" " + w, add_special_tokens=False).input_ids[0] for w in LOCATIONS]
    P = ctx["P"]
    pairs, hb, hs = s8.das_items(m, tok, P["TSET"][:17], 5)
    assert all(pr.target == LOCATIONS.index(PAIR_SWAP[c["source"]]) for pr, c in zip(pairs, P["TSET"][:17]))
    U0 = das_at.init_basis("pca", torch.stack([s - b for b, s in zip(hb, hs)]), m.config.hidden_size, 101)
    from ckeys.interventions import pca_basis
    assert torch.allclose(U0, pca_basis(torch.stack([s - b for b, s in zip(hb, hs)]), 16).float())
    assert U0.shape == (16, m.config.hidden_size) and torch.allclose(U0 @ U0.T, torch.eye(16), atol=1e-4)
    with pytest.raises(AssertionError):     # a PCA init needs at least 16 training pairs
        das_at.train_remap_at(m, pairs[:2], hb[:2], hs[:2], 5, forms, seed=101, log=lambda s: None)
    old = das_at.LR
    try:
        das_at.LR = 0.0                      # lr 0: the fitted basis is the initial one (the init is kept)
        U, losses = das_at.train_remap_at(m, pairs[:2], hb[:2], hs[:2], 5, forms, seed=102, log=lambda s: None)
    finally:
        das_at.LR = old
    Ur = das_at.init_basis("random", None, m.config.hidden_size, 102)
    assert torch.allclose(U, Ur, atol=1e-5) and len(losses) == 2 and all(x == x for x in losses)
    U2, losses2 = das_at.train_remap_at(m, pairs[:3], hb[:3], hs[:3], 5, forms, seed=102, log=lambda s: None)
    assert torch.allclose(U2 @ U2.T, torch.eye(16), atol=1e-4) and len(losses2) == 3
    assert not torch.is_grad_enabled()
    # with h_src = h_base the remap is the identity: the patched scores equal the clean scores
    with torch.no_grad():
        sc = das_at.patched_scores(m, pairs[0], 5, hb[0], hb[0], U2, forms)
        lg = m(pairs[0].base_ids, use_cache=False).logits[0, -1]
    assert close(sc, das_at.cand_scores(lg, forms))
    # the eval-time interchange is the same map
    from ckeys.interventions import fixed_subspace_interchange
    e = fixed_subspace_interchange(U2)(hb[0][None], hs[0][None])[0]
    assert torch.allclose(e, hb[0] + (hs[0] - hb[0]) @ U2.T @ U2, atol=1e-5)


def test_reader_sets_and_columns():
    class A:
        test, key, heads_dir = True, "qwen7", "results/gpu_stage6/heads"
    Hs, rand, elig, src = s8.reader_sets(A, 24, 14, 7)
    assert all(l > 7 for l, _ in Hs) and all(len(r) == len(Hs) and set(r) <= set(elig) for r in rand)
    assert s8.reader_sets(A, 24, 14, 7)[1] == rand
    cols = s8.reader_kv_columns([(9, 0), (9, 13), (10, 7)], 14, 2, 64)
    assert cols[9].tolist() == list(range(0, 128)) and cols[10].tolist() == list(range(64, 128))
    A.test = False
    Hs7, _, _, src7 = s8.reader_sets(A, 28, 28, 7)
    import json
    J = json.load(open("results/gpu_stage6/heads/Qwen2.5-7B-Instruct.json"))
    top = [tuple(c) for c in J["arms"]["P1"]["rankings"]["a3"][:J["provenance"]["kstar"]]]
    assert Hs7 == [c for c in top if c[0] > 7] and src7["kstar"] == 40 and len(Hs7) <= 40


def test_families_vectors():
    """The edit vectors from a calibration record: T, R, E1, E2, E4, E5 and the components, against direct formulas."""
    D = 16
    g = torch.Generator().manual_seed(3)
    l = 7
    T = {"mu1": {l: torch.randn(6, D, generator=g)}, "mu5": {f: {l: torch.randn(6, D, generator=g)} for f in ("EN", "FR")},
         "R": edits.random_dirs(D, [l])}
    T["Q"] = {l: edits.lexical_span(T["mu5"]["EN"][l])}

    class C:
        pass
    cal = C()
    cal.T, cal.J = T, {"e5_alpha": {"FR": {str(l): 2.0}, "NL": {str(l): 1.0}}}

    class A:
        key = "mistral7"
    U = torch.linalg.qr(torch.randn(D, 4, generator=g))[0].T
    fam = s8.Families(A, cal, {(l, 101): U, (l, 102): U}, {})
    h = {k: {l: torch.randn(D, generator=g)} for k in ("B", "S", "X", "piS", "piX")}
    vec = fam.vectors(0, l, h, 2, {"S": 4, "X": 5})
    hB = h["B"][l]
    assert torch.allclose(vec["E1|S"], hB + T["mu1"][l][4] - T["mu1"][l][2])
    assert torch.allclose(vec["E2|X"], hB + T["mu5"]["EN"][l][5] - T["mu5"]["EN"][l][2])
    assert torch.allclose(vec["E5FR|S"], hB + 2.0 * (T["mu5"]["FR"][l][4] - T["mu5"]["FR"][l][2]))
    assert torch.allclose(vec["E5NL|S"], hB + edits.lex_split(T["mu1"][l][4] - T["mu1"][l][2], T["Q"][l])[1])
    assert torch.allclose(vec["E4a|X"], hB + (h["piX"][l] - hB) @ U.T @ U, atol=1e-6)
    r = vec["R|S"] - hB
    assert abs(float(r.norm()) - float((T["mu1"][l][4] - T["mu1"][l][2]).norm())) < 1e-4
    assert torch.equal(vec["T|X"], h["X"][l])
    dz, dn = vec["E1|S"] - hB, h["S"][l] - hB
    assert torch.allclose(vec["PAR:E1|S"] - hB + vec["PERP:E1|S"] - hB, dz, atol=1e-5)
    assert torch.allclose(vec["LEX:nat|S"] - hB + vec["NONLEX:nat|S"] - hB, dn, atol=1e-5)
    assert set(fam.instances()) == {"T", "R", "E1", "E2", "E4a", "E4b", "E5FR", "E5NL"}


# --------------------------------------------------------------------------- Prakash et al.'s material
def test_prakash_caa_with_natural_difference_equals_id():
    """(ix) on Prakash et al.'s material: the CAA arm with mu(S) - mu(s_q) = h_S - h_B at [p, p+1] (block 12 of the 0.5B
    model) gives the ID arm's exchange rows (1e-4), so the CAA code differs from ID only through its vector."""
    from ckeys import causaltom as ct
    from experiments import prakash_caa as pc
    from experiments import prakash_swap as ps
    try:
        rel = ct.load()
    except AssertionError as ex:
        pytest.fail(f"release checkout unusable ({ex}); set PRAKASH_REPO to a checkout of {ct.RELEASE_URL} at {ct.RELEASE_SHA}")
    pairs = ct.pool(rel)
    tok = AutoTokenizer.from_pretrained(NAME)
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval()
    nL, dpt = len(blocks(m)), 12
    L = pc.caa_item(ps.item(tok, rel, pairs[0], "OPTIONS-AFTER", 0))
    with ps.capture_resid(m, [dpt], [L["id_span"]]) as rB:
        ps.logprobs(m, L["ids"]["B"])
    with ps.capture_resid(m, [dpt], [L["id_span"]]) as rS:
        ps.logprobs(m, L["ids"]["S"])
    meta = L["meta"]
    mu = {x: torch.zeros(2, m.config.hidden_size) for x in rel.states}
    mu[meta["S"]], mu[meta["s_q"]] = rS[dpt][0].float(), rB[dpt][0].float()
    cells = {c["arm"]: c for c in pc.run_pair(m, tok, L, dpt, mu, nL, True, lambda s: None)}
    for r in ps.ROWS:
        assert abs(cells["CAA"]["m"][r] - cells["ID"]["m"][r]) <= TOL, r
    assert cells["CAA"]["nu_CAA_ID"] < 1e-3
    del m
    gc.collect()
