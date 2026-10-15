"""Stage 9, edit and dose arm (docs/stage9/SPEC.md 8, 9.1-9.2, 11.3, 11.5; 13.3): experiments/stage9_edits.py against
independent references in FP32 on the CPU (1e-4 in log-probabilities, 1e-5 in K/V).

Models: a tiny random Qwen2 (24 blocks, hidden 32, the Qwen2.5-0.5B tokenizer), where every row of eval_core is checked,
and Qwen2.5-0.5B-Instruct for the lemma and the all-position hook with real stop rules.
1. The lemma (0.5B): B with an E1 vector (and E1x{c}, E4b) written at (p, l) by ckeys.edits.write_resid in a full
   forward equals B with the edit prefix pass's tables clamped from l + 1 (E and L 1e-4 against the trie scores and, for
   one row per block, ckeys.surface.score_reference; tables 1e-5 of their scale), l in (3, 7), NONE, AFTER, LETTER,
   POST; T|S's tables equal the natural S tables bitwise (the padded 5-row shape) and T_check is 0.
   Every row of eval_core (tiny model; blocks 3 and 11; the four formats): each table row's E and L equal a batch-1
   forward under a separately built reference (the row's tables written out from SPEC 8.1 over K/V captured from full
   prompts; the residual edits by write_resid of vectors built here from full-prompt residuals), each CAA row's those
   under the test's own all-position hook; the generations equal greedy_reference of each row alone.
2. Dose tables: J{a} = B + a(t - B) exactly; r_K (proj_k) of J{a} and K{a} equals a and of V{a} equals 0 (1e-6); J1 = nat.
3. E4b: the vector equals fixed_subspace_interchange(U) with a random orthonormal U; the stored-basis loader asserts the
   pinned sha256 and the shape on the committed files (no model) and refuses a file with another hash.
4. CAA: greedy decoding with the KV cache under the all-position hook equals greedy_reference with the test's hook active
   in every cache-free forward; the trie scores under the hook equal score_reference under the test's hook; the hook at
   index p alone equals single-site E1 (scores 1e-4).
5. prep9: one p and an identical prefix through p in NONE, AFTER, LETTER, POST for every E9 core (the study tokenizers,
   skipped when not cached); the check stage on them; the calib and das pins on the committed files.
6. TEST_MODE subprocess runs of check and eval (n = 1) write every key of SPEC 11.3 and 11.5; resume by core; exit 3 at
   the deadline; the llama8 layout without E4b.
"""
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from ckeys import edits, fresh, stage9
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.generate import greedy_reference
from ckeys.interventions import blocks, capture, fixed_subspace_interchange
from ckeys.story import LOCATIONS, PAIR_SWAP, pick_x
from ckeys.surface import FormSet, score, score_reference
from experiments import stage9_edits as se
from experiments.stage8_edits import NAT_ROWS, edit_pass, natural_pass, split_tables

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL, KVTOL = 1e-4, 1e-5
ROOT = Path(__file__).resolve().parents[1]
TOKENIZERS = {"qwen7": "Qwen/Qwen2.5-7B-Instruct", "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
              "llama8": "unsloth/Meta-Llama-3.1-8B-Instruct"}     # llama8: the build's byte-identical tokenizer copy
HIDDEN = {"qwen7": 3584, "mistral7": 4096, "llama8": 4096}
torch.set_grad_enabled(False)


def tiny_model(nL=24):
    c = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=nL, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=8, max_position_embeddings=4096, tie_word_embeddings=False,
                    initializer_range=0.5)
    c._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(c).float().eval()


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(NAME)


@pytest.fixture(scope="module")
def tiny():
    return tiny_model()


@pytest.fixture(scope="module")
def qwen():
    return AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval()


@pytest.fixture(scope="module")
def e9():
    return stage9.edit_cores(), stage9.edit_items()


def close(a, b, tol=TOL):
    return torch.allclose(torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64), atol=tol, rtol=0)


def maxdiff(a, b):
    return float((torch.as_tensor(a, dtype=torch.float64) - torch.as_tensor(b, dtype=torch.float64)).abs().max())


# --------------------------------------------------------------------------- independent references
def full_kv(model, ids, p, nL):
    """{(l, ch): [D]} K/V at p of a full forward of ``ids`` (every layer)."""
    with capture_kv(model, [p], range(nL)) as t:
        model(ids, use_cache=False, logits_to_keep=1)
    return {k: v[0, 0] for k, v in t.items()}


def full_resid(model, ids, layers, p):
    with capture(model, layers, "resid") as st:
        model(ids, use_cache=False, logits_to_keep=1)
    return {l: st[l][0, p].float() for l in layers}


@contextlib.contextmanager
def add_after(model, l, p, v):
    """The test's own all-position steering: h[:, p:] += v at block l's output (full-sequence forwards only)."""
    def fn(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        assert h.shape[1] > p, "the reference runs full-sequence forwards only"
        h = h.clone()
        h[:, p:] = h[:, p:] + v.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hd = blocks(model)[l].register_forward_hook(fn)
    try:
        yield
    finally:
        hd.remove()


def alphas(kind):
    """(a_K, a_V) of an interpolation row written out from SPEC 8.1 (nat = J1; K1 / V1 keep B's other channel)."""
    if kind == "nat":
        return 1.0, 1.0
    for pat, f in ((r"J([0-9.]+)", lambda m: (float(m[1]), float(m[1]))),
                   (r"K([0-9.]+)V([0-9.]+)", lambda m: (float(m[1]), float(m[2]))),
                   (r"K([0-9.]+)", lambda m: (float(m[1]), 0.0)), (r"V([0-9.]+)", lambda m: (0.0, float(m[1])))):
        m = re.fullmatch(pat, kind)
        if m:
            return f(m)
    return None


def lerp(x, y, a):
    return x if a == 0.0 else y if a == 1.0 else x + a * (y - x)


class Ref:
    """Independent material of one core on one model: K/V at p and block outputs at p from full NONE prompts."""

    def __init__(self, model, d, mu1, U, nL, layers_resid):
        self.model, self.d, self.mu1, self.U, self.nL, self.p = model, d, mu1, U, nL, d["p"]
        ids = d["ids"]["NONE"]
        self.kv = {v: full_kv(model, ids[v], self.p, nL) for v in ("B", "S", "X")}
        self.h = {v: full_resid(model, ids[v], layers_resid, self.p) for v in NAT_ROWS}

    def idx(self, t):
        return self.d["iS"] if t == "S" else self.d["iX"]

    def delta(self, l, t):
        return self.mu1[l][self.idx(t)] - self.mu1[l][self.d["iB"]]

    def vec(self, l, kind, t):
        hB = self.h["B"][l]
        if kind == "E4b":
            U = self.U[l]
            return hB + (self.h["pi" + t][l] - hB) @ U.T @ U
        c = 1.0 if kind == "E1" else float(kind[3:])
        return hB + c * self.delta(l, t)

    def ctx(self, name, l):
        """The single-row reference context of a row (table rows, residual edits and CAA rows)."""
        m, p = self.model, self.p
        if name in ("self", "self2"):
            return contextlib.nullcontext()
        kind, t = name.split("|")
        if kind == "CAA":
            return add_after(m, l, p, self.delta(l, t))
        if kind.startswith("E1") or kind == "E4b":
            return edits.write_resid(m, l, p, self.vec(l, kind, t)[None])
        ak, av = alphas(kind)
        B, T = self.kv["B"], self.kv[t]
        layers = range(l + 1, self.nL)
        tab = {(q, "k"): lerp(B[(q, "k")], T[(q, "k")], ak)[None] for q in layers}
        tab |= {(q, "v"): lerp(B[(q, "v")], T[(q, "v")], av)[None] for q in layers}
        return clamp_kv(m, [p], tab, layers)


def ref_scores(model, ids, fs, ctx, reference=False):
    with ctx:
        r = (score_reference if reference else score)(model, ids, fs)
    return [float(r["E"][w][0]) for w in fs.words], [float(r["L"][w][0]) for w in fs.words]


def small_fs(tok, fmt, core, frames=(" ", " The ")):
    return se.letter_form_set(tok) if fmt == "LETTER" else FormSet(tok, LOCATIONS, {"E": frames}, names=edits.names_of(core))


# --------------------------------------------------------------------------- 1. the lemma at Qwen2.5-0.5B
LEMMA_ROWS = ["self", "E1|S", "E1|X", "E1x1.5|S", "E4b|X", "self2"]


def test_lemma_e1_write_equals_table_rows(qwen, tok, e9):
    model, nL = qwen, len(blocks(qwen))
    core, it = e9[0][0], e9[1][0]
    d = se.prep9(tok, core)
    p = d["p"]
    mu1 = se.test_means(model, tok)
    U = {l: se.test_basis(model.config.hidden_size) for l in (3, 7)}
    res, kv = natural_pass(model, d, [3, 7], nL)
    h = {k: {l: res[l][i] for l in (3, 7)} for i, k in enumerate(NAT_ROWS)}
    ref = Ref(model, d, mu1, U, nL, [3, 7])
    for l in (3, 7):
        assert all(close(h[v][l], ref.h[v][l], 1e-4) for v in NAT_ROWS)       # prefix pass = full-prompt residuals
        layers = list(range(l + 1, nL))
        vecs = se.vectors(l, h, mu1, U, d)
        vn = list(vecs)
        kvE = edit_pass(model, d, l, torch.stack([vecs[n] for n in vn]), layers)
        tabE = {n: split_tables(kvE, i, layers) for i, n in enumerate(vn)}
        tabB = split_tables(kv, 0, layers)
        tabT = {"S": split_tables(kv, 1, layers), "X": split_tables(kv, 2, layers)}
        # T|S: its tables equal the natural S tables bitwise (forwards of the natural pass's 5-row shape)
        assert all(torch.equal(tabE["T|S"][k], tabT["S"][k]) for k in tabT["S"])
        # the edit's tables equal the K/V of a full forward with the vector written at (p, l) (1e-5 of their scale)
        names = [n for n in vn if n != "T|S"]
        with edits.write_resid(model, l, p, torch.stack([ref.vec(l, n.split("|")[0], n.split("|")[1]) for n in names])):
            with capture_kv(model, [p], layers) as full:
                model(d["ids"]["POST"]["B"].expand(len(names), -1), use_cache=False, logits_to_keep=1)
        scale = max(1.0, max(float(v.abs().max()) for v in full.values()))
        for i, n in enumerate(names):
            dm = max(float((full[k][i, 0] - tabE[n][k]).abs().max()) for k in full)
            assert dm <= KVTOL * scale, (l, n, dm, scale)
        specs = [s for s in se.row_specs(l, True) if s[0] in LEMMA_ROWS]
        rows = se.build_rows(specs, tabB, tabT, tabE, layers)
        for f in stage9.EDIT_FORMATS:
            fs = small_fs(tok, f, core)
            ids = d["ids"][f]["B"]
            out = se.run_batch(model, tok, ids, fs, [r[0] for r in rows], it, f, se.table_ctx(model, p, rows, layers),
                               gen=False)
            for n in LEMMA_ROWS:
                E, L = ref_scores(model, ids, fs, ref.ctx(n, l))
                assert close(out[n]["E"], E) and close(out[n]["L"], L), (l, f, n, maxdiff(out[n]["E"], E))
            assert not close(out["E1|S"]["E"], out["self"]["E"], 1e-2), "the edit must change the scores"
            if f == ("AFTER" if l == 3 else "LETTER"):     # one plain forward per form
                E, L = ref_scores(model, ids, fs, ref.ctx("E1|X", l), reference=True)
                assert close(out["E1|X"]["E"], E) and close(out["E1|X"]["L"], L), (l, f)


# --------------------------------------------------------------------------- every row of eval_core (tiny model)
TINY_BLOCKS = (3, 11)
GEN_ALL = {"NONE@3", "LETTER@11"}
GEN_SOME = ["self", "nat|X", "E1|S", "E4b|X", "E1x0.75|X", "J0.65|S", "K0.8V1|X", "V1|S", "self2"]


@pytest.fixture(scope="module")
def tiny_rec(tiny, tok, e9):
    model, nL = tiny, len(blocks(tiny))
    core, it = e9[0][1], e9[1][1]
    mu1 = {l: 3.0 * m for l, m in se.test_means(model, tok).items()}     # a visible edit on the random model
    U = {l: se.test_basis(model.config.hidden_size) for l in stage9.EDIT_BLOCKS}
    keep = {}
    rec = se.eval_core(model, tok, 1, core, it, mu1, U, [], nL, TINY_BLOCKS, stage9.EDIT_FORMATS, True, keep=keep)
    d = se.prep9(tok, core)
    return rec, Ref(model, d, mu1, U, nL, list(TINY_BLOCKS)), keep, core, it


def test_record_layout_and_t_check(tiny_rec):
    rec, ref, keep, core, it = tiny_rec
    d = ref.d
    assert rec["index"] == 1 and rec["core"] == core and rec["p"] == d["p"] and rec["X"] == pick_x(core)
    assert (rec["iB"], rec["iS"], rec["iX"]) == (LOCATIONS.index(core["base"]), LOCATIONS.index(core["source"]),
                                                 LOCATIONS.index(pick_x(core)))
    assert rec["iPiS"] == LOCATIONS.index(PAIR_SWAP[core["source"]]) and rec["iPiX"] == LOCATIONS.index(PAIR_SWAP[pick_x(core)])
    assert rec["cluster"] == [core["base"], core["source"]]
    assert rec["T_check"] == {"3": 0.0, "11": 0.0}
    assert set(rec["cells"]) == {f"{f}@{l}" for f in stage9.EDIT_FORMATS for l in TINY_BLOCKS}
    for key, cell in rec["cells"].items():
        l = int(key.split("@")[1])
        assert list(cell) == se.row_names(l, True) + ["caa"] and list(cell["caa"]) == ["self", "CAA|S", "CAA|X"]
        letter = key.startswith("LETTER@")
        for r in list(cell.values())[:-1] + list(cell["caa"].values()):
            assert set(r) == {"E", "L", "gen", "ans", "how", "n_tok"} | ({"word"} if letter else set())
            assert len(r["E"]) == len(r["L"]) == 6
            assert r["ans"] in LOCATIONS + ("other",) and r["how"] in ("word", "letter", "none")
            if letter:
                assert r["word"] == fresh.parse(r["gen"], LOCATIONS)
    for l in TINY_BLOCKS:
        st = rec["stats"][str(l)]
        assert list(st) == [n for n in se.row_names(l, True) if n not in ("self", "self2")]
        keys = {"proj_k", "proj_v", "cos_k", "cos_v", "ratio_k", "ratio_v", "nu", "cos_resid", "ratio_resid"}
        assert all(set(v) == keys for v in st.values())
        for t in "SX":
            assert abs(st[f"nat|{t}"]["proj_k"] - 1) < 1e-5 and st[f"nat|{t}"]["nu"] < 1e-5
            assert abs(st[f"nat|{t}"]["cos_resid"] - 1) < 1e-5 and abs(st[f"nat|{t}"]["ratio_resid"] - 1) < 1e-5
            assert st[f"E1|{t}"]["cos_resid"] is not None and st[f"E4b|{t}"]["ratio_resid"] is not None
            if l == 3:
                assert st[f"J0.5|{t}"]["cos_resid"] is None and abs(st[f"J0.5|{t}"]["proj_k"] - 0.5) < 1e-5
                # the multipliers scale E1's residual displacement exactly
                r1 = st[f"E1|{t}"]["ratio_resid"]
                assert all(abs(st[f"E1x{c}|{t}"]["ratio_resid"] - c * r1) < 1e-4 for c in se.MULTS)
    # E1's delivered key ratio from tables captured independently (full forward under write_resid)
    l, layers = 3, list(range(4, ref.nL))
    with edits.write_resid(ref.model, l, d["p"], ref.vec(l, "E1", "S")[None]):
        with capture_kv(ref.model, [d["p"]], layers) as t:
            ref.model(d["ids"]["NONE"]["B"], use_cache=False, logits_to_keep=1)
    dE = torch.cat([(t[(q, "k")][0, 0] - ref.kv["B"][(q, "k")]).double() for q in layers])
    dN = torch.cat([(ref.kv["S"][(q, "k")] - ref.kv["B"][(q, "k")]).double() for q in layers])
    assert abs(rec["stats"]["3"]["E1|S"]["proj_k"] - float(dE @ dN / (dN @ dN))) < 1e-4


@pytest.mark.parametrize("fmt", stage9.EDIT_FORMATS)
def test_every_row_equals_its_reference(tiny_rec, tok, fmt):
    rec, ref, keep, core, it = tiny_rec
    model, d = ref.model, ref.d
    fs = se.form_set9(tok, fmt, core, [])
    assert len(fs) > 0
    ids = d["ids"][fmt]["B"]
    stop = stage9.letter_stop(tok, it) if fmt == "LETTER" else fresh.stop_fn(tok, LOCATIONS)
    moved = 0
    for l in TINY_BLOCKS:
        cell = rec["cells"][f"{fmt}@{l}"]
        rows = {**{n: r for n, r in cell.items() if n != "caa"}, **{n: r for n, r in cell["caa"].items() if n != "self"}}
        caa_self = cell["caa"]["self"]
        for n, r in list(rows.items()) + [("self", caa_self)]:
            E, L = ref_scores(model, ids, fs, ref.ctx(n, l))
            assert close(r["E"], E) and close(r["L"], L), (fmt, l, n, maxdiff(r["E"], E), maxdiff(r["L"], L))
            check_gen = f"{fmt}@{l}" in GEN_ALL or n in GEN_SOME or n.startswith("CAA")
            if check_gen:
                with ref.ctx(n, l):
                    g = greedy_reference(model, tok, ids, max_new=16, stop=stop)[0]
                text = fresh.gen_text(tok, ids[0].tolist(), g)
                assert (r["gen"], r["n_tok"]) == (text, len(g)), (fmt, l, n, r["gen"], text)
                w = fresh.parse(text, LOCATIONS)
                want = stage9.parse_letter(text, it) if fmt == "LETTER" else (w, "word" if w != "other" else "none")
                assert (r["ans"], r["how"]) == want
                assert r.get("word", None) == (w if fmt == "LETTER" else None)
            moved += not close(r["E"], cell["self"]["E"], 1e-3)
        assert close(cell["self"]["E"], cell["self2"]["E"], 1e-5)        # one batch: self and self2 agree
        assert close(caa_self["E"], cell["self"]["E"])                    # two batch shapes, each rounded to 1e-5
    assert moved > 20, moved                                                 # not vacuous
    if fmt == "NONE":   # one plain forward per form: a table row, a residual row and a CAA row
        cell = rec["cells"]["NONE@3"]
        for n, r in (("K0.5V1|X", cell["K0.5V1|X"]), ("E4b|S", cell["E4b|S"]), ("CAA|X", cell["caa"]["CAA|X"])):
            E, L = ref_scores(model, ids, fs, ref.ctx(n, 3), reference=True)
            assert close(r["E"], E) and close(r["L"], L), (n, maxdiff(r["E"], E))


# --------------------------------------------------------------------------- 2. dose tables
def synthetic_tables(layers, D=8, dtype=torch.bfloat16, seed=1):
    g = torch.Generator().manual_seed(seed)
    B = {(q, ch): torch.randn(D, generator=g).to(dtype) for q in layers for ch in "kv"}
    T = {t: {k: (v.float() + torch.randn(D, generator=g)).to(dtype) for k, v in B.items()} for t in "SX"}
    E = {}
    return B, T, E


def test_dose_tables_known_answers():
    layers = [4, 5, 6]
    B, T, _ = synthetic_tables(layers)
    for l, e4b in ((3, True), (7, False), (11, True), (15, False)):
        names = se.row_names(l, e4b)
        dose = l in stage9.DOSE_BLOCKS
        assert names[0] == "self" and names[-1] == "self2" and len(names) == (46 if dose else 8) - 2 * (not e4b)
        assert len(set(names)) == len(names)
        assert ("J0.65|S" in names) == dose and ("E4b|X" in names) == e4b and "J1|S" not in names and "J1.0|S" not in names
    want = {f"J{a}" for a in (0.25, 0.5, 0.65, 0.8, 0.9, 1.25)} | {"K0.5", "K0.8", "V0.5", "V0.8", "K0.5V1", "K0.8V1",
                                                                  "K1V0.5", "K1V0.8", "K1", "V1", "E1x0.5", "E1x0.75",
                                                                  "E1x1.5", "nat", "E1", "E4b"}
    assert {n.split("|")[0] for n in se.row_names(3, True)} - {"self", "self2"} == want
    tabE = {f"{z}|{t}": {k: (v.float() + 0.1).to(torch.bfloat16) for k, v in T[t].items()}
            for z in ("E1", "E4b", "E1x0.5", "E1x0.75", "E1x1.5") for t in "SX"}
    specs = se.row_specs(3, True)
    rows = {r[0]: r for r in se.build_rows(specs, B, T, tabE, layers)}
    f = lambda tab: {k: v.float() for k, v in tab.items()}  # noqa: E731
    Bf, Tf = f(B), {t: f(T[t]) for t in "SX"}
    for t in "SX":
        for q in layers:
            for a in se.ALPHAS_J:
                for j, ch in ((1, "k"), (2, "v")):
                    assert torch.equal(rows[f"J{a}|{t}"][j][q], Bf[(q, ch)] + a * (Tf[t][(q, ch)] - Bf[(q, ch)]))
            for a in se.ALPHAS_KV:
                Ik = Bf[(q, "k")] + a * (Tf[t][(q, "k")] - Bf[(q, "k")])
                Iv = Bf[(q, "v")] + a * (Tf[t][(q, "v")] - Bf[(q, "v")])
                assert torch.equal(rows[f"K{a}|{t}"][1][q], Ik) and torch.equal(rows[f"K{a}|{t}"][2][q], Bf[(q, "v")])
                assert torch.equal(rows[f"V{a}|{t}"][1][q], Bf[(q, "k")]) and torch.equal(rows[f"V{a}|{t}"][2][q], Iv)
                assert torch.equal(rows[f"K{a}V1|{t}"][1][q], Ik) and torch.equal(rows[f"K{a}V1|{t}"][2][q], Tf[t][(q, "v")])
                assert torch.equal(rows[f"K1V{a}|{t}"][1][q], Tf[t][(q, "k")]) and torch.equal(rows[f"K1V{a}|{t}"][2][q], Iv)
            assert torch.equal(rows[f"K1|{t}"][1][q], Tf[t][(q, "k")]) and torch.equal(rows[f"K1|{t}"][2][q], Bf[(q, "v")])
            assert torch.equal(rows[f"V1|{t}"][1][q], Bf[(q, "k")]) and torch.equal(rows[f"V1|{t}"][2][q], Tf[t][(q, "v")])
            assert torch.equal(rows[f"nat|{t}"][1][q], Tf[t][(q, "k")]) and torch.equal(rows[f"nat|{t}"][2][q], Tf[t][(q, "v")])
            assert torch.equal(rows[f"E1x0.5|{t}"][1][q], tabE[f"E1x0.5|{t}"][(q, "k")].float())
            for n in ("self", "self2"):
                assert torch.equal(rows[n][1][q], Bf[(q, "k")]) and torch.equal(rows[n][2][q], Bf[(q, "v")])
        # J1 = nat: B + 1 (t - B) is t up to rounding; the a = 1 table is t's own
        J1 = edits.interp(Bf, Tf[t], 1.0, layers, "k")
        assert all(torch.allclose(J1[q], Tf[t][(q, "k")], atol=1e-6) for q in layers)
        assert all(torch.equal(se.chan(("I", 1.0), "k", B, T[t], tabE, layers)[q], Tf[t][(q, "k")]) for q in layers)
    # r_K = proj_k: a for J{a} and K{a}, 0 for V{a}; proj_v the mirror image
    vecs = {n: torch.zeros(4) for n in tabE} | {"E1|S": -torch.ones(4), "E1|X": -torch.ones(4)}
    h = {v: {3: torch.zeros(4)} for v in NAT_ROWS}
    h["S"][3], h["X"][3] = torch.ones(4), -torch.ones(4)       # h_B = 0: E1|S points against S, E1|X along X
    st = se.row_stats(list(rows.values()), specs, B, T, layers, vecs, h, 3)
    for t in "SX":
        for a in se.ALPHAS_J:
            assert abs(st[f"J{a}|{t}"]["proj_k"] - a) < 1e-6 and abs(st[f"J{a}|{t}"]["proj_v"] - a) < 1e-6
            assert abs(st[f"J{a}|{t}"]["nu"] - abs(1 - a)) < 1e-5 and st[f"J{a}|{t}"]["cos_resid"] is None
        for a in se.ALPHAS_KV:
            assert abs(st[f"K{a}|{t}"]["proj_k"] - a) < 1e-6 and abs(st[f"K{a}|{t}"]["proj_v"]) < 1e-6
            assert abs(st[f"V{a}|{t}"]["proj_k"]) < 1e-6 and abs(st[f"V{a}|{t}"]["proj_v"] - a) < 1e-6
            assert abs(st[f"K{a}V1|{t}"]["proj_k"] - a) < 1e-6 and abs(st[f"K{a}V1|{t}"]["proj_v"] - 1) < 1e-6
            assert abs(st[f"K1V{a}|{t}"]["proj_k"] - 1) < 1e-6 and abs(st[f"K1V{a}|{t}"]["proj_v"] - a) < 1e-6
        assert abs(st[f"K1|{t}"]["proj_k"] - 1) < 1e-6 and abs(st[f"V1|{t}"]["proj_k"]) < 1e-6
        assert st[f"nat|{t}"]["proj_k"] == 1.0 and st[f"nat|{t}"]["cos_resid"] == 1.0 and st[f"nat|{t}"]["ratio_resid"] == 1.0
        assert st[f"E1|{t}"]["cos_resid"] == (-1.0 if t == "S" else 1.0) and st[f"E1|{t}"]["ratio_resid"] == 1.0
    assert "self" not in st and "self2" not in st


# --------------------------------------------------------------------------- 3. E4b and the stored files
def test_e4b_and_e1_vectors():
    D = 32
    g = torch.Generator().manual_seed(4)
    U0 = torch.linalg.qr(torch.randn(D, 16, generator=g))[0].T
    assert torch.allclose(U0 @ U0.T, torch.eye(16), atol=1e-5)
    mu1 = {l: torch.randn(6, D, generator=g) for l in stage9.EDIT_BLOCKS}
    h = {v: {l: torch.randn(D, generator=g) for l in stage9.EDIT_BLOCKS} for v in NAT_ROWS}
    d = {"iB": 2, "iS": 4, "iX": 0}
    for l in stage9.EDIT_BLOCKS:
        v = se.vectors(l, h, mu1, {l: U0}, d)
        hB = h["B"][l]
        for t, i in (("S", 4), ("X", 0)):
            assert torch.equal(v[f"E4b|{t}"], fixed_subspace_interchange(U0)(hB[None], h["pi" + t][l][None])[0])
            assert torch.allclose(v[f"E4b|{t}"], hB + (h["pi" + t][l] - hB) @ U0.T @ U0, atol=1e-5)
            assert torch.allclose(v[f"E1|{t}"], hB + mu1[l][i] - mu1[l][2], atol=1e-5)
            for c in se.MULTS:
                assert (f"E1x{c}|{t}" in v) == (l in stage9.DOSE_BLOCKS)
                if l in stage9.DOSE_BLOCKS:
                    assert torch.allclose(v[f"E1x{c}|{t}"], hB + c * (mu1[l][i] - mu1[l][2]), atol=1e-5)
        assert torch.equal(v["T|S"], h["S"][l])
        assert "E4b|S" not in se.vectors(l, h, mu1, None, d)
    U = se.test_basis(48)
    assert U.shape == (16, 48) and torch.allclose(U @ U.T, torch.eye(16), atol=1e-5) and torch.equal(U, se.test_basis(48))


def committed(rel):
    f = ROOT / rel
    if not f.exists():
        pytest.skip(f"{rel} is not in this copy")
    return f


@pytest.mark.parametrize("key", ["qwen7", "mistral7"])
def test_das_loader_pins(key, tmp_path):
    rel = f"{se.STAGE8C}/das/{key}.pt"
    f = committed(rel)
    pin = stage9.ARTEFACT_SHA256[rel]
    U = se.load_das(f, pin, hidden=HIDDEN[key])
    assert set(U) == set(stage9.EDIT_BLOCKS) and all(u.shape == (16, HIDDEN[key]) and u.dtype == torch.float32 for u in U.values())
    raw = torch.load(f, weights_only=False)
    assert all(torch.equal(U[l], raw[(l, 102)].float()) and not torch.equal(U[l], raw[(l, 101)].float()) for l in U)
    with pytest.raises(AssertionError, match="differs from the pin"):
        se.load_das(f, "0" * 64)
    with pytest.raises(AssertionError):
        se.load_das(f, pin, hidden=HIDDEN[key] + 1)
    bad = tmp_path / f"{key}.pt"
    shutil.copy(f, bad)
    with open(bad, "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(AssertionError, match="differs from the pin"):
        se.load_das(bad, pin)
    Ua, prov = se.das_of(Namespace(key=key, stage8c=se.STAGE8C), HIDDEN[key])
    assert prov == {"das_file": rel, "das_sha256": pin, "das_seed": 102} and all(torch.equal(Ua[l], U[l]) for l in U)
    assert se.das_of(Namespace(key="llama8", stage8c=se.STAGE8C)) == (None, {"das_file": None, "das_sha256": None,
                                                                             "das_seed": 102})


@pytest.mark.parametrize("key", ["qwen7", "mistral7", "llama8"])
def test_calib_loader_pins(key, tmp_path):
    rel = f"{se.STAGE8C}/calib/{key}.pt"
    f = committed(rel)
    c = se.load_calib(Namespace(key=key, stage8c=se.STAGE8C), HIDDEN[key])
    assert c["calib_sha256"] == stage9.ARTEFACT_SHA256[rel] and c["calib_file"] == rel and c["frames"] == []
    assert set(c["mu1"]) == set(stage9.EDIT_BLOCKS) and all(m.shape == (6, HIDDEN[key]) for m in c["mu1"].values())
    with pytest.raises(AssertionError):
        se.load_calib(Namespace(key=key, stage8c=se.STAGE8C), HIDDEN[key] + 1)
    (tmp_path / "calib").mkdir()
    shutil.copy(f, tmp_path / "calib" / f"{key}.pt")
    shutil.copy(f.with_suffix(".json"), tmp_path / "calib" / f"{key}.json")
    assert se.load_calib(Namespace(key=key, stage8c=str(tmp_path)))["calib_sha256"] == c["calib_sha256"]
    with open(tmp_path / "calib" / f"{key}.pt", "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(AssertionError, match="differs from the pin"):
        se.load_calib(Namespace(key=key, stage8c=str(tmp_path)))


# --------------------------------------------------------------------------- 4. the all-position CAA hook (0.5B)
@pytest.mark.parametrize("fmt", ["AFTER", "LETTER"])
def test_caa_hook(qwen, tok, e9, fmt):
    model, nL = qwen, len(blocks(qwen))
    core, it = e9[0][2], e9[1][2]
    d = se.prep9(tok, core)
    p, l = d["p"], 7
    mu1 = se.test_means(model, tok)
    delta = torch.zeros(3, model.config.hidden_size)
    delta[1] = 2.0 * (mu1[l][d["iS"]] - mu1[l][d["iB"]])
    delta[2] = 2.0 * (mu1[l][d["iX"]] - mu1[l][d["iB"]])
    names = ["self", "CAA|S", "CAA|X"]
    fs = small_fs(tok, fmt, core)
    ids = d["ids"][fmt]["B"]
    out = se.run_batch(model, tok, ids, fs, names, it, fmt, lambda: se.caa_hook(model, l, p, delta), gen=True)
    stop = se.stop9(tok, it, fmt)
    for i, n in enumerate(names):
        hk = add_after(model, l, p, delta[i])
        E, L = ref_scores(model, ids, fs, hk)
        assert close(out[n]["E"], E) and close(out[n]["L"], L), (fmt, n, maxdiff(out[n]["E"], E))
        with add_after(model, l, p, delta[i]):
            g = greedy_reference(model, tok, ids, max_new=16, stop=stop)[0]
        assert out[n]["gen"] == fresh.gen_text(tok, ids[0].tolist(), g) and out[n]["n_tok"] == len(g), (n, out[n]["gen"])
        assert (out[n]["ans"], out[n]["how"]) == se.answer9(out[n]["gen"], it, fmt)
        assert {k: out[n][k] for k in ("ans", "how", "word") if k in out[n]} == se.gen_fields(out[n]["gen"], it, fmt)
    assert not close(out["CAA|S"]["E"], out["self"]["E"], 1e-2)
    if fmt == "AFTER":     # the trie under the hook against one plain forward per form under the test's hook
        fs1 = small_fs(tok, fmt, core, frames=(" ",))
        got = se.run_batch(model, tok, ids, fs1, names, it, fmt, lambda: se.caa_hook(model, l, p, delta), gen=False)
        E, L = ref_scores(model, ids, fs1, add_after(model, l, p, delta[1]), reference=True)
        assert close(got["CAA|S"]["E"], E) and close(got["CAA|S"]["L"], L)
    # at index p alone the hook is the single-site E1 (written by the table pipeline from the edit prefix pass)
    res, kv = natural_pass(model, d, [l], nL)
    h = {k: {l: res[l][i]} for i, k in enumerate(NAT_ROWS)}
    mu2 = {l: 2.0 * mu1[l]}
    vecs = se.vectors(l, h, mu2, None, d)
    layers = list(range(l + 1, nL))
    vn = list(vecs)
    kvE = edit_pass(model, d, l, torch.stack([vecs[n] for n in vn]), layers)
    tabE = {n: split_tables(kvE, i, layers) for i, n in enumerate(vn)}
    tabB = split_tables(kv, 0, layers)
    tabT = {"S": split_tables(kv, 1, layers), "X": split_tables(kv, 2, layers)}
    specs = [s for s in se.row_specs(l, False) if s[0] in ("self", "E1|S", "E1|X")]
    rows = se.build_rows(specs, tabB, tabT, tabE, layers)
    e1 = se.run_batch(model, tok, ids, fs, [r[0] for r in rows], it, fmt, se.table_ctx(model, p, rows, layers), gen=False)
    at_p = se.run_batch(model, tok, ids, fs, names, it, fmt, lambda: se.caa_hook(model, l, p, delta, only_p=True), gen=False)
    for t in "SX":
        assert close(at_p[f"CAA|{t}"]["E"], e1[f"E1|{t}"]["E"]) and close(at_p[f"CAA|{t}"]["L"], e1[f"E1|{t}"]["L"]), t


def test_caa_decode_branch_tiny(tiny, tok, e9):
    """16-token generation (a random model never stops early): every cached decode step adds the vector."""
    model = tiny
    core, it = e9[0][3], e9[1][3]
    d = se.prep9(tok, core)
    p, l = d["p"], 3
    g = torch.Generator().manual_seed(9)
    delta = torch.cat([torch.zeros(1, 32), 4.0 * torch.randn(2, 32, generator=g)])
    n_tok = []
    for fmt in ("NONE", "LETTER"):
        ids = d["ids"][fmt]["B"]
        out = se.run_batch(model, tok, ids, se.form_set9(tok, fmt, core), ["self", "CAA|S", "CAA|X"], it, fmt,
                           lambda: se.caa_hook(model, l, p, delta), gen=True)
        for i, n in enumerate(["self", "CAA|S", "CAA|X"]):
            with add_after(model, l, p, delta[i]):
                gr = greedy_reference(model, tok, ids, max_new=16, stop=se.stop9(tok, it, fmt))[0]
            assert out[n]["gen"] == fresh.gen_text(tok, ids[0].tolist(), gr) and out[n]["n_tok"] == len(gr), n
            n_tok.append(len(gr))
        assert out["CAA|S"]["gen"] != out["self"]["gen"]
    assert sum(k == 16 for k in n_tok) >= 4, n_tok


# --------------------------------------------------------------------------- answers
class Pieces:
    def decode(self, ids):
        return "".join(ids)


def test_answers_and_stops(e9):
    it = e9[1][0]
    assert it.words == LOCATIONS and it.order == tuple(range(6))
    for text, want in (("A", ("box", "letter")), (" F)", ("closet", "letter")), ("**C**", ("shelf", "letter")),
                       (" The drawer", ("drawer", "word")), ("Bob", ("other", "none")), ("", ("other", "none"))):
        assert se.answer9(text, it, "LETTER") == want, text
    assert se.answer9(" In the baskets.", it, "AFTER") == ("basket", "word")
    assert se.answer9(" B", it, "NONE") == ("other", "none") and se.answer9(" Cabinet", it, "POST") == ("cabinet", "word")
    # LETTER rows also store the word parse: an article and a word is option A by SPEC 9.2, and the conflict is countable
    for text, want in ((" A closet", {"ans": "box", "how": "letter", "word": "closet"}),
                       (" A) box", {"ans": "box", "how": "letter", "word": "box"}),
                       ("**C**", {"ans": "shelf", "how": "letter", "word": "other"}),
                       (" The drawer", {"ans": "drawer", "how": "word", "word": "drawer"}),
                       ("Bob", {"ans": "other", "how": "none", "word": "other"})):
        assert se.gen_fields(text, it, "LETTER") == want, text
    assert se.gen_fields(" A closet", it, "AFTER") == {"ans": "closet", "how": "word"}       # no word field outside LETTER
    tk = Pieces()
    assert se.stop9(tk, it, "LETTER")(0, [" D", ")"]) and not se.stop9(tk, it, "AFTER")(0, [" D", ")"])
    assert se.stop9(tk, it, "NONE")(0, [" the", " box", "."]) and not se.stop9(tk, it, "NONE")(0, [" the", " bo"])


def test_out_default_follows_test_mode(monkeypatch):
    """A manual run without --out writes TEST files under results/gpu_stage9_test, real ones under results/gpu_stage9."""
    monkeypatch.delenv("TEST_MODE", raising=False)
    base = ["--stage", "check", "--key", "qwen7", "--model", NAME]
    assert se.args_of(base).out == str(ROOT / "results/gpu_stage9")
    a = se.args_of(base + ["--test"])
    assert a.out == str(ROOT / "results/gpu_stage9_test") and a.tag == "TEST_qwen7" and a.n == 2
    assert se.args_of(base + ["--out", "elsewhere"]).out == "elsewhere"
    monkeypatch.setenv("TEST_MODE", "1")
    assert se.args_of(base).out == str(ROOT / "results/gpu_stage9_test")
    assert se.args_of(base + ["--out", "elsewhere"]).out == "elsewhere"


# --------------------------------------------------------------------------- 5. prep9 and the check on the study tokenizers
def _tok(name):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return AutoTokenizer.from_pretrained(name)
    except OSError as ex:
        pytest.skip(f"tokenizer {name} not cached: {type(ex).__name__}")


def test_prep9_qwen05(tok, e9):
    for core in e9[0][:10]:
        d = se.prep9(tok, core)
        p = d["p"]
        for v, loc in d["vals"].items():
            ref = {f: edits.prompt_ids(tok, core, f, loc) for f in stage9.EDIT_FORMATS}
            assert all(torch.equal(ref[f], d["ids"][f][v]) for f in ref)
            assert all(torch.equal(ref[f][0, :p + 1], ref["NONE"][0, :p + 1]) for f in ref)
            if v != "B":
                assert all(int((ref[f][0] != d["ids"][f]["B"][0]).sum()) == (loc != core["base"]) for f in ref)
    with pytest.raises(AssertionError):
        se.prep9(tok, e9[0][0], formats=("AFTER", "LETTER"))


@pytest.mark.parametrize("key", list(TOKENIZERS))
def test_prep9_every_core_cached_tokenizer(key, e9):
    tok = _tok(TOKENIZERS[key])
    ps = set()
    for core in e9[0]:
        d = se.prep9(tok, core)
        p = d["p"]
        ps.add(p)
        for f in stage9.EDIT_FORMATS:      # written out again: B's ids differ from each value's at p only
            for v in ("S", "X", "piS", "piX"):
                diff = (d["ids"][f][v][0] != d["ids"][f]["B"][0]).nonzero().flatten().tolist()
                assert diff == [p], (key, f, v, diff)
                assert torch.equal(d["ids"][f][v][0, :p + 1], d["ids"]["NONE"][v][0, :p + 1])
    assert len(e9[0]) == 80 and ps


@pytest.mark.parametrize("key", list(TOKENIZERS))
def test_check_stage_cached_tokenizer(key, tmp_path, monkeypatch):
    _tok(TOKENIZERS[key])
    committed(f"{se.STAGE8C}/calib/{key}.pt")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(TOKENIZERS[key])
    except OSError as ex:
        pytest.skip(f"config of {TOKENIZERS[key]} not cached: {type(ex).__name__}")
    rc = se.main(["--stage", "check", "--key", key, "--model", TOKENIZERS[key], "--out", str(tmp_path)])
    rep = json.loads((tmp_path / "check" / f"{key}_edits.json").read_text())
    assert rc == 0 and rep["pass"] and not rep["fails"] and not rep["skipped"], rep["fails"]
    P = rep["provenance"]
    assert P["n_items"] == 80 and P["calib_sha256"] == stage9.ARTEFACT_SHA256[f"{se.STAGE8C}/calib/{key}.pt"]
    assert (P["das_sha256"] is not None) == (key in se.E4B_KEYS) and P["frames"] == [] and P["hidden"] == HIDDEN[key]
    assert rep["n_cores"] == 80 and rep["p_positions"] and rep["rows"]["3"] == se.row_names(3, key in se.E4B_KEYS)


# --------------------------------------------------------------------------- 6. TEST_MODE files
PROV = {"args", "git_commit", "torch", "transformers", "python", "device", "wrapper", "stage9", "arm", "stage", "git_dirty",
        "freeze_commit", "predictions_sha256", "spec_sha256", "started_utc", "finished_utc", "model", "model_key",
        "revision", "verified", "test_mode", "population", "population_sha256", "n_items", "skipped_items", "formats",
        "formats_not_run", "onsets", "kv_only_onsets", "max_new", "reserve_min", "timings", "calib_file", "calib_sha256",
        "das_file", "das_sha256", "das_seed", "blocks", "dose_blocks", "frames"}
PROV_RUN = PROV | {"dtype", "attn_implementation", "n_layers", "hidden", "n_heads", "n_kv_heads"}
CORE = {"index", "core", "X", "p", "iB", "iS", "iX", "iPiS", "iPiX", "cluster", "T_check", "stats", "cells"}


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory, tok):
    d = tmp_path_factory.mktemp("tiny") / "model"
    tiny_model().save_pretrained(d)
    tok.save_pretrained(d)
    return d


def cli(tiny_dir, out, key, *args, deadline=None):
    env = dict(os.environ, PYTHONPATH=str(ROOT), TEST_MODE="1", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    env.pop("STAGE8_DEADLINE", None)
    env.pop("STAGE9_FREEZE", None)
    if deadline is not None:
        env["STAGE8_DEADLINE"] = str(deadline)
    cmd = [sys.executable, "-m", "experiments.stage9_edits", "--key", key, "--model", str(tiny_dir), "--out", str(out),
           "--n", "1", *args]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_test_mode_files(tiny_dir, tmp_path):
    out = tmp_path / "out"
    rc, log = cli(tiny_dir, out, "qwen7", "--stage", "check")
    assert rc == 0, log
    chk = json.loads((out / "check" / "TEST_qwen7_edits.json").read_text())
    assert {"provenance", "pass", "fails", "skipped"} <= set(chk) and chk["pass"] and not chk["fails"] and not chk["skipped"]
    assert PROV <= set(chk["provenance"]) and chk["provenance"]["stage"] == "check" and chk["provenance"]["arm"] == "edits"
    assert chk["provenance"]["calib_file"] == chk["provenance"]["das_file"] == "TEST" and chk["provenance"]["test_mode"]
    # the deadline before the first core: exit 3, nothing run
    rc, log = cli(tiny_dir, out, "qwen7", "--stage", "eval", deadline=time.time() - 60)
    assert rc == 3, log
    ev0 = json.loads((out / "edits" / "TEST_qwen7.json").read_text())
    assert ev0["cores"] == [] and ev0["provenance"]["cores_not_run"] == [0]
    rc, log = cli(tiny_dir, out, "qwen7", "--stage", "eval")
    assert rc == 0, log
    ev = json.loads((out / "edits" / "TEST_qwen7.json").read_text())
    assert {"provenance", "population", "rows", "cores"} <= set(ev) and ev["population"] == "E9"
    P = ev["provenance"]
    assert PROV_RUN <= set(P) and P["population_sha256"] == stage9.POP_SHA256["E9"] and P["n_items"] == 1
    assert P["calib_file"] == "TEST" and P["das_file"] == "TEST" and P["das_seed"] == 102 and P["e4b"] is True
    assert P["blocks"] == [3, 7, 11, 15] and P["dose_blocks"] == [3, 7] and P["formats"] == list(stage9.EDIT_FORMATS)
    assert P["formats_not_run"] == [] and P["cores_not_run"] == [] and P["n_done"] == 1 and P["freeze_commit"] is None
    assert P["dtype"] == "torch.float32" and P["attn_implementation"] == "sdpa" and P["n_layers"] == 24
    assert ev["rows"] == {str(l): se.row_names(l, True) for l in stage9.EDIT_BLOCKS}
    assert len(ev["rows"]["3"]) == 46 and len(ev["rows"]["11"]) == 8
    (c,) = ev["cores"]
    assert set(c) == CORE and c["core"] == stage9.edit_cores()[0] and c["T_check"] == {str(l): 0.0 for l in stage9.EDIT_BLOCKS}
    assert set(c["cells"]) == {f"{f}@{l}" for f in stage9.EDIT_FORMATS for l in stage9.EDIT_BLOCKS}
    for key, cell in c["cells"].items():
        l = key.split("@")[1]
        assert list(cell)[:-1] == ev["rows"][l] and set(cell["caa"]) == {"self", "CAA|S", "CAA|X"}
        assert all({"E", "L", "gen", "ans", "how"} <= set(r) for r in list(cell.values())[:-1] + list(cell["caa"].values()))
        assert all(("word" in r) == key.startswith("LETTER@") for r in list(cell.values())[:-1] + list(cell["caa"].values()))
    assert set(c["stats"]) == {"3", "7", "11", "15"} and "J0.25|X" in c["stats"]["7"] and "J0.25|X" not in c["stats"]["11"]
    # a rerun keeps the core
    rc, log = cli(tiny_dir, out, "qwen7", "--stage", "eval")
    assert rc == 0 and "1 cores kept" in log, log
    assert json.loads((out / "edits" / "TEST_qwen7.json").read_text())["cores"] == ev["cores"]
    # llama8: no E4b basis, the rows without E4b
    rc, log = cli(tiny_dir, out, "llama8", "--stage", "eval", "--blocks", "3", "--formats", "LETTER")
    assert rc == 0, log
    ev8 = json.loads((out / "edits" / "TEST_llama8.json").read_text())
    P8 = ev8["provenance"]
    assert P8["das_file"] is None and P8["e4b"] is False and P8["formats_not_run"] == ["NONE", "AFTER", "POST"]
    assert ev8["rows"] == {"3": se.row_names(3, False)} and len(ev8["rows"]["3"]) == 44
    assert set(ev8["cores"][0]["cells"]) == {"LETTER@3"} and "E4b|S" not in ev8["cores"][0]["cells"]["LETTER@3"]
