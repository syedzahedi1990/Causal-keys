"""Identity edits at the writing token and their clamp rows (stage 8, part C of preregistration J).

An edit writes one vector into the residual stream at the writing token p, at the output of 0-based block l. B and S
differ only at p, so (the lemma, tests/test_stage8_edits.py):
  (i) an edit confined to (p, l) reaches every other position only through p's key and value in blocks >= l + 1; hence
      B with the edit's K/V written at p from block l + 1 equals the edited run, and the key-only and value-only rows are
      exact;
  (ii) writing h_S,l(p) gives S's K/V at p from l + 1, i.e. the natural clamp KV(S) from onset l + 1.
Every edit is therefore applied as a clamp row from tables captured in a prefix pass (positions 0..p of the B prompt with
the vector written at (p, l)); the prefix through p is the same in the four formats, so the tables are the same tensors in
LETTER, P1, POST and NONE, and only the readers after p differ.

Populations (rule G6: disjoint by full core tuple from U and from each other; U = the union of make_cores(1000,
Random(s)) for s = 0..3). A stream is make_cores(1, rng) drawn repeatedly from one Random(seed) (the same cores as one
make_cores(n, rng) call); a population takes the first cores of its stream that are in none of the excluded sets and not
repeated, and satisfy its condition:
  E      Random(8101): pi(S) != B and pi(X) != B (X = story.pick_x, pi = story.PAIR_SWAP); the first 80 (136 draws;
         24 ordered (base, source) clusters)
  H      Random(8102): no condition, excluding E; the first 200 (201 draws); H_fit = the first 150, H_cal = the last 50
  TSET   Random(8103): pi(S) != B, excluding E and H; the first 1,000 (DAS training); THOLD = the next 100 (held out)
Hashes: sha256(json.dumps([[*tuple] for each core in order])), pinned in POP_SHA256 (U in U_SHA256).

Families (vectors written at (p, l); t is the target value S or X of the story, B its base value; mu are means over H_fit):
  T       h_t,l(p)                                    the equivalence control (= the natural clamp by (ii))
  R       h_B + |mu1(t) - mu1(B)| r_t / |r_t|         r_t a fixed Gaussian direction (Generator seed 8104, per layer and value)
  E1      h_B + mu1(t) - mu1(B)                       CAA-in: mu1 = mean h_x,l(p) over H_fit with each value x written at p
  E2      h_B + mu2(t) - mu2(B)                       CAA-out: mu2 = mean block-l output at " x" in the 24 neutral sentences
  E3      the SAE edit of ckeys.sae (Qwen2.5-7B only)
  E4      h_B + U^T U (h_pi(t),l - h_B)               the rank-16 remap at p (ckeys.das_at), seeds 101 (PCA init) and 102
  E5      h_B + alpha (mu5(form(t)) - mu5(form(B)))   the French, German or synonym form of the value (ckeys.neutral)
  E5 NL   h_B + alpha NONLEX(mu1(t) - mu1(B))         E1 without its component in the English lexical span
Components of a displacement d = h_E - h_B against the story's natural displacement dn = h_t - h_B and the 5-dimensional
English lexical span L_l = span{mu2(x) - mu2(box) : x != box} (orthonormal basis Q_l):
  PAR = h_B + c dn, c = <d, dn> / <dn, dn>;  PERP = h_B + d - c dn;  LEX = h_B + Q Q^T d;  NONLEX = h_B + d - Q Q^T d.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import random

import torch

from .clamp import capture_kv, clamp_kv
from .encoding import LETTERS, encode, raw_prompt
from .interventions import _out_tensor, _with_tensor, blocks, hooks
from .story import LOCATIONS, PAIR_SWAP, make_cores, pick_x, record
from .surface import FRAMES_E_FIXED, FormSet, score

STORY_FIELDS = ("agent", "other", "object", "distractor", "initial", "distractor_location", "base", "source")
FORMATS = ("LETTER", "P1", "POST", "NONE")
LETTER_FRAMES = (" ", "", " **", "**")
SEEDS = {"E": 8101, "H": 8102, "TSET": 8103}
N_POP = {"E": 80, "H": 200, "TSET": 1000, "THOLD": 100}
N_HFIT = 150
U_SHA256 = "abd1f0530a3d08a2058743f59102f8dd6dd360af06fc65dd3277c5b5eb176c3d"
POP_SHA256 = {"E": "a23465a211577f4c6e9efe78d4c7a588b598c05437406001145a420fec9d8c1b",
              "H": "8cc62cefc9865a475f51fd554ece829981b4b46dcdaa7ae66e9261cae3b6b584",
              "TSET": "06c56f93dce02d93a957a4e78556f4694ea3d994385a80212a2da1edd3bff16f",
              "THOLD": "136b7d940050222b1737d20df4aba78e1e327eac16ab86b50730b471d5e261f4"}
R_SEED = 8104


# --------------------------------------------------------------------------- populations
def key(c: dict) -> tuple:
    return tuple(c[f] for f in STORY_FIELDS)


def pop_hash(cores) -> str:
    return hashlib.sha256(json.dumps([list(key(c)) for c in cores]).encode()).hexdigest()


def universe() -> set:
    U = set()
    for s in range(4):
        U |= {key(c) for c in make_cores(1000, random.Random(s))}
    return U


def stream(seed: int):
    rng = random.Random(seed)
    while True:
        yield make_cores(1, rng)[0]


def _take(seed, n, excluded, cond):
    out, seen = [], set()
    for c in stream(seed):
        k = key(c)
        if k in excluded or k in seen:
            continue
        seen.add(k)
        if cond(c):
            out.append(c)
            if len(out) == n:
                return out


def cond_E(c):
    return PAIR_SWAP[c["source"]] != c["base"] and PAIR_SWAP[pick_x(c)] != c["base"]


def cond_T(c):
    return PAIR_SWAP[c["source"]] != c["base"]


def populations(check: bool = True) -> dict:
    """{"E", "H", "TSET", "THOLD"}: lists of cores (hashes asserted when ``check``)."""
    U = universe()
    E = _take(SEEDS["E"], N_POP["E"], U, cond_E)
    H = _take(SEEDS["H"], N_POP["H"], U | {key(c) for c in E}, lambda c: True)
    T = _take(SEEDS["TSET"], N_POP["TSET"] + N_POP["THOLD"], U | {key(c) for c in E} | {key(c) for c in H}, cond_T)
    P = {"E": E, "H": H, "TSET": T[:N_POP["TSET"]], "THOLD": T[N_POP["TSET"]:]}
    if check:
        for k, v in P.items():
            assert pop_hash(v) == POP_SHA256[k], f"population {k}: {pop_hash(v)} != pinned {POP_SHA256[k]}"
    return P


# --------------------------------------------------------------------------- prompts
def prompt_ids(tok, core: dict, fmt: str, loc: str) -> torch.Tensor:
    rec = record(core, "direct", loc)
    return encode(tok, raw_prompt(fmt, rec["story"], rec["query"]), prefill="Answer:")


def names_of(core: dict) -> dict:
    return {"a": core["agent"], "b": core["other"], "o": core["object"], "d": core["distractor"]}


def prep(tok, core: dict, values=None) -> dict:
    """Ids of every format for each value written at p (default: B, S, X, pi(S), pi(X)); p; the shared prefix.
    Asserts one length per format, a single differing position p, the same p in every format and an identical prefix
    through p in every format (so the edit tables are the same tensors in all four formats)."""
    X = pick_x(core)
    vals = values or {"B": core["base"], "S": core["source"], "X": X, "piS": PAIR_SWAP[core["source"]], "piX": PAIR_SWAP[X]}
    ids = {f: {k: prompt_ids(tok, core, f, v) for k, v in vals.items()} for f in FORMATS}
    p = None
    for f in FORMATS:
        L = {v.shape for v in ids[f].values()}
        assert len(L) == 1, (f, core, L)
        B = ids[f]["B"][0]
        for k, v in ids[f].items():
            if vals[k] == vals["B"]:
                continue
            d = (v[0] != B).nonzero().flatten().tolist()
            assert len(d) == 1, (f, k, d)
            p = d[0] if p is None else p
            assert d[0] == p, (f, k, d, p)
    pre = ids["NONE"]["B"][:, :p + 1]
    for f in FORMATS:
        for k in vals:
            assert torch.equal(ids[f][k][:, :p + 1], ids["NONE"][k][:, :p + 1]), (f, k, "prefix through p differs")
    return {"core": core, "X": X, "vals": vals, "ids": ids, "p": p, "prefix": pre,
            "iB": LOCATIONS.index(vals["B"]), "iS": LOCATIONS.index(core["source"]), "iX": LOCATIONS.index(X)}


def form_set(tok, fmt: str, core: dict, frames=()) -> FormSet:
    """The E form set of one item: the letter forms under LETTER, else the 32 fixed frames plus the discovered frames
    (instantiated with the item's names)."""
    if fmt == "LETTER":
        return FormSet(tok, LETTERS, {"E": LETTER_FRAMES})
    return FormSet(tok, LOCATIONS, {"E": tuple(FRAMES_E_FIXED) + tuple(frames)}, names=names_of(core))


# --------------------------------------------------------------------------- hooks and captures
@contextlib.contextmanager
def write_resid(model, layer: int, pos: int, vecs: torch.Tensor):
    """At the output of block ``layer`` set out[r, pos] = vecs[r] ([R, D]) for every batch row (full-sequence passes
    that reach ``pos`` only)."""
    def fn(_m, _i, out):
        h = _out_tensor(out)
        if h.shape[1] <= pos:
            return out
        h = h.clone()
        h[:, pos] = vecs.to(h.device, h.dtype)
        return _with_tensor(out, h)
    with hooks([blocks(model)[layer].register_forward_hook(fn)]):
        yield


@contextlib.contextmanager
def capture_resid(model, layers, pos: int):
    """Yields {layer: [R, D]} block outputs at ``pos`` (registered after any write hook on the same block, so it sees the
    written value)."""
    store, hs = {}, []
    for l in layers:
        def fn(_m, _i, out, l=l):
            h = _out_tensor(out)
            if h.shape[1] > pos:
                store[l] = h[:, pos].detach().clone()
        hs.append(blocks(model)[l].register_forward_hook(fn))
    with hooks(hs):
        yield store


@torch.no_grad()
def prefix_pass(model, prefix: torch.Tensor, p: int, resid_layers=(), kv_layers=(), write=None, chunk: int = 128):
    """One prefix pass (positions 0..p) per row. ``prefix`` [R or 1, p+1]; ``write`` = (layer, vecs [R, D]) or None.
    Returns ({layer: [R, D]} residuals at p, {(layer, ch): [R, D]} K/V at p). Rows are run in chunks of ``chunk``."""
    dev = next(model.parameters()).device
    R = write[1].shape[0] if write is not None else prefix.shape[0]
    res, kv = {}, {}
    for a in range(0, R, chunk):
        b = min(R, a + chunk)
        ids = (prefix if prefix.shape[0] > 1 else prefix.expand(R, -1))[a:b].to(dev)
        with contextlib.ExitStack() as st:
            if write is not None:
                st.enter_context(write_resid(model, write[0], p, write[1][a:b]))
            r = st.enter_context(capture_resid(model, resid_layers, p))
            k = st.enter_context(capture_kv(model, [p], kv_layers))
            model(ids, use_cache=False, logits_to_keep=1)
        for l, v in r.items():
            res.setdefault(l, []).append(v.float().cpu())
        for q, v in k.items():
            kv.setdefault(q, []).append(v[:, 0].cpu())
    return {l: torch.cat(v) for l, v in res.items()}, {q: torch.cat(v) for q, v in kv.items()}


# --------------------------------------------------------------------------- means
@torch.no_grad()
def class_means(model, tok, cores, layers) -> dict:
    """E1: {layer: [6, D]} mean over ``cores`` of h_x,l(p) with each of the six values x written at p (NONE prompt;
    the prefix is format-independent)."""
    D = model.config.hidden_size
    acc = {l: torch.zeros(6, D, dtype=torch.float64) for l in layers}
    for c in cores:
        d = prep(tok, c, {"B": c["base"]} | {f"v{i}": x for i, x in enumerate(LOCATIONS) if x != c["base"]})
        pre = torch.cat([d["ids"]["NONE"][k][:, :d["p"] + 1] for k in d["vals"]])
        res, _ = prefix_pass(model, pre, d["p"], resid_layers=layers)
        order = [LOCATIONS.index(v) for v in d["vals"].values()]
        for l in layers:
            acc[l][order] += res[l].double()
    return {l: (v / len(cores)).float() for l, v in acc.items()}


@torch.no_grad()
def lexical_means(model, tok, family: str, layers, sentences=None, min_sentences=None) -> dict:
    """E2 (family "EN") and E5 ("FR", "DE", "SYN"): {layer: [6, D]} mean block-l output at the last token of " form"
    over the neutral sentences (indices ``sentences``; default every sentence clean in this tokenizer)."""
    from . import neutral
    idx = neutral.clean_sentences(tok, family) if sentences is None else sentences
    assert len(idx) >= (neutral.MIN_SENTENCES if min_sentences is None else min_sentences), (family, len(idx))
    D = model.config.hidden_size
    acc = {l: torch.zeros(6, D, dtype=torch.float64) for l in layers}
    groups = {}                       # prompts of one length run as one batch (prefix through the form's last token)
    for i in idx:
        for x, form in enumerate(neutral.FORMS[family]):
            ids, pos = neutral.sentence_ids(tok, neutral.SENTENCES[i], form)
            groups.setdefault(pos, []).append((x, ids[0, :pos + 1]))
    for pos, items in groups.items():
        res, _ = prefix_pass(model, torch.stack([t for _, t in items]), pos, resid_layers=layers)
        for j, (x, _) in enumerate(items):
            for l in layers:
                acc[l][x] += res[l][j].double()
    return {l: (v / len(idx)).float() for l, v in acc.items()}


def lexical_span(mu_en: torch.Tensor) -> torch.Tensor:
    """Q [D, 5]: orthonormal basis of span{mu2(x) - mu2(box) : x != box}."""
    M = torch.stack([mu_en[i] - mu_en[0] for i in range(1, 6)], 1).double()
    Q, _ = torch.linalg.qr(M)
    return Q.float()


def random_dirs(D: int, layers) -> dict:
    """{layer: [6, D]} unit Gaussian directions, one per value, Generator seed 8104, drawn layer by layer in order."""
    g = torch.Generator().manual_seed(R_SEED)
    out = {}
    for l in layers:
        r = torch.randn(6, D, generator=g)
        out[l] = r / r.norm(dim=1, keepdim=True)
    return out


# --------------------------------------------------------------------------- vectors and components
def par_perp(d: torch.Tensor, dn: torch.Tensor):
    c = float((d.double() @ dn.double()) / (dn.double() @ dn.double()))
    par = c * dn
    return par, d - par, c


def lex_split(d: torch.Tensor, Q: torch.Tensor):
    lex = Q @ (Q.T @ d)
    return lex, d - lex


def cos(a, b) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


# --------------------------------------------------------------------------- table statistics
def kv_stats(tab_e, tab_t, tab_b, layers, heads_k=None) -> dict:
    """Distances of an edit's K/V at p from the natural ones over ``layers`` (all KV heads concatenated):
    nu = |[K;V]_E - [K;V]_t| / |[K;V]_t - [K;V]_B| (one pooled ratio over the concatenated blocks), the K and V cosines
    of the displacements, their norm ratios, and the projections of the edit's displacement on the natural one,
    <dE, dN> / <dN, dN>, for K (also over the KV-head columns ``heads_k`` {layer: column index tensor}, the reader
    heads' KV groups) and V (exploratory O1)."""
    dE = {ch: torch.cat([(tab_e[(q, ch)] - tab_b[(q, ch)]).double().flatten() for q in layers]) for ch in "kv"}
    dN = {ch: torch.cat([(tab_t[(q, ch)] - tab_b[(q, ch)]).double().flatten() for q in layers]) for ch in "kv"}
    num = sum(float(((dE[ch] - dN[ch]) ** 2).sum()) for ch in "kv")
    den = sum(float((dN[ch] ** 2).sum()) for ch in "kv")
    out = {"nu": (num / max(den, 1e-30)) ** 0.5}
    for ch in "kv":
        n = float((dN[ch] ** 2).sum())
        out[f"cos_{ch}"] = cos(dE[ch], dN[ch])
        out[f"ratio_{ch}"] = float(dE[ch].norm()) / max(n ** 0.5, 1e-30)
        out[f"proj_{ch}"] = float(dE[ch] @ dN[ch]) / max(n, 1e-30)
    if heads_k is not None:
        qs = [q for q in layers if q in heads_k]
        if qs:
            e = torch.cat([(tab_e[(q, "k")] - tab_b[(q, "k")]).double()[heads_k[q]] for q in qs])
            nn_ = torch.cat([(tab_t[(q, "k")] - tab_b[(q, "k")]).double()[heads_k[q]] for q in qs])
            out["proj_k_readers"] = float(e @ nn_) / max(float(nn_ @ nn_), 1e-30)
        else:
            out["proj_k_readers"] = float("nan")
    return out


# --------------------------------------------------------------------------- clamp rows and scoring
def interp(tab_b: dict, tab_t: dict, lam: float, layers, ch: str) -> dict:
    """The synthetic table B + lam (t - B) of channel ``ch`` (J-C-G8)."""
    return {q: tab_b[(q, ch)] + lam * (tab_t[(q, ch)] - tab_b[(q, ch)]) for q in layers}


@torch.no_grad()
def score_rows(model, ids: torch.Tensor, fs: FormSet, p: int, rows: list, layers, chunk: int = 64, letters: bool = False):
    """Score clamp rows on one prompt. ``rows``: list of (name, ktab {layer: [D]}, vtab {layer: [D]}); rows[0] is the self
    row (B's own tables), which is placed first in every chunk so that every row is differenced against an in-batch self
    row. Returns {"self": {"E": [6], "L": [6], "mass": m, "argmax": i}, name: {"dE": [6], "dL": [6], "mass": m, "argmax": i}}
    with dE, dL the row's E and L scores minus those of its chunk's self row; ``letters`` reads the six letters."""
    words = list(LETTERS) if letters else list(LOCATIONS)
    layers = list(layers)
    self_row, rest = rows[0], rows[1:]
    out = {}
    step = max(1, chunk - 1)
    for a in range(0, max(1, len(rest)), step):
        ch = [self_row] + rest[a:a + step]
        tabs = {}
        for q in layers:
            tabs[(q, "k")] = torch.stack([r[1][q] for r in ch])[:, None]
            tabs[(q, "v")] = torch.stack([r[2][q] for r in ch])[:, None]
        with clamp_kv(model, [p], tabs, layers):
            res = score(model, ids.expand(len(ch), -1), fs)
        E = torch.stack([res["E"][w] for w in words], 1).double().cpu()
        L = torch.stack([res["L"][w] for w in words], 1).double().cpu()
        mass = E.exp().sum(1)
        for i, r in enumerate(ch):
            if i == 0:
                if "self" not in out:
                    out["self"] = {"E": E[0].tolist(), "L": L[0].tolist(), "mass": float(mass[0]), "argmax": int(E[0].argmax())}
                continue
            out[r[0]] = {"dE": (E[i] - E[0]).tolist(), "dL": (L[i] - L[0]).tolist(), "mass": float(mass[i]),
                         "argmax": int(E[i].argmax())}
    return out
