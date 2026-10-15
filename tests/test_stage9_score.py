"""analysis/stage9_score.py on synthetic results in the layouts of docs/stage9/SPEC.md section 11, with known answers
(SPEC 13.4): every prediction's SUPPORTED, SUPPORTED AT THE POINT, NOT SUPPORTED and NOT EVALUABLE paths; l_c with
censoring; l_b recomputation and its mismatch; the "2 of 3" rule; the falsifier triggers and the pre-written readings;
the AUC with ties and its evaluability floor; psi in nats and in logits; containment primary with strict printed;
competence filtering; the bootstrap (two-stage structure, determinism, paired across formats); the provenance checks
(MISMATCH, exit 2: two commits, a TEST_ file outside TEST, a population hash, a frames / calib / das sha256, a freeze
commit or note sha256 differing from COMMIT.txt, an eval that read another lb file, a row grid other than SPEC 7.1 / 8.1);
a deadline-partial edits file (NOT EVALUABLE, no MISMATCH) and deadline-partial stories, passages and calibration files
(listed as PARTIAL, the parts that ran evaluable); the CONTROLS floor against the clean B run, the in-batch determinism
and the LETTER letter/word conflict counts; the pytest-log gate; a missing model; the per-model P2 entry (the stories
part; P2p counts 2 of 3); the main reading withheld when a falsifier part cannot be read.

The generator writes, per model, item records whose generated answers are fixed by a scenario: for each (format, row) a
share r of the competent items answers S (edits: the target t), a share o "other", the rest B, assigned by a fixed rank of
the items, so every rate is known exactly. The default scenario follows the claim: free-form formats flip at every late
onset; listed formats (AFTER, LETTER; OPTA) flip only before the onset LC of the model, and their early window flips from
LC on; the edit rows transfer to the listed formats only at block 3 and only above r_K = 0.8.
"""
import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
import stage9_score as SC  # noqa: E402
from stage9_parts import checks as CK  # noqa: E402
from stage9_parts import data as DA  # noqa: E402
from stage9_parts import preds as PR  # noqa: E402
from stage9_parts import stats as ST  # noqa: E402

from ckeys import stage9 as s9  # noqa: E402
from ckeys.story import LOCATIONS, pick_x  # noqa: E402

KEYS = ("mistral7", "llama8", "qwen7")
NL = {"mistral7": 32, "llama8": 32, "qwen7": 28}
LC = {"mistral7": 8, "llama8": 10, "qwen7": 14}
HEAD, F, NOTE, SPEC = "d1" * 20, "a1" * 20, "b" * 64, "c" * 64
REV = {k: v["revision"] for k, v in json.loads((ROOT / "scripts/stage8_models.json").read_text())["models"].items()}
PINS = s9.ARTEFACT_SHA256
S8A = {k: json.loads((ROOT / f"results/gpu_stage8a/factorial/{k}.json").read_text())["provenance"]["population_sha256"]
       for k in KEYS}
FAST_NB = 2000
REAL_NB = ST.NB


@pytest.fixture(autouse=True)
def _fast_boot(monkeypatch):
    """Fewer resamples in the verdict-path tests (the determinism test and one full run use the real 10,000)."""
    monkeypatch.setattr(ST, "NB", FAST_NB)


@pytest.fixture
def tmp_path(tmp_path, request):  # noqa: F811  (overrides pytest's tmp_path and uses it)
    """pytest's tmp_path, deleted after the test unless the test failed: a synthetic run (up to about 60 MB a copy of the
    full run) would otherwise stay on disk for the session, and pytest keeps the last sessions' directories."""
    failed = request.session.testsfailed
    yield tmp_path
    if request.session.testsfailed == failed:
        shutil.rmtree(tmp_path, ignore_errors=True)


# --------------------------------------------------------------------------- generator
def story_rows(L):
    on = list(range(2, L - 1, 2))
    kv = [l for l in (4, 8, 12, 16) if l < L]
    return (["none", "KV@0"] + [f"late@{l}" for l in on] + [f"early@{l}" for l in on] + ["K@0", "V@0"]
            + [f"Klate@{l}" for l in kv] + [f"Vlate@{l}" for l in kv] + ["none2"])


def passage_rows(L):
    on = [round(x * L) for x in (0.15, 0.3, 0.45, 0.6)]
    return (["none", "KV@0"] + [f"late@{l}" for l in on] + [f"early@{l}" for l in on] + ["K@0", "V@0"]
            + [f"Klate@{l}" for l in on] + [f"Vlate@{l}" for l in on] + ["none2"])


def default_rate(key, fmt, row, listed):
    """The share of answers that flip (S / the target) for a story or passage window row."""
    if row in ("none", "none2"):
        return 0.0
    if row == "KV@0":
        return 1.0
    kind, l = row.split("@")
    l = int(l)
    lc = LC[key] if fmt not in ("NOM", "OPTB", "OPTA") else [round(x * NL[key]) for x in (0.15, 0.3, 0.45, 0.6)][1]
    return {"late": (1.0 if l < lc else 0.0) if listed else 1.0, "early": (1.0 if l >= lc else 0.0) if listed else 0.0,
            "K": 0.9 if listed else 0.1, "V": 0.05 if listed else 0.95,
            "Klate": 1.0 if listed and l < lc else 0.0, "Vlate": 0.0 if listed else 1.0}[kind]


def rk_of(Z):
    if Z == "nat":
        return 1.0
    if Z == "E1":
        return 0.95
    if Z.startswith("E1x"):
        return 0.95 * float(Z[3:])
    if Z == "E4b":
        return 0.3
    if Z.startswith("J"):
        return float(Z[1:])
    m = re.fullmatch(r"K([\d.]+)(V1)?", Z)
    if m:
        return float(m.group(1))
    if re.fullmatch(r"V[\d.]+", Z):
        return 0.0
    if re.fullmatch(r"K1V[\d.]+", Z):
        return 1.0
    raise KeyError(Z)


def default_edit_rate(key, fmt, l, Z):
    listed = fmt in ("AFTER", "LETTER")
    if not listed:
        return 1.0
    if Z in ("E4b", "CAA") and l != 3:
        return 0.0
    if Z == "E4b":
        return 0.0
    if l != 3:
        return 0.0
    return 1.0 if rk_of("E1" if Z == "CAA" else Z) >= 0.8 else 0.0


def id_scale(fmt, Z):
    """The identity effect of a row as a fraction of the natural one (psi): key rows convex in listed formats."""
    listed = fmt in ("AFTER", "LETTER")
    m = re.fullmatch(r"K([\d.]+)", Z)
    if m:
        a = float(m.group(1))
        return 1.0 if a == 1.0 else ({0.5: 0.1, 0.8: 0.48}[a] if listed else a)
    m = re.fullmatch(r"V([\d.]+)", Z)
    if m:
        a = float(m.group(1))
        return 1.0 if a == 1.0 else {0.5: 0.6, 0.8: 0.85}[a]
    m = re.fullmatch(r"J([\d.]+)", Z)
    if m:
        a = float(m.group(1))
        return a ** 3 if listed else a
    return 1.0


def edit_kinds(l, e4b):
    k = ["nat", "E1"] + (["E4b"] if e4b else [])
    if l in (3, 7):
        k += [f"E1x{c}" for c in (0.5, 0.75, 1.5)] + [f"J{a}" for a in (0.25, 0.5, 0.65, 0.8, 0.9, 1.25)]
        k += ["K1", "V1"] + [f"K{a}" for a in (0.5, 0.8)] + [f"V{a}" for a in (0.5, 0.8)]
        k += [f"K{a}V1" for a in (0.5, 0.8)] + [f"K1V{a}" for a in (0.5, 0.8)]
    return k


class Scen:
    """Overrides of the default scenario. stories / passages: {(key or "*", fmt, row): r or (r, o) or dict(r, o, pB,
    comp)}; edits: {(key or "*", fmt, block, Z): r}; plus switches."""

    def __init__(self, stories=None, cal=None, passages=None, edits=None, strict_other=False, scale=None, rk=None):
        self.stories, self.cal, self.passages, self.edits = stories or {}, cal or {}, passages or {}, edits or {}
        self.strict_other, self.scale, self.rk = strict_other, scale or {}, rk or {}

    @staticmethod
    def get(d, key, *rest):
        return d.get((key, *rest), d.get(("*", *rest)))


def spec_of(v, default):
    if v is None:
        v = default
    if isinstance(v, dict):
        return v.get("r", default), v.get("o", 0.0), v.get("pB", 0.99), v.get("comp", 1.0)
    if isinstance(v, tuple):
        return v[0], v[1], 0.99, 1.0
    return v, 0.0, 0.99, 1.0


def assign(n, r, o, comp, rank):
    """Per item: "S", "other" or "B" (competent items: the first round(comp n) ranks; within them the shares r, o)."""
    nc = int(round(comp * n))
    kS, kO = int(round(r * nc)), int(round(o * nc))
    out = []
    for i in range(n):
        q = rank[i]
        if q < nc:
            out.append("S" if q < kS else "other" if q < kS + kO else "B")
        else:
            q2, m = q - nc, n - nc
            out.append("S" if q2 < round(r * m) else "B")
    return out


def lp6(words, ans_word, pB, iS, iB, iX):
    e = [-10.0] * 6
    if ans_word == "S":
        e[iS] = math.log(0.98)
    elif ans_word == "B":
        e[iB] = math.log(pB)
    else:
        e[iX] = math.log(0.6)
    return e


def story_file(key, pop_items, pop, scen_d, test, L, fmts=s9.STORY_FORMATS, comp_none=None):
    rows = story_rows(L)
    n = len(pop_items)
    rng = np.random.default_rng(seed_of(key, pop))
    recs = []
    for fmt in fmts:
        listed = fmt in ("AFTER", "LETTER")
        rank = list(rng.permutation(n))
        comp = spec_of(Scen.get(scen_d, key, fmt, "none"), 0.0)[3]
        per_row = {}
        for row in rows:
            r, o, pB, _ = spec_of(Scen.get(scen_d, key, fmt, row), default_rate(key, fmt, row, listed))
            per_row[row] = (assign(n, r, o, comp, rank), pB)
        for i, it in enumerate(pop_items):
            m = it.meta()
            W, tr = m["words"], m["track"]
            iS, iB, iX = W.index(tr["S"]), W.index(tr["B"]), W.index(tr["X"])
            rr = {}
            for row in rows:
                a, pB = per_row[row][0][i], per_row[row][1]
                if row in ("none", "none2") and rank[i] >= int(round(comp * n)):
                    a = "other"
                word = {"S": tr["S"], "B": tr["B"], "other": "other"}[a]
                e = lp6(W, a, pB, iS, iB, iX)
                rr[row] = {"E": e, "L": e, "gen": " " + word, "ans": word, "how": "word" if a != "other" else "none", "n_tok": 2}
            e = lp6(W, "S", 0.99, iS, iB, iX)
            cb = [x + 0.001 for x in rr["none"]["E"]]          # the clean B run: the BF16 floor is 0.001 in every entry
            recs.append({**m, "pop": pop, "idx": i, "fmt": fmt, "p": 40, "T": 90, "rows": rr,
                         "clean_S": {"E": e, "L": e, "gen": " " + tr["S"], "ans": tr["S"], "how": "word", "n_tok": 2},
                         "clean_B": {"E": cb, "L": cb}})
    return recs, rows


def prov(key, test, arm, stage, **kw):
    p = {"stage9": True, "arm": arm, "stage": stage, "git_commit": HEAD, "git_dirty": False,
         "freeze_commit": None if test else F, "predictions_sha256": NOTE, "spec_sha256": SPEC,
         "started_utc": "2026-10-15T10:00:00Z", "finished_utc": "2026-10-15T10:05:00Z", "model": f"/models/{key}",
         "model_key": key, "revision": REV[key], "verified": {"repo": "r", "revision": REV[key], "sha256": "0" * 64},
         "transformers": "5.18.0", "torch": "2.x", "test_mode": test, "args": {}}
    if stage != "check":
        p |= {"dtype": "torch.float32" if test else "torch.bfloat16", "attn_implementation": "sdpa", "device": "x",
              "n_layers": NL[key], "hidden": 64, "n_heads": 4, "n_kv_heads": 2, "timings": {}}
    return p | kw


def seed_of(*parts) -> int:
    """A stable seed (Python's str hash is randomised per process)."""
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16)


def write(path: Path, J):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(J))


def make_run(root: Path, scen: Scen = None, test=True, keys=KEYS, n_st=40, n_c=30, n_ed=30, n_e=40, n_r=20,
             pytest_ok=True, drop=()):
    """Write a synthetic results directory. ``drop``: (key, arm) pairs whose results are not written."""
    scen = scen or Scen()
    pre = "TEST_" if test else ""
    root.mkdir(parents=True, exist_ok=True)
    (root / "COMMIT.txt").write_text("==== 2026-10-15T09:00:00Z stage 9\n" + ("" if test else
                                     f"stage-9 predictions frozen at {F} (2026-10-14T20:00:00+00:00); note sha256 {NOTE}; "
                                     f"spec sha256 {SPEC}; HEAD {HEAD}\n"))
    if pytest_ok is not None:
        lines = ["==== 2026-10-15T09:00:00Z python -m pytest", "=" * 20 + " test session starts " + "=" * 20]
        for f in CK.STAGE9_TESTS:
            lines.append(f"{f}::test_one {'PASSED' if pytest_ok is True or f != pytest_ok else 'FAILED'} [ 50%]")
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "logs" / "pytest.log").write_text("\n".join(lines) + "\n")
    N9, C, E9 = s9.stories()[:n_st], s9.calibration()[:n_c], s9.edit_cores()[:n_ed]
    for key in keys:
        t, L = pre + key, NL[key]
        st8 = {"frames_file": "x", "frames_sha256": PINS[f"results/gpu_stage8b/frames/{key}.json"], "g3_file": "x",
               "g3_sha256": PINS[f"results/gpu_stage8b/g3/{key}.json"], "scorer": "trie", "frames": []}
        if (key, "stories") not in drop:
            write(root / "check" / f"{t}_stories.json", {"provenance": prov(key, test, "stories", "check", **st8), "pass": True, "fails": []})
            crecs, rows = story_file(key, C, "C", scen.cal, test, L)
            fl = DA.late_flips_all(DA.StoryRun({"provenance": {"n_layers": L}, "rows": rows, "items": crecs}), "AFTER")
            lb = DA.lb_of(fl)
            P = prov(key, test, "stories", "cal", population="C", population_sha256=s9.POP_SHA256["C"], n_items=len(C),
                     skipped_items=[], formats=list(s9.STORY_FORMATS), formats_done=list(s9.STORY_FORMATS), formats_not_run=[],
                     onsets=list(range(2, L - 1, 2)), l_b=lb, **st8, finished_utc="2026-10-15T10:05:00Z")
            write(root / "lb" / f"{t}.json", {"provenance": P, "population": "C", "rows": rows, "l_b": lb,
                                             "late_after_flip": {str(l): round(v, 5) for l, v in fl.items()},
                                             "rule": "l_b = first onset with late AFTER flip <= 0.2 over all C cores",
                                             "items": crecs})
            lbf = root / "lb" / f"{t}.json"
            recs, _ = story_file(key, N9, "N9", scen.stories, test, L)
            P = prov(key, test, "stories", "eval", population="N9", population_sha256=s9.POP_SHA256["N9"], n_items=len(N9),
                     skipped_items=[], formats=list(s9.STORY_FORMATS), formats_done=list(s9.STORY_FORMATS), formats_not_run=[],
                     onsets=list(range(2, L - 1, 2)), l_b=lb, lb_file=str(lbf), lb_sha256=CK.sha_file(lbf), **st8,
                     started_utc="2026-10-15T10:06:00Z")
            write(root / "stories" / f"{t}.json", {"provenance": P, "population": "N9", "rows": rows, "items": recs})
        if (key, "passages") not in drop:
            pp = {"frame": " ", "frames_file": "x", "frames_sha256": PINS[f"results/gpu_stage8a/frames/{key}.json"]}
            write(root / "check" / f"{t}_passages.json", {"provenance": prov(key, test, "passages", "check", **pp), "pass": True, "fails": []})
            J = passage_file(key, scen, n_e, n_r)
            J["provenance"] = prov(key, test, "passages", "eval", **pp, valid_sha256={"E": S8A[key], "R": "e" * 64},
                                   stage8a_population_sha256=S8A[key], population=["E", "R"], population_sha256={},
                                   n_items={"E": n_e, "R": n_r}, skipped_items={"E": [], "R": []},
                                   formats=["NOM", "OPTB", "OPTA"], formats_not_run=[], onsets=[round(x * L) for x in (0.15, 0.3, 0.45, 0.6)])
            write(root / "passages" / f"{t}.json", J)
        if (key, "edits") not in drop:
            e4b = key in ("qwen7", "mistral7")
            ep = {"calib_file": "x", "calib_sha256": PINS[f"results/gpu_stage8c/calib/{key}.pt"],
                  "das_file": "x" if e4b else None, "das_sha256": PINS[f"results/gpu_stage8c/das/{key}.pt"] if e4b else None,
                  "das_seed": 102}
            write(root / "check" / f"{t}_edits.json", {"provenance": prov(key, test, "edits", "check", **ep), "pass": True, "fails": []})
            cores = edit_file(key, E9, scen, e4b)
            P = prov(key, test, "edits", "eval", **ep, population="E9", population_sha256=s9.POP_SHA256["E9"],
                     n_items=len(E9), skipped_items=[], cores_not_run=[], blocks=[3, 7, 11, 15], dose_blocks=[3, 7],
                     formats=list(s9.EDIT_FORMATS), formats_not_run=[], e4b=e4b)
            write(root / "edits" / f"{t}.json", {"provenance": P, "population": "E9", "rows": {}, "cores": cores})
    return root


def passage_file(key, scen, n_e, n_r):
    L = NL[key]
    rows = passage_rows(L)
    rng = np.random.default_rng(seed_of(key, "ps"))
    splits = {}
    for split, n, n_art in (("E", n_e, 12), ("R", n_r, 6)):
        fm = {}
        for fmt in ("NOM", "OPTB", "OPTA"):
            listed = fmt == "OPTA"
            rank = list(rng.permutation(n))
            per_row = {row: assign(n, *(lambda s: (s[0], s[1], s[3]))(spec_of(Scen.get(scen.passages, key, f"{split}/{fmt}", row),
                                                                            default_rate(key, fmt, row, listed))), rank)
                       for row in rows}
            recs = []
            for i in range(n):
                rr = {}
                for row in rows:
                    a = per_row[row][i]
                    Y = {"S": "S", "B": "B", "other": "other"}[a]
                    lp = {y: -8.0 for y in ("B", "S", "X", "Z", "D")}
                    lp["X" if Y == "other" else Y] = -0.01
                    strict = "other" if (scen.strict_other and Y == "S") else Y
                    rr[row] = {"lp": lp, "mass": 1.0, "argmax": 7, "text": f"ans {Y}", "ids": [1, 2], "n_tok": 2,
                               "who_strict": strict, "who_contain": Y}
                recs.append({"id": f"{split}{i:03d}", "art": i % n_art, "title": "t", "sub": "s", "D_in": True,
                             "stratum": "x", "split": split, "fmt": fmt, "nP": 2, "P": [50, 51], "T": 300, "j": 1, "w": [3],
                             "dec": {y: 1 for y in ("B", "S", "X", "Z", "D")}, "rows": rr,
                             "S_run": {"text": "S", "ids": [1], "n_tok": 1, "who_strict": "other" if scen.strict_other else "S",
                                       "who_contain": "S"},
                             "clean_B": {"lp": {y: v + 0.002 for y, v in rr["none"]["lp"].items()}}})
            fm[fmt] = recs
        splits[split] = fm
    return {"rows": rows, "frame": " ", "splits": splits}


def edit_file(key, E9, scen, e4b):
    n = len(E9)
    rng = np.random.default_rng(seed_of(key, "ed"))
    rank = list(rng.permutation(n))
    cores = []
    plan = {}
    for fmt in s9.EDIT_FORMATS:
        for l in (3, 7, 11, 15):
            for Z in edit_kinds(l, e4b) + ["CAA"]:
                r = Scen.get(scen.edits, key, fmt, l, Z)
                r = default_edit_rate(key, fmt, l, Z) if r is None else r
                plan[(fmt, l, Z)] = assign(n, r, 0.0, 1.0, rank)
    for i, core in enumerate(E9):
        X = pick_x(core)
        iB, iS, iX = LOCATIONS.index(core["base"]), LOCATIONS.index(core["source"]), LOCATIONS.index(X)
        word = {"S": core["source"], "X": X, "B": core["base"]}
        self_e = [-10.0] * 6
        self_e[iB] = math.log(0.99)
        rec = {"index": i, "core": core, "X": X, "p": 30, "iB": iB, "iS": iS, "iX": iX, "iPiS": 0, "iPiX": 1,
               "cluster": [core["base"], core["source"]], "T_check": {str(l): 0.0 for l in (3, 7, 11, 15)},
               "stats": {}, "cells": {}}
        for l in (3, 7, 11, 15):
            rec["stats"][str(l)] = {f"{Z}|{t}": {"proj_k": scen.rk.get(Z, rk_of(Z)), "proj_v": 1.0, "cos_k": 1.0, "cos_v": 1.0,
                                                 "ratio_k": 1.0, "ratio_v": 1.0, "nu": 1.0, "cos_resid": None, "ratio_resid": None}
                                    for Z in edit_kinds(l, e4b) for t in ("S", "X")}
            for fmt in s9.EDIT_FORMATS:
                cell = {"self": {"E": self_e, "L": self_e, "gen": " " + core["base"], "ans": core["base"], "how": "word"}}

                def row(Z, t):
                    a = plan[(fmt, l, Z)][i]
                    ans = word[t] if a == "S" else word["B"]
                    e = list(self_e)
                    sc = scen.scale.get((fmt, Z), id_scale(fmt, Z))
                    e[iS if t == "S" else iX] += 5.0 * sc
                    lg = list(e)
                    if (fmt, Z) in scen.scale and ("logit", fmt, Z) in scen.scale:
                        lg = list(self_e)
                        lg[iS if t == "S" else iX] += 5.0 * scen.scale[("logit", fmt, Z)]
                    return {"E": e, "L": lg, "gen": " " + ans, "ans": ans, "how": "word"}
                for Z in edit_kinds(l, e4b):
                    for t in ("S", "X"):
                        cell[f"{Z}|{t}"] = row(Z, t)
                cell["self2"] = dict(cell["self"])
                cell["caa"] = {"self": cell["self"], "CAA|S": row("CAA", "S"), "CAA|X": row("CAA", "X")}
                rec["cells"][f"{fmt}@{l}"] = cell
        cores.append(rec)
    return cores


def score(root, test=True):
    rc = SC.main(["--results", str(root)] + (["--test"] if test else []))
    text = (root / "STAGE9_SCORE.txt").read_text()
    J = json.loads((root / "stage9_summary.json").read_text())
    return rc, text, J


def verdict(J, code, key="ALL"):
    return J[f"verdict/{key}/{code}"]["verdict"]


def fal(J, code, key="ALL"):
    return J[f"falsifier/{key}/{code}"]["triggered"]


ALL_LINES = ("P1", "P1s", "P1p", "P2", "P2s", "P2p", "P3", "P3s", "P3p", "P4", "P5a", "P5b", "POST", "B1", "B2", "B3")


# --------------------------------------------------------------------------- statistics
def test_locations_and_onsets_match_ckeys():
    assert DA.LOCATIONS == tuple(LOCATIONS)
    for L in (24, 28, 32):
        assert DA.story_onsets(L) == s9.story_onsets(L) and DA.passage_onsets(L) == s9.passage_onsets(L)
    assert DA.STORY_FORMATS == s9.STORY_FORMATS and DA.EDIT_FORMATS == s9.EDIT_FORMATS
    assert ST.SEED == s9.BOOT_SEED and ST.NB == FAST_NB
    assert CK.STAGE9_TESTS == tuple(f"tests/test_stage9_{x}.py" for x in ("pops", "stories", "passages", "edits", "score", "guard", "pre"))


def test_bootstrap_two_stage_deterministic_and_seeded(monkeypatch):
    monkeypatch.setattr(ST, "NB", 10000)
    cl = [("a",)] * 5 + [("b",)] * 3 + [("c",)] * 7
    B1 = ST.Boot(cl, nb=10000, seed=20261015)
    B2 = ST.boot_of(cl)
    assert B2.W.shape == (10000, 15) and np.array_equal(B1.W, B2.W)          # seed 20261015, 10,000 resamples
    assert ST.boot_of(cl) is B2                                                # one index set per population, cached
    draws = 0
    for ii in ([0, 1, 2, 3, 4], [5, 6, 7], list(range(8, 15))):               # two-stage: whole draws of a cluster,
        tot = B1.W[:, ii].sum(1)                                              # then items within it
        assert np.all(tot % len(ii) == 0)
        draws = draws + tot // len(ii)
    assert np.all(draws == 3)                                                 # 3 clusters drawn with replacement
    assert B1.W[:, :5].std() > 0 and (B1.W[:, 5:8].sum(1) > 0).mean() < 1
    x = np.arange(15) % 2
    q1, q2 = ST.rate(B1, x), ST.rate(B2, x)
    assert np.array_equal(q1.bs, q2.bs) and q1.pt == pytest.approx(x.mean())
    # a rate over a subset is the ratio of weighted sums; a resample with zero subset weight is dropped
    m = np.zeros(15)
    m[[0, 1]] = 1
    q = ST.rate(B1, x, m)
    den = B1.W @ m
    assert np.isnan(q.bs[den == 0]).all() and q.drop == pytest.approx((den == 0).mean())
    assert q.drop > 0.05 and not q.ok() and "resamples" in q.reason()
    assert not ST.rate(B1, x, np.zeros(15)).ok()


def test_weighted_median_and_auc():
    B = ST.boot_of([(i % 4,) for i in range(20)])
    v = np.linspace(0, 1, 20) ** 2
    m = np.ones(20, bool)
    q = ST.wmedian(B, v, m)
    assert q.pt == pytest.approx(np.median(v))
    for b in range(25):            # every resample equals numpy's median of the repeated values
        w = B.W[b].astype(int)
        assert q.bs[b] == pytest.approx(np.median(np.repeat(v, w)))
    # AUC with ties against brute force, at the point and in a resample
    rng = np.random.default_rng(3)
    core = rng.integers(0, 20, 300)
    r = np.round(rng.random(300) * 4) / 4
    y = rng.random(300) < 0.3 + 0.5 * r
    qa, npos, nneg = ST.auc(B, core, r, y)

    def brute(w):
        num = den = 0.0
        for i in np.flatnonzero(y):
            for j in np.flatnonzero(~y):
                ww = w[core[i]] * w[core[j]]
                num += ww * ((r[i] > r[j]) + 0.5 * (r[i] == r[j]))
                den += ww
        return num / den
    assert qa.pt == pytest.approx(brute(np.ones(20)))
    assert qa.bs[7] == pytest.approx(brute(B.W[7]))
    assert npos == y.sum() and nneg == (~y).sum()
    q9, p9, _ = ST.auc(B, core[:40], r[:40], np.arange(40) < 9)      # 9 positives: below the floor of 10
    assert p9 == 9 and not q9.ok() and "at least 10" in q9.reason()
    qt, _, _ = ST.auc(B, core[:40], np.ones(40), np.arange(40) < 20)  # all tied: 0.5
    assert qt.pt == pytest.approx(0.5)


def test_verdict_levels_combination_and_k_of():
    B = ST.boot_of([(i,) for i in range(50)])
    sure = ST.rate(B, np.ones(50))
    near = ST.rate(B, np.r_[np.ones(48), np.zeros(2)])             # 0.96, interval crossing 0.95
    low = ST.rate(B, np.r_[np.ones(45), np.zeros(5)])              # 0.90
    assert ST.check(sure, ">=", 0.95, "x").level == ST.SUP
    assert ST.check(near, ">=", 0.95, "x").level == ST.PT and near.lo < 0.95
    assert ST.check(low, ">=", 0.95, "x").level == ST.NOT
    assert ST.check(ST.missing("gone"), ">=", 0.95, "x").level == ST.NE
    exact = ST.rate(B, np.r_[np.ones(15), np.zeros(35)])            # 0.3 exactly: <= 0.3 holds despite rounding
    assert ST.check(exact, "<=", 0.3, "x").level in (ST.SUP, ST.PT)
    assert ST.trig(ST.diff(ST.rate(B, np.r_[np.ones(15), np.zeros(35)]), ST.rate(B, np.r_[np.ones(10), np.zeros(40)])), "<", 0.1) is False
    assert ST.combine([ST.SUP, ST.SUP]) == ST.SUP and ST.combine([ST.SUP, ST.PT]) == ST.PT
    assert ST.combine([ST.SUP, ST.NE]) == ST.NE and ST.combine([ST.NE, ST.NOT]) == ST.NOT and ST.combine([]) == ST.NE
    # "2 of 3", counted at the point
    assert ST.k_of([ST.SUP, ST.SUP, ST.NOT], 2) == ST.SUP
    assert ST.k_of([ST.SUP, ST.PT, ST.NOT], 2) == ST.PT
    assert ST.k_of([ST.PT, ST.NOT, ST.NOT], 2) == ST.NOT
    assert ST.k_of([ST.PT, ST.NE, ST.NOT], 2) == ST.NE
    assert ST.k_of([ST.NE, ST.NE, ST.NE], 2) == ST.NE


def test_lc_with_censoring_and_competence(tmp_path):
    """l_c: the first onset with late flip < 0.5 on the competent items, L (censored) when none; recomputed per resample."""
    scen = Scen(stories={("*", "NONE", "none"): {"r": 0.0, "comp": 0.75},          # a quarter of NONE incompetent
                         ("*", "NONE", "late@2"): {"r": 1.0, "comp": 0.75},
                         ("*", "AFTER", "late@6"): 0.45})                            # mistral7: l_c(AFTER) = 6, not LC = 8
    root = make_run(tmp_path / "r", scen, keys=("mistral7",), n_st=40)
    J = json.loads((root / "stories" / "TEST_mistral7.json").read_text())
    S = DA.StoryRun(J)
    assert S.comp("NONE").sum() == 30 and S.comp("AFTER").sum() == 40
    ctx = PR.Ctx({"mistral7": PR.Model("mistral7", st=S)}, CK.Log(), True, True)
    q, l = ctx.lc(ctx.models["mistral7"], "NONE")
    assert l == 32 and q.pt == 1.0 and np.allclose(q.bs, 1.0)                         # censored: no onset below 0.5
    q, l = ctx.lc(ctx.models["mistral7"], "AFTER")
    assert l == 6 and q.pt == pytest.approx(6 / 32)
    assert set(np.unique(q.bs)) <= {6 / 32, 8 / 32}                                  # a resample may move it to 8
    # incompetent items are outside the rate: NONE late@2 over the 30 competent items
    assert ctx.sflip(ctx.models["mistral7"], "NONE", "late@2").pt == 1.0
    assert ctx.sflip(ctx.models["mistral7"], "NONE", "late@2").n == 30
    assert ctx.sflip(ctx.models["mistral7"], "NONE", "late@2", sub="all").n == 40
    # paired across formats: one weight matrix per population and model
    assert S.B is DA.StoryRun(J).B


def test_psi_in_nats_and_logits(tmp_path):
    scen = Scen(scale={("AFTER", "K0.5"): 0.1, ("logit", "AFTER", "K0.5"): 0.3})
    root = make_run(tmp_path / "r", scen, keys=("qwen7",), n_ed=20)
    E = DA.EditRun(json.loads((root / "edits" / "TEST_qwen7.json").read_text()))
    ctx = PR.Ctx({"qwen7": PR.Model("qwen7", ed=E)}, CK.Log(), True, True)
    m = ctx.models["qwen7"]
    assert ctx.psi(m, "AFTER", 3, "K", 0.5, "E").pt == pytest.approx(0.1)
    assert ctx.psi(m, "AFTER", 3, "K", 0.5, "logit").pt == pytest.approx(0.3)
    assert ctx.psi(m, "NONE", 3, "V", 0.5).pt == pytest.approx(0.6)
    assert ctx.psi(m, "AFTER", 3, "J", 0.5).pt == pytest.approx(0.125)
    assert E.ident("AFTER", 3, "K1").mean() == pytest.approx(5.0)
    assert ctx.psi(m, "AFTER", 11, "K", 0.5).why                                     # no dose rows at block 11


# --------------------------------------------------------------------------- full runs: verdict paths
def test_at_the_point_paths(tmp_path):
    """Near-threshold numbers in one model per line: every line SUPPORTED AT THE POINT."""
    m, ll, q = "mistral7", "llama8", "qwen7"
    scen = Scen(
        stories={(m, "NONE", "late@4"): 0.96,                                   # P1s: >= 0.95 at the point only
                 (ll, "LETTER", "late@8"): 0.47, (ll, "LETTER", "late@10"): 1.0,  # P2s: l_c(LETTER) 8/32 at the point,
                 (ll, "LETTER", "late@12"): 0.0,                                 # 12/32 > 0.35 in the resamples where late@8 >= 0.5
                 (q, "AFTER", "early@14"): 0.72,                                # P3s: >= 0.7 at the point only
                 (m, "AFTER", "late@8"): {"r": 0.0, "o": 0.17},                 # P4: share B among non-S 0.83
                 (ll, "POST", "late@6"): 0.82},                                 # POST
        passages={(m, "E/NOM", "late@5"): 0.82,                                 # P1p
                  (ll, "E/OPTA", "late@10"): 0.38,                              # P2p (2 of 3: still SUP elsewhere)
                  (q, "E/OPTA", "early@8"): 0.53},                              # P3p
        edits={(m, "POST", 7, "E1"): 0.92,                                      # P5a: |0.92 - 1| = 0.08
               (q, "AFTER", 3, "E4b"): 0.68, (q, "LETTER", 3, "E4b"): 0.0,     # P5b: tau 0.68
               (ll, "AFTER", 3, "J0.5"): 0.33,                                  # B1
               (m, "AFTER", 3, "E1x0.5"): 0.48})                                # B2
    root = make_run(tmp_path / "r", scen, n_st=60, n_e=60, n_ed=40)
    rc, text, J = score(root)
    for c, k in (("P1s", m), ("P2s", ll), ("P3s", q), ("P4", m), ("POST", ll), ("P1p", m), ("P3p", q), ("P5a", m),
                 ("P5b", q), ("B1", ll), ("B2", m)):
        assert verdict(J, c, k) == ST.PT, (c, k, J[f"verdict/{k}/{c}"])
        assert verdict(J, c) == ST.PT, c
    assert verdict(J, "P2p", ll) == ST.PT and verdict(J, "P2p") == ST.SUP     # 2 of 3 SUPPORTED: SUPPORTED
    assert verdict(J, "P1") == ST.PT and verdict(J, "P3") == ST.PT


def test_not_supported_paths(tmp_path):
    m, ll, q = "mistral7", "llama8", "qwen7"
    scen = Scen(
        stories={(m, "BEFORE", "late@2"): 0.9,                                  # P1s
                 (ll, "LETTER", "late@10"): 1.0, (ll, "LETTER", "late@12"): 1.0, # P2s: l_c(LETTER) 14/32 > 0.35
                 (q, "NONE", "early@14"): 0.1,                                  # P3s (F1 not: 0.1 < 0.2)
                 (ll, "AFTER", "late@10"): {"r": 0.0, "o": 0.4},                # P4: share B 0.6
                 (q, "POST", "late@4"): 0.7},                                   # POST
        passages={(m, "E/OPTB", "late@10"): 0.6,                                # P1p
                  (m, "E/OPTA", "late@10"): 0.5, (ll, "E/OPTA", "late@10"): 0.45,  # P2p: 1 of 3 at the point
                  (q, "E/NOM", "early@8"): 0.3},                                # P3p
        edits={(ll, "LETTER", 7, "E1"): 0.3,                                    # P5a
               (m, "AFTER", 7, "E4b"): 0.9, (m, "AFTER", 3, "E4b"): 0.1,        # P5b at block 7: tau 0.9
               (q, "AFTER", 3, "J0.5"): 0.5,                                    # B1 (B-F1 not: 0.5 < 0.6)
               (ll, "NONE", 3, "E1x0.5"): 0.7})                                 # B2
    root = make_run(tmp_path / "r", scen, n_st=60, n_e=60, n_ed=40)
    rc, text, J = score(root)
    for c, k in (("P1s", m), ("P2s", ll), ("P3s", q), ("P4", ll), ("POST", q), ("P1p", m), ("P3p", q), ("P5a", ll),
                 ("P5b", m), ("B1", q), ("B2", ll)):
        assert verdict(J, c, k) == ST.NOT, (c, k, J[f"verdict/{k}/{c}"])
        assert verdict(J, c) == ST.NOT, c
        assert re.search(rf"^  {re.escape(c)} .*-> NOT SUPPORTED \(NOT SUPPORTED in: {k}\)", text, re.M), c
    assert verdict(J, "P2p") == ST.NOT and verdict(J, "P2p", q) == ST.SUP          # only qwen7 at the point
    assert verdict(J, "P2") == ST.NOT
    assert fal(J, "F1") is False and fal(J, "B-F1") is False


def test_b3_not_supported_and_not_evaluable(tmp_path):
    # transfer unrelated to r_K: the J rows transfer regardless of alpha, the E1 multipliers never
    ed = {("*", "AFTER", 3, Z): 1.0 for Z in ("J0.25", "J0.5", "J0.65")}
    ed |= {("*", "AFTER", 3, Z): 0.0 for Z in ("E1", "E1x1.5", "J0.8", "J0.9", "J1.25")}
    root = make_run(tmp_path / "a", Scen(edits=ed), n_st=20, n_e=20, n_ed=30)
    rc, text, J = score(root)
    assert verdict(J, "B3") == ST.NOT and J["edits/qwen7/B3:auc"]["pt"] < 0.5
    # no transfer cell (the natural AFTER flip < 0.9 at blocks 3 and 7): NOT EVALUABLE
    root = make_run(tmp_path / "b", Scen(edits={("*", "AFTER", 3, "nat"): 0.5}), n_st=20, n_e=20, n_ed=30)
    rc, text, J = score(root)
    assert verdict(J, "B3") == ST.NE and "no dose block where the natural AFTER flip >= 0.9" in text
    # too few negatives: below the floor
    root = make_run(tmp_path / "c", Scen(rk={Z: 0.9 for Z in ("E1x0.5", "E1x0.75", "E4b", "J0.25", "J0.5", "J0.65")},
                                         edits={("*", "AFTER", 3, Z): 1.0 for Z in ("E1x0.5", "E1x0.75", "E4b", "J0.25", "J0.5", "J0.65")}),
                    n_st=20, n_e=20, n_ed=30)
    rc, text, J = score(root)
    assert verdict(J, "B3") == ST.NE and "0 negatives (at least 10 of each needed)" in text


def test_b3_at_the_point(tmp_path, monkeypatch):
    """B3 SUPPORTED AT THE POINT: an AUC just above 0.8 whose interval crosses it (units varying between cores)."""
    root = make_run(tmp_path / "r", keys=("llama8",), n_st=10, n_e=10, n_ed=40)
    E = DA.EditRun(json.loads((root / "edits" / "TEST_llama8.json").read_text()))
    ctx = PR.Ctx({"llama8": PR.Model("llama8", ed=E)}, CK.Log(), True, True)
    rng = np.random.default_rng(5)
    core = np.repeat(np.arange(E.n), 12)
    r = np.tile(np.linspace(0.1, 1.4, 12), E.n)
    y = rng.random(core.size) < np.clip(0.5 + 0.7 * (r - 0.75), 0.02, 0.98)
    monkeypatch.setattr(PR, "auc_units", lambda ctx, m: ([3], core, r, y))
    v, q, acc, txt = PR.b3(ctx, ctx.models["llama8"])
    assert q.pt >= 0.8 and q.lo < 0.8 and v.level == ST.PT, (q.txt(), v.level)
    assert "rule r_K >= 0.8 accuracy" in txt and 0 < acc.pt < 1


def test_not_evaluable_paths(tmp_path):
    """A missing arm, a missing format, l_b None, a failed check file, the P5b precondition."""
    m, ll, q = "mistral7", "llama8", "qwen7"
    scen = Scen(cal={(q, "AFTER", f"late@{l}"): 0.5 for l in range(2, 27, 2)},     # qwen7: l_b None
                edits={(m, "AFTER", 3, "nat"): 0.5, (m, "AFTER", 3, "E1"): 0.5})   # mistral7: tau_nat(3) 0.5 < 0.9
    root = make_run(tmp_path / "r", scen, n_st=20, n_e=20, n_ed=20, drop=((ll, "passages"),))
    J = json.loads((root / "stories" / "TEST_llama8.json").read_text())             # llama8: POST not run
    J["items"] = [r for r in J["items"] if r["fmt"] != "POST"]
    J["provenance"]["formats_not_run"] = ["POST"]
    write(root / "stories" / "TEST_llama8.json", J)
    C = json.loads((root / "check" / "TEST_qwen7_edits.json").read_text())         # qwen7: the edits check failed
    C["pass"], C["fails"] = False, ["synthetic failure"]
    write(root / "check" / "TEST_qwen7_edits.json", C)
    rc, text, J = score(root)
    assert verdict(J, "P3s", q) == ST.NE and verdict(J, "P4", q) == ST.NE and "l_b is None" in J["verdict/qwen7/P3s"]["why"]
    assert verdict(J, "P1p", ll) == ST.NE and "no passages results for llama8" in J["verdict/llama8/P1p"]["why"]
    assert verdict(J, "POST", ll) == ST.NE and verdict(J, "POST") == ST.NE
    assert verdict(J, "P5b", m) == ST.NE and "tau_nat(3)" in J["verdict/mistral7/P5b"]["why"]
    for c in ("P5a", "P5b", "B1", "B2", "B3"):
        assert verdict(J, c, q) == ST.NE and "the check failed" in J[f"verdict/qwen7/{c}"]["why"], c
    assert verdict(J, "P5b") == ST.NE and verdict(J, "P5a") == ST.NE
    for c in ("P1p", "P3p", "P3s", "P4", "POST", "P5a", "P5b", "B1", "B2", "B3"):
        assert verdict(J, c) == ST.NE, c
    assert verdict(J, "P2p") == ST.SUP                                               # 2 of 3 with two models
    assert fal(J, "F1") is None and fal(J, "F1", q) is None and fal(J, "F1", m) is False
    assert J["verdict/qwen7/P5c"]["transfers"] is None


def test_falsifiers_trigger_and_the_f2_reading(tmp_path):
    m, ll, q = "mistral7", "llama8", "qwen7"
    st = {(m, "NONE", "early@8"): 0.25}                                             # F1
    st |= {(ll, "BEFORE", f"late@{l}"): (1.0 if l < 12 else 0.0) for l in range(2, 31, 2)}   # F2s: BEFORE closes at 12, AFTER 10
    st |= {(q, "NONE", f"late@{l}"): (1.0 if l < 16 else 0.0) for l in range(2, 27, 2)}      # F3: NONE closes at 16, AFTER 14
    scen = Scen(stories=st, passages={(ll, "E/OPTB", "late@10"): 0.05},             # F2p: OPTB - OPTA = 0.05
                edits={(q, "AFTER", 3, "J0.5"): 0.65,                               # B-F1
                       }, scale={("NONE", "V0.5"): 0.15})                           # B-F2: psi_V 0.15 - psi_K 0.1
    root = make_run(tmp_path / "r", scen, n_st=40, n_e=40, n_ed=30)
    rc, text, J = score(root)
    for c, k in (("F1", m), ("F2s", ll), ("F2p", ll), ("F3", q), ("B-F1", q)):
        assert fal(J, c, k) is True and fal(J, c) is True, c
    assert all(fal(J, "B-F2", k) is True for k in KEYS) and fal(J, "F2") is True
    assert "F2 triggers: report only the evaluation warning" in text
    assert "describe the convexity without a threshold claim" in text
    assert "a model has finished using a context token" not in text


def test_reading_when_p3_fails_and_p5c_transfers(tmp_path):
    scen = Scen(stories={("*", "AFTER", f"early@{l}"): 0.3 for l in range(2, 31, 2)},
                edits={("*", "AFTER", 11, "CAA"): 0.6})
    root = make_run(tmp_path / "r", scen, n_st=30, n_e=20, n_ed=20)
    rc, text, J = score(root)
    assert verdict(J, "P3s") == ST.NOT and verdict(J, "P1s") == ST.SUP and verdict(J, "P2s") == ST.SUP
    assert "Late edits fail silently with listed candidates" in text
    assert all(J[f"verdict/{k}/P5c"]["transfers"] is True for k in KEYS)
    assert "P5c transfers: state the claim for single-site edits" in text


def test_containment_primary_strict_printed(tmp_path):
    root = make_run(tmp_path / "r", Scen(strict_other=True), n_st=20, n_e=40, n_ed=20)
    rc, text, J = score(root)
    assert verdict(J, "P1p") == ST.SUP and verdict(J, "P3p") == ST.SUP              # containment decides
    assert re.search(r"^  P1p \[E, strict\] .*-> NOT EVALUABLE", text, re.M)        # strict: S runs never competent
    assert re.search(r"^  P1p \[R, containment\] .*-> SUPPORTED", text, re.M)
    assert "passages/qwen7/E:NOM:late@4:flip" in J and "passages/qwen7/E:NOM:KV@0:flip_strict_all" not in J
    assert "strict" in text and "competent (containment)" in text


def test_competence_filtering_changes_the_rate(tmp_path):
    """Items whose none row does not answer B are outside C9(f): their answers do not count."""
    scen = Scen(stories={("*", "NONE", "none"): {"r": 0.0, "comp": 0.5}, ("*", "NONE", "late@4"): {"r": 1.0, "comp": 0.5}})
    root = make_run(tmp_path / "r", scen, keys=("llama8",), n_st=40)
    J = json.loads((root / "stories" / "TEST_llama8.json").read_text())
    for r in J["items"]:                       # the incompetent half answers B under late@4
        if r["fmt"] == "NONE" and r["rows"]["none"]["ans"] == "other":
            r["rows"]["late@4"]["ans"] = r["track"]["B"]
    S = DA.StoryRun(J)
    ctx = PR.Ctx({"llama8": PR.Model("llama8", st=S)}, CK.Log(), True, True)
    assert ctx.sflip(ctx.models["llama8"], "NONE", "late@4").pt == 1.0
    assert ctx.sflip(ctx.models["llama8"], "NONE", "late@4", sub="all").pt == pytest.approx(0.5)


# --------------------------------------------------------------------------- checks, provenance, gate
@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One full-size results directory outside TEST (the real population sizes), copied by each check test (each copy is
    deleted after its test by the tmp_path fixture above; this one after the module)."""
    root = tmp_path_factory.mktemp("full") / "r"
    old = ST.NB
    ST.NB = FAST_NB
    try:
        make_run(root, test=False, n_st=150, n_c=30, n_ed=80, n_e=185, n_r=71)
    finally:
        ST.NB = old
    yield root
    shutil.rmtree(root.parent, ignore_errors=True)


def mutate(src, dst, fn):
    shutil.copytree(src, dst)
    fn(dst)
    return dst


def edit_json(path, fn):
    J = json.loads(path.read_text())
    fn(J)
    path.write_text(json.dumps(J))


def setp(rel, **kw):
    return lambda r: edit_json(r / rel, lambda J: J["provenance"].update(**kw))


# per (model, arm) mutations, grouped so that each group touches distinct (model, arm) pairs:
# (mutation, the text it must produce, (model, line) that becomes NOT EVALUABLE)
GROUPS = {
    "shas": [(setp("stories/qwen7.json", population_sha256="0" * 64), "!= the pin of N9", ("qwen7", "P1s")),
             (setp("stories/mistral7.json", frames_sha256="0" * 64), "frames sha256 000000000000 != the pin", ("mistral7", "P2s")),
             (setp("passages/llama8.json", frames_sha256="1" * 64), "frames sha256 111111111111 != the pin", ("llama8", "P1p")),
             (setp("edits/qwen7.json", calib_sha256="0" * 64), "calib sha256 000000000000 != the pin", ("qwen7", "P5a")),
             (setp("edits/mistral7.json", das_sha256="0" * 64), "das sha256 000000000000 != the pin", ("mistral7", "B1")),
             (lambda r: shutil.copy(r / "edits" / "llama8.json", r / "edits" / "TEST_llama8.json"),
              "a TEST_MODE file outside TEST", ("llama8", "B2"))],
    "freeze": [(setp("stories/llama8.json", freeze_commit="f" * 40), "freeze_commit ffffffffff", ("llama8", "P3s")),
               (setp("passages/qwen7.json", predictions_sha256="9" * 64), "predictions_sha256 differs", ("qwen7", "P3p")),
               (setp("passages/mistral7.json", spec_sha256="9" * 64), "spec_sha256 differs", ("mistral7", "P1p")),
               (setp("stories/mistral7.json", lb_sha256="0" * 64), "the eval read another lb file", ("mistral7", "P4")),
               (setp("edits/qwen7.json", dtype="torch.float32"), "is not BF16", ("qwen7", "B2")),
               (setp("edits/llama8.json", revision="0" * 40), "!= the manifest's", ("llama8", "P5a"))],
    "lb": [(lambda r: edit_json(r / "lb" / "llama8.json", lambda J: J.update(l_b=12)),
            "the lb file's l_b 12 != the recomputed 10", ("llama8", "P3s")),
           (setp("lb/qwen7.json", finished_utc="2026-10-15T11:00:00Z"), "after the eval started", ("qwen7", "P4")),
           (lambda r: edit_json(r / "passages" / "llama8.json", lambda J: J["provenance"]["valid_sha256"].update(E="0" * 64)),
            "differs from stage 8a's population", ("llama8", "P2p")),
           (setp("stories/qwen7.json", git_dirty=True), "modified tracked files", ("qwen7", "POST")),
           (setp("passages/mistral7.json", attn_implementation="eager"), "attention eager is not sdpa", ("mistral7", "P3p")),
           (lambda r: (r / "lb" / "mistral7.json").unlink(), "the stories eval exists without it", ("mistral7", "P2s"))],
    "grid": [(lambda r: edit_json(r / "stories" / "qwen7.json", lambda J: J["rows"].remove("late@14")),
              "the row grid differs from SPEC 7.1 at L = 28: 38 rows, 39 expected; missing ['late@14']", ("qwen7", "P1s")),
             (lambda r: edit_json(r / "passages" / "mistral7.json",
                                  lambda J: J.update(rows=["late@11" if x == "late@10" else x for x in J["rows"]])),
              "missing ['late@10']; unexpected ['late@11']", ("mistral7", "P1p")),
             (lambda r: edit_json(r / "edits" / "llama8.json", lambda J: J["cores"][5]["cells"]["AFTER@3"].pop("K0.5|S")),
              "the edit rows differ from SPEC 8.1 in 1 cell(s): cell AFTER@3 (core 5): 43 rows, 44 expected; missing "
              "['K0.5|S']", ("llama8", "P5a")),
             (setp("edits/mistral7.json", e4b=False), "e4b False, but mistral7 has stored E4b basis", ("mistral7", "B1")),
             (setp("lb/llama8.json", n_layers=30), "n_layers 30 != 32 (SPEC 4)", ("llama8", "P3s"))],
}
RUN_LEVEL = {   # every verdict NOT EVALUABLE
    "two commits": (setp("edits/llama8.json", git_commit="e" * 40), "the results files carry 2 commits"),
    "COMMIT.txt freeze": (lambda r: (r / "COMMIT.txt").write_text((r / "COMMIT.txt").read_text().replace(F, "f" * 40)),
                          "freeze_commit"),
    "no freeze record": (lambda r: (r / "COMMIT.txt").write_text("==== a session without the guard\n"),
                         "COMMIT.txt has no unique stage-9 freeze record"),
}


def test_all_supported_on_the_clean_full_run(full_run, tmp_path):
    """The default scenario at the real population sizes outside TEST: every check passes, every line SUPPORTED, no
    falsifier, the window reading selected, exit 0, the output files and sections written."""
    root = mutate(full_run, tmp_path / "r", lambda r: None)
    rc, text, J = score(root, test=False)
    assert rc == 0 and "checks: OK" in text and "  MISMATCH" not in text
    for c in ALL_LINES:
        assert verdict(J, c) == ST.SUP, (c, J[f"verdict/ALL/{c}"])
    for c in ("F1", "F2s", "F2p", "F2", "F3", "B-F1", "B-F2"):
        assert fal(J, c) is False, c
    assert all(J[f"verdict/{k}/P5c"]["transfers"] is False for k in KEYS)
    heads = [l[5:].split(" (")[0] for l in text.splitlines() if l.startswith("#### ")]
    assert heads == list(SC.SECTIONS)
    assert "a model has finished using a context token" in text and "add the edits clause" in text
    assert "an edit at a context token reaches a listed answer only if" in text
    assert J["stories/mistral7/AFTER:lc_frac"]["pt"] == pytest.approx(8 / 32)
    assert J["stories/qwen7/AFTER:lc_frac"]["pt"] == pytest.approx(14 / 28)
    assert J["stories/llama8/NONE:lc_frac"]["pt"] == 1.0 and "censored: > last onset" in text
    assert J["edits/qwen7/B3:auc"]["pt"] == 1.0 and J["edits/llama8/B3:auc"]["pt"] == 1.0
    assert J["edits/mistral7/B3:rule_accuracy"]["pt"] == 1.0
    assert J["dose/qwen7/AFTER@3:psi_K(0.5):nats"]["pt"] == pytest.approx(0.1)
    assert J["dose/qwen7/NONE@3:psi_V(0.5):logit"]["pt"] == pytest.approx(0.6)
    for k in KEYS:
        assert re.search(rf"{k}: l_b = {LC[k]} \(recomputed from 30 C cores", text)
    for f, head in (("fig_window.csv", "model,arm,split,format,window,onset,onset_frac,n,flip,flip_lo,flip_hi,stale,stale_lo,stale_hi,eflip"),
                    ("fig_edits.csv", "model,format,block,row,n,flip,lo,hi,stale,r_K"),
                    ("fig_dose.csv", "model,format,block,kind,alpha,psi_nats,psi_nats_lo,psi_nats_hi,psi_logit,psi_logit_lo,psi_logit_hi")):
        lines = (root / f).read_text().splitlines()
        assert lines[0] == head and len(lines) > 10
    assert all(set(v) >= {"pt", "lo", "hi", "n"} for k, v in J.items() if k.split("/")[0] in ("stories", "passages", "edits", "dose"))
    assert REAL_NB == 10000                                   # the scorer's resamples (the tests use fewer)


@pytest.mark.parametrize("group", list(GROUPS))
def test_per_model_mismatches_exit_2(full_run, tmp_path, group):
    def apply(r):
        for fn, _, _ in GROUPS[group]:
            fn(r)
    root = mutate(full_run, tmp_path / "r", apply)
    rc, text, J = score(root, test=False)
    assert rc == 2 and J["meta/run/checks"]["mismatch"] is True
    hit = set()
    for _, needle, (k, c) in GROUPS[group]:
        assert needle in text, (group, needle)
        assert verdict(J, c, k) == ST.NE and "check failed" in J[f"verdict/{k}/{c}"]["why"], (group, k, c)
        hit.add((k, c))
    untouched = [(k, c) for k in KEYS for c in ("P1s", "P1p", "P5a") if all(x[0] != k or x[1] != c for x in hit)]
    assert untouched and all(verdict(J, c, k) in (ST.SUP, ST.NE) for k, c in untouched)
    assert any(verdict(J, c, k) == ST.SUP for k, c in untouched)


@pytest.mark.parametrize("name", list(RUN_LEVEL))
def test_run_level_mismatch_makes_every_verdict_not_evaluable(full_run, tmp_path, name):
    fn, needle = RUN_LEVEL[name]
    root = mutate(full_run, tmp_path / "r", fn)
    rc, text, J = score(root, test=False)
    assert rc == 2 and needle in text and "MISMATCH" in text
    assert all(verdict(J, c) == ST.NE for c in ALL_LINES)


def test_partial_stories_passages_and_cal_are_listed(full_run, tmp_path):
    """A deadline-partial stories file (formats not run), passages file (pieces not run) and calibration (only
    lb/<tag>.json.partial) are listed as PARTIAL in CHECKS, in SUMMARY and in meta/run/checks "partial", are not a
    MISMATCH (exit 0), and leave the parts that ran evaluable: only the statistics that need a missing part are NOT
    EVALUABLE."""
    def apply(r):
        def drop_post(J):
            J["items"] = [x for x in J["items"] if x["fmt"] != "POST"]
            J["provenance"].update(formats_not_run=["POST"], formats_done=[f for f in s9.STORY_FORMATS if f != "POST"])
        edit_json(r / "stories" / "llama8.json", drop_post)

        def drop_r_opta(J):
            del J["splits"]["R"]["OPTA"]
            J["provenance"].update(formats_not_run=["R/OPTA"], complete=False)
        edit_json(r / "passages" / "qwen7.json", drop_r_opta)
        lb = json.loads((r / "lb" / "mistral7.json").read_text())
        lb["provenance"]["formats_done"] = ["NONE", "BEFORE"]
        (r / "lb" / "mistral7.json.partial").write_text(json.dumps(lb))
        (r / "lb" / "mistral7.json").unlink()
        (r / "stories" / "mistral7.json").unlink()                 # the eval waits for the lb file
    root = mutate(full_run, tmp_path / "r", apply)
    rc, text, J = score(root, test=False)
    assert rc == 0 and "checks: OK" in text and J["meta/run/checks"]["mismatch"] is False, text[-3000:]
    assert J["meta/run/checks"]["partial"] == ["llama8/stories", "mistral7/stories", "qwen7/passages"]
    assert J["meta/run/checks"]["partial_whole_arm"] == []
    checks = text[text.index("#### CHECKS"):text.index("#### CONTROLS")]
    summ = text[text.index("#### SUMMARY"):]
    assert "PARTIAL llama8 stories: stories/llama8.json: incomplete at the deadline: formats ['POST'] not run" in checks
    assert "PARTIAL qwen7 passages: passages/qwen7.json: incomplete at the deadline: pieces ['R/OPTA'] not run" in checks
    assert "PARTIAL mistral7 stories: lb/mistral7.json.partial: the calibration (cal) is incomplete" in checks
    assert "the statistics that need the parts not run are NOT EVALUABLE (not a MISMATCH)" in checks
    assert "Partial at the deadline" in summ and "llama8 stories (the parts not run NOT EVALUABLE)" in summ
    assert "qwen7 passages (the parts not run NOT EVALUABLE)" in summ and "mistral7 stories" in summ
    assert verdict(J, "POST", "llama8") == ST.NE and verdict(J, "POST") == ST.NE
    for c in ("P1s", "P2s", "P3s", "P4"):                           # the formats that ran stay evaluable
        assert verdict(J, c, "llama8") == ST.SUP, c
    assert verdict(J, "P1p", "qwen7") == ST.SUP and verdict(J, "P2p") == ST.SUP and verdict(J, "P3p", "qwen7") == ST.SUP
    lines = text.splitlines()
    i = next(i for i, x in enumerate(lines) if x.startswith("  P2p [R, containment]"))
    assert "qwen7: NOT EVALUABLE" in lines[i + 1] and "R/OPTA not run" in lines[i + 1], lines[i:i + 2]
    assert verdict(J, "P1s", "mistral7") == ST.NE and "no stories results" in J["verdict/mistral7/P1s"]["why"]
    assert verdict(J, "P5a") == ST.SUP and verdict(J, "B1") == ST.SUP          # other arms untouched


def test_controls_floor_and_letter_conflicts(tmp_path):
    """CONTROLS: the BF16 floor against the clean B run (stories 0.001 and passages 0.002 by construction; edits: self
    against the CAA batch's self, 0), the in-batch determinism (none against none2: 0), and the LETTER letter/word
    conflicts (stories: 3 of 5 letter answers; edits: 1 of 2) in the text and in stage9_summary.json."""
    root = make_run(tmp_path / "r", keys=("llama8",), n_st=10, n_c=10, n_e=10, n_r=6, n_ed=6)

    def letters(J):
        k = 0
        for r in J["items"]:
            if r["fmt"] != "LETTER":
                continue
            for name, d in list(r["rows"].items()) + [("clean_S", r["clean_S"])]:
                d["word"] = "other"
            if k < 5:
                d = r["rows"]["late@2"]
                d.update(how="letter", ans=r["track"]["S"], word=r["track"]["B"] if k < 3 else "other")
            k += 1
    edit_json(root / "stories" / "TEST_llama8.json", letters)

    def eletters(J):
        for c in J["cores"]:
            cell = c["cells"]["LETTER@3"]
            for d in [v for n, v in cell.items() if n != "caa"] + list(cell["caa"].values()):
                d["word"] = "other"
        c = J["cores"][0]["cells"]["LETTER@3"]
        c["nat|S"] = dict(c["nat|S"], how="letter", word=J["cores"][0]["core"]["base"])
        c["E1|S"] = dict(c["E1|S"], how="letter", word=c["E1|S"]["ans"])
    edit_json(root / "edits" / "TEST_llama8.json", eletters)
    rc, text, J = score(root)
    ctl = text[text.index("#### CONTROLS"):text.index("#### PREDICTIONS")]
    assert "BF16 floor mean|E(none) - E(clean B)| 0.0010, E-argmax differs 0.000 (n=10)" in ctl
    assert "in-batch determinism max|E(none) - E(none2)| 0.0e+00, answers differ 0.000" in ctl
    assert "BF16 floor mean|lp(none) - lp(clean B)| 0.0020, decision argmax differs 0.000 (n=10)" in ctl
    assert "floor mean|E(self) - E(clean B)| 0.0000 (E-argmax differs 0.000, answers differ 0.000)" in ctl
    assert "stories LETTER    letter/word conflicts: 0.600 (3/5) of the letter answers name another candidate word" in ctl
    assert "by row: late@2 3" in ctl
    assert "edits LETTER@3 letter/word conflicts: 0.500 (1/2) of the letter answers" in ctl and "by row: nat 1" in ctl
    assert J["stories/llama8/NONE:bf16_floor_mean_abs_E"]["pt"] == pytest.approx(0.001)
    assert J["stories/llama8/NONE:bf16_floor_mean_abs_E"]["argmax_differs"] == 0.0
    assert J["passages/llama8/E:NOM:bf16_floor_mean_abs_lp"]["pt"] == pytest.approx(0.002)
    assert J["edits/llama8/NONE@3:bf16_floor_mean_abs_E"]["pt"] == 0.0
    assert J["stories/llama8/LETTER:letter_word_conflicts"]["count"] == 3 and J["stories/llama8/LETTER:letter_word_conflicts"]["n"] == 5
    assert J["edits/llama8/LETTER@3:letter_word_conflicts"]["count"] == 1
    S = DA.StoryRun(json.loads((root / "stories" / "TEST_llama8.json").read_text()))
    assert S.e_floor("AFTER")[0] == pytest.approx(0.001) and S.determinism("AFTER")[:2] == (0.0, 0.0)


def test_partial_edits_is_not_evaluable_without_a_mismatch(full_run, tmp_path):
    """A deadline-partial edits file (cores_not_run) makes that model's edits lines NOT EVALUABLE (scoring a subset of
    the cores would change the population) but, like a partial stories or passages file, is not a MISMATCH: exit 0."""
    root = mutate(full_run, tmp_path / "r", setp("edits/llama8.json", cores_not_run=[79]))
    rc, text, J = score(root, test=False)
    assert rc == 0 and "checks: OK" in text and J["meta/run/checks"]["mismatch"] is False
    assert "PARTIAL llama8 edits: edits/llama8.json: incomplete at the deadline: 1 cores not run" in text
    assert J["meta/run/checks"]["partial"] == ["llama8/edits"] and "Partial at the deadline" in text
    assert J["meta/run/checks"]["partial_whole_arm"] == ["llama8/edits"] and "llama8 edits (the whole arm NOT EVALUABLE)" in text
    for c in ("P5a", "B1", "B2", "B3"):
        assert verdict(J, c, "llama8") == ST.NE and "partial at the deadline" in J[f"verdict/llama8/{c}"]["why"], c
        assert verdict(J, c) == ST.NE and verdict(J, c, "qwen7") == ST.SUP, c
    assert J["verdict/llama8/P5c"]["transfers"] is None and verdict(J, "P5b") == ST.SUP and verdict(J, "P1") == ST.SUP


def test_row_grids_equal_the_builders():
    """The scorer's own rebuild of SPEC 7.1 and 8.1 (the lists every onset is read from) equals the builders' row lists
    and the SPEC's counts: 43 / 39 story rows at L = 32 / 28, 21 passage rows, 46 / 8 edit rows at the dose / late blocks
    (2 fewer without E4b)."""
    from experiments import stage9_edits as SE, stage9_passages as SP, stage9_stories as SS
    for L in (24, 28, 32):
        assert CK.story_rows(L) == SS.row_names(L) == story_rows(L)
        assert CK.passage_rows(L) == list(SP.window_rows(L)) == passage_rows(L)
    assert (len(CK.story_rows(32)), len(CK.story_rows(28)), len(CK.passage_rows(32)), len(CK.passage_rows(28))) == (43, 39, 21, 21)
    for l in (3, 7, 11, 15):
        for e4b in (True, False):
            assert CK.edit_rows(l, e4b) == SE.row_names(l, e4b) == ["self"] + [f"{k}|{t}" for k in edit_kinds(l, e4b)
                                                                                for t in ("S", "X")] + ["self2"]
    assert [len(CK.edit_rows(l, e)) for l in (3, 11) for e in (True, False)] == [46, 44, 8, 6]


def test_p2_per_model_is_the_stories_part(tmp_path):
    """P2p is a "2 of 3" line: a model whose OPTA flip is above 0.4 does not make P2 fail, so the per-model P2 entry is
    the stories part, with that model's P2p level beside it (not a NOT SUPPORTED that contradicts the combined line)."""
    root = make_run(tmp_path / "r", Scen(passages={("llama8", "E/OPTA", "late@10"): 0.6}), n_st=20, n_e=20, n_ed=10)
    rc, text, J = score(root)
    assert verdict(J, "P2p", "llama8") == ST.NOT and verdict(J, "P2p") == ST.SUP and verdict(J, "P2") == ST.SUP
    assert all(verdict(J, "P2", k) == ST.SUP == verdict(J, "P2s", k) for k in KEYS)
    assert "P2p here NOT SUPPORTED" in J["verdict/llama8/P2"]["note"] and "P2p here SUPPORTED" in J["verdict/qwen7/P2"]["note"]
    assert re.search(r"^    P2 +SUPPORTED +mistral7 SUPPORTED, llama8 SUPPORTED, qwen7 SUPPORTED$", text, re.M)
    assert "llama8 SUPPORTED (P2p here NOT SUPPORTED)" in text
    assert verdict(J, "P1", "llama8") == ST.SUP and verdict(J, "P3", "llama8") == ST.SUP     # P1, P3: both parts per model


def test_reading_withheld_when_a_falsifier_part_is_not_evaluable(tmp_path):
    """No passages for llama8 (e.g. a deadline cut in the last step): F2p cannot be read there, so F2 is NOT EVALUABLE;
    the note's main reading needs F1-F3 not triggered, so it is withheld and the missing part is named."""
    root = make_run(tmp_path / "r", n_st=20, n_e=20, n_ed=20, drop=(("llama8", "passages"),))
    rc, text, J = score(root)
    assert verdict(J, "P1s") == ST.SUP and verdict(J, "P2s") == ST.SUP and verdict(J, "P3s") == ST.SUP
    assert fal(J, "F2p") is None and fal(J, "F2p", "llama8") is None and fal(J, "F2s") is False and fal(J, "F2") is None
    assert fal(J, "F1") is False and fal(J, "F3") is False
    R = J["meta/run/checks"]["readings"]
    assert any("so it is withheld" in r and "F2p not evaluable in llama8" in r and "F2 NOT EVALUABLE" in r for r in R), R
    assert "a model has finished using a context token" not in text and "No pre-written reading" not in text


def test_t_check_is_a_mismatch_without_changing_a_verdict(full_run, tmp_path):
    root = mutate(full_run, tmp_path / "r", lambda r: edit_json(r / "edits" / "qwen7.json",
                                                               lambda J: J["cores"][0]["T_check"].update({"3": 0.0625})))
    rc, text, J = score(root, test=False)
    assert rc == 2 and "T_check nonzero" in text and verdict(J, "P5a") == ST.SUP and verdict(J, "B3") == ST.SUP


def test_pytest_gate(full_run, tmp_path):
    root = mutate(full_run, tmp_path / "a", lambda r: (r / "logs" / "pytest.log").unlink())
    rc, text, J = score(root, test=False)
    assert rc == 2 and "FP32 tests not passed: no outcome" in text
    assert all(verdict(J, c) == ST.NE for c in ALL_LINES)
    assert all(verdict(J, c, k) == ST.NE for c in ("P1s", "P5a") for k in KEYS)
    # a failed stage-9 test file in the last session
    root = mutate(full_run, tmp_path / "b", lambda r: None)
    make_log = (root / "logs" / "pytest.log").read_text().replace("tests/test_stage9_edits.py::test_one PASSED",
                                                                  "tests/test_stage9_edits.py::test_one FAILED")
    (root / "logs" / "pytest.log").write_text(make_log)
    rc, text, J = score(root, test=False)
    assert rc == 2 and "tests/test_stage9_edits.py 0 passed, 1 failed or error" in text and verdict(J, "P3") == ST.NE
    # the gate reads the last session that ran the files
    (root / "logs" / "pytest.log").write_text(make_log + (full_run / "logs" / "pytest.log").read_text())
    rc, text, J = score(root, test=False)
    assert rc == 0 and verdict(J, "P3") == ST.SUP
    # a stage-9 file absent from the session
    (root / "logs" / "pytest.log").write_text(make_log.replace("tests/test_stage9_pre.py::test_one FAILED", "")
                                              .replace("tests/test_stage9_edits.py::test_one FAILED", "tests/test_stage9_edits.py::test_one PASSED")
                                              .replace("tests/test_stage9_pre.py::test_one PASSED", ""))
    rc, text, J = score(root, test=False)
    assert rc == 2 and "tests/test_stage9_pre.py 0 passed" in text


def test_test_mode_waives_the_gate_and_the_freeze(tmp_path):
    root = make_run(tmp_path / "r", n_st=20, n_e=20, n_ed=20, pytest_ok=None)
    rc, text, J = score(root)
    assert rc == 0 and "TEST MODE" in text and "(TEST: waived)" in text and verdict(J, "P1s") == ST.SUP
    assert "TEST MODE: the verdicts above are plumbing checks, not results." in text
    # the scorer detects TEST_ files without --test
    assert SC.main(["--results", str(root)]) == 0


def test_missing_model(tmp_path):
    root = make_run(tmp_path / "r", keys=("mistral7", "qwen7"), n_st=20, n_e=20, n_ed=20)
    (root / "FETCH_FAILED.txt").write_text("2026-10-15T10:00:00Z FETCH REFUSED llama8 (exit 1)\n")
    rc, text, J = score(root)
    assert verdict(J, "P1s", "llama8") == ST.NE and "FETCH_FAILED.txt: llama8" in J["verdict/llama8/P1s"]["why"]
    assert verdict(J, "P1s") == ST.NE and verdict(J, "P5b") == ST.SUP                   # P5b: qwen7 and mistral7 only
    assert verdict(J, "P2s") == ST.NE and verdict(J, "P1") == ST.NE and verdict(J, "P2") == ST.NE
    assert verdict(J, "P2p") == ST.SUP
    assert re.search(r"^  P1s .*-> NOT EVALUABLE \(2 of 3 evaluable; not evaluable: llama8\)", text, re.M)


def test_scorer_error_exits_1(tmp_path, monkeypatch):
    root = make_run(tmp_path / "r", keys=("qwen7",), n_st=10, n_e=10, n_ed=10)
    monkeypatch.setattr(PR, "p5c", lambda ctx, m: 1 / 0)
    rc = SC.main(["--results", str(root), "--test"])
    assert rc == 0                                            # a model's line that raises is NOT EVALUABLE, not an error
    monkeypatch.setattr(SC.RP, "controls", lambda ctx, out: 1 / 0)
    assert SC.main(["--results", str(root), "--test"]) == 1
    assert "CONTROLS ERROR" in (root / "STAGE9_SCORE.txt").read_text()


# --------------------------------------------------------------------------- the pipeline's lists
def test_pipeline_runs_every_stage9_test_and_scores_with_this_scorer():
    sh = (ROOT / "scripts" / "gpu_stage9.sh").read_text()
    m = re.search(r"s8_pytest ((?:[^\n]*\\\n)*[^\n]*)", sh)
    files = re.findall(r"tests/\S+\.py", m.group(1))
    assert set(CK.STAGE9_TESTS) <= set(files)
    assert set(files) >= {"tests/test_fresh.py", "tests/test_fresh_factorial.py", "tests/test_generate.py", "tests/test_surface.py",
                          "tests/test_clamp.py", "tests/test_natural_clamp.py", "tests/test_stage8_edits.py",
                          "tests/test_stage8_populations.py"}
    assert all((ROOT / f).exists() for f in files if "stage9_pre" not in f)
    # tests/test_paper_s9p.py checks paper/numbers.tex, which may be regenerated after the freeze: not a gate (SPEC 12.2)
    assert "tests/test_paper_s9p.py" not in files and "tests/test_paper_s9p.py" not in CK.STAGE9_TESTS
    assert "s9_finish analysis/stage9_score.py" in sh
    for f in ("gpu_stage9.sh", "stage9_common.sh"):
        assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / f)]).returncode == 0
    gi = (ROOT / ".gitignore").read_text().splitlines()
    assert {"results/gpu_stage9_test/", "gpu_stage9_results.tgz", "gpu_stage9_test_results.tgz"} <= set(gi)
