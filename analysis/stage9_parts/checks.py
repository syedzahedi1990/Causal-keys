"""Checks of the stage-9 scorer (docs/stage9/SPEC.md sections 11.1, 12.1, 14): provenance, populations, the stage-8
artefact pins, the row grids (SPEC 7.1 and 8.1, rebuilt here independently of the builders), l_b, T_check and the
pytest-log gate. Each failure is recorded against the model and arm it concerns (its predictions are then NOT
EVALUABLE) or against the run; every failure prints MISMATCH and makes the scorer exit 2 outside TEST. A deadline-partial
arm is recorded apart, as PARTIAL (listed in CHECKS and SUMMARY and in stage9_summary.json's meta/run/checks "partial"),
never as a MISMATCH, and does not make the scorer exit 2: an edits file with cores_not_run makes the model's edits
predictions NOT EVALUABLE (a subset of the cores would change the population); a stories file with formats not run, a
passages or LETA file with (split, format) pieces not run, and a calibration left as lb/<tag>.json.partial leave the
parts that were run evaluable, and every statistic that needs a missing part is NOT EVALUABLE.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path

from .data import PASSAGE_FORMATS, STORY_FORMATS, lb_of, late_flips_all, passage_onsets, story_onsets

KEYS = ("mistral7", "llama8", "qwen7")
E4B_KEYS = ("qwen7", "mistral7")
N_POP = {"N9": 150, "C": 30, "E9": 80, "E": 185}
N_LAYERS = {"mistral7": 32, "llama8": 32, "qwen7": 28}      # SPEC 4 (TEST_MODE: the 0.5B model's 24, not checked)
KV_ONLY_ONSETS = (4, 8, 12, 16)                              # SPEC 7.1
EDIT_BLOCKS, DOSE_BLOCKS = (3, 7, 11, 15), (3, 7)            # SPEC 8.1
MULTS, ALPHAS_J, ALPHAS_KV = (0.5, 0.75, 1.5), (0.25, 0.5, 0.65, 0.8, 0.9, 1.25), (0.5, 0.8)
CAA_ROWS = ["self", "CAA|S", "CAA|X"]
STAGE9_TESTS = ("tests/test_stage9_pops.py", "tests/test_stage9_stories.py", "tests/test_stage9_passages.py",
                "tests/test_stage9_edits.py", "tests/test_stage9_score.py", "tests/test_stage9_guard.py",
                "tests/test_stage9_pre.py")
FREEZE_RE = re.compile(r"stage-9 predictions frozen at ([0-9a-f]{7,40}) \((.*?)\); note sha256 ([0-9a-f]{64}); "
                       r"spec sha256 ([0-9a-f]{64}); HEAD ([0-9a-f]{7,40})")


def sha_file(f) -> str:
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def keyof(rel: str) -> str:
    """The model key of a results file: its tag without TEST_ and without the arm suffix."""
    stem = Path(rel).stem
    for suf in ("_stories", "_edits", "_passages", "_LETA"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem[5:] if stem.startswith("TEST_") else stem


def armof(rel: str) -> str:
    d, stem = rel.split("/")[0], Path(rel).stem
    if d == "check":
        return stem.rsplit("_", 1)[1]
    if d in ("lb", "stories"):
        return "stories"
    if d == "passages":
        return "leta" if stem.endswith("_LETA") else "passages"
    return d


class Freeze:
    """The freeze record(s) the guard appended to COMMIT.txt."""

    def __init__(self, root: Path):
        t = (root / "COMMIT.txt").read_text(errors="replace") if (root / "COMMIT.txt").exists() else ""
        self.recs = FREEZE_RE.findall(t)
        u = {(F, n, s) for F, _, n, s, _ in self.recs}
        self.unique = len(u) == 1
        self.F, self.note, self.spec = (sorted(u)[0] if len(u) == 1 else (None, None, None))
        self.heads = sorted({h for *_, h in self.recs})


class Log:
    """MISMATCH bookkeeping: per (model, arm) and for the whole run; deadline-partial arms apart (not a MISMATCH)."""

    def __init__(self):
        self.bad: dict = {}    # (model, arm) -> reasons: that arm's predictions in that model are NOT EVALUABLE
        self.run: list = []    # reasons that make every verdict NOT EVALUABLE (one commit, the freeze record, ...)
        self.warn: list = []   # MISMATCH lines that make the scorer exit 2 without touching a verdict (T_check)
        self.part: dict = {}   # (model, arm) -> reasons: deadline-partial, listed as PARTIAL; no MISMATCH (no exit 2)
        self.whole: set = set()  # the (model, arm) of part whose whole arm is NOT EVALUABLE (edits: cores not run)

    def flag(self, key, arm, why):
        self.bad.setdefault((key, arm), []).append(why)

    def partial(self, key, arm, why, whole: bool = False):
        """A deadline-partial arm. ``whole``: every prediction of the arm in that model is NOT EVALUABLE (edits); else only
        the statistics that need the parts not run are (stories formats, passage pieces), which they are by themselves."""
        self.part.setdefault((key, arm), []).append(why)
        if whole:
            self.whole.add((key, arm))

    def why(self, key, arm) -> list:
        return self.bad.get((key, arm), [])

    def why_partial(self, key, arm) -> list:
        """The reasons that make the whole arm NOT EVALUABLE (an arm whose missing parts are NOT EVALUABLE by themselves
        gives none)."""
        return self.part.get((key, arm), []) if (key, arm) in self.whole else []

    def any(self) -> bool:
        return bool(self.run) or bool(self.warn) or any(self.bad.values())


# --------------------------------------------------------------------------- pytest gate
def pytest_gate(root: Path, files=STAGE9_TESTS):
    """(ok True / False / None (no log or no session that ran them), text) from the last pytest session in the logs that
    ran any of the files: every file with at least one pass, no failure or error in the session."""
    logs = sorted((root / "logs").glob("*pytest*")) if (root / "logs").is_dir() else []
    best = None
    for f in logs:
        t = f.read_text(errors="replace")
        for sess in re.split(r"=+ test session starts =+", t)[1:]:
            if any(x in sess for x in files):
                best = (f.name, sess)
    if best is None:
        return None, "FP32 tests not passed: no outcome (no pytest log that ran the stage-9 test files)"
    sess = best[1]
    parts, ok = [], True
    for x in files:
        st = {}
        for m in re.finditer(rf"({re.escape(x)}::\S+) (PASSED|FAILED|ERROR|SKIPPED)|(PASSED|FAILED|ERROR|SKIPPED) "
                             rf"({re.escape(x)}::\S+)", sess):
            tid, w = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
            st[tid] = w
        p = sum(w == "PASSED" for w in st.values())
        bad = sum(w in ("FAILED", "ERROR") for w in st.values())
        s = sum(w == "SKIPPED" for w in st.values())
        good = p >= 1 and bad == 0
        ok &= good
        parts.append(f"{x} {p} passed, {bad} failed or error, {s} skipped" + ("" if good else " NOT PASSED"))
    other = len(re.findall(r"(?:^|\s)(?:FAILED|ERROR) tests/\S+", sess)) + len(re.findall(r"tests/\S+::\S+ (?:FAILED|ERROR)", sess))
    if other and ok:
        ok = False
        parts.append(f"{other} failures or errors in other files of the session")
    return ok, f"{best[0]} (its last session that ran them): " + "; ".join(parts)


# --------------------------------------------------------------------------- provenance
def manifest_revisions(repo: Path) -> dict:
    try:
        J = json.loads((repo / "scripts" / "stage8_models.json").read_text())
        return {k: v["revision"] for k, v in J["models"].items()}
    except Exception:  # noqa: BLE001
        return {}


def _utc(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:  # noqa: BLE001
        return None


def provenance(root: Path, repo: Path, F: dict, test: bool, log: Log, out, pins: dict):
    """The PROVENANCE section: every results file's provenance against the run's rules; returns the commits seen."""
    fr = Freeze(root)
    for name in ("COMMIT.txt", "ENV.txt", "REVISIONS.txt", "SKIPPED.txt", "FETCH_FAILED.txt", "FAILED.txt"):
        f = root / name
        out(f"  {name}:" + ("".join(f"\n    {l.rstrip()}" for l in f.read_text(errors="replace").splitlines()[-40:]
                                     if l.strip()) if f.exists() else " absent"))
    if fr.recs:
        out(f"  freeze record(s) in COMMIT.txt: {len(fr.recs)}; F {fr.F}, note sha256 {str(fr.note)[:16]}, spec sha256 "
            f"{str(fr.spec)[:16]}, HEAD {', '.join(h[:10] for h in fr.heads)}"
            + ("" if fr.unique else "  MISMATCH: the records name more than one freeze commit or sha256"))
    else:
        out("  freeze record in COMMIT.txt: none" + (" (TEST: no guard)" if test else "  MISMATCH: the guard's record is missing"))
    if not test and (not fr.recs or not fr.unique):
        log.run.append("COMMIT.txt has no unique stage-9 freeze record")
    revs = manifest_revisions(repo)
    commits = {}
    for rel, J in sorted(F.items()):
        if rel.startswith("verified/"):
            continue
        key, arm = keyof(rel), armof(rel)
        if "error" in J or "provenance" not in J:
            out(f"  {rel}: " + (f"UNREADABLE {J.get('error')}" if "error" in J else "NO PROVENANCE") + "  MISMATCH")
            log.flag(key, arm, f"{rel} unreadable or without provenance")
            continue
        p = J["provenance"]
        commits.setdefault(p.get("git_commit"), []).append(rel)
        flag = []
        if p.get("stage9") is not True:
            flag.append("not a stage-9 file (provenance.stage9 is not true)")
        if p.get("model_key") not in (None, key):
            flag.append(f"model_key {p.get('model_key')} != the file's key {key}")
        ver = p.get("verified") or {}
        rev = p.get("revision") or ver.get("revision")
        if not test:
            if Path(rel).name.startswith("TEST_") or p.get("test_mode"):
                flag.append("a TEST_MODE file outside TEST")
            if p.get("git_dirty"):
                flag.append("written from a tree with modified tracked files (code, note or spec; SPEC 11.1)")
            if fr.unique and fr.recs:
                if p.get("freeze_commit") != fr.F:
                    flag.append(f"freeze_commit {p.get('freeze_commit')} != COMMIT.txt's {fr.F}")
                if p.get("predictions_sha256") != fr.note:
                    flag.append("predictions_sha256 differs from COMMIT.txt's note sha256")
                if p.get("spec_sha256") != fr.spec:
                    flag.append("spec_sha256 differs from COMMIT.txt's spec sha256")
            if "dtype" in p and not str(p.get("dtype")).endswith("bfloat16"):
                flag.append(f"dtype {p.get('dtype')} is not BF16")
            if "attn_implementation" in p and p.get("attn_implementation") != "sdpa":
                flag.append(f"attention {p.get('attn_implementation')} is not sdpa")
            if key in revs and rev != revs[key]:
                flag.append(f"revision {rev} != the manifest's {revs[key]}")
            if not ver:
                flag.append("no verified model files recorded")
            flag += artefact_flags(rel, arm, key, p, pins)
        for x in flag:
            log.flag(key, arm, f"{rel}: {x}")
        out(f"  {rel}: commit {str(p.get('git_commit'))[:10]}, dirty {p.get('git_dirty')}, freeze "
            f"{str(p.get('freeze_commit'))[:10]}, model {p.get('model')} (key {p.get('model_key')}), revision {str(rev)[:10]}, "
            f"dtype {p.get('dtype', '-')}, attn {p.get('attn_implementation', '-')}, device {p.get('device', '-')}, "
            f"transformers {p.get('transformers')}, test_mode {p.get('test_mode')}" + "".join(f"  MISMATCH: {x}" for x in flag))
    if len(commits) > 1:
        log.run.append(f"the results files carry {len(commits)} commits")
    out(f"  commits: {sorted(str(c)[:10] for c in commits)}" + ("  MISMATCH: not one commit" if len(commits) > 1 else ""))
    if not test and fr.heads and commits and set(commits) - set(fr.heads):
        log.run.append("a results file's commit is not a HEAD recorded with the freeze record")
        out("  MISMATCH: a results file's commit is not among the HEADs of the freeze records")
    return commits


def artefact_flags(rel: str, arm: str, key: str, p: dict, pins: dict) -> list:
    """The stage-8 artefacts a file records against their pins (SPEC section 4)."""
    flag = []
    if arm == "stories":
        for kind in ("frames", "g3"):
            if f"{kind}_sha256" in p:
                want = pins.get(f"results/gpu_stage8b/{kind}/{key}.json")
                if p.get(f"{kind}_sha256") != want:
                    flag.append(f"{kind} sha256 {str(p.get(f'{kind}_sha256'))[:12]} != the pin {str(want)[:12]}")
        if "scorer" in p and p.get("scorer") != "trie":
            flag.append(f"scorer {p.get('scorer')} is not the trie G3 chose")
    elif arm in ("passages", "leta"):
        want = pins.get(f"results/gpu_stage8a/frames/{key}.json")
        if p.get("frames_sha256") != want:
            flag.append(f"frames sha256 {str(p.get('frames_sha256'))[:12]} != the pin {str(want)[:12]}")
    elif arm == "edits":
        want = pins.get(f"results/gpu_stage8c/calib/{key}.pt")
        if p.get("calib_sha256") != want:
            flag.append(f"calib sha256 {str(p.get('calib_sha256'))[:12]} != the pin {str(want)[:12]}")
        want = pins.get(f"results/gpu_stage8c/das/{key}.pt") if key in E4B_KEYS else None
        if p.get("das_sha256") != want:
            flag.append(f"das sha256 {str(p.get('das_sha256'))[:12]} != the pin {str(want)[:12]}")
        if p.get("das_seed", 102) != 102:
            flag.append(f"das seed {p.get('das_seed')} is not 102")
    return flag


# --------------------------------------------------------------------------- populations, l_b, T_check
def population(F: dict, test: bool, log: Log, out, pop_pins: dict, stage8a: dict):
    """The POPULATION section: pinned populations, sizes, skipped items, partial files, the passage E set (TEST: the pins,
    sizes and the stage-8a comparison are waived; a failed check file still counts)."""
    ok_all = True
    for rel, J in sorted(F.items()):
        if "provenance" not in J:
            continue
        p, key, arm = J["provenance"], keyof(rel), armof(rel)
        flag, info, part = [], [], []
        d = rel.split("/")[0]
        if d == "check":
            info.append(f"pass {J.get('pass')}")
            if not J.get("pass"):
                flag.append(f"the check failed: {J.get('fails')}")
        elif d in ("lb", "stories"):
            pop = "C" if d == "lb" else "N9"
            sk = p.get("skipped_items") or []
            info.append(f"population {p.get('population')}, {p.get('n_items')} items, formats done {p.get('formats_done')}, "
                        f"not run {p.get('formats_not_run') or 'none'}, skipped {len(sk)}")
            nr = [f for f in (p.get("formats_not_run") or [])]
            have = {r.get("fmt") for r in J.get("items") or []}
            absent = [f for f in STORY_FORMATS if f not in have and f not in nr]
            if nr:
                part.append(f"incomplete at the deadline: formats {nr} not run (exit 3; the next session resumes it); "
                            "their statistics are NOT EVALUABLE")
            if absent:
                part.append(f"formats {absent} absent from the file (not run); their statistics are NOT EVALUABLE")
            if not test and (p.get("population") != pop or p.get("population_sha256") != pop_pins.get(pop)):
                flag.append(f"population {p.get('population')} sha256 {str(p.get('population_sha256'))[:12]} != the pin of {pop}")
            if not test and p.get("n_items") != N_POP[pop]:
                flag.append(f"{p.get('n_items')} items, not {N_POP[pop]}")
            if sk:
                flag.append(f"{len(sk)} skipped items (0 expected)")
        elif d == "edits":
            sk = p.get("skipped_items") or []
            nr = p.get("cores_not_run") or []
            idx = [c["index"] for c in J.get("cores", [])]
            info.append(f"{len(idx)} of {p.get('n_items')} cores, not run {len(nr)}, skipped {len(sk)}, blocks {p.get('blocks')}, "
                        f"formats {p.get('formats')}, e4b {p.get('e4b')}")
            if not test and p.get("population_sha256") != pop_pins.get("E9"):
                flag.append(f"population sha256 {str(p.get('population_sha256'))[:12]} != the pin of E9")
            if not test and p.get("n_items") != N_POP["E9"]:
                flag.append(f"{p.get('n_items')} cores, not {N_POP['E9']}")
            if sk:
                flag.append(f"{len(sk)} skipped cores (0 expected)")
            if nr:
                part.append(f"incomplete at the deadline: {len(nr)} cores not run (exit 3; the next session resumes it)")
            if idx != list(range(len(idx))):
                flag.append("core indices are not 0..n-1 in order")
        elif d == "passages":
            vs = p.get("valid_sha256") or {}
            ni = p.get("n_items") or {}
            sk = p.get("skipped_items") or {}
            s8 = stage8a.get(key)
            info.append(f"valid E {ni.get('E')} (sha256 {str(vs.get('E'))[:12]}), R {ni.get('R')} (sha256 {str(vs.get('R'))[:12]}); "
                        f"stage 8a E {str(s8)[:12]}; pieces not run {p.get('formats_not_run') or 'none'}")
            nr = [x for x in (p.get("formats_not_run") or [])]
            want = ([("E", "LETA")] if arm == "leta" else [(s, f) for s in ("E", "R") for f in PASSAGE_FORMATS])
            got = {(s, f) for s, fm in (J.get("splits") or {}).items() for f in fm}
            absent = [f"{s}/{f}" for s, f in want if (s, f) not in got and f"{s}/{f}" not in nr]
            if nr:
                part.append(f"incomplete at the deadline: pieces {nr} not run (exit 3; the next session resumes it); "
                            "their statistics are NOT EVALUABLE")
            if absent:
                part.append(f"pieces {absent} absent from the file (not run); their statistics are NOT EVALUABLE")
            if not test:
                if vs.get("E") != s8 or p.get("stage8a_population_sha256") != s8:
                    flag.append("the valid E set differs from stage 8a's population (MISMATCH)")
                if ni.get("E") != N_POP["E"]:
                    flag.append(f"{ni.get('E')} valid E items, not {N_POP['E']}")
                if sk.get("E"):
                    flag.append(f"{len(sk['E'])} E items skipped (0 expected)")
        else:
            continue
        if d != "check":   # lb, stories, edits, passages (and LETA)
            flag = grid_flags(rel, J, key, test) + flag
        for x in flag:
            log.flag(key, arm, f"{rel}: {x}")
        for x in part:
            log.partial(key, arm, f"{rel}: {x}", whole=(d == "edits"))
        ok_all &= not flag
        out(f"  {rel}: " + "; ".join(info) + "".join(f"  MISMATCH: {x}" for x in flag) + "".join(f"  PARTIAL: {x}" for x in part))
    return ok_all


def partial_cal(root: Path, test: bool, log: Log, out):
    """A calibration step stopped at the deadline leaves lb/<tag>.json.partial and no lb file (its formats are resumed by
    the next session): PARTIAL stories (l_b and the stories eval are then missing, so their statistics are NOT
    EVALUABLE by themselves)."""
    d = root / "lb"
    for f in sorted(d.glob("*.json.partial")) if d.is_dir() else []:
        tag = f.name[: -len(".json.partial")]
        if test != tag.startswith("TEST_") or (d / f"{tag}.json").exists():
            continue
        try:
            done = json.loads(f.read_text())["provenance"].get("formats_done")
        except Exception:  # noqa: BLE001
            done = "unreadable"
        key = tag[5:] if tag.startswith("TEST_") else tag
        why = f"lb/{f.name}: the calibration (cal) is incomplete at the deadline (formats done {done}); no lb file yet"
        log.partial(key, "stories", why)
        out(f"  {why}  PARTIAL")


# --------------------------------------------------------------------------- the row grids (SPEC 7.1, 8.1)
def story_rows(nL: int) -> list[str]:
    """SPEC 7.1, stories: none, KV@0, late@l, early@l (l in story_onsets), K@0, V@0, Klate@l, Vlate@l (l in 4, 8, 12, 16
    below L), none2: 43 rows at L = 32, 39 at 28."""
    on, kv = story_onsets(nL), [l for l in KV_ONLY_ONSETS if l < nL]
    return (["none", "KV@0"] + [f"late@{l}" for l in on] + [f"early@{l}" for l in on] + ["K@0", "V@0"]
            + [f"Klate@{l}" for l in kv] + [f"Vlate@{l}" for l in kv] + ["none2"])


def passage_rows(nL: int) -> list[str]:
    """SPEC 7.1, passages: the same pattern with passage_onsets(L) for every windowed row (21 rows)."""
    on = passage_onsets(nL)
    return (["none", "KV@0"] + [f"late@{l}" for l in on] + [f"early@{l}" for l in on] + ["K@0", "V@0"]
            + [f"Klate@{l}" for l in on] + [f"Vlate@{l}" for l in on] + ["none2"])


def edit_rows(l: int, e4b: bool) -> list[str]:
    """SPEC 8.1, the table rows of a cell at block l in batch order: self, then per kind the targets S and X, then self2
    (8 rows at blocks 11 and 15, 46 at the dose blocks; 2 fewer without E4b)."""
    kinds = ["nat", "E1"] + (["E4b"] if e4b else [])
    if l in DOSE_BLOCKS:
        kinds += [f"E1x{c}" for c in MULTS] + [f"J{a}" for a in ALPHAS_J] + ["K1", "V1"]
        kinds += [f"K{a}" for a in ALPHAS_KV] + [f"V{a}" for a in ALPHAS_KV]
        kinds += [f"K{a}V1" for a in ALPHAS_KV] + [f"K1V{a}" for a in ALPHAS_KV]
    return ["self"] + [f"{k}|{t}" for k in kinds for t in ("S", "X")] + ["self2"]


def _diff(got: list, want: list) -> str:
    miss, extra = [x for x in want if x not in got], [x for x in got if x not in want]
    return (f"{len(got)} rows, {len(want)} expected" + (f"; missing {miss[:4]}" if miss else "")
            + (f"; unexpected {extra[:4]}" if extra else "") + ("; another order" if not miss and not extra else ""))


def grid_flags(rel: str, J: dict, key: str, test: bool) -> list:
    """The file's row grid against SPEC 7.1 (stories, lb, passages, LETA: the "rows" list for the recorded n_layers) or
    8.1 (edits: every cell's table rows for its block and the file's e4b, the CAA batch, e4b = the key has a stored
    basis); outside TEST also n_layers against SPEC 4. The onsets every statistic reads come from these lists."""
    p, d = J.get("provenance") or {}, rel.split("/")[0]
    nL = p.get("n_layers")
    flag = []
    if not isinstance(nL, int) or nL <= 0:
        return [f"n_layers {nL!r} is not a number of blocks: the row grid cannot be checked"]
    if not test and key in N_LAYERS and nL != N_LAYERS[key]:
        flag.append(f"n_layers {nL} != {N_LAYERS[key]} (SPEC 4)")
    if d in ("lb", "stories", "passages"):
        want = story_rows(nL) if d in ("lb", "stories") else passage_rows(nL)
        got = list(J.get("rows") or [])
        if got != want:
            flag.append(f"the row grid differs from SPEC 7.1 at L = {nL}: {_diff(got, want)}")
    elif d == "edits":
        e4b = p.get("e4b")
        if e4b is not (key in E4B_KEYS):
            flag.append(f"e4b {e4b!r}, but {key} {'has' if key in E4B_KEYS else 'has no'} stored E4b basis (SPEC 4)")
        bad = {}
        for c in J.get("cores") or []:
            for name, cell in (c.get("cells") or {}).items():
                fmt, _, l = name.partition("@")
                try:
                    want = edit_rows(int(l), bool(e4b))
                except ValueError:
                    bad.setdefault(name, f"cell {name} is not <format>@<block>")
                    continue
                if int(l) not in EDIT_BLOCKS:
                    bad.setdefault(name, f"block {l} is not one of {EDIT_BLOCKS}")
                got = [k for k in cell if k != "caa"]
                if got != want:
                    bad.setdefault(name, f"cell {name} (core {c.get('index')}): {_diff(got, want)}")
                if list(cell.get("caa") or {}) != CAA_ROWS:
                    bad.setdefault(name + "/caa", f"cell {name} (core {c.get('index')}): CAA batch {list(cell.get('caa') or {})} "
                                                  f"!= {CAA_ROWS}")
        if bad:
            flag.append(f"the edit rows differ from SPEC 8.1 in {len(bad)} cell(s): " + "; ".join(list(bad.values())[:3]))
    return flag


def lb_checks(root: Path, key: str, tag: str, lb, st, test: bool, log: Log, out):
    """l_b recomputed from the stored C rows (late AFTER flip over all C cores, the first onset <= 0.2) against the lb
    file, the eval file's record of it, the sha256 of the file the eval read, and the order (lb written before the eval
    started). Returns the recomputed l_b (or None)."""
    if lb is None:
        if st is not None:   # the eval reads the lb file before it starts: an eval without one read a file now gone
            log.flag(key, "stories", "l_b: the eval file exists but its lb file does not")
        out(f"  {key}: no lb file -> every statistic at l_b is NOT EVALUABLE" + ("  MISMATCH: the stories eval exists "
                                                                                 "without it" if st is not None else ""))
        return None
    fl = late_flips_all(lb, "AFTER")
    lb2 = lb_of(fl) if fl else None
    J = lb.J
    flag = []
    if J.get("l_b") != lb2:
        flag.append(f"the lb file's l_b {J.get('l_b')} != the recomputed {lb2}")
    rec = {str(l): round(v, 5) for l, v in fl.items()}
    if J.get("late_after_flip") and {k: round(float(v), 5) for k, v in J["late_after_flip"].items()} != rec:
        flag.append("the lb file's late AFTER flips differ from the stored rows")
    f = root / "lb" / f"{tag}.json"
    if st is not None:
        sp = st.P
        if sp.get("l_b") != lb2:
            flag.append(f"the eval recorded l_b {sp.get('l_b')} != {lb2}")
        if f.exists() and sp.get("lb_sha256") != sha_file(f):
            flag.append("the eval read another lb file (lb_sha256 differs from the file's)")
        if not str(sp.get("lb_file", "")).replace("\\", "/").endswith(f"lb/{tag}.json"):
            flag.append(f"the eval's lb_file {sp.get('lb_file')} is not lb/{tag}.json")
        a, b = _utc(lb.P.get("finished_utc")), _utc(sp.get("started_utc"))
        if a and b and a > b:
            flag.append(f"the lb file was finished ({lb.P.get('finished_utc')}) after the eval started ({sp.get('started_utc')})")
    for x in flag:
        log.flag(key, "stories", f"l_b: {x}")
    out(f"  {key}: l_b = {lb2} (recomputed from {lb.n} C cores; late AFTER flips "
        + ", ".join(f"{l}: {v:.3f}" for l, v in fl.items()) + f"); lb file {J.get('l_b')}; eval "
        + (f"{st.P.get('l_b')}, lb sha256 {str(st.P.get('lb_sha256'))[:12]}" if st is not None else "absent")
        + ("" if not flag else "".join(f"  MISMATCH: {x}" for x in flag)))
    return lb2


def t_check(ed, key, test: bool, log: Log, out):
    """The T|S pipeline check: max |T tables - natural S tables| per block over the cores (0 expected)."""
    if ed is None:
        return
    mx = {}
    for c in ed.cores:
        for l, v in (c.get("T_check") or {}).items():
            mx[l] = max(mx.get(l, 0.0), float(v))
    bad = {l: v for l, v in mx.items() if v > 0}
    if bad and not test:
        log.warn.append(f"{key}: T_check nonzero {bad}")
    out(f"  {key}: T_check max |T|S - nat S| per block {mx}" + (" (0 expected: OK)" if not bad else " (nonzero; 0 expected; "
                                                                 "TEST: not a check)" if test else "  MISMATCH: nonzero (0 "
                                                                 "expected: the edit pass has the natural pass's shape)"))
