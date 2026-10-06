"""Score preregistration P-2026-10-05-G (docs/PREREGISTRATION.md, GPU stage 5): the entry point over the five part
scorers of analysis/stage5_parts (a attention, b knockout, c subsets, d variants, e ioi), each exactly as written there.

Inputs under --root (default results/gpu_stage5), as written by scripts/gpu_stage5.sh: attention/, knockout/, factorial/,
row_restricted/, competence/, form_attention/, ioi/, ioi_attention/ (a part whose directory is absent is NOT EVALUABLE).
Output, also written to --out (default {root}/STAGE5_SCORE.txt): the gates of every part first, then one verdict line
per prediction G1-G22 in order (a prediction with sub-verdicts is MET when every sub-verdict is met, NOT EVALUABLE when
none can be evaluated, else NOT MET: not evaluable counts as not met, so an anchor gated out of part (a) counts as not
met in G1 and G4a, whereas G3 and G4b take their minimum over the gated-in anchors; a sub-verdict whose stated precondition failed,
G4b when G2 declared no H_diss (H_track, mixed, or no gated-in small model) or when G4a is not met, is NOT APPLICABLE and
left out; G2 is MET only under H_diss, H_track printing NOT MET as the declared alternative; G7's verdict is H_redundant, H_replaced is reported
beside it; an M8 failure of a present knockout model makes G5-G8 NOT EVALUABLE, a missing knockout file only counts as
not met in the k/k lines), the provenance of every results file (commit, model and revision, dtype, device, transformers,
torch, attention implementation, skipped items; MISMATCH when the files hold more than one commit or one model's files more
than one revision) with every line of the pipeline's COMMIT.txt / ENV.txt / REVISIONS.txt, the population checks (n per
arm against the preregistered sizes, the same cores in every arm of a file, and across the parts sharing a model: a
smaller part must hold the first k cores of the larger one in item order), and then each part's full report (tables,
exploratory block).
--tag TEST_<model> (what scripts/gpu_stage5.sh passes in TEST_MODE): every model found stands in for the preregistered
ones, the verdict lines are not results, and what the tiny model cannot be scored on prints NOT EVALUABLE.
Exit status 1 if a part scorer raised (its traceback is in the report and its predictions are NOT EVALUABLE); 2 if,
outside a TEST_ tag, the provenance or the population check reports MISMATCH (the score file is still written).
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from stage5_parts import attention, ioi, knockout, subsets, variants  # noqa: E402

MAIN = {"a": "attention", "b": "knockout", "c": "factorial", "d": "factorial", "e": "ioi"}
DIRS = ("attention", "knockout", "factorial", "row_restricted", "competence", "form_attention", "ioi", "ioi_attention")
NAME = {"a": "re-mention attention", "b": "attention knockout", "c": "membership and dose", "d": "non-identical re-mentions", "e": "IOI"}
TITLE = {"G1": "anchors attend: E, F/E at 7B/14B under SENTENCE-AFTER", "G2": "dissociation at the small models (H_diss; H_track = the alternative)",
         "G3": "magnitude under the declared account (R_A, story-paired anchor ratio)", "G4": "hop 2: (a) anchors G, (b) cross-scale Q",
         "G5": "necessity of the candidate-word edges, r_K(M1)", "G6": "matched control column M2", "G7": "the copy takes over and the answer stays (H_redundant)",
         "G8": "routes at the answer position: (a) NONE M3, (b) AFTER M3, (c) M4 vs M1", "G9": "membership (sentence S, list L)",
         "G10": "proportionality refuted (S, L)", "G11": "the copy carries the identity again (L3out vs L3)", "G12": "reader rows at k = 2 and 3 (splice, 12 cells)",
         "G13": "wrappers keep the read (THE, MODIF)", "G14": "an exact repeat reads most (6 variants)", "G15": "token-level on the paper's measure (SYN, FRMIX, DEMIX)",
         "G16": "disambiguation: token pattern on r_K^any and a_v (SYN, FRMIX, DEMIX)", "G17": "value compensation where the read is lost (FRMIX, DEMIX)",
         "G18": "no key read in plain IOI", "G19": "a later list opens a key read: (a) existence, (b) positive sign", "G20": "the lookup replaces the copy",
         "G21": "controls: (a) BEFORE, (b) QUESTION, (c) GPT-2 INLINE_BEFORE", "G22": "GPT-2 in-sentence re-mention: (a) key read, (b) inhibitory"}


def V(ok):
    return "NOT APPLICABLE" if ok is NA else "NOT EVALUABLE" if ok is None else "MET" if ok else "NOT MET"


NA = "n/a"  # a sub-verdict whose stated precondition failed: printed NOT APPLICABLE, left out of the combination
EXPECTED_N = {"attention": 150, "knockout": 150, "factorial": 150, "row_restricted": 60, "form_attention": 60, "ioi": 200}


def comb(subs):
    subs = [s for s in subs if s is not NA]
    return None if all(s is None for s in subs) else all(s is not None and bool(s) for s in subs)


def run_part(label, fn, *args, **kw):
    buf = []
    try:
        return buf, fn(*args, out=buf.append, **kw)
    except Exception:
        buf.append(f"  SCORER ERROR in part ({label}); its predictions are NOT EVALUABLE\n" + traceback.format_exc())
        return buf, None


def score_parts(root, test):
    """{part: (lines, result)} with result None when the directory is missing or the scorer raised."""
    R = {}
    for p in "abcde":
        if not (root / MAIN[p]).is_dir():
            R[p] = ([f"  no results: {root / MAIN[p]} missing"], None)
            continue
        if p == "a":
            R[p] = run_part(p, attention.score, root / "attention")
        elif p == "b":
            R[p] = run_part(p, knockout.score, root / "knockout", **({"models": [f.stem[:-3] for f in sorted((root / "knockout").glob("*_s0.json"))]} if test else {}))
        elif p == "c":
            R[p] = run_part(p, subsets.score, root, test=test)
        elif p == "d":
            R[p] = run_part(p, variants.score, root, test=test)
        else:
            R[p] = run_part(p, ioi.score, root, test=test)
    return R


def all_none(d, keys, models=None):
    """None when no model has a value for any of ``keys`` (nothing evaluable), else True iff every value is True."""
    vals = [d.get(m, {}).get(k) for m in (models or d) for k in keys]
    return comb(vals)


def verdicts(R, out):
    """One line per prediction G1-G22 in order; returns {name: True / False / None}."""
    res = {p: r for p, (_, r) in R.items()}
    F = {}

    def line(g, subs, note=""):
        """``subs`` = [(label, ok, text)]; one line, the sub-verdicts in order."""
        overall = comb(ok for _, ok, _ in subs)
        F[g] = overall
        parts = "; ".join(f"{lab}{': ' if lab else ''}{V(ok)}" + (f" ({txt})" if txt else "") for lab, ok, txt in subs)
        out(f"  {g:4s} {TITLE[g]:70s} {parts} -> {V(overall)}" + (f"  [{note}]" if note else ""))

    a = res.get("a") or {}
    v = a.get("verdicts", {})
    na = "" if res.get("a") else "part (a) not scored"
    for g in ("G1", "G2", "G3"):
        line(g, [("", *v.get(g, (None, "")))], na)
    g4a, g4b = v.get("G4a", (None, "")), v.get("G4b", (None, ""))
    if g4b[0] is None and g4b[1].endswith(("G2 did not declare H_diss", "G4a not met")):  # the entry's precondition of (b) failed (also: no small model gated in)
        g4b = (NA, g4b[1])
    line("G4", [("(a)", *g4a), ("(b)", *g4b)], na)
    b = res.get("b")
    nb = "" if b else "part (b) not scored"
    gate_b = bool(b) and all(v is not False for v in b["gate"].values())  # a MISSING model (None) is not an M8 failure
    some = bool(b) and bool(b["gate"])  # no model at all (a TEST root whose knockout files were moved aside): nothing measured
    nb = nb or ("" if gate_b else "Gate b FAILED: NOT EVALUABLE until the knockout is fixed and re-run") or ("" if some else "no knockout model found")
    g = lambda k: b[k] if gate_b and some else None  # noqa: E731
    line("G5", [("", g("G5"), "")], nb)
    line("G6", [("", g("G6"), "")], nb)
    hr = V(g("H_replaced"))
    line("G7", [("H_redundant", g("G7"), "")], nb or (f"H_replaced {hr}" + ("; partial takeover" if not b["G7"] and not b["H_replaced"] else
                                                                            "; both patterns: contradictory" if b["G7"] and b["H_replaced"] else "")))
    line("G8", [(lab, None if g("G8") is None else b["G8"][i], "") for i, lab in enumerate(("(a)", "(b)", "(c)"))], nb)
    c = res.get("c")
    nc = "" if c else "part (c) not scored"
    if c:
        ev = lambda k, models=None: None if all_none(c["pred"], [k], models) is None else c["final"][k]  # noqa: E731
        s12 = comb(c["cells"][m][a] for m in c["cells"] for a in subsets.SPLICE_ARMS) if c["cells"] else None
        line("G9", [("S", ev("G9 S"), ""), ("L", ev("G9 L"), "")])
        line("G10", [("S", ev("G10 S"), ""), ("L", ev("G10 L"), "")])
        line("G11", [("", ev("G11"), "")])
        line("G12", [("", None if s12 is None else c["final"]["G12"], "")])
    else:
        for g in ("G9", "G10", "G11", "G12"):
            line(g, [("", None, "")], nc)
    d = res.get("d")
    nd = "" if d else "part (d) not scored"
    fd = (d or {}).get("final", {})
    frame = "" if not d else "" if fd.get("G13 THE") is True else "G13 THE not met or not evaluable: G14-G17 frame-confounded"
    line("G13", [(k, fd.get(f"G13 {k}"), "") for k in ("THE", "MODIF")], nd)
    line("G14", [(k, fd.get(f"G14 {k}"), "") for k in ("TITLE", "UPPER", "PLURAL", "SYN", "FRMIX", "DEMIX")], nd or frame)
    line("G15", [(k, fd.get(f"G15 {k}"), "") for k in ("SYN", "FRMIX", "DEMIX")], nd or frame)
    line("G16", [(k, fd.get(f"G16 {k}"), (d or {}).get("decisions", {}).get(k, "").replace("not evaluable", "")) for k in ("SYN", "FRMIX", "DEMIX")], nd or frame)
    line("G17", [(k, fd.get(f"G17 {k}"), "") for k in ("FRMIX", "DEMIX")], nd or frame)
    e = res.get("e")
    ne = "" if e else "part (e) not scored"
    pe, fe, me = ((e or {}).get(k, {}) for k in ("pred", "final", "models"))
    ev = lambda k: None if not e or all_none(pe, [k], me[k]) is None else fe[k]  # noqa: E731  evaluability over the scored models only
    line("G18", [("", ev("G18"), "")], ne)
    line("G19", [("(a)", ev("G19a"), ""), ("(b)", ev("G19b"), "")], ne)
    line("G20", [("", ev("G20"), "")], ne)
    line("G21", [("(a)", ev("G21a"), ""), ("(b)", ev("G21b"), ""), ("(c)", ev("G21c"), "")], ne)
    line("G22", [("(a)", ev("G22a"), ""), ("(b)", ev("G22b"), "")], ne)
    return F


def gates(R, out):
    res = {p: r for p, (_, r) in R.items()}
    out("GATES (per part and model; a failing model is not evaluable in that part, which counts as not met)")
    g0 = {}
    for p in ("c", "d"):
        for m, v in ((res.get(p) or {}).get("gate0") or {}).items():
            g0.setdefault(m, v)
    out("  Gate 0 (reproduction of stage 3b; parts c and d): " + (", ".join(f"{m} {'passed' if v else 'no reference (TEST)' if v is None else 'FAILED'}" for m, v in g0.items()) or "NOT EVALUATED (no factorial)"))
    a = res.get("a")
    out("  Gate a (part a, per model): " + (", ".join(f"{m} ({a['role'][m]}) {'gated in' if v else 'GATED OUT'}" for m, v in a["gated"].items()) if a and a["gated"] else "NOT EVALUATED (no attention results)"))
    b = res.get("b")
    out("  Gate b (part b, M8 sanity): " + (", ".join(f"{m} {'MISSING' if v is None else V(v)}" for m, v in b["gate"].items())
                                            + f" -> {V(False if any(v is False for v in b['gate'].values()) else None if any(v is None for v in b['gate'].values()) else True)}"
                                            if b and b["gate"] else "NOT EVALUATED (no knockout results)"))
    d = res.get("d")
    if d:
        for m, md in d["models"].items():
            if not md.ok:
                out(f"  Gates d1-d3 (part d) {m}: factorial MISSING")
                continue
            d1 = [x for x, v in md.d1.items() if min(v) >= 0.95]
            out(f"  Gates d1-d3 (part d) {m}: d1 clean accuracy >= 0.95 in {len(d1)}/{len(md.d1)} arms"
                + (f" (failing: {sorted(set(md.d1) - set(d1))})" if len(d1) < len(md.d1) else "")
                + "; d2 competence " + (", ".join(f"{f} {sum(v.values())}/6" for f, v in md.d2.items()) if md.d2 else "MISSING")
                + f"; d3 A_POST > 0: {'pass' if md.d3 else 'MISSING' if md.d3 is None else 'FAIL'}")
    else:
        out("  Gates d1-d3 (part d): NOT EVALUATED (no factorial)")
    e = res.get("e")
    out("  Gate e (part e, per model x arm): " + ("; ".join(f"{m} " + ",".join(f"{arm}:{'ok' if ok else 'FAIL'}" for arm, ok in g.items()) for m, g in e["gates"].items() if g) if e and any(e["gates"].values()) else "NOT EVALUATED (no IOI results)"))


def load_json(f):
    try:
        return json.load(open(f))
    except Exception as ex:  # a truncated file is reported, not fatal
        return {"error": repr(ex)}


def provenance(root, out, test=False):
    out("PROVENANCE (every results file; the pipeline's COMMIT.txt, ENV.txt and REVISIONS.txt, every line)")
    for name in ("COMMIT.txt", "ENV.txt", "REVISIONS.txt"):
        f = root / name
        out(f"  {name}:" + ("".join(f"\n    {l.rstrip()}" for l in f.read_text().splitlines() if l.strip()) if f.exists() else " absent"))
    commits, bad, revs = set(), [], {}
    for d in DIRS:
        for f in sorted((root / d).glob("*.json")) if (root / d).is_dir() else []:
            j = load_json(f)
            if "error" in j:
                out(f"  {d}/{f.name}: UNREADABLE {j['error']}"); bad.append(f.name); continue
            if not isinstance(j, dict) or "provenance" not in j:
                out(f"  {d}/{f.name}: NO PROVENANCE ({'list of' if isinstance(j, list) else 'dict without a provenance block,'} {len(j)} items)"
                    + ("" if test else "  MISMATCH")); bad.append(f.name) if not test else None; continue
            p, a = j["provenance"], j["provenance"].get("args", {})
            commits.add(p.get("git_commit"))
            revs.setdefault(str(a.get("model") or f.stem.split("_s")[0].removesuffix("_direct").removesuffix("_ioi")).split("/")[-1], {}).setdefault(a.get("revision"), []).append(f"{d}/{f.name}")
            out(f"  {d}/{f.name}: commit {str(p.get('git_commit'))[:10]}, model {a.get('model')} rev {a.get('revision')}, dtype {a.get('dtype', p.get('dtype'))}, "
                f"device {p.get('device')}, transformers {p.get('transformers')}, torch {p.get('torch')}, python {p.get('python')}, attn {p.get('attn_implementation', '-')}, "
                f"skipped {p.get('skipped_items', '-')}, n {a.get('n', '-')}, seed {a.get('seed', '-')}" + (f", label {p['label']!r}" if p.get("label") else ""))
    mixed = {m: r for m, r in revs.items() if len(r) > 1}
    for m, r in mixed.items():
        out(f"  {m}: MISMATCH: files with different revisions: " + "; ".join(f"{str(v)[:10]} ({', '.join(fs)})" for v, fs in r.items()))
    out(f"  commits: {sorted(str(c)[:10] for c in commits)}" + ("  MISMATCH: not one commit" if len(commits) > 1 else "")
        + (f"; revisions: one per model ({len(revs)} models)" if not mixed else f"; revisions MISMATCH in {sorted(mixed)}")
        + ("  UNREADABLE or without provenance: " + ", ".join(bad) if bad else ""))
    return len(commits) <= 1 and not bad and not mixed


def items_of(j):
    items = j.get("results") or j.get("items") if isinstance(j, dict) else j
    return [it for it in (items or []) if isinstance(it, dict) and "core" in it and "arm" in it]


def population(root, out, test=False):
    out("POPULATION (n per arm" + ("" if test else " against the preregistered sizes") + "; the same cores in every arm of a file, and across the parts sharing a model)")
    cores, ok = {}, True
    for d in DIRS:
        for f in sorted((root / d).glob("*.json")) if (root / d).is_dir() else []:
            items = items_of(load_json(f))
            if not items:
                continue
            by = {}  # arm -> the distinct cores in item order
            for it in items:
                c = json.dumps(it["core"], sort_keys=True)
                c in by.setdefault(it["arm"], []) or by[it["arm"]].append(c)
            ns = {a: len(s) for a, s in by.items()}
            same = len({frozenset(s) for s in by.values()}) == 1
            n_ok = test or all(n == EXPECTED_N[d] for n in ns.values())
            ok &= same and n_ok
            out(f"  {d}/{f.name}: " + (f"n = {next(iter(ns.values()))} in each of {len(ns)} arms" if len(set(ns.values())) == 1 else f"n per arm {ns}")
                + ("" if test else f" (expected {EXPECTED_N[d]}): {'OK' if n_ok else 'MISMATCH'}") + f"; same cores across arms: {'OK' if same else 'MISMATCH'}")
            if same and (d == "ioi" or f.stem.endswith("_ioi") or not f.stem.endswith("_s1")):  # belief parts: seed 0 (a model's cores); IOI: the seed-1 cores shared by the factorial and the splice
                m = f.stem.split("_s")[0] if d in ("knockout", "factorial", "ioi") else f.stem.removesuffix("_direct").removesuffix("_ioi")
                cores.setdefault(m + (" (IOI)" if d == "ioi" or f.stem.endswith("_ioi") else ""), {})[d] = next(iter(by.values()))
    for m, parts in cores.items():
        ref = max(parts.values(), key=len)  # the seed-0 cores in item order: a smaller part must hold the first k of them
        rel = {d: "same" if set(s) == set(ref) else f"the first {len(s)} of {len(ref)}" if s == ref[:len(s)]
               else f"a subset of {len(ref)}, NOT the first {len(s)}: MISMATCH" if set(s) <= set(ref) else "MISMATCH" for d, s in parts.items()}
        ok &= all("MISMATCH" not in v for v in rel.values())
        out(f"  {m}: cores across parts: " + ", ".join(f"{d} {v}" for d, v in rel.items()))
    out(f"  population: {'OK' if ok else 'MISMATCH (a cell does not hold the expected n or the same cores)'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/gpu_stage5")
    ap.add_argument("--tag", default="stage5", help="TEST_<model> in TEST_MODE: every model found is scored, verdicts are not results")
    ap.add_argument("--out", default=None, help="default {root}/STAGE5_SCORE.txt")
    a = ap.parse_args(argv)
    root, test = Path(a.root), a.tag.startswith("TEST_")
    lines = []
    out = lines.append
    out(f"Stage 5 scoring, preregistration P-2026-10-05-G; root {root}; tag {a.tag}")
    if test:
        out("TEST MODE: the models found stand in for the preregistered ones; verdict lines below are plumbing checks, not results")
    R = score_parts(root, test)
    out("")
    gates(R, out)
    out("\nVERDICTS (G1-G22; sub-verdicts in order; not evaluable counts as not met in every k/k line, a gated-out anchor included (G1, G4a); a line with nothing evaluable is NOT EVALUABLE;"
        " a sub-verdict whose precondition failed is not applicable and left out)")
    F = verdicts(R, out)
    out("")
    prov_ok = provenance(root, out, test)
    out("")
    pop_ok = population(root, out, test)
    errors = [p for p, (ls, r) in R.items() if r is None and (root / MAIN[p]).is_dir()]
    out(f"\nSUMMARY: {sum(v is True for v in F.values())} MET, {sum(v is False for v in F.values())} NOT MET, {sum(v is None for v in F.values())} NOT EVALUABLE of {len(F)}; "
        f"provenance {'OK' if prov_ok else 'MISMATCH'}; population {'OK' if pop_ok else 'MISMATCH'}" + (f"; SCORER ERROR in parts {errors}" if errors else ""))
    for p in "abcde":
        out(f"\n######## PART ({p}) {NAME[p]}: full report")
        lines.extend(R[p][0])
    text = "\n".join(lines) + "\n"
    print(text, end="")
    f = Path(a.out) if a.out else root / "STAGE5_SCORE.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return 1 if errors else 2 if not test and not (prov_ok and pop_ok) else 0


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        sys.exit(main())
