"""Score preregistration P-2026-10-05-H (docs/PREREGISTRATION.md, GPU stage 6): the entry point over the two part
scorers of analysis/stage6_parts (a heads: Gates a2-a3, H1-H6; b prakash: Gates b0-b3, H7-H12), each as written there.

Inputs under --root (default results/gpu_stage6), as written by scripts/gpu_stage6.sh: heads/<model>.json
(experiments/stage6_heads.py), prakash/<model>/{preflight,filter,sweep_*,lstar,exchange,clamp}.json
(experiments/prakash_swap.py), and the pipeline's COMMIT.txt, ENV.txt, REVISIONS.txt, RELEASE.txt, log_pytest_stage6.txt.
Output, also written to --out (default {root}/STAGE6_SCORE.txt), in this order: the provenance (every line of the
pipeline's COMMIT/ENV/REVISIONS/RELEASE files; per results file the commit, model and revision, dtype, attention, device,
versions; the release commit and file hashes against ckeys.causaltom; MISMATCH when the files hold more than one commit,
one model's files more than one revision, or outside a TEST_ tag a dtype/attention other than the entry's), the population
checks (part a: n_rank = n_eval = 60 per arm and the cores of R = make_cores(n, Random(0)) and E = make_cores(n,
Random(1)) in order in both arms; part b: n = 150 or every passing pair, the first passing pairs in pool order, every
preregistered exchange cell and clamp onset present; DEVIATION, not MISMATCH, when fewer than 150 pairs pass), the gates
(a1 from the last pytest run in the pipeline's log, a2, a3, b0-b3), then one verdict line per prediction H1-H12 in order
(H12 NOT RUN without Llama-3-70B results), a summary, and each part's full report. Verdict rules are the part scorers' (a
prediction over both models of part (a) is MET when met in both, NOT MET when not met in at least one evaluable model,
else NOT EVALUABLE); outside a TEST_ tag Gate a1 failing or not run makes H1-H6 NOT EVALUABLE.
--tag TEST_<model> (what scripts/gpu_stage6.sh passes in TEST_MODE): the models found stand in for the preregistered ones,
the population sizes are not checked and the verdict lines are plumbing checks, not results.
Exit status 1 if a part scorer raised (its traceback is in the report and its predictions are NOT EVALUABLE); 2 if,
outside a TEST_ tag, the provenance or the population check reports MISMATCH, or part-(a) results exist without a passing
Gate a1 (the score file is still written).
"""
import argparse
import json
import random
import re
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from stage6_parts import heads, prakash  # noqa: E402

from ckeys import causaltom as ct  # noqa: E402
from ckeys.story import make_cores  # noqa: E402

TITLE = {"H1": "sparsity: the top-k* heads by a3 suffice", "H2": "necessity and specificity (knockout, random sets)",
         "H3": "mean-ablation of the top-k* heads", "H4": "canonical duplicate-token heads", "H5": "the second hop at the answer",
         "H6": "the same readers for list and sentence", "H7": "the law on the binding swap, depth-matched",
         "H8": "shared prediction under NO-MENTION", "H9": "crossover on their intervention (H_read)",
         "H10": "the rival (H_binding)", "H11": "positive control: the law on an identity edit", "H12": "optional Llama-3-70B"}
N_HEADS, N_PRAKASH, SEEDS = 60, 150, {"rank": 0, "eval": 1}
FORMATS_EX = ct.FORMATS + ct.EXTRA_FORMATS
A1_TESTS = 7


def V(ok):
    return "NOT EVALUABLE" if ok is None else "MET" if ok else "NOT MET"


def load_json(f):
    try:
        return json.load(open(f))
    except Exception as ex:  # a truncated file is reported, not fatal
        return {"error": repr(ex)}


def run_part(label, fn, *args, **kw):
    buf = []
    try:
        return buf, fn(*args, out=buf.append, **kw)
    except Exception:
        buf.append(f"  SCORER ERROR in part ({label}); its predictions are NOT EVALUABLE\n" + traceback.format_exc())
        return buf, None


def files(root):
    """[(label, path)] of every stage-6 results file."""
    fs = [(f"heads/{f.name}", f) for f in sorted((root / "heads").glob("*.json"))] if (root / "heads").is_dir() else []
    for d in sorted((root / "prakash").iterdir()) if (root / "prakash").is_dir() else []:
        fs += [(f"prakash/{d.name}/{f.name}", f) for f in sorted(d.glob("*.json"))] if d.is_dir() else []
    return fs


# --------------------------------------------------------------------------- provenance
def provenance(root, out, test):
    out("PROVENANCE (the pipeline's COMMIT.txt, ENV.txt, REVISIONS.txt and RELEASE.txt, every line; every results file)")
    for name in ("COMMIT.txt", "ENV.txt", "REVISIONS.txt", "RELEASE.txt"):
        f = root / name
        out(f"  {name}:" + ("".join(f"\n    {l.rstrip()}" for l in f.read_text().splitlines() if l.strip()) if f.exists() else " absent"))
    commits, bad, revs, ok = set(), [], {}, True
    for lab, f in files(root):
        j = load_json(f)
        if "error" in j or "provenance" not in j:
            out(f"  {lab}: " + (f"UNREADABLE {j['error']}" if "error" in j else "NO PROVENANCE") + "  MISMATCH")
            bad.append(lab)
            continue
        p = j["provenance"]
        a = p.get("args", {})
        model = str(p.get("model") or a.get("model"))
        rev = p.get("revision", a.get("revision"))
        commits.add(p.get("git_commit"))
        revs.setdefault(model.split("/")[-1], {}).setdefault(rev, []).append(lab)
        dt, attn = p.get("dtype"), p.get("attn_implementation")
        want = None if test or dt is None or a.get("label") else ("eager" if lab.startswith("heads/") else "sdpa")   # a labelled re-run (FP32 re-check) is exploratory
        flag = "" if want is None or (str(dt).endswith("bfloat16") and attn == want) else f"  MISMATCH: the entry fixes BF16 and {want} attention"
        ok &= not flag
        out(f"  {lab}: commit {str(p.get('git_commit'))[:10]}, model {model} rev {rev}, dtype {dt or '-'}, attn {attn or '-'}, device {p.get('device', '-')}, "
            f"transformers {p.get('transformers')}, torch {p.get('torch')}, python {p.get('python')}"
            + (f", release {str(p['release_sha'])[:10]}, pool {str(p['pool_sha256'])[:12]}" if "release_sha" in p else "")
            + (f", label {a['label']!r}" if a.get("label") else "") + flag)
    for d in sorted((root / "prakash").glob("*/preflight.json")) if (root / "prakash").is_dir() else []:
        pf = load_json(d)
        r = pf.get("release", {})
        good = r.get("sha") == ct.RELEASE_SHA and r.get("files") == ct.FILES and pf.get("pool_sha256") == ct.POOL_SHA256
        ok &= good
        out(f"  release used by {d.parent.name}: {r.get('url', ct.RELEASE_URL)} at {r.get('sha')}; "
            + ", ".join(f"{k} {str(v)[:12]}" for k, v in (r.get("files") or {}).items())
            + f"; pool sha256 {str(pf.get('pool_sha256'))[:16]} (n {pf.get('n_pool')}): " + ("OK (ckeys.causaltom pins)" if good else "MISMATCH against ckeys.causaltom"))
    mixed = {m: r for m, r in revs.items() if len(r) > 1}
    for m, r in mixed.items():
        out(f"  {m}: MISMATCH: files with different revisions: " + "; ".join(f"{str(v)[:10]} ({', '.join(fs)})" for v, fs in r.items()))
    out(f"  commits: {sorted(str(c)[:10] for c in commits)}" + ("  MISMATCH: not one commit" if len(commits) > 1 else "")
        + (f"; revisions: one per model ({len(revs)} models)" if not mixed else f"; revisions MISMATCH in {sorted(mixed)}")
        + ("; UNREADABLE or without provenance: " + ", ".join(bad) if bad else ""))
    return ok and len(commits) <= 1 and not bad and not mixed


# --------------------------------------------------------------------------- population
def pop_heads(root, out, test):
    ok = True
    for f in sorted((root / "heads").glob("*.json")) if (root / "heads").is_dir() else []:
        j = load_json(f)
        if "arms" not in j:
            continue
        arms, line = j["arms"], []
        miss = [a for a in ("P1", "POST") if a not in arms]
        ok &= not miss
        for arm, A in arms.items():
            for part, key in (("rank", "rank"), ("eval", "eval")):
                cores = [r["core"] for r in A[key]]
                n_ok = test or len(cores) == N_HEADS
                same = cores == make_cores(len(cores), random.Random(SEEDS[part]))
                ok &= n_ok and same
                line.append(f"{arm} {'R' if part == 'rank' else 'E'} n={len(cores)}" + ("" if test else f" ({'OK' if n_ok else 'MISMATCH'})")
                            + f" make_cores(n, Random({SEEDS[part]})) {'OK' if same else 'MISMATCH'}")
        nseq = j.get("provenance", {}).get("dup", {}).get("n_seq")
        ok &= test or nseq == 100
        out(f"  heads/{f.name}: " + "; ".join(line) + (f"; arms MISSING {miss}" if miss else "") + f"; duplicate-score sequences {nseq}"
            + ("" if test or nseq == 100 else " MISMATCH (100)"))
    return ok


def pop_prakash(root, out, test):
    ok = True
    for d in sorted((root / "prakash").iterdir()) if (root / "prakash").is_dir() else []:
        if not d.is_dir() or (d / "SKIPPED.txt").exists() or not (d / "lstar.json").exists():
            out(f"  prakash/{d.name}: " + ("SKIPPED" if (d / "SKIPPED.txt").exists() else "no lstar.json (incomplete)"))
            continue
        lj, fl = load_json(d / "lstar.json"), load_json(d / "filter.json")
        pop, ls, li = lj.get("population", []), lj.get("lstar"), lj.get("lstar_ID")
        pas = [x["i"] for x in fl.get("pairs", []) if x["ok"]]
        if fl.get("pairs"):
            n_ok = len(pop) == min(N_PRAKASH, len(pas)) and len(fl["pairs"]) == ct.N_POOL
            first = pop == pas[:len(pop)]
            txt = (f"LM filter {len(pas)}/{len(fl['pairs'])} pass; population n = {len(pop)} ({'OK' if n_ok else 'MISMATCH'}), the first passing pairs in pool order: {'OK' if first else 'MISMATCH'}"
                   + (f"; DEVIATION: fewer than {N_PRAKASH} passing pairs, every passing pair is scored" if len(pas) < N_PRAKASH and not test else ""))
        else:
            n_ok, first = test, pop == list(range(len(pop)))
            txt = f"no LM filter (TEST only): population = the first {len(pop)} pool pairs" + ("" if test else "  MISMATCH")
        ok &= n_ok and first
        nL = next((p["provenance"].get("n_layers") for p in (load_json(d / "exchange.json"), load_json(d / "clamp.json")) if "provenance" in p), None)
        ex, cl = load_json(d / "exchange.json"), load_json(d / "clamp.json")
        fmts = ex.get("formats") or cl.get("formats") or []
        need_f = list(fmts) if test else list(prakash.LAW) if d.name == prakash.OPTIONAL else list(FORMATS_EX)   # 70B: H12's formats only
        want_ex = {("BIND", ls, f) for f in need_f} | {("ID", dep, f) for f in need_f if f != "QNAMES2" for dep in {li, ls}}
        prim = {o for o in (ls + 1, li + 1, *([0] if test else [0, 3, 14])) if nL is None or o < nL}
        want_cl = {(f, o) for f in need_f if f != "QNAMES2" for o in prim}
        cnt_ex, cnt_cl = {}, {}
        for c in ex.get("cells", []):
            cnt_ex.setdefault((c["arm"], c["depth"], c["format"]), []).append(c["i"])
        for c in cl.get("cells", []):
            cnt_cl.setdefault((c["format"], c["l0"]), []).append(c["i"])
        miss_ex, miss_cl = sorted(map(str, want_ex - set(cnt_ex))), sorted(map(str, want_cl - set(cnt_cl)))
        wrong = [k for k, v in {**cnt_ex, **cnt_cl}.items() if v != pop]
        sw = [f.name for f in sorted(d.glob("sweep_*.json")) if load_json(f).get("population", pop) != pop]
        ok &= not (miss_ex or miss_cl or wrong or sw)
        out(f"  prakash/{d.name}: {txt}; l* = {ls}, l*_ID = {li}; exchange {len(cnt_ex)} cells (arm, depth, format), clamp {len(cnt_cl)} cells (format, onset), "
            f"each over the population: {'OK' if not wrong and not sw else f'MISMATCH {wrong + sw}'}; preregistered cells present: "
            + ("OK" if not (miss_ex or miss_cl) else f"MISMATCH (missing exchange {miss_ex}, clamp {miss_cl})"))
    return ok


def population(root, out, test):
    out("POPULATION (" + ("TEST: sizes not checked; " if test else "") + "part a: R and E cores per arm; part b: the filtered population and every preregistered cell)")
    ok = pop_heads(root, out, test) & pop_prakash(root, out, test)
    out(f"  population: {'OK' if ok else 'MISMATCH (a cell does not hold the expected n, cores or cells)'}")
    return ok


# --------------------------------------------------------------------------- gates and verdicts
def run_commit(root, ts):
    """The commit of the pipeline run that started last at or before ts (COMMIT.txt: a dated header, then the sha)."""
    f, sha = root / "COMMIT.txt", None
    L = f.read_text().splitlines() if f.exists() else []
    for i, l in enumerate(L[:-1]):
        m = re.match(r"==== (\S+) PART=", l)
        if m and (ts is None or m.group(1) <= ts):
            sha = L[i + 1].strip()
    return sha


def gate_a1(root, out):
    f = root / "log_pytest_stage6.txt"
    if not f.exists():
        out("  Gate a1  FP32 exactness (tests/test_head_splice.py): no pytest log in the root -> NOT EVALUABLE (not run)")
        return None
    t = f.read_text()
    hd = list(re.finditer(r"^==== (\S+) .*-m pytest.*$", t, re.M))   # run() writes one header per attempt; the last attempt counts
    ts = hd[-1].group(1) if hd else None
    t = t[hd[-1].end():] if hd else t
    p, x = (len(re.findall(rf"tests/test_head_splice\.py::\S+ {s}", t)) for s in ("PASSED", "(?:FAILED|ERROR)"))
    ok = p >= A1_TESTS and x == 0
    out(f"  Gate a1  FP32 exactness, tests/test_head_splice.py in log_pytest_stage6.txt (last of {len(hd)} run(s), {ts or 'undated'}, "
        f"commit {str(run_commit(root, ts))[:10]}): {p} passed, {x} failed (>= {A1_TESTS} tests, none failing) -> {V(ok)}")
    return ok


def gates(root, R, out):
    out("GATES (part a: a1 before any model, a2 per model and format, a3 per model; part b: per model, arm, depth and format)")
    a1 = gate_a1(root, out)
    la, _ = R["a"]
    out("\n".join(l for l in la if l.startswith("  Gate a")) or "  Gates a2, a3: NOT EVALUATED (no part-(a) results)")
    lb, rb = R["b"]
    keep, m, on = [], None, False   # the lines between a model's GATES header and its first prediction line
    for l in lb:
        if l.startswith("\n-- "):
            m, on = None if "EXPLORATORY" in l else l.split(":")[0][4:], False
        elif l.startswith("   GATES"):
            on = True
            keep += [f"  {m or prakash.PRIMARY + ' (missing)'}:"] + ["   " + x.strip() for x in l.split("\n")[1:]]
        elif on and l.startswith(("   (H7", "\n   PREDICTIONS")):
            on = False
        elif on:
            keep.append(" " + l)
    out("\n".join(keep) if keep else "  Gates b0-b3: NOT EVALUATED (no part-(b) results" + (")" if rb is not None or not lb else "; see the part-(b) report)"))
    return a1


def verdicts(R, a1, l70, out, test=False):
    out("\nVERDICTS (H1-H12 in order; part (a) over the required models (Qwen2.5-7B, Mistral-7B): MET if met in both, NOT MET if not met in an "
        "evaluable model, else NOT EVALUABLE; part (b) at Qwen2.5-14B)")
    la, ra = R["a"]
    _, rb = R["b"]
    F = {}
    for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
        if ra is None:
            F[h] = None
            out(f"  {h:4s} {TITLE[h]:50s} no part-(a) results -> NOT EVALUABLE")
            continue
        per, G = ra["per_model"].get(h, {}), ra["gates"]
        def note(m, h=h):
            if m not in G:
                return " (missing)"
            c = next((re.search(r"computed (NOT EVALUABLE|NOT MET|MET), gate failed", l) for l in la if l.startswith(f"  {h:4s} {m:28s}")), None)
            return f" (gate failed; computed {c.group(1)})" if c else ""
        F[h] = ra["verdicts"].get(h) if a1 or test else None
        extra = "" if a1 or test else f"  [computed {V(ra['verdicts'].get(h))}; Gate a1 failed or not run -> NOT EVALUABLE]"
        if h == "H5" and "H5s" in ra["verdicts"]:
            extra += "  [SENTENCE-AFTER reported: " + "; ".join(f"{m} {V(v)}" for m, v in ra["per_model"]["H5s"].items()) + f" -> {V(ra['verdicts']['H5s'])}]"
        out(f"  {h:4s} {TITLE[h]:50s} " + "; ".join(f"{m} {V(v)}{note(m)}" for m, v in per.items()) + f" -> {V(F[h])}" + extra)
    if ra is not None and not a1:
        out("       Gate a1 " + ("FAILED" if a1 is False else "NOT RUN") + ": the instrument did not pass its exactness tests"
            + (" (TEST: verdicts kept as plumbing checks)" if test else "; H1-H6 are NOT EVALUABLE"))
    for h in ("H7", "H8", "H9", "H10", "H11", "H12"):
        ok, txt = (rb or {}).get("verdicts", {}).get(h, (None, "NOT RUN (optional; no Llama-3-70B results)" if h == "H12" and not l70 else "no part-(b) results"))
        F[h] = "NOT RUN" if h == "H12" and txt.startswith("NOT RUN") else ok
        word = "NOT RUN" if F[h] == "NOT RUN" else V(ok)
        out(f"  {h:4s} {TITLE[h]:50s} -> {word}\n         {txt}")
    return F


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/gpu_stage6")
    ap.add_argument("--tag", default="stage6", help="TEST_<model> in TEST_MODE: every model found is scored, verdicts are not results")
    ap.add_argument("--out", default=None, help="default {root}/STAGE6_SCORE.txt")
    a = ap.parse_args(argv)
    root, test = Path(a.root), a.tag.startswith("TEST_")
    lines = []
    out = lines.append
    out(f"Stage 6 scoring, preregistration P-2026-10-05-H; root {root}; tag {a.tag}")
    if test:
        out("TEST MODE: the models found stand in for the preregistered ones; verdict lines below are plumbing checks, not results")
    R = {"a": run_part("a", heads.score, root / "heads", test=test) if (root / "heads").is_dir() else ([f"  no results: {root / 'heads'} missing"], None),
         "b": run_part("b", prakash.score, root / "prakash", test=test) if (root / "prakash").is_dir() else ([f"  no results: {root / 'prakash'} missing"], None)}
    out("")
    prov_ok = provenance(root, out, test) and not (R["a"][1] or {}).get("mismatch")
    out("")
    pop_ok = population(root, out, test)
    out("")
    a1 = gates(root, R, out)
    F = verdicts(R, a1, (root / "prakash" / prakash.OPTIONAL).is_dir(), out, test)
    a1_bad = not test and bool((R["a"][1] or {}).get("gates")) and not a1
    errors = [p for p, (ls, r) in R.items() if r is None and any("SCORER ERROR" in l for l in ls)]
    n = lambda v: sum(x is v if not isinstance(v, str) else x == v for x in F.values())  # noqa: E731
    out(f"\nSUMMARY: {n(True)} MET, {n(False)} NOT MET, {n(None)} NOT EVALUABLE, {n('NOT RUN')} NOT RUN of {len(F)}; "
        f"provenance {'OK' if prov_ok else 'MISMATCH'}; population {'OK' if pop_ok else 'MISMATCH'}" + (f"; SCORER ERROR in parts {errors}" if errors else "")
        + ("; Gate a1 not passed with part-(a) results (exit 2)" if a1_bad else ""))
    for p, name in (("a", "reader heads and the second hop"), ("b", "the exchange on Prakash et al.'s intervention")):
        out(f"\n######## PART ({p}) {name}: full report")
        lines.extend(R[p][0])
    text = "\n".join(lines) + "\n"
    print(text, end="")
    f = Path(a.out) if a.out else root / "STAGE6_SCORE.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return 1 if errors else 2 if not test and not (prov_ok and pop_ok and not a1_bad) else 0


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        sys.exit(main())
