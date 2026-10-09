"""Build an anonymised copy of the repository for double-blind review.

Usage (from the repository root): python scripts/make_anonymous_release.py OUT_DIR

The copy contains the tracked files at HEAD, minus internal planning notes, with
  * file and directory names containing "paper1" renamed to "prior" (and every reference updated),
  * "Paper 1" and "our predecessor" replaced by "Anonymous (2026)" and "Paper 2" by "this paper",
  * the GitHub remote replaced by a placeholder,
and adds PREREGISTRATION_HISTORY.txt: for every commit that touched the preregistration log or a scoring
script, its short hash, commit time, subject and diff, and for every commit that touched results, experiment
code or GPU scripts, its hash, time and subject (no author information), so that reviewers can check
that each preregistration was committed before its run and each scoring script before its outputs were read.
Check the output by hand before uploading it.
"""
import re
import subprocess
import sys
from pathlib import Path

EXCLUDE = ("paper/BRIEF.md", "docs/PAPER2_PROPOSAL.md", "docs/GPU_RUNBOOK.md", "docs/V3_STAGE5_IMPLICATIONS.md", "docs/V3_PLAN.md", "paper/sections/old/", "notebooks/",
           "scripts/make_anonymous_release.py")  # the script would scrub (and so expose) its own patterns
TRACKED_HISTORY = ["docs/PREREGISTRATION.md", "analysis/stage1_prereg.py", "analysis/stage2_score.py",
                   "analysis/stage3_score.py", "analysis/stage3b_score.py", "analysis/stage4_score.py",
                   "analysis/stage5_score.py", "analysis/stage5_parts", "analysis/stage6_score.py", "analysis/stage6_parts",
                   "analysis/stage7_score.py", "analysis/stage7_parts"]
TEXT = (".py", ".md", ".sh", ".tex", ".txt", ".json", ".bib", ".sty", ".bst", ".cfg", ".toml", ".ipynb", ".yml", ".yaml")
SUBS = [(re.compile(r"https://github\.com/[^/\s]+/Causal-keys(\.git)?"), "<anonymous repository>"),
        (re.compile(r"/tmp/claude-0/[^\s\"']*?/scratchpad/"), "<scratch>/"),
        (re.compile(r"(?<![\w-])Causal-keys(?![\w-])"), "repo"),
        (re.compile(r"Paper 1's"), "Anonymous (2026)'s"),
        (re.compile(r"Paper 1"), "Anonymous (2026)"),
        (re.compile(r"Paper 2"), "this paper"),
        (re.compile(r"of our\n# predecessor, Anonymous \(2026\)"), "of\n# Anonymous (2026)"),
        (re.compile(r"[Oo]ur own predecessor's"), "Anonymous (2026)'s"),
        (re.compile(r"[Oo]ur predecessor, Anonymous \(2026\)"), "Anonymous (2026)"),
        (re.compile(r"(?:[Oo]ur|[Tt]he) predecessor's"), "Anonymous (2026)'s"),
        (re.compile(r"[Oo]ur predecessor"), "Anonymous (2026)"),
        (re.compile(r"\b[Tt]he predecessor\b"), "Anonymous (2026)"),
        (re.compile(r"predecessor"), "prior work"),
        (re.compile(r"\(the author's, not thresholds\)"), "(the authors', not thresholds)"),
        (re.compile(r"that the author downloaded"), "that the authors downloaded"),
        (re.compile(r"paper2_v"), "paper_v"),
        (re.compile(r"\bP1R\b"), "PRIOR_R"),
        (re.compile(r"\bP1_ROOT\b"), "PRIOR_ROOT"),
        (re.compile(r"\bp1([-_])root\b"), r"prior\1root"),
        (re.compile(r"\bp1\.zip\b"), "prior.zip"),
        (re.compile(r"paper1"), "prior"),
        (re.compile(r"claude/paper2-research"), "main")]


def scrub(text):
    for pat, rep in SUBS:
        text = pat.sub(rep, text)
    return text


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    changed = []
    for f in git("ls-files").splitlines():
        if f.startswith(EXCLUDE):
            continue
        dst = out / scrub(f)
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = Path(f).read_bytes()
        if f.endswith(TEXT):
            t0 = data.decode("utf-8")
            t1 = scrub(t0)
            if t1 != t0:
                changed.append(scrub(f))
            data = t1.encode("utf-8")
        dst.write_bytes(data)
    log = git("log", "--reverse", "--format=%h", "--", *TRACKED_HISTORY).split()
    parts = []
    for h in log:
        head = git("show", "-s", "--format=commit %h%ncommitted %cI%n%n    %s%n", h)
        diff = git("show", "--format=", h, "--", *TRACKED_HISTORY)
        parts.append(scrub(head + diff))
    runs = git("log", "--reverse", "--format=%h  %cI  %s", "--", "results/", "experiments/", "scripts/")
    (out / "PREREGISTRATION_HISTORY.txt").write_text(
        "== Commits that touched the preregistration log or a scoring script (with diffs)\n\n" + "\n".join(parts)
        + "\n\n== Commits that touched results, experiment code or GPU scripts (hash, commit time, subject)\n\n" + scrub(runs))
    (out / "ANONYMISED_FILES.txt").write_text(
        "Text files whose content was anonymised for this release (author-identifying names, paths and identifiers\n"
        "replaced); their entries in the results' MANIFEST.sha256 files therefore do not verify:\n\n" + "\n".join(changed) + "\n")
    leaks = [str(p) for p in out.rglob("*") if p.is_file() and p.suffix in TEXT
             and re.search(r"Paper 1|paper1|predecessor|[Oo]ur\W+(?:own\W+)?prior work|claude-0|(?<![\w-])Causal-keys(?![\w-])|github\.com/[^<\s]*Causal",
                           p.read_text(errors="ignore"))]
    print(f"wrote {out}; {len(log)} history commits; possible leaks: {leaks or 'none'}")


if __name__ == "__main__":
    main(sys.argv[1])
