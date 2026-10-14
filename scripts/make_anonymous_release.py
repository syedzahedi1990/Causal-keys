"""Build an anonymised copy of the repository for double-blind review.

Usage (from anywhere in the repository): python scripts/make_anonymous_release.py OUT_DIR [--identifier TEXT ...]
       python scripts/make_anonymous_release.py --scan-only DIR [--identifier TEXT ...]

OUT_DIR must not exist and must lie outside the repository. The copy contains the tracked files at HEAD, minus
internal planning notes and the public third-party data in OMIT (named, with its source, in EDITED_FILES.txt), with
  * file and directory names containing "paper1" renamed to "prior" (and every reference updated, so imports of a
    renamed module and the tests that name it stay consistent),
  * "Paper 1" and "our predecessor" replaced by "Anonymous (2026)" and "Paper 2" by "this paper",
  * the GitHub remote replaced by a placeholder, the repository name by "repo", local scratch paths by "<scratch>",
    the predecessor's "reviewer repository" by "code release" (paths included), and the download URL of that
    release (which carries the repository name) by $PRIOR_ZIP_URL in the GPU scripts,
  * personal identifiers replaced: the owner of the origin remote, the e-mail addresses of git's user and of every
    commit's author and committer, and every --identifier (or comma-separated $ANON_IDENTIFIERS) given,
and adds PREREGISTRATION_HISTORY.txt: for every commit that touched the preregistration log or a scoring
script, its short hash, commit time, subject and diff, and for every commit that touched results, experiment
code or GPU scripts, its hash, time and subject (no author information), so that reviewers can check
that each preregistration was committed before its run and each scoring script before its outputs were read.
It also adds EDITED_FILES.txt, which lists every file whose content was edited (so a reader can tell why its
line in a results MANIFEST.sha256 no longer verifies), every renamed file, and every file a manifest lists that the
release does not contain.

Every file that decodes as UTF-8 (whatever its suffix: logs, step markers, CSV, manifests, files without a
suffix) is scrubbed; a file with a known text suffix that does not decode is an error. Binary files are copied
as they are. Afterwards the whole output is scanned, every file regardless of suffix: text files as text,
binary files as raw bytes (also as UTF-16/UTF-32 and, for zip archives such as .npz/.pt, member by member; PDFs
through pdftotext when it is installed). Any remaining "paper 1", "reviewer", repository name, "predecessor",
scratch path, branch name, unknown GitHub account or e-mail address, or personal identifier makes the script
exit with status 1. Check the output by hand before uploading it.
"""
import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

EXCLUDE = ("paper/BRIEF.md", "docs/PAPER2_PROPOSAL.md", "docs/GPU_RUNBOOK.md", "docs/V3_STAGE5_IMPLICATIONS.md", "docs/V3_PLAN.md", "paper/sections/old/", "notebooks/",
           "scripts/make_anonymous_release.py", "docs/stage8_design/")  # the script would scrub (and so expose) its own patterns
# Public third-party data whose own text contains a scanned word ("predecessor") and whose sha256 the code asserts: left out
# rather than edited (an edited copy would also stop scripts/gpu_stage8a.sh, which downloads the file only when it is absent).
OMIT = {"results/gpu_stage8a/squad/dev-v1.1.json":
        "SQuAD v1.1 dev, https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json (sha256 asserted by ckeys/squad_items.py)"}
TRACKED_HISTORY = ["docs/PREREGISTRATION.md", "analysis/stage1_prereg.py", "analysis/stage2_score.py",
                   "analysis/stage3_score.py", "analysis/stage3b_score.py", "analysis/stage4_score.py",
                   "analysis/stage5_score.py", "analysis/stage5_parts", "analysis/stage6_score.py", "analysis/stage6_parts",
                   "analysis/stage7_score.py", "analysis/stage7_parts",
                   "analysis/stage8a_score.py", "analysis/stage8a_parts", "analysis/stage8b_score.py", "analysis/stage8b_parts",
                   "analysis/stage8c_score.py", "analysis/stage8c_parts", "analysis/stage8d_score.py", "analysis/stage8d_parts"]
# Suffixes that must decode as UTF-8 (anything else that decodes is treated as text too) and suffixes never decoded.
TEXT = (".py", ".md", ".sh", ".tex", ".txt", ".json", ".jsonl", ".bib", ".sty", ".bst", ".cfg", ".toml", ".ipynb", ".yml",
        ".yaml", ".log", ".done", ".csv", ".tsv", ".sha256", ".gitignore", ".ini", ".html", ".svg")
BINARY = (".npz", ".npy", ".pt", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".tgz", ".gz", ".zip", ".pkl", ".safetensors")


def _sub(key, pattern, rep, flags=0):
    """A substitution; it is tried only when the lower-cased text contains `key` (which every match contains)."""
    return key, re.compile(pattern, flags), rep


SUBS = [_sub("github.com", r"(?:https://github\.com/|git@github\.com:)[^/\s]+/Causal-keys(?:\.git)?", "<anonymous repository>", re.I),
        # the predecessor's anonymous release is named after this repository: its URL is withheld (see EDITED_FILES.txt)
        _sub("4open.science", r"https?://anonymous\.4open\.science/[^\s\"'<>]*?causal-keys[^\s\"'<>]*", "${PRIOR_ZIP_URL:-}", re.I),
        _sub("/tmp/claude-", r"/tmp/claude-\d+/[^\s\"']*?/scratchpad/", "<scratch>/"),
        _sub("/tmp/claude-", r"/tmp/claude-\d+/[^/\s\"']+(?:/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?", "<scratch>"),
        _sub("causal-keys", r"causal-keys", "repo", re.I),  # checkout and cache directories, pytest rootdir
        _sub("paper 1", r"Paper 1's", "Anonymous (2026)'s"),
        _sub("paper 1", r"Paper 1", "Anonymous (2026)"),
        _sub("paper 2", r"Paper 2", "this paper"),
        _sub("predecessor", r"sentence and its predecessor\b", "sentence and the sentence before it"),  # not the earlier work
        _sub("predecessor", r"of our\n# predecessor, Anonymous \(2026\)", "of\n# Anonymous (2026)"),
        _sub("predecessor", r"[Oo]ur own predecessor's", "Anonymous (2026)'s"),
        _sub("predecessor", r"[Oo]ur predecessor, Anonymous \(2026\)", "Anonymous (2026)"),
        _sub("predecessor", r"(?:[Oo]ur|[Tt]he) predecessor's", "Anonymous (2026)'s"),
        _sub("predecessor", r"[Oo]ur predecessor", "Anonymous (2026)"),
        _sub("predecessor", r"\b[Tt]he predecessor\b", "Anonymous (2026)"),
        _sub("predecessor", r"predecessor", "prior work"),
        # the predecessor's release for review ("reviewer repository"), its unpacked directory, and the predecessor's
        # code-identity label; and the reviewer-named sets of surface forms
        _sub("reviewer", r"derived_reviewer_implementation", "derived_release_implementation"),
        _sub("reviewer", r"reviewer-repository", "code-release"),
        _sub("reviewer", r"released reviewer repository", "released code"),
        _sub("reviewer", r"reviewer repository", "code release"),
        _sub("reviewer", r"reviewer-named", "pre-specified"),
        _sub("reviewer", r"reviewer-driven controls", "additional controls"),
        _sub("the author", r"\(the author's, not thresholds\)", "(the authors', not thresholds)"),
        _sub("the author", r"that the author downloaded", "that the authors downloaded"),
        _sub("paper2_v", r"paper2_v", "paper_v"),
        _sub("p1r", r"\bP1R\b", "PRIOR_R"),
        _sub("p1_root", r"\bP1_ROOT\b", "PRIOR_ROOT"),
        _sub("p1", r"\bp1([-_])root\b", r"prior\1root"),
        _sub("p1.zip", r"\bp1\.zip\b", "prior.zip"),
        _sub("paper1", r"paper1", "prior"),  # also renames paper1_*.py modules, their imports and test_paper1_* tests
        _sub("claude/paper2-research", r"claude/paper2-research", "main")]

# Third-party GitHub accounts and e-mail addresses that may remain (cited releases, LaTeX style-file authors, placeholders).
GITHUB_OK = ("Nix07", "rajpurkar", "anonymous")
EMAIL_OK = {"anon.email@domain.com", "anon@example.com", "anonymous@example.com", "test@example.invalid",
            "peter.williams@dsto.defence.gov.au", "peterwilliams@dsto.defence.gov.au", "alex.smola@anu.edu.au",
            "pieter@vanoostrum.org", "daly@mps.mpg.de"}
EMAIL = r"(?<![\w.%+-])[a-z0-9][\w.%+-]*@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}"
# Patterns that must not survive anywhere (matched case-insensitively): (label, regex, literal key for a fast pre-check)
LEAKS = [("paper 1", r"paper[\s_~-]*1(?!\d)", "paper"),
         ("reviewer", r"reviewer", "reviewer"),
         ("repository name", r"causal[\s_-]?keys", "causal"),
         ("predecessor", r"predecessor", "predecessor"),
         ("our prior work", r"\bour\W+(?:own\W+)?prior\W+work", "prior"),
         ("scratch path", r"claude-\d", "claude-"),
         ("branch name", r"claude/paper2", "claude/"),
         ("GitHub account", r"github\.com[/:](?!(?:%s)\b)[\w.-]+" % "|".join(GITHUB_OK), "github"),
         ("GitHub account", r"\b(?!(?:%s)\.)[\w-]+\.github\.io" % "|".join(GITHUB_OK), "github"),
         ("e-mail address", EMAIL, "@")]
# Literal patterns for binary files, searched in their raw bytes as ASCII, UTF-16 and UTF-32 (plus EMAIL as ASCII).
BIN_LEAKS = [("paper 1", "paper1"), ("paper 1", "paper 1"), ("reviewer", "reviewer"), ("repository name", "causal-keys"),
             ("predecessor", "predecessor"), ("scratch path", "claude-0"), ("branch name", "claude/paper2")]


def git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True).stdout


def personal_identifiers(extra):
    """(text, replacement) for the origin's owner, git's e-mail addresses and the given identifiers."""
    ids = {}
    m = re.search(r"github\.com[/:]([^/\s]+)/", git("remote", "get-url", "origin", check=False))
    if m:
        ids[m.group(1)] = "anonymous"
    emails = git("config", "user.email", check=False).split() + git("log", "--format=%ae%n%ce").split()
    for e in emails:
        ids[e] = "anonymous@example.com"
    for e in extra:
        if len(e) < 4:
            sys.exit(f"identifier {e!r} is too short to scan for safely")
        ids.setdefault(e, "anonymous@example.com" if "@" in e else "anonymous")
    return ids


def scrub(text, subs):
    low = text.lower()
    for key, pat, rep in subs:
        if key in low:
            new = pat.sub(rep, text)
            if new != text:
                text, low = new, new.lower()
    return text


def as_text(path, data, strict=True):
    """The file's content as text, or None for a binary file; with strict, a file with a text suffix must be UTF-8."""
    suffix = Path(path).suffix.lower()
    if suffix in BINARY:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        if strict and (suffix in TEXT or path.endswith(TEXT)):
            sys.exit(f"{path}: a text file that is not UTF-8; it cannot be scrubbed (convert it or exclude it)")
        return None
    return None if "\0" in text and suffix not in TEXT else text


def allowed_email(address):
    return address.lower() in EMAIL_OK or re.search(r"@example\.(?:com|org|net|invalid)$", address, re.I) is not None


def text_leaks(text, leaks):
    low, found = text.lower(), []
    for label, pat, key in leaks:
        if key not in low:
            continue
        for m in pat.finditer(text):
            if label == "e-mail address" and allowed_email(m.group()):
                continue
            line = text.count("\n", 0, m.start()) + 1
            snippet = text[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")
            found.append(f"line {line}: {label}: ...{snippet}...")
    return found


def binary_leaks(data, bin_leaks):
    """Leaks in raw bytes (ASCII/UTF-8, UTF-16 and UTF-32), in compressed zip members and in inflated PDF streams."""
    found = []
    for label, pat in bin_leaks:
        for m in pat.finditer(data):
            if label == "e-mail address" and allowed_email(m.group().decode("ascii", "replace")):
                continue
            found.append(f"{label}: raw bytes at offset {m.start()} match {pat.pattern[:60]!r}")
            break
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    found += [f"zip member name {info.filename}: {f}" for f in binary_leaks(info.filename.encode(), bin_leaks)]
                    if info.compress_type != zipfile.ZIP_STORED:
                        found += [f"zip member {info.filename}: {f}" for f in binary_leaks(z.read(info), bin_leaks)]
        except zipfile.BadZipFile:
            pass
    if data[:5] == b"%PDF-":
        for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
            try:
                found += [f"PDF stream: {f}" for f in binary_leaks(zlib.decompress(m.group(1)), bin_leaks)]
            except zlib.error:
                pass
    return found


def compile_leaks(identifiers):
    leaks = [(label, re.compile(p, re.I), key) for label, p, key in LEAKS]
    leaks += [("personal identifier", re.compile(re.escape(i), re.I), i.lower()) for i in identifiers]
    raw = [(label, s) for label, s in BIN_LEAKS] + [("personal identifier", i) for i in identifiers]
    bin_leaks = []
    for label, s in raw:
        for gap in (b"", b"\x00", b"\x00\x00\x00"):  # ASCII/UTF-8, UTF-16 and UTF-32 (either byte order)
            bin_leaks.append((label, re.compile(gap.join(re.escape(c.encode()) for c in s), re.I)))
    bin_leaks.append(("e-mail address", re.compile(EMAIL.encode(), re.I)))
    return leaks, bin_leaks


def scan(root, identifiers):
    """Every leak in every file under root (file names included); text files as text, others as bytes."""
    leaks, bin_leaks = compile_leaks(identifiers)
    pdftotext = shutil.which("pdftotext")
    found = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        found += [f"{rel}: file name: {f}" for f in text_leaks(rel, leaks)]
        if not p.is_file():
            continue
        data = p.read_bytes()
        text = as_text(rel, data, strict=False)
        if text is not None:
            found += [f"{rel}: {f}" for f in text_leaks(text, leaks)]
            continue
        found += [f"{rel}: {f}" for f in binary_leaks(data, bin_leaks)]
        if data[:5] == b"%PDF-" and pdftotext:
            out = subprocess.run([pdftotext, "-q", str(p), "-"], capture_output=True).stdout.decode("utf-8", "replace")
            found += [f"{rel}: PDF text: {f}" for f in text_leaks(out, leaks)]
    return found, pdftotext is not None


def check_manifest(name, before, after):
    """A scrubbed MANIFEST.sha256 may change only the file names it lists, never a hash."""
    a, b = before.splitlines(), after.splitlines()
    if len(a) != len(b) or any(x.split(None, 1)[:1] != y.split(None, 1)[:1] for x, y in zip(a, b)):
        sys.exit(f"{name}: scrubbing would change a hash in this manifest")


def missing_from_manifests(out):
    """Entries of the release's MANIFEST.sha256 files whose file is not in the release (e.g. never committed)."""
    missing = []
    for m in sorted(out.rglob("MANIFEST.sha256")):
        for line in m.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                path = parts[1].lstrip("*")
                f = m.parent / path if path.startswith("./") else out / path
                if not f.exists():
                    missing.append(Path(os.path.relpath(f, out)).as_posix())
    return missing


def build(out, identifiers):
    subs = SUBS + [(i.lower(), re.compile(re.escape(i), re.I), rep) for i, rep in identifiers.items()]
    out.mkdir(parents=True, exist_ok=False)
    edited, renamed, seen, left_out = [], [], {}, set()
    for f in git("ls-files").splitlines():
        if f.startswith(EXCLUDE) or f in OMIT:
            left_out.add(scrub(f, subs))
            continue
        name = scrub(f, subs)
        if name in seen:
            sys.exit(f"{f} and {seen[name]} would both be written to {name}")
        seen[name] = f
        if name != f:
            renamed.append(name)
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = Path(f).read_bytes()
        t0 = as_text(f, data)
        if t0 is not None:
            t1 = scrub(t0, subs)
            if t1 != t0:
                if f.endswith(".sha256"):
                    check_manifest(f, t0, t1)
                edited.append(name)
                data = t1.encode("utf-8")
        dst.write_bytes(data)
    log = git("log", "--reverse", "--format=%h", "--", *TRACKED_HISTORY).split()
    parts = []
    for h in log:
        head = git("show", "-s", "--format=commit %h%ncommitted %cI%n%n    %s%n", h)
        diff = git("show", "--format=", h, "--", *TRACKED_HISTORY)
        parts.append(scrub(head + diff, subs))
    runs = git("log", "--reverse", "--format=%h  %cI  %s", "--", "results/", "experiments/", "scripts/")
    (out / "PREREGISTRATION_HISTORY.txt").write_text(
        "== Commits that touched the preregistration log or a scoring script (with diffs)\n\n" + "\n".join(parts)
        + "\n\n== Commits that touched results, experiment code or GPU scripts (hash, commit time, subject)\n\n" + scrub(runs, subs))
    (out / "EDITED_FILES.txt").write_text(
        "Text files whose content was edited for this release (local paths, repository and account names, and other\n"
        "identifying wording replaced); their entries in the results' MANIFEST.sha256 files therefore do not verify.\n"
        "A MANIFEST.sha256 listed here was edited only in the file names it lists; its hashes are unchanged.\n\n"
        + "\n".join(edited) + "\n\n"
        "Files renamed for this release (new names shown; the MANIFEST.sha256 files list them under these names):\n\n"
        + "\n".join(renamed) + "\n\n"
        "Public third-party files left out of this release (their MANIFEST.sha256 entries hold the sha256 of the copy used):\n\n"
        + "\n".join(f"{f}: {src}" for f, src in OMIT.items()) + "\n\n"
        "Other files listed in a MANIFEST.sha256 that are not in this release (never committed to the repository):\n\n"
        + "\n".join(f + (" (excluded from this release)" if f in left_out else "")
                     for f in missing_from_manifests(out) if f not in OMIT) + "\n\n"
        "The GPU scripts read the download URL of Anonymous (2026)'s code release from $PRIOR_ZIP_URL; set PRIOR_R to\n"
        "an unpacked copy of that release (or PRIOR_ZIP_URL to its zip URL) before running them.\n")
    return len(log), len(edited), len(renamed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="output directory (must not exist; outside the repository)")
    ap.add_argument("--scan-only", action="store_true", help="only scan an existing directory for leaks")
    ap.add_argument("--identifier", action="append", default=[],
                    help="a further personal identifier (name, handle, e-mail) to replace and scan for; repeatable")
    args = ap.parse_args(argv)
    root = Path(git("rev-parse", "--show-toplevel").strip()).resolve()
    out = Path(args.out).resolve()
    os.chdir(root)
    extra = args.identifier + [s.strip() for s in os.environ.get("ANON_IDENTIFIERS", "").split(",") if s.strip()]
    identifiers = personal_identifiers(extra)
    if not args.scan_only:
        if out == root or root in out.parents:
            sys.exit(f"{out} is inside the repository; write the release elsewhere")
        dirty = [ln[3:] for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if not ln[3:].startswith(EXCLUDE)]
        if dirty:
            print(f"warning: the release copies the working tree, which differs from HEAD in {dirty}", file=sys.stderr)
        n_log, n_edited, n_renamed = build(out, identifiers)
        print(f"wrote {out}; {n_log} history commits; {n_edited} files edited, {n_renamed} renamed")
    elif not out.is_dir():
        sys.exit(f"{out} is not a directory")
    found, used_pdftotext = scan(out, identifiers)
    if not used_pdftotext:
        print("note: pdftotext not found, so PDF text was scanned only through its raw and inflated streams", file=sys.stderr)
    if found:
        print(f"LEAK SCAN FAILED: {len(found)} possible leaks in {out}; do not upload it:", file=sys.stderr)
        for f in found:
            print("  " + f, file=sys.stderr)
        sys.exit(1)
    print(f"leak scan of {out}: no leaks found")


if __name__ == "__main__":
    main()
