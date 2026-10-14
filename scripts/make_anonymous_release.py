"""Build an anonymised copy of the repository for double-blind review.

Usage (from anywhere in the repository): python scripts/make_anonymous_release.py OUT_DIR [--identifier TEXT ...] [--offline]
       python scripts/make_anonymous_release.py --scan-only DIR [--identifier TEXT ...] [--offline]

OUT_DIR must not exist and must lie outside the repository. The copy is built from the HEAD commit, never from the
working tree: the file list comes from `git ls-tree HEAD` and every file's content from `git cat-file` (its blob at
HEAD), so uncommitted edits never ship; a working tree that differs from HEAD only gives a warning naming the files
whose edits are left out. Executable bits are kept. The copy holds the files at HEAD, minus internal planning notes,
the GPU runbook and the public third-party data in OMIT (named, with its source, in EDITED_FILES.txt), with
  * file and directory names containing "paper1" renamed to "prior" (and every reference updated, so imports of a
    renamed module and the tests that name it stay consistent),
  * "Paper 1" and "our predecessor" replaced by "Anonymous (2026)" and "Paper 2" by "this paper",
  * the GitHub remote replaced by a placeholder, the repository name by "repo", local scratch paths by "<scratch>",
    the predecessor's "reviewer repository" by "code release" (paths included), and the download URL of that
    release (which carries the repository name) by $PRIOR_ZIP_URL in the GPU scripts,
  * the singular self-references "the author predicts", "(the author's, not thresholds)" and "that the author
    downloaded" made plural,
  * personal identifiers replaced: the owner of the origin remote, the e-mail addresses of git's user and of every
    commit's author and committer (all branches), and every --identifier (or comma-separated $ANON_IDENTIFIERS) given,
and adds PREREGISTRATION_HISTORY.txt: for every commit that touched the preregistration log or a scoring
script, its short hash, commit time, subject and diff, and for every commit that touched results, experiment
code or GPU scripts, its hash, time and subject (no author information), so that reviewers can check
that each preregistration was committed before its run and each scoring script before its outputs were read.
It also adds EDITED_FILES.txt, which lists every file whose content was edited (so a reader can tell why its
line in a results MANIFEST.sha256 no longer verifies), every renamed file, every file that records the sha256 of an
edited file (a scorer that checks that hash reports a mismatch on the release; what the stage-4 and stage-8a
scorers do on the release is spelled out), every file a manifest lists that the release does not contain, the
withheld runbook and the tests that skip without it or without git history (found by their needs_runbook and
needs_git markers), and where the GPU scripts expect Anonymous (2026)'s unpacked code release.

Authors-only notes go to OUT_DIR.AUTHORS.txt beside the release, never into it: the commit built, the uncommitted
changes left out, how many of this repository's commits the release names by hash, and the origin's visibility on
GitHub (asked through `gh api`, else GitHub's unauthenticated REST API; --offline skips the question). Those hashes
identify the repository while it is public (GitHub finds a public repository's commits by hash), so the GitHub
repository must be private before the release reaches reviewers; the script prints that warning prominently.

Every file that decodes as UTF-8 (whatever its suffix: logs, step markers, CSV, manifests, files without a
suffix) is scrubbed; a file with a known text suffix that does not decode is an error. Binary files are copied
as they are. Afterwards the whole output is scanned, every file regardless of suffix: text files as text, binary
files as raw bytes with every pattern (as ASCII/UTF-8) and with every pattern but the e-mail one as UTF-16 and
UTF-32 in either byte order; recursively also zip members, gzip/bzip2/xz streams (.gz, .tgz), tar members,
inflated PDF streams and decoded PDF hex strings; and PDFs' text and metadata through pdftotext and pdfinfo when
they are installed. Any remaining "paper 1" (also with an underscore, hyphen, tilde, backslash, {} or nothing
between the words), "reviewer", repository name, "predecessor", scratch path, branch name, assistant attribution
(Co-Authored-By, Claude-Session, claude.ai/code, claude.com/claude-code, noreply@anthropic), unknown GitHub account
or e-mail address, or personal identifier makes the script exit with status 1. Check the output by hand before
uploading it.
"""
import argparse
import bz2
import datetime
import gzip
import hashlib
import io
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

RUNBOOK = "docs/GPU_RUNBOOK.md"
EXCLUDE = ("paper/BRIEF.md", "docs/PAPER2_PROPOSAL.md", RUNBOOK, "docs/V3_STAGE5_IMPLICATIONS.md", "docs/V3_PLAN.md", "paper/sections/old/", "notebooks/",
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
# Scorers that compare a recorded sha256 with a file this release edits (both checked by running them on a release built
# from 289192c and on the repository): results directory, scorer, the edited files whose hash it checks (a regex under
# the results directory), why they are edited, and what the scorer does on the release. The checking files (.json files
# under the results directory that record such a hash) are found at build time.
SCORER_CHECKS = [
    ("results/gpu_stage8a", "analysis/stage8a_score.py", r"frames/[^/]+\.json", "a local checkout path in their provenance "
     "records; the frames themselves are unchanged", 'reports "MISMATCH: frames file differs from the one this file used" in '
     "its PROVENANCE section for {refs}, and \"provenance MISMATCH\" in its summary, and exits with status 2; every gate "
     "and verdict is unchanged (7 MET, 6 NOT MET, 7 NOT EVALUABLE of 20 counted lines, as in "
     "results/gpu_stage8a/STAGE8A_SCORE.txt)"),
    ("results/gpu_stage4", "analysis/stage4_score.py", r"fit_[^/]+/run/COMPLETE\.json", "the code-identity label of "
     "Anonymous (2026)'s code in their inventory; the fitted bases are unchanged", 'reports "provenance: MISMATCH" for '
     "{refs} and stops with \"PROVENANCE MISMATCH ... not scored\" (exit status 1) before any evaluation; its output on "
     "the unedited files is results/gpu_stage4/STAGE4_SCORE.txt")]


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
        # singular self-references (the release is for double-blind review): docs/PREREGISTRATION.md, G16 and the
        # "Expected values" paragraphs of entries, and the release copy of stage 7
        _sub("the author", r"\b([Tt])he author predicts\b", r"\1he authors predict"),
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
# What may stand between "paper" and "1": white space, _ ~ -, a backslash alone or before one of them or , ; : !
# (LaTeX "Paper\ 1", "Paper\,1", "paper\_1"), an empty group {}, a URL-encoded space or an HTML non-breaking space.
PAPER_GAP = r"(?:[\s_~-]|\\[\s_~,;:!]?|\{\}|%20|&nbsp;|&#160;)*"
# Patterns that must not survive anywhere (matched case-insensitively): (label, regex, literal key for a fast pre-check;
# every match contains the key, lower-cased)
LEAKS = [("paper 1", r"paper" + PAPER_GAP + r"1(?!\d)", "paper"),
         ("reviewer", r"reviewer", "reviewer"),
         ("repository name", r"causal[\s_-]?keys", "causal"),
         ("predecessor", r"predecessor", "predecessor"),
         ("our prior work", r"\bour\W+(?:own\W+)?prior\W+work", "prior"),
         ("scratch path", r"claude-\d", "claude-"),
         ("branch name", r"claude/paper2", "claude/"),
         ("assistant attribution", r"co[\s_-]*authored[\s_-]*by", "authored"),
         ("assistant attribution", r"claude[\s_-]*session", "session"),
         ("assistant attribution", r"claude\.ai/code", "claude.ai"),
         ("assistant attribution", r"claude\.com/claude-code", "claude.com"),
         ("assistant attribution", r"noreply@anthropic", "noreply"),
         ("GitHub account", r"github\.com[/:](?!(?:%s)\b)[\w.-]+" % "|".join(GITHUB_OK), "github"),
         ("GitHub account", r"\b(?!(?:%s)\.)[\w-]+\.github\.io" % "|".join(GITHUB_OK), "github"),
         ("e-mail address", EMAIL, "@")]
# Leaks not searched in UTF-16/UTF-32 views of binary data (a lone "@" key would decode almost every binary file)
ASCII_ONLY = {"e-mail address"}
# Tests that need what a release copy lacks; they carry these pytest markers and skip there (listed in EDITED_FILES.txt)
SKIP_MARKERS = {"needs_git": "this repository's git history (a copy without .git)",
                "needs_runbook": f"{RUNBOOK}, which is withheld from this release"}
BIN_HITS = 3  # matches listed per pattern and view of a binary file
COMMIT_HEX = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])")
SHA256_HEX = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


def git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True).stdout


def head_files(rev):
    """(path, mode, blob id) of every file in the tree of `rev` (an exit on a symlink or submodule entry)."""
    raw = subprocess.run(["git", "ls-tree", "-r", "-z", "--full-tree", rev], check=True, capture_output=True).stdout
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, blob = meta.decode().split()
        path = path.decode("utf-8")
        if kind != "blob" or mode not in ("100644", "100755"):
            sys.exit(f"{path}: a {kind} with mode {mode} (symlink or submodule) cannot be copied; exclude it")
        yield path, mode, blob


class Blobs:
    """Blob contents from one `git cat-file --batch` process."""

    def __init__(self):
        self.p = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def read(self, blob):
        self.p.stdin.write(blob.encode() + b"\n")
        self.p.stdin.flush()
        header = self.p.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            sys.exit(f"git cat-file: no blob {blob} ({b' '.join(header).decode(errors='replace')})")
        data = self.p.stdout.read(int(header[2]))
        self.p.stdout.read(1)  # the newline after the content
        return data

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def personal_identifiers(extra):
    """(text, replacement) for the origin's owner, git's e-mail addresses and the given identifiers."""
    ids = {}
    m = re.search(r"github\.com[/:]([^/\s]+)/", git("remote", "get-url", "origin", check=False))
    if m:
        ids[m.group(1)] = "anonymous"
    emails = git("config", "user.email", check=False).split() + git("log", "--all", "--format=%ae%n%ce").split()
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


def _hits(text, label, pat, where, limit=None):
    n = 0
    for m in pat.finditer(text):
        if label == "e-mail address" and allowed_email(m.group()):
            continue
        n += 1
        if limit and n > limit:
            yield f"{where(m.start())}: {label}: (further matches not listed)"
            return
        snippet = re.sub(r"[^\x20-\x7e]", ".", text[max(0, m.start() - 30):m.end() + 30])
        yield f"{where(m.start())}: {label}: ...{snippet}..."


def text_leaks(text, leaks):
    low, found = text.lower(), []
    for label, pat, key in leaks:
        if key in low:
            found += _hits(text, label, pat, lambda i: f"line {text.count(chr(10), 0, i) + 1}")
    return found


def compile_leaks(identifiers):
    """Text leaks (Unicode, case-insensitive) and, for binary data, (label, ASCII str regex, key, UTF-16 and UTF-32
    byte regexes of the key with NUL gaps, or None for an ASCII-only leak)."""
    raw = [(label, p, key) for label, p, key in LEAKS] + [("personal identifier", re.escape(i), i.lower()) for i in identifiers]
    leaks = [(label, re.compile(p, re.I), key) for label, p, key in raw]
    bin_leaks = []
    for label, p, key in raw:
        wide = None if label in ASCII_ONLY else tuple(
            re.compile(gap.join(re.escape(c.encode()) for c in key), re.I) for gap in (b"\x00", b"\x00\x00\x00"))
        bin_leaks.append((label, re.compile(p, re.I | re.A), key, wide))
    return leaks, bin_leaks


def binary_leaks(data, bin_leaks, depth=0):
    """Leaks in raw bytes (every pattern, as ASCII/UTF-8; and, where the key occurs with NUL gaps, as UTF-16/UTF-32 in
    either byte order) and, recursively, in what containers() finds in them."""
    found = []
    view = data.decode("latin-1")  # one character per byte: offsets are byte offsets
    low = view.lower()
    for label, pat, key, wide in bin_leaks:
        if key in low:
            found += _hits(view, label, pat, lambda i: f"raw bytes at offset {i}", BIN_HITS)
        for unit, kpat in zip((2, 4), wide or ()):
            if not kpat.search(data):
                continue
            for off in range(unit):  # both byte orders at every alignment (a shifted LE view loses a BE string's last character)
                d = data[off:]
                d = d[:len(d) // unit * unit]
                for enc in (f"utf-{8 * unit}-le", f"utf-{8 * unit}-be"):
                    where = lambda i, off=off, unit=unit, enc=enc: f"{enc.upper()} text at byte offset {off + unit * i}"  # noqa: E731
                    found += _hits(d.decode(enc, "replace"), label, pat, where, BIN_HITS)
    if depth < 4:
        for label, name, content in containers(data):
            if name:
                found += [f"{label}: name: {f}" for lab, pat, key, _ in bin_leaks if key in name.lower()
                          for f in _hits(name, lab, pat, lambda i: name)]
            if content:
                found += [f"{label}: {f}" for f in binary_leaks(content, bin_leaks, depth + 1)]
    return found


def containers(data):
    """(label, member name or None, content or None) for what a blob contains: zip members (the content of a stored
    one only when it is itself a container, since its bytes are in the raw data), decompressed gzip/bzip2/xz data, tar
    members, a PDF's decoded hex strings and its inflated streams (and their hex strings)."""
    out = []
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    content = z.read(info)
                    keep = info.compress_type != zipfile.ZIP_STORED or is_container(content[:600])
                    out.append((f"zip member {info.filename}", info.filename, content if keep else None))
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError, OSError):
            pass
    for magic, mod, label in ((b"\x1f\x8b", gzip, "gunzipped"), (b"BZh", bz2, "bunzipped"), (b"\xfd7zXZ\x00", lzma, "unxz-ed")):
        if data.startswith(magic):
            try:
                out.append((label, None, mod.decompress(data)))
            except (OSError, EOFError, ValueError, lzma.LZMAError):
                pass
    if data[257:262] == b"ustar":
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
                for m in t.getmembers():
                    out.append((f"tar member {m.name}", m.name, t.extractfile(m).read() if m.isfile() else None))
        except (tarfile.TarError, OSError):
            pass
    if data[:5] == b"%PDF-":
        out.append(("PDF hex strings", None, pdf_hex(data)))
        for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
            try:
                inflated = zlib.decompress(m.group(1))
            except zlib.error:
                continue
            out += [(f"PDF stream at offset {m.start()}", None, inflated),
                    (f"PDF stream at offset {m.start()}, hex strings", None, pdf_hex(inflated))]
    return out


def pdf_hex(data):
    """The PDF hex strings (<FEFF0041...>, as in metadata) in data, decoded and joined by newlines."""
    out = []
    for m in re.finditer(rb"<([0-9A-Fa-f\s]{4,})>", data):
        h = re.sub(rb"\s", b"", m.group(1))
        out.append(bytes.fromhex((h + b"0" * (len(h) % 2)).decode()))
    return b"\n".join(out)


def is_container(head):
    return (head[:4] == b"PK\x03\x04" or head[:2] == b"\x1f\x8b" or head[:3] == b"BZh" or head[:6] == b"\xfd7zXZ\x00"
            or head[:5] == b"%PDF-" or head[257:262] == b"ustar")


def scan(root, identifiers):
    """Every leak in every file under root (file names included); text files as text, others as bytes. Also the
    hexadecimal tokens of the text files (for the commit-hash count)."""
    leaks, bin_leaks = compile_leaks(identifiers)
    pdftotext, pdfinfo = shutil.which("pdftotext"), shutil.which("pdfinfo")
    found, tokens = [], {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        found += [f"{rel}: file name: {f}" for f in text_leaks(rel, leaks)]
        if not p.is_file():
            continue
        data = p.read_bytes()
        text = as_text(rel, data, strict=False)
        if text is not None:
            found += [f"{rel}: {f}" for f in text_leaks(text, leaks)]
            t = set(COMMIT_HEX.findall(text))
            if t:
                tokens[rel] = t
            continue
        found += [f"{rel}: {f}" for f in binary_leaks(data, bin_leaks)]
        if data[:5] == b"%PDF-":
            for tool, args in ((pdftotext, ["-q", str(p), "-"]), (pdfinfo, ["-meta", str(p)]), (pdfinfo, [str(p)])):
                if tool:
                    out = subprocess.run([tool, *args], capture_output=True).stdout.decode("utf-8", "replace")
                    found += [f"{rel}: PDF {Path(tool).name}: {f}" for f in text_leaks(out, leaks)]
    return found, (pdftotext, pdfinfo), tokens


def commits_named(tokens):
    """{commit: [files]} for the commits of this repository (any branch) that the text files name by hash."""
    full = git("rev-list", "--all").split()
    by7 = {}
    for h in full:
        by7.setdefault(h[:7], []).append(h)
    named = {}
    for rel, toks in tokens.items():
        for t in toks:
            for h in by7.get(t[:7], ()):
                if h.startswith(t):
                    named.setdefault(h, set()).add(rel)
    return named


def origin_repo():
    m = re.search(r"github\.com[/:]([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", git("remote", "get-url", "origin", check=False).strip())
    return f"{m.group(1)}/{m.group(2)}" if m else None


def visibility(repo):
    """'public', 'private' or 'internal' from GitHub's REST API (through gh, else unauthenticated), 'not public' on an
    unauthenticated 404, None when it could not be asked."""
    if shutil.which("gh"):
        try:
            r = subprocess.run(["gh", "api", f"repos/{repo}", "--jq", ".visibility"], capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip().lower() in ("public", "private", "internal"):
                return r.stdout.strip().lower()
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}", headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
        return d.get("visibility") or ("private" if d.get("private") else "public")
    except urllib.error.HTTPError as e:
        return "not public" if e.code == 404 else None
    except (OSError, ValueError):
        return None


def hash_warning(named, repo, vis, offline):
    """The authors' warning about the commit hashes the release carries (never written into the release)."""
    if not named:
        return "The release names no commit of this repository by hash."
    files = sorted({f for fs in named.values() for f in fs})
    top = sorted(files, key=lambda f: -sum(f in fs for fs in named.values()))[:5]
    when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    lines = [f"The release names {len(named)} commits of this repository by hash, in {len(files)} files "
             f"(most in {', '.join(top)}). Commit hashes identify the repository while it is public: GitHub finds a "
             "public repository's commits by their hash. The records are not rewritten."]
    where = f"github.com/{repo}" if repo else "the repository"
    if vis == "public":
        lines.append(f"{where} IS PUBLIC (GitHub API, {when}). Make it private (Settings > General > Danger Zone > Change "
                     "visibility) BEFORE the release reaches reviewers, and keep it private until the review ends.")
    elif vis in ("private", "internal", "not public"):
        lines.append(f"{where} is {vis} (GitHub API, {when}). Keep it so until the review ends; check again just before "
                     "uploading the release.")
    else:
        lines.append(f"The visibility of {where} could not be checked{' (--offline)' if offline else ''}. Make sure it is "
                     "private BEFORE the release reaches reviewers, and keep it private until the review ends.")
    bar = "!" * 100
    return "\n".join([bar, "!! FOR THE AUTHORS (not in the release): COMMIT HASHES", bar, *lines, bar])


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


def skipping_tests(out):
    """{marker: [test ids]} of the release's tests that carry a SKIP_MARKERS marker."""
    tests = {k: [] for k in SKIP_MARKERS}
    for f in sorted((out / "tests").glob("test_*.py")):
        for marker, name in re.findall(r"^@(\w+)\n(?:@.*\n)*def (test_\w+)", f.read_text(), re.M):
            if marker in tests:
                tests[marker].append(f"tests/{f.name}::{name}")
    return tests


def prior_release_note(out):
    """Where the GPU scripts and tests expect Anonymous (2026)'s unpacked code release, and which read PRIOR_R."""
    expect, take = [], []
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.suffix in (".sh", ".py") and "v5.5-code-release" in (t := f.read_text(errors="replace")):
            rel = f.relative_to(out).as_posix()
            expect.append(rel)
            if re.search(r"\$\{PRIOR_R:-|environ\.get\(\"PRIOR_R\"", t):
                take.append(rel)
    if not expect:
        return ""
    other = [f for f in expect if f not in take]
    return ("Anonymous (2026)'s code release. The GPU scripts read its download URL from $PRIOR_ZIP_URL (not included in\n"
            "this release) and expect the unpacked release in a folder named v5.5-code-release under $PRIOR_ROOT\n"
            "(default ~/prior), i.e. ~/prior/v5.5-code-release; the folder in the release's own zip has another name. Either\n"
            "rename the unpacked folder to $PRIOR_ROOT/v5.5-code-release, or set PRIOR_R to the unpacked folder.\n"
            + fill("Files that expect the folder name: " + ", ".join(expect) + ". Of these, PRIOR_R is read by "
                   + (", ".join(take) or "none") + "; "
                   + (", ".join(other) + " read only PRIOR_ROOT (rename the folder for them)" if other else "every one reads it")
                   + ".") + "\n\n")


def fill(text):
    return textwrap.fill(text, 118, break_long_words=False, break_on_hyphens=False)


def scorer_notes(out, sha_refs):
    """What each scorer in SCORER_CHECKS reports on the release, with the files it flags and the edited files behind it."""
    notes = []
    for res, scorer, trigger, why, what in SCORER_CHECKS:
        pat = re.compile(re.escape(res + "/") + trigger + "$")
        refs = {f: [x for x in e if pat.match(x)] for f, e in sha_refs.items() if f.startswith(res + "/") and f.endswith(".json")}
        refs = {f: e for f, e in refs.items() if e}
        if not refs or not (out / scorer).exists():
            continue
        edited = sorted({x for e in refs.values() for x in e})
        notes.append(f"{scorer} run on this release " + what.format(refs=", ".join(sorted(refs))) + f". The cause: the "
                     f"{len(edited)} files " + ", ".join(edited) + f" were edited ({why}), and the files it flags record "
                     "their sha256.")
    return "".join(fill(n) + "\n\n" for n in notes)


def build(out, identifiers, head):
    subs = SUBS + [(i.lower(), re.compile(re.escape(i), re.I), rep) for i, rep in identifiers.items()]
    out.mkdir(parents=True, exist_ok=False)
    edited, renamed, seen, left_out = [], [], {}, set()
    edited_sha, hexes = {}, {}
    blobs = Blobs()
    for f, mode, blob in head_files(head):
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
        data = blobs.read(blob)
        t0 = as_text(f, data)
        if t0 is not None:
            if not f.endswith(".sha256"):
                h = set(SHA256_HEX.findall(data))
                if h:
                    hexes[name] = h
            t1 = scrub(t0, subs)
            if t1 != t0:
                if f.endswith(".sha256"):
                    check_manifest(f, t0, t1)
                edited.append(name)
                edited_sha[hashlib.sha256(data).hexdigest().encode()] = name
                data = t1.encode("utf-8")
        dst.write_bytes(data)
        if mode == "100755":
            dst.chmod(0o755)
    blobs.close()
    # files that record the sha256 of an edited file (as it was before the edit)
    sha_refs = {}
    for name, h in hexes.items():
        e = sorted(edited_sha[x] for x in h if x in edited_sha and edited_sha[x] != name)
        if e:
            sha_refs[name] = e
    log = git("log", "--reverse", "--format=%h", head, "--", *TRACKED_HISTORY).split()
    parts = []
    for h in log:
        head_txt = git("show", "-s", "--format=commit %h%ncommitted %cI%n%n    %s%n", h)
        diff = git("show", "--format=", h, "--", *TRACKED_HISTORY)
        parts.append(scrub(head_txt + diff, subs))
    runs = git("log", "--reverse", "--format=%h  %cI  %s", head, "--", "results/", "experiments/", "scripts/")
    (out / "PREREGISTRATION_HISTORY.txt").write_text(
        "== Commits that touched the preregistration log or a scoring script (with diffs)\n\n" + "\n".join(parts)
        + "\n\n== Commits that touched results, experiment code or GPU scripts (hash, commit time, subject)\n\n" + scrub(runs, subs))
    skips = skipping_tests(out)
    edited_txt = (
        "Text files whose content was edited for this release (local paths, repository and account names, and other\n"
        "identifying wording replaced); their entries in the results' MANIFEST.sha256 files therefore do not verify.\n"
        "A MANIFEST.sha256 listed here was edited only in the file names it lists; its hashes are unchanged.\n\n"
        + "\n".join(edited) + "\n\n"
        "Files renamed for this release (new names shown; the MANIFEST.sha256 files list them under these names):\n\n"
        + "\n".join(renamed) + "\n\n"
        "Files that record (in provenance records or logs) the sha256 of a file edited above, as it was before the edit; a\n"
        "scorer that checks such a hash against the file reports a mismatch on this release, although the data are those\n"
        "scored:\n\n"
        + "\n".join(f"{f}: {', '.join(e)}" for f, e in sorted(sha_refs.items())) + "\n\n"
        + scorer_notes(out, sha_refs)
        + "Public third-party files left out of this release (their MANIFEST.sha256 entries hold the sha256 of the copy used):\n\n"
        + "\n".join(f"{f}: {src}" for f, src in OMIT.items()) + "\n\n"
        "Other files listed in a MANIFEST.sha256 that are not in this release (never committed to the repository):\n\n"
        + "\n".join(f + (" (excluded from this release)" if f in left_out else "")
                    for f in missing_from_manifests(out) if f not in OMIT) + "\n\n"
        f"Withheld: {RUNBOOK}, the authors' operating notes for the rented GPU machines.\n\n"
        + ("Tests that skip in this copy (each says why; their outcomes on the GPU machines are in the results' pytest\n"
           "logs):\n\n" + "\n".join(f"{t}: needs {SKIP_MARKERS[k]}" for k in SKIP_MARKERS for t in skips[k]) + "\n\n"
           if any(skips.values()) else "")
        + prior_release_note(out))
    (out / "EDITED_FILES.txt").write_text(edited_txt)
    warnings = [f"no test in the release carries the marker {k} (for tests that need {v}): such tests fail in the release "
                "instead of skipping (are the test changes committed?)" for k, v in SKIP_MARKERS.items() if not skips[k]]
    return len(log), len(edited), len(renamed), edited_txt, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="output directory (must not exist; outside the repository)")
    ap.add_argument("--scan-only", action="store_true", help="only scan an existing directory for leaks")
    ap.add_argument("--identifier", action="append", default=[],
                    help="a further personal identifier (name, handle, e-mail) to replace and scan for; repeatable")
    ap.add_argument("--offline", action="store_true", help="do not ask GitHub whether the origin repository is public")
    args = ap.parse_args(argv)
    root = Path(git("rev-parse", "--show-toplevel").strip()).resolve()
    out = Path(args.out).resolve()
    os.chdir(root)
    extra = args.identifier + [s.strip() for s in os.environ.get("ANON_IDENTIFIERS", "").split(",") if s.strip()]
    identifiers = personal_identifiers(extra)
    notes = []
    if not args.scan_only:
        if out == root or root in out.parents:
            sys.exit(f"{out} is inside the repository; write the release elsewhere")
        head = git("rev-parse", "--verify", "HEAD^{commit}").strip()
        notes.append(f"Built from HEAD {head} ({git('show', '-s', '--format=%cI  %s', head).strip()}).")
        dirty = [ln[3:] for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if not ln[3:].startswith(EXCLUDE)]
        if dirty:
            msg = (f"the working tree differs from HEAD in {len(dirty)} files; the release is built from HEAD, so these "
                   f"uncommitted edits are NOT in it (commit them first if they should ship): {dirty}")
            print("warning: " + msg, file=sys.stderr)
            notes.append("Uncommitted: " + msg)
        n_log, n_edited, n_renamed, edited_txt, warnings = build(out, identifiers, head)
        for w in warnings:
            print("warning: " + w, file=sys.stderr)
            notes.append("Warning: " + w)
        print(f"wrote {out} from HEAD {head[:12]}; {n_log} history commits; {n_edited} files edited, {n_renamed} renamed")
    elif not out.is_dir():
        sys.exit(f"{out} is not a directory")
    found, (pdftotext, pdfinfo), tokens = scan(out, identifiers)
    if not (pdftotext and pdfinfo):
        print("note: pdftotext or pdfinfo not found, so PDF text or metadata was scanned only through raw and inflated "
              "bytes", file=sys.stderr)
    repo = origin_repo()
    vis = visibility(repo) if repo and not args.offline else None
    warning = hash_warning(commits_named(tokens), repo, vis, args.offline)
    if not args.scan_only:
        authors = out.parent / (out.name + ".AUTHORS.txt")
        authors.write_text(warning + "\n\n" + "\n".join(notes) + "\n\nLeak scan: "
                           + (f"FAILED, {len(found)} possible leaks" if found else "no leaks found")
                           + "\n\n==== EDITED_FILES.txt as shipped in the release ====\n\n" + edited_txt)
        print(f"authors' notes (not part of the release): {authors}")
    if found:
        print(f"LEAK SCAN FAILED: {len(found)} possible leaks in {out}; do not upload it:", file=sys.stderr)
        for f in found:
            print("  " + f, file=sys.stderr)
    else:
        print(f"leak scan of {out}: no leaks found")
    print(warning, file=sys.stderr)  # last, so that it is not scrolled away
    if found:
        sys.exit(1)


if __name__ == "__main__":
    main()
