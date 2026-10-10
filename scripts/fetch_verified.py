"""Fetch the files of one stage-8 model (preregistration J, P-2026-10-10-J) and verify every byte against the hashes
pinned in scripts/stage8_models.json, captured from the Hugging Face API on 2026-10-10.

    python scripts/fetch_verified.py --key llama8 --dest ~/stage8_models/llama8 [--manifest scripts/stage8_models.json]

Sources, per file, in this order (the first copy that verifies is used):
  1. a copy already in --dest (re-verified on every call, so the command is idempotent);
  2. the local Hugging Face cache, if it holds the file at the pinned revision of a listed repo (a symlink is made;
     --no-cache skips this);
  3. downloads: the official repo at the pinned revision (for a gated repo only when HF_TOKEN is set), then, for a gated
     repo, the ungated repos the manifest lists for that file, each at its pinned revision. A mirror is listed only
     where its file has the official hash and size, so a verified copy is byte-identical to the official file.
Verification: the size, then sha256 of the content for LFS files, or the git blob id sha1(b"blob {size}\\0" + content)
for small files, as the Hub reports them. Every file of the manifest entry must verify, and --dest may hold no other
regular file, so the directory a model loads from is exactly the official file set at the pinned revision.
Output: DEST/VERIFIED.json (written atomically, only when every file verifies; an earlier one is removed first) with the
key, repo, revision, attention implementation, the manifest's sha256, and per file the expected and observed hash and
the source used. A transient download error is retried three times per source (30 s, 60 s back-off); an error that a
retry cannot fix (401, 403, 404, a gated or missing repo or file) moves on to the next source at once.
Exit status: 0 verified; 1 refused (a file not verified from any source, with DEST/VERIFY_FAILED.json listing every
attempt, or a file in DEST that the manifest does not list); 2 a bad manifest, key or usage; 4 not enough free disk at
DEST for the files still to fetch (2 GiB margin; nothing is fetched). Only status 1 is a verification failure.
The token is read from HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) and never printed.
Downloads use huggingface_hub when it is importable, else plain HTTPS (urllib) with retries. --local-root R replaces
every download by a copy from R/<owner>/<name>/<revision>/<file> (offline use and tests). --verify-only never fetches.
"""
from __future__ import annotations

import argparse
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "stage8_models.json"
KEYS = ("qwen7", "qwen14", "qwen1.5", "qwen3b", "qwen0.5", "mistral7", "olmo7", "llama8", "gemma9", "gemma2b",
        "phi4", "falcon7", "yi9", "mistral24")
HEX40, HEX64 = re.compile(r"^[0-9a-f]{40}$"), re.compile(r"^[0-9a-f]{64}$")
CHUNK = 8 << 20
OWN = {"VERIFIED.json", "VERIFY_FAILED.json", ".fetch.lock"}   # files of this script inside --dest


# --------------------------------------------------------------------------- hashes
def git_blob_sha1(path) -> str:
    """The git blob id of a file: sha1 of b"blob {size}\\0" followed by the content (what the Hub calls blobId)."""
    size = os.path.getsize(path)
    h = hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def kind(spec) -> str:
    return "sha256" if "sha256" in spec else "git_blob"


def verify_file(path, spec):
    """(ok, observed) for one file against its manifest spec {"sha256" | "git_blob": hex, "size": int}."""
    path = Path(path)
    if not path.exists():
        return False, {"error": "missing"}
    size = os.path.getsize(path)
    if size != spec["size"]:
        return False, {"size": size, "error": f"size {size} != {spec['size']}"}
    k = kind(spec)
    obs = sha256_file(path) if k == "sha256" else git_blob_sha1(path)
    return obs == spec[k], {"size": size, k: obs}


# --------------------------------------------------------------------------- manifest
def load_manifest(path=MANIFEST) -> dict:
    return json.loads(Path(path).read_text())


def manifest_sha256(path=MANIFEST) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(man: dict) -> list:
    """Schema problems of a manifest (empty list = valid). Checked by tests/test_fetch_verified.py on the committed file
    and by every fetch."""
    bad = []
    if man.get("schema") != "stage8_models/1":
        bad.append("schema is not stage8_models/1")
    models = man.get("models")
    if not isinstance(models, dict) or not models:
        return bad + ["no models"]
    if man.get("fallback") not in models:
        bad.append("fallback key not in models")
    for key, e in models.items():
        p = f"{key}: "
        for fld in ("repo", "revision", "gated", "attn", "allow_patterns", "files", "sources"):
            if fld not in e:
                bad.append(p + f"no {fld}")
        if any(f not in e for f in ("repo", "revision", "files", "sources")):
            continue
        if not re.match(r"^[\w.-]+/[\w.-]+$", e["repo"]):
            bad.append(p + f"bad repo {e['repo']!r}")
        if not HEX40.match(e["revision"]):
            bad.append(p + "revision is not a 40-hex commit")
        if e.get("attn") not in ("sdpa", "eager"):
            bad.append(p + f"attn {e.get('attn')!r}")
        files = e["files"]
        if not files or not any(f.endswith(".safetensors") for f in files) or "config.json" not in files:
            bad.append(p + "files must include config.json and safetensors")
        for f, s in files.items():
            if "/" in f or f in OWN or f.startswith("."):
                bad.append(p + f"bad file name {f!r}")
            ks = [k for k in ("sha256", "git_blob") if k in s]
            if len(ks) != 1:
                bad.append(p + f"{f}: needs exactly one of sha256 / git_blob")
            elif not (HEX64 if ks[0] == "sha256" else HEX40).match(str(s[ks[0]])):
                bad.append(p + f"{f}: malformed {ks[0]}")
            if not isinstance(s.get("size"), int) or s["size"] <= 0:
                bad.append(p + f"{f}: bad size")
            if not any(fnmatch.fnmatch(f, pat) for pat in e.get("allow_patterns", [])):
                bad.append(p + f"{f}: matches no allow_pattern")
            if any(fnmatch.fnmatch(f, pat) for pat in e.get("ignore_patterns", [])):
                bad.append(p + f"{f}: matches an ignore_pattern")
        cover = {f: 0 for f in files}
        for s in e["sources"]:
            if not re.match(r"^[\w.-]+/[\w.-]+$", str(s.get("repo"))) or not HEX40.match(str(s.get("revision"))):
                bad.append(p + f"bad source {s.get('repo')!r}@{s.get('revision')!r}")
                continue
            if s["repo"] == e["repo"]:
                bad.append(p + "the official repo is not a mirror source")
            for f in s.get("files", []):
                if f not in files:
                    bad.append(p + f"source {s['repo']} lists {f!r}, which is not a manifest file")
                else:
                    cover[f] += 1
        unver = e.get("unverifiable_without_token", {})
        if e["gated"]:
            for f, n in cover.items():
                if n == 0 and f not in unver:
                    bad.append(p + f"{f}: gated, no mirror source and not listed as unverifiable_without_token")
        elif e["sources"]:
            bad.append(p + "a non-gated model is fetched from its official repo only")
    return bad


def sources_for(e: dict, fname: str, token: bool) -> list:
    """Ordered (repo, revision) candidates for one file: the official repo (gated: only with a token), then mirrors."""
    out = [] if (e["gated"] and not token) else [(e["repo"], e["revision"])]
    return out + [(s["repo"], s["revision"]) for s in e["sources"] if fname in s.get("files", [])]


# --------------------------------------------------------------------------- downloads
def _hub_download(repo, rev, fname, tmpdir: Path, token):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=repo, filename=fname, revision=rev, local_dir=str(tmpdir), token=token or False)
    return Path(p)


def _https_download(repo, rev, fname, tmpdir: Path, token):
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{urllib.parse.quote(fname)}"
    dst = tmpdir / fname
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:   # HTTPError carries .code
        shutil.copyfileobj(r, f, CHUNK)
    return dst


PERMANENT = {"GatedRepoError", "RepositoryNotFoundError", "EntryNotFoundError", "RevisionNotFoundError",
             "RemoteEntryNotFoundError", "FileNotFoundError"}


def permanent(ex) -> bool:
    """A download error that a retry of the same source cannot fix (no access, no such repo, revision or file)."""
    code = getattr(ex, "code", None) or getattr(getattr(ex, "response", None), "status_code", None)
    return type(ex).__name__ in PERMANENT or code in (401, 403, 404)


def default_download(repo, rev, fname, tmpdir: Path, token):
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return _https_download(repo, rev, fname, tmpdir, token)
    return _hub_download(repo, rev, fname, tmpdir, token)


def local_root_download(root):
    """A downloader that copies R/<owner>/<name>/<revision>/<file> (offline sources and tests)."""
    def dl(repo, rev, fname, tmpdir: Path, token):
        src = Path(root) / repo / rev / fname
        if not src.is_file():
            raise FileNotFoundError(f"{src} not found")
        dst = tmpdir / fname
        shutil.copyfile(src, dst)
        return dst
    return dl


def cache_lookup(repo, rev, fname):
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(repo_id=repo, filename=fname, revision=rev)
    except Exception:
        return None
    return Path(p) if isinstance(p, str) and os.path.isfile(p) else None


# --------------------------------------------------------------------------- fetch
def write_atomic(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


MARGIN = 2 << 30   # bytes of free disk kept beyond the files still to fetch


def fetch(key, dest, manifest=MANIFEST, token=None, download=None, use_cache=True, verify_only=False, log=None,
          tries=3, wait=30):
    """Assemble and verify DEST for one manifest key. Returns the VERIFIED record. Raises SystemExit(1) when a file
    cannot be verified (its bytes do not match from any source that delivered it, or no source delivered it after
    `tries` attempts each, with `wait`-second back-off), or DEST holds a file the manifest does not list; SystemExit(4)
    when the disk cannot hold the files still to fetch (an environment problem, not a verification failure);
    SystemExit(<message>) for a bad manifest or key."""
    log = log or (lambda *a: print(*a, file=sys.stderr, flush=True))
    man = load_manifest(manifest)
    bad = validate_manifest(man)
    if bad:
        raise SystemExit(f"manifest {manifest} is invalid: " + "; ".join(bad[:10]))
    if key not in man["models"]:
        raise SystemExit(f"unknown key {key!r} (one of {', '.join(man['models'])})")
    e = man["models"][key]
    download = download or default_download
    dest = Path(dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    lock = open(dest / ".fetch.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)   # a background prefetch and a later fetch of the same key do not interleave
    try:
        for f in ("VERIFIED.json", "VERIFY_FAILED.json"):
            (dest / f).unlink(missing_ok=True)
        def present(f, sp):   # a copy of the right size in DEST or (to be linked) in the local HF cache needs no space
            if (dest / f).is_file() and os.path.getsize(dest / f) == sp["size"]:
                return True
            return use_cache and any((c := cache_lookup(r, v, f)) is not None and os.path.getsize(c) == sp["size"]
                                     for r, v in sources_for(e, f, bool(token)))
        need = sum(sp["size"] for f, sp in e["files"].items() if not present(f, sp))
        free = shutil.disk_usage(dest).free
        if not verify_only and need + MARGIN > free:
            log(f"{key}: not enough disk at {dest}: {free / 1e9:.1f} GB free, {need / 1e9:.1f} GB to fetch "
                f"(+{MARGIN / 1e9:.1f} GB margin); nothing fetched")
            raise SystemExit(4)
        part = dest / ".partial"
        recs, failed = {}, []
        for fname, spec in e["files"].items():
            target = dest / fname
            rec = {"kind": kind(spec), "expected": spec[kind(spec)], "size": spec["size"], "attempts": []}
            if target.exists() or target.is_symlink():
                ok, obs = verify_file(target, spec)
                if ok:
                    rec.update(ok=True, source="existing copy (re-verified)", observed=obs[rec["kind"]])
                    recs[fname] = rec
                    continue
                rec["attempts"].append({"source": "existing copy", "observed": obs})
                log(f"{key}/{fname}: the existing copy does not verify ({obs}); fetched again")
                target.unlink()
            srcs = [] if verify_only else sources_for(e, fname, bool(token))
            for repo, rev in (srcs if use_cache else []):   # first a verified copy in the local HF cache (linked)
                src = f"{repo}@{rev} (local HF cache)"
                if (c := cache_lookup(repo, rev, fname)) is not None:
                    ok, obs = verify_file(c, spec)
                    rec["attempts"].append({"source": src, "ok": ok, "observed": obs})
                    if ok:
                        os.symlink(os.path.realpath(c), target)
                        rec.update(ok=True, source=src, observed=obs[rec["kind"]])
                        break
            for repo, rev in ([] if rec.get("ok") else srcs):   # then downloads, source by source
                src = f"{repo}@{rev}"
                shutil.rmtree(part, ignore_errors=True)
                part.mkdir()
                p = None
                for t in range(tries):   # transient errors are retried on the same source; then the next source
                    try:
                        p = download(repo, rev, fname, part, token)
                        break
                    except Exception as ex:
                        rec["attempts"].append({"source": src, "error": f"{type(ex).__name__}: {ex}"[:300]})
                        log(f"{key}/{fname}: {src} failed (try {t + 1}): {type(ex).__name__}: {str(ex)[:200]}")
                        if permanent(ex) or t == tries - 1:
                            break
                        time.sleep(wait * (t + 1))
                if p is None:
                    continue
                ok, obs = verify_file(p, spec)
                rec["attempts"].append({"source": src, "ok": ok, "observed": obs})
                if ok:
                    os.replace(p, target)
                    rec.update(ok=True, source=src, observed=obs[rec["kind"]])
                    log(f"{key}/{fname}: verified ({rec['kind']}) from {src}")
                    break
                log(f"{key}/{fname}: {src} gave bytes that do not verify ({obs}); refused")
            shutil.rmtree(part, ignore_errors=True)
            rec.setdefault("ok", False)
            if not rec["ok"]:
                why = e.get("unverifiable_without_token", {}).get(fname) if not token else None
                bad_bytes = any(a.get("ok") is False for a in rec["attempts"])
                rec["reason"] = why or ("not fetched (--verify-only)" if verify_only else
                                        "bytes that do not match the official hash" if bad_bytes else
                                        "unavailable from every listed source")
                failed.append(fname)
            recs[fname] = rec
        extra = sorted(p.name for p in dest.iterdir() if p.name not in e["files"] and p.name not in OWN
                       and not p.name.startswith(".") and (p.is_file() or p.is_symlink()))
        out = {"key": key, "repo": e["repo"], "revision": e["revision"], "attn": e["attn"], "gated": e["gated"],
               "token_used": bool(token), "manifest": str(Path(manifest)), "manifest_sha256": manifest_sha256(manifest),
               "fetcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "dest": str(dest), "host": socket.gethostname(),
               "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "sources_used": sorted({r["source"] for r in recs.values() if r.get("ok")}),
               "files": recs, "failed": failed, "unexpected_files": extra}
        if failed or extra:
            write_atomic(out, dest / "VERIFY_FAILED.json")
            msg = []
            if failed:
                msg.append(f"{len(failed)} file(s) not verified: " + ", ".join(f"{f} ({recs[f]['reason']})" for f in failed))
            if extra:
                msg.append("files not in the manifest in " + str(dest) + ": " + ", ".join(extra))
            log(f"{key}: REFUSED. " + " ".join(msg))
            raise SystemExit(1)
        write_atomic(out, dest / "VERIFIED.json")
        log(f"{key}: {len(recs)} files verified against {e['repo']}@{e['revision']} -> {dest}/VERIFIED.json")
        return out
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--no-cache", action="store_true", help="do not take verified files from the local HF cache")
    ap.add_argument("--verify-only", action="store_true", help="verify DEST as it is; fetch nothing")
    ap.add_argument("--local-root", help="copy from R/<owner>/<name>/<revision>/<file> instead of downloading")
    a = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    try:
        fetch(a.key, a.dest, a.manifest, token=token,
              download=local_root_download(a.local_root) if a.local_root else None,
              use_cache=not a.no_cache, verify_only=a.verify_only)
    except SystemExit as ex:
        if isinstance(ex.code, str):   # a manifest or usage error
            print(ex.code, file=sys.stderr)
            return 2
        return int(ex.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
