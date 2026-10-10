"""scripts/fetch_verified.py and scripts/stage8_common.sh (preregistration J, rule G2 and the shared GPU-script library),
offline: the git blob id against known values (git hash-object, and the Hub's id of a cached file when present);
sha256 and size checks; a gated model assembled from fake mirror directories, in source order, with a corrupted copy
refused and the next source used, an existing copy re-verified (idempotent) or replaced when corrupted, a stray file
refused, VERIFIED.json only on success; transient download errors retried on the same source, permanent ones (404)
not; too little disk is status 4, not a refusal; the CLI's exit codes (0, 1, 2); the committed scripts/stage8_models.json
(schema, the fourteen keys and repos, attention per family, every gated file covered by a mirror, revisions equal to
those of stages 5-7); the library's guards (DRAFT entry, no entry, modified tree, no finalising commit, code or entry
changed since it) on throw-away git repositories; and a TEST_MODE dry run of s8_init, s8_fetch, s8_step (keep,
FORCE_STEPS, a failed step, a failed exploratory step, a partial step, S8_OUTPUTS), s8_time_left, s8_skip, s8_drop
and s8_finish with a fake scorer (GATES and SUMMARY printed, MANIFEST.sha256, archive, exit status); s8_fetch outside
TEST_MODE with a stand-in fetcher (a refusal returns 1 and lets the script go on; a disk error stops it at the next
library call; s8_drop deletes a model fetched in this session and keeps one verified before it)."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_verified as fv  # noqa: E402

PY = sys.executable
REV = {"official": "1" * 40, "m1": "2" * 40, "m2": "3" * 40, "plain": "4" * 40}


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def spec(data: bytes, lfs: bool) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)} if lfs else {"git_blob": blob(data), "size": len(data)}


FILES = {"config.json": (b'{"model_type": "toy"}\n', False), "model.safetensors": (os.urandom(4096) + b"w", True),
         "tokenizer.json": (b'{"tok": [1, 2, 3]}\n', False)}


def put(root: Path, repo: str, rev: str, files: dict):
    for name, data in files.items():
        p = root / repo / rev / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def fake_world(tmp_path, corrupt_m1=()):
    """A local root with the gated official repo, two mirrors (m1 lacks tokenizer.json; corrupt_m1 files hold wrong
    bytes of the right size) and a non-gated repo; and a manifest describing them."""
    root = tmp_path / "hub"
    good = {n: d for n, (d, _) in FILES.items()}
    put(root, "org/gated", REV["official"], good)
    bad = {n: (bytes([d[0] ^ 1]) + d[1:]) if n in corrupt_m1 else d for n, d in good.items() if n != "tokenizer.json"}
    put(root, "mirror/one", REV["m1"], bad)
    put(root, "mirror/two", REV["m2"], good)
    put(root, "org/plain", REV["plain"], good)
    files = {n: spec(d, lfs) for n, (d, lfs) in FILES.items()}
    base = {"attn": "sdpa", "allow_patterns": ["*.json", "*.safetensors"], "ignore_patterns": [], "files": files}
    man = {"schema": "stage8_models/1", "fallback": "plain", "models": {
        "gated": dict(base, repo="org/gated", revision=REV["official"], gated="manual", unverifiable_without_token={},
                      sources=[{"repo": "mirror/one", "revision": REV["m1"], "files": ["config.json", "model.safetensors"]},
                               {"repo": "mirror/two", "revision": REV["m2"], "files": list(files)}]),
        "plain": dict(base, repo="org/plain", revision=REV["plain"], gated=False, sources=[])}}
    mp = tmp_path / "models.json"
    mp.write_text(json.dumps(man))
    return root, mp, man


def run_fetch(key, dest, mp, root, token=None):
    return fv.fetch(key, dest, mp, token=token, download=fv.local_root_download(root), use_cache=False, log=lambda *a: None)


# --------------------------------------------------------------------------- hashes
def test_git_blob_known_values(tmp_path):
    cases = {b"": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", b"hello\n": "ce013625030ba8dba906f756967f9e9ca394464a"}
    for data, want in cases.items():
        p = tmp_path / "f"
        p.write_bytes(data)
        assert fv.git_blob_sha1(p) == want
    p.write_bytes(os.urandom(3 * fv.CHUNK // 2))   # spans chunk boundaries
    if shutil.which("git"):
        assert fv.git_blob_sha1(p) == subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True,
                                                     check=True).stdout.strip()
    assert fv.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_git_blob_matches_hub_id_of_cached_file():
    """The Hub's blobId of Qwen2.5-0.5B-Instruct's config.json (captured in the manifest) equals our sha1 of the cached
    file (skipped when the model is not in the local HF cache)."""
    e = fv.load_manifest()["models"]["qwen0.5"]
    p = fv.cache_lookup(e["repo"], e["revision"], "config.json")
    if p is None:
        pytest.skip("Qwen2.5-0.5B-Instruct config.json not cached at the pinned revision")
    assert fv.verify_file(p, e["files"]["config.json"]) == (True, {"size": 659, "git_blob": e["files"]["config.json"]["git_blob"]})


def test_verify_file_size_and_hash(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert fv.verify_file(p, spec(b"abc", True))[0] and fv.verify_file(p, spec(b"abc", False))[0]
    assert not fv.verify_file(p, spec(b"abd", True))[0]                     # same size, other bytes
    assert not fv.verify_file(p, spec(b"abd", False))[0]
    ok, obs = fv.verify_file(p, spec(b"abcd", True))
    assert not ok and "size" in obs["error"]
    assert fv.verify_file(tmp_path / "missing", spec(b"abc", True)) == (False, {"error": "missing"})


# --------------------------------------------------------------------------- fetch with fake sources
def test_gated_assembled_from_mirrors_in_order(tmp_path):
    root, mp, man = fake_world(tmp_path)
    dest = tmp_path / "d"
    out = run_fetch("gated", dest, mp, root)
    v = json.loads((dest / "VERIFIED.json").read_text())
    assert v == json.loads(json.dumps(out)) and not v["failed"] and not v["token_used"]
    assert v["files"]["config.json"]["source"] == f"mirror/one@{REV['m1']}"           # first listed source
    assert v["files"]["tokenizer.json"]["source"] == f"mirror/two@{REV['m2']}"        # the only source for it
    assert v["manifest_sha256"] == hashlib.sha256(mp.read_bytes()).hexdigest()
    for n, (d, _) in FILES.items():
        assert (dest / n).read_bytes() == d
    assert not (dest / ".partial").exists() and not (dest / "VERIFY_FAILED.json").exists()
    # idempotent: a second call re-verifies the copies in place and fetches nothing
    calls = []
    out2 = fv.fetch("gated", dest, mp, download=lambda *a: calls.append(a), use_cache=False, log=lambda *a: None)
    assert not calls and all(r["source"].startswith("existing copy") for r in out2["files"].values())


def test_official_first_with_token(tmp_path):
    root, mp, man = fake_world(tmp_path)
    e = man["models"]["gated"]
    assert fv.sources_for(e, "config.json", token=False)[0] == ("mirror/one", REV["m1"])
    assert fv.sources_for(e, "config.json", token=True)[0] == ("org/gated", REV["official"])
    assert fv.sources_for(man["models"]["plain"], "config.json", token=False) == [("org/plain", REV["plain"])]
    out = run_fetch("gated", tmp_path / "d", mp, root, token="t0k")
    assert {r["source"] for r in out["files"].values()} == {f"org/gated@{REV['official']}"} and out["token_used"]
    assert "t0k" not in (tmp_path / "d" / "VERIFIED.json").read_text()
    out = run_fetch("plain", tmp_path / "p", mp, root)
    assert {r["source"] for r in out["files"].values()} == {f"org/plain@{REV['plain']}"}


def test_corrupt_mirror_refused_next_source_used(tmp_path):
    root, mp, man = fake_world(tmp_path, corrupt_m1=("model.safetensors",))
    out = run_fetch("gated", tmp_path / "d", mp, root)
    r = out["files"]["model.safetensors"]
    assert r["ok"] and r["source"] == f"mirror/two@{REV['m2']}"
    assert r["attempts"][0]["source"] == f"mirror/one@{REV['m1']}" and r["attempts"][0]["ok"] is False


def test_mismatch_everywhere_refused(tmp_path):
    root, mp, man = fake_world(tmp_path, corrupt_m1=("config.json",))
    (root / "mirror/two" / REV["m2"] / "config.json").write_bytes(b'{"model_type": "toY"}\n')   # same size, wrong bytes
    dest = tmp_path / "d"
    with pytest.raises(SystemExit) as ex:
        run_fetch("gated", dest, mp, root)
    assert ex.value.code == 1
    assert not (dest / "VERIFIED.json").exists() and not (dest / "config.json").exists()
    f = json.loads((dest / "VERIFY_FAILED.json").read_text())
    assert f["failed"] == ["config.json"] and len(f["files"]["config.json"]["attempts"]) == 2
    # an earlier VERIFIED.json does not survive a failed re-verification
    (root / "mirror/two" / REV["m2"] / "config.json").write_bytes(FILES["config.json"][0])
    run_fetch("gated", dest, mp, root)
    assert (dest / "VERIFIED.json").exists()
    (root / "mirror/two" / REV["m2"] / "config.json").write_bytes(b'{"model_type": "toY"}\n')
    (dest / "config.json").write_bytes(b'{"model_type": "toY"}\n')
    with pytest.raises(SystemExit):
        run_fetch("gated", dest, mp, root)
    assert not (dest / "VERIFIED.json").exists()


def test_missing_source_refused(tmp_path):
    root, mp, man = fake_world(tmp_path)
    shutil.rmtree(root / "mirror/two")
    with pytest.raises(SystemExit) as ex:
        run_fetch("gated", tmp_path / "d", mp, root)
    assert ex.value.code == 1
    f = json.loads((tmp_path / "d" / "VERIFY_FAILED.json").read_text())
    assert f["failed"] == ["tokenizer.json"] and "FileNotFoundError" in f["files"]["tokenizer.json"]["attempts"][0]["error"]


def test_corrupted_copy_in_dest_is_replaced(tmp_path):
    root, mp, man = fake_world(tmp_path)
    dest = tmp_path / "d"
    run_fetch("gated", dest, mp, root)
    w = dest / "model.safetensors"
    w.write_bytes(b"\0" * w.stat().st_size)
    out = run_fetch("gated", dest, mp, root)
    r = out["files"]["model.safetensors"]
    assert r["attempts"][0]["source"] == "existing copy" and r["source"] == f"mirror/one@{REV['m1']}"
    assert w.read_bytes() == FILES["model.safetensors"][0]


def test_stray_file_refused(tmp_path):
    root, mp, man = fake_world(tmp_path)
    dest = tmp_path / "d"
    run_fetch("gated", dest, mp, root)
    (dest / "tokenizer.model").write_bytes(b"stray")
    with pytest.raises(SystemExit) as ex:
        run_fetch("gated", dest, mp, root)
    assert ex.value.code == 6   # the directory's state, not a verification failure of the model's files: no fallback
    assert json.loads((dest / "VERIFY_FAILED.json").read_text())["unexpected_files"] == ["tokenizer.model"]
    assert not (dest / "VERIFIED.json").exists()


def test_cache_copy_used_only_when_it_verifies(tmp_path, monkeypatch):
    root, mp, man = fake_world(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "config.json").write_bytes(FILES["config.json"][0])
    (cache / "model.safetensors").write_bytes(b"\1" * len(FILES["model.safetensors"][0]))   # wrong bytes
    monkeypatch.setattr(fv, "cache_lookup", lambda repo, rev, f: (cache / f) if (cache / f).exists() and repo == "mirror/one" else None)
    dest = tmp_path / "d"
    out = fv.fetch("gated", dest, mp, download=fv.local_root_download(root), use_cache=True, log=lambda *a: None)
    r = out["files"]["config.json"]
    assert r["source"].endswith("(local HF cache)") and r["link"] == "hard"
    assert not (dest / "config.json").is_symlink() and (dest / "config.json").stat().st_ino == (cache / "config.json").stat().st_ino
    assert not (dest / "model.safetensors").is_symlink() and out["files"]["model.safetensors"]["source"] == f"mirror/one@{REV['m1']}"
    assert (dest / "model.safetensors").stat().st_ino != (cache / "model.safetensors").stat().st_ino   # the bad copy not linked
    # the cache entry replaced later (a re-download renames a new file over the blob) or removed: the verified bytes stay
    (cache / "new").write_bytes(b'{"model_type": "toY"}\n')
    os.replace(cache / "new", cache / "config.json")
    assert (dest / "config.json").read_bytes() == FILES["config.json"][0]
    (cache / "config.json").unlink()
    assert (dest / "config.json").read_bytes() == FILES["config.json"][0]


def test_cache_symlink_only_across_file_systems(tmp_path, monkeypatch):
    """Where a hard link is impossible (another file system) the blob is symbolically linked, and the link is verified
    after it is made; a later call re-verifies it like any existing copy."""
    root, mp, man = fake_world(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "config.json").write_bytes(FILES["config.json"][0])
    monkeypatch.setattr(fv, "cache_lookup", lambda repo, rev, f: (cache / f) if (cache / f).exists() and repo == "mirror/one" else None)
    def no_link(*a):
        raise OSError(18, "Invalid cross-device link")
    monkeypatch.setattr(fv.os, "link", no_link)
    dest = tmp_path / "d"
    out = fv.fetch("gated", dest, mp, download=fv.local_root_download(root), use_cache=True, log=lambda *a: None)
    assert (dest / "config.json").is_symlink() and out["files"]["config.json"]["link"] == "symbolic"
    # the cache file changes behind the symbolic link: the next call re-verifies, drops the link and fetches again
    (cache / "config.json").write_bytes(b'{"model_type": "toY"}\n')
    out = fv.fetch("gated", dest, mp, download=fv.local_root_download(root), use_cache=False, log=lambda *a: None)
    r = out["files"]["config.json"]
    assert r["attempts"][0]["source"] == "existing copy" and r["source"] == f"mirror/one@{REV['m1']}"
    assert not (dest / "config.json").is_symlink() and (dest / "config.json").read_bytes() == FILES["config.json"][0]


def test_verify_only_fetches_nothing(tmp_path):
    root, mp, man = fake_world(tmp_path)
    with pytest.raises(SystemExit):
        fv.fetch("gated", tmp_path / "d", mp, download=lambda *a: pytest.fail("downloaded"), verify_only=True,
                 use_cache=False, log=lambda *a: None)
    assert json.loads((tmp_path / "d" / "VERIFY_FAILED.json").read_text())["files"]["config.json"]["reason"].startswith("not fetched")


def test_cli_exit_codes(tmp_path):
    root, mp, man = fake_world(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    cli = [PY, str(ROOT / "scripts" / "fetch_verified.py"), "--manifest", str(mp), "--local-root", str(root), "--no-cache"]
    r = subprocess.run(cli + ["--key", "gated", "--dest", str(tmp_path / "ok")], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and (tmp_path / "ok" / "VERIFIED.json").exists() and r.stdout == "", r.stderr
    (root / "mirror/two" / REV["m2"] / "tokenizer.json").write_bytes(b"{}")
    r = subprocess.run(cli + ["--key", "gated", "--dest", str(tmp_path / "bad")], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "REFUSED" in r.stderr
    r = subprocess.run(cli + ["--key", "nokey", "--dest", str(tmp_path / "x")], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "unknown key" in r.stderr
    (tmp_path / "afile").write_text("x")   # --dest is a regular file: an unexpected error (exit 3), never a refusal (1)
    r = subprocess.run(cli + ["--key", "gated", "--dest", str(tmp_path / "afile")], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "not a verification failure" in r.stderr
    man["models"]["gated"]["revision"] = "main"
    mp.write_text(json.dumps(man))
    r = subprocess.run(cli + ["--key", "gated", "--dest", str(tmp_path / "x")], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "40-hex" in r.stderr


def test_validate_manifest_catches_errors(tmp_path):
    _, _, man = fake_world(tmp_path)
    assert fv.validate_manifest(man) == []
    def broken(f):
        m = json.loads(json.dumps(man))
        f(m["models"])
        return " | ".join(fv.validate_manifest(m))
    assert "no mirror source" in broken(lambda m: m["gated"]["sources"][1]["files"].remove("tokenizer.json"))
    assert "malformed sha256" in broken(lambda m: m["gated"]["files"]["model.safetensors"].update(sha256="ab"))
    assert "exactly one" in broken(lambda m: m["gated"]["files"]["config.json"].update(sha256="0" * 64))
    assert "official repo only" in broken(lambda m: m["plain"]["sources"].append({"repo": "a/b", "revision": "5" * 40, "files": []}))
    assert "allow_pattern" in broken(lambda m: m["plain"]["files"].update({"x.bin": {"git_blob": "6" * 40, "size": 1}}))
    assert "ignore_pattern" in broken(lambda m: m["plain"].update(ignore_patterns=["model*"]))
    assert "not a manifest file" in broken(lambda m: m["gated"]["sources"][0]["files"].append("other.json"))
    assert "attn" in broken(lambda m: m["plain"].update(attn="flash"))
    assert "official repo is not a mirror" in broken(lambda m: m["gated"]["sources"][0].update(repo="org/gated"))


# --------------------------------------------------------------------------- the committed manifest
SPEC = {"qwen7": "Qwen/Qwen2.5-7B-Instruct", "qwen14": "Qwen/Qwen2.5-14B-Instruct", "qwen1.5": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen0.5": "Qwen/Qwen2.5-0.5B-Instruct",
        "mistral7": "mistralai/Mistral-7B-Instruct-v0.3", "olmo7": "allenai/OLMo-2-1124-7B-Instruct",
        "llama8": "meta-llama/Llama-3.1-8B-Instruct", "gemma9": "google/gemma-2-9b-it", "gemma2b": "google/gemma-2-2b-it",
        "phi4": "microsoft/phi-4", "falcon7": "tiiuae/Falcon3-7B-Instruct", "yi9": "01-ai/Yi-1.5-9B-Chat",
        "mistral24": "mistralai/Mistral-Small-24B-Instruct-2501"}


def test_committed_manifest():
    man = fv.load_manifest()
    assert fv.validate_manifest(man) == []
    m = man["models"]
    assert tuple(m) == fv.KEYS and {k: e["repo"] for k, e in m.items()} == SPEC and man["fallback"] == "yi9"
    for k, e in m.items():
        assert e["attn"] == ("eager" if k.startswith("gemma") else "sdpa"), k
        assert not any(f.startswith("consolidated") or f.endswith((".bin", ".pth", ".gguf")) for f in e["files"]), k
        assert "tokenizer_config.json" in e["files"] and ("model.safetensors.index.json" in e["files"] or "model.safetensors" in e["files"])
        idx = [f for f in e["files"] if f.endswith(".safetensors")]
        assert idx and all(re.match(r"^model(-\d{5}-of-\d{5})?\.safetensors$", f) for f in idx), k
        if e["gated"]:
            assert not e["unverifiable_without_token"], k   # every gated file has a byte-identical ungated copy
            cov = {f: [s["repo"] for s in e["sources"] if f in s["files"]] for f in e["files"]}
            assert all(len(v) >= 2 for v in cov.values()), (k, cov)   # at least two mirrors per file
        else:
            assert e["sources"] == [], k
    assert {k for k, e in m.items() if e["gated"]} == {"llama8", "gemma9", "gemma2b"}
    # Mistral-7B's tokenizer.model is committed as an LFS pointer (a 130-byte git blob); the Hub serves the object,
    # so the manifest pins the object's sha256 and size, with a note
    t = m["mistral7"]["files"]["tokenizer.model"]
    assert t["sha256"].startswith("37f00374") and t["size"] == 587404 and "LFS pointer" in t["note"]


def test_lfs_pointer_object_is_the_served_file():
    """The cached tokenizer.model of Mistral-7B-Instruct-v0.3 (what the Hub served) has the pinned sha256, and equals
    tokenizer.model.v3, whose git blob id the manifest pins (skipped when not cached)."""
    e = fv.load_manifest()["models"]["mistral7"]
    p, p3 = (fv.cache_lookup(e["repo"], e["revision"], f) for f in ("tokenizer.model", "tokenizer.model.v3"))
    if p is None or p3 is None:
        pytest.skip("Mistral-7B-Instruct-v0.3 tokenizer files not cached at the pinned revision")
    assert fv.verify_file(p, e["files"]["tokenizer.model"])[0] and fv.verify_file(p3, e["files"]["tokenizer.model.v3"])[0]
    assert Path(p).read_bytes() == Path(p3).read_bytes()


def test_manifest_revisions_equal_earlier_stages():
    m = {e["repo"]: e["revision"] for e in fv.load_manifest()["models"].values()}
    seen = 0
    for f in sorted(ROOT.glob("results/gpu_stage[5-7]/REVISIONS.txt")):
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in m:
                assert m[parts[0]] == parts[1], (f, line)
                seen += 1
    assert seen >= 7


# --------------------------------------------------------------------------- the shell library
LIB = ROOT / "scripts" / "stage8_common.sh"


def test_library_syntax():
    assert subprocess.run(["bash", "-n", str(LIB)]).returncode == 0


ENTRY = """# Preregistrations

## P-2026-10-08-I: an earlier entry

text

## P-2026-10-10-J: GPU stage 8, toy entry

{status}
{filler}

### Part A

lines

## Outcome of P-2026-10-08-I

done
"""


def git(repo, *a):
    return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                           *a], cwd=repo, capture_output=True, text=True, check=True).stdout


def toy_repo(tmp_path, status="**Final.**", subject="Finalise preregistration J", entry=True):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(LIB, repo / "scripts")
    shutil.copy(ROOT / "scripts" / "stage8_models.json", repo / "scripts")
    for d in ("ckeys", "tests", "docs"):
        (repo / d).mkdir()
    (repo / "ckeys" / "x.py").write_text("X = 1\n")
    filler = "\n".join(f"line {i} of a long entry" for i in range(20000))   # far more than a pipe buffer
    (repo / "docs" / "PREREGISTRATION.md").write_text(ENTRY.format(status=status, filler=filler) if entry else "# none\n")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", subject)
    return repo


def init(repo, tmp_path, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("TEST_MODE", "FORCE", "FORCE_STEPS", "OUT", "TGZ", "DEADLINE_H")}
    e.update(PY=PY, OUT=str(tmp_path / "out"), TGZ=str(tmp_path / "out.tgz"), S8_MODELS=str(tmp_path / "models"), **env)
    cmd = "PART=a; S8_DEADLINE_H_DEFAULT=1; source scripts/stage8_common.sh; s8_init; echo INIT_OK"
    return subprocess.run(["bash", "-c", cmd], cwd=repo, capture_output=True, text=True, env=e, timeout=300)


def test_guard_draft_entry(tmp_path):
    repo = toy_repo(tmp_path, status="**DRAFT, not yet final.** Under construction.")
    r = init(repo, tmp_path)
    assert r.returncode == 1 and "is still a DRAFT" in r.stdout and "INIT_OK" not in r.stdout
    assert "still a DRAFT" in (tmp_path / "out" / "FAILED.txt").read_text() and (tmp_path / "out.tgz").exists()
    assert init(repo, tmp_path, TEST_MODE="1").returncode == 0   # TEST_MODE: no guards


def test_guard_final_entry_that_quotes_the_marker_runs(tmp_path):
    """The final entry describes the guard in prose and so quotes the marker's words; only a line that starts with the
    marker (**DRAFT, not yet final.**) makes the entry a draft. The common part of entry J has such a prose line."""
    prose = '1. It refuses to start if:\n   - this section still says "DRAFT, not yet final" (the check reads the section);'
    repo = toy_repo(tmp_path, status="**Final.** Fixed in the commit.\n\n" + prose)
    r = init(repo, tmp_path, MINGIB="100000")
    assert "still a DRAFT" not in r.stdout and "final at" in r.stdout, r.stdout + r.stderr
    repo2 = toy_repo(tmp_path / "t2", status="**DRAFT, not yet final.** Under construction.\n\n" + prose)
    r = init(repo2, tmp_path / "t2")
    assert r.returncode == 1 and "is still a DRAFT" in r.stdout


def test_guard_no_entry_and_bad_switches(tmp_path):
    repo = toy_repo(tmp_path, entry=False)
    r = init(repo, tmp_path)
    assert r.returncode == 1 and "no entry P-2026-10-10-J" in r.stdout
    assert init(repo, tmp_path, TEST_MODE="yes").returncode == 2
    assert init(repo, tmp_path, DEADLINE_H="0").returncode == 2
    assert init(repo, tmp_path, FORCE_STEPS="a;b").returncode == 2


def test_guard_modified_tree_and_subject(tmp_path):
    repo = toy_repo(tmp_path, subject="Finalise preregistration J (wip)")
    r = init(repo, tmp_path)
    assert r.returncode == 1 and "no commit with the subject 'Finalise preregistration J'" in r.stdout
    git(repo, "commit", "-q", "--allow-empty", "-m", "Finalise preregistration J")
    (repo / "ckeys" / "x.py").write_text("X = 2\n")
    r = init(repo, tmp_path)
    assert r.returncode == 1 and "not a clean git checkout" in r.stdout


def test_guard_code_or_entry_changed(tmp_path):
    repo = toy_repo(tmp_path)
    (repo / "ckeys" / "x.py").write_text("X = 2\n")
    git(repo, "commit", "-qam", "change code after finalising")
    r = init(repo, tmp_path)
    assert r.returncode == 1 and "differs from the commit 'Finalise preregistration J'" in r.stdout
    git(repo, "revert", "--no-edit", "HEAD")
    p = repo / "docs" / "PREREGISTRATION.md"
    p.write_text(p.read_text().replace("### Part A\n\nlines", "### Part A\n\nlines, edited"))
    git(repo, "commit", "-qam", "edit the entry after finalising")
    r = init(repo, tmp_path)
    assert r.returncode == 1 and "section of docs/PREREGISTRATION.md differs" in r.stdout
    git(repo, "revert", "--no-edit", "HEAD")
    p.write_text(p.read_text() + "\n## Outcome of P-2026-10-10-J, part A\n\nnumbers\n")   # outside the entry: allowed
    git(repo, "commit", "-qam", "outcome of part A")
    r = init(repo, tmp_path, MINGIB="100000")
    assert "final at" in r.stdout and ("no CUDA device with >= 100000 GiB" in r.stdout), r.stdout + r.stderr


FAKE_SCORER = '''import argparse, pathlib
a = argparse.ArgumentParser(); a.add_argument("--results"); a.add_argument("--test", action="store_true"); a = a.parse_args()
assert a.test
pathlib.Path(a.results, "STAGE8B_SCORE.txt").write_text("""PROVENANCE
  commit abc
GATES (J-B-G0 before any model)
  J-B-G0 exactness: MET
PREDICTIONS
  J-B1 L prior 0.95: MET
REPORTED
  x
SUMMARY: 1 MET of 1
  class L: 1 met, sum of priors 0.95
######## EXPLORATORY (not scored)
  y
""")
'''

DRY = r'''
PART=b; S8_DEADLINE_H_DEFAULT=0.5; source scripts/stage8_common.sh; s8_init
s8_pytest tests/test_fetch_verified.py::test_git_blob_known_values
DIR=$(s8_fetch llama8); echo "DIR=$DIR"
s8_step ok_step $PY -c "import os; assert os.environ['TEST_MODE'] == '1' and int(os.environ['STAGE8_DEADLINE']) > 0; print('hello')"
s8_step fail_step $PY -c "import sys; sys.exit(5)"
S8_OUTPUTS="$OUT/x.json" s8_step fail_out $PY -c "import json, sys; json.dump({}, open(sys.argv[1], 'w')); sys.exit(1)" "$OUT/x.json"
s8_step partial_step $PY -c "import sys; sys.exit(3)"
S8_EXPLORATORY=1 s8_step explo_fail false
s8_time_left 1 && echo TIME_OK
s8_time_left 100000 || s8_skip long_step "less than 100000 minutes left"
s8_skip ok_step "would skip"
echo "ATTN=$(s8_attn gemma9) $(s8_attn llama8)"
s8_drop llama8
s8_finish "$FAKE"
echo "FINISH_RC=$?"
'''


def test_test_mode_dry_run(tmp_path):
    fake = tmp_path / "fake_score.py"
    fake.write_text(FAKE_SCORER)
    out = tmp_path / "out"
    env = {k: v for k, v in os.environ.items() if k not in ("FORCE", "FORCE_STEPS", "DEADLINE_H", "KEEP_CACHE")}
    env.update(PY=PY, TEST_MODE="1", OUT=str(out), TGZ=str(tmp_path / "t.tgz"), FAKE=str(fake), S8_MODELS=str(tmp_path / "m"))
    run = lambda **kw: subprocess.run(["bash", "-c", DRY], cwd=ROOT, capture_output=True, text=True, env={**env, **kw}, timeout=600)
    r = run()
    s = r.stdout
    assert "DIR=Qwen/Qwen2.5-0.5B-Instruct" in s and "TIME_OK" in s and "ATTN=eager sdpa" in s, s + r.stderr
    assert "FINISH_RC=1" in s   # a step FAILED
    # GATES and SUMMARY printed, nothing else of the score file
    assert "J-B-G0 exactness: MET" in s and "SUMMARY: 1 MET of 1" in s and "class L: 1 met" in s
    assert "J-B1 L prior" not in s and "commit abc" not in s and "  y" not in s.splitlines()
    assert (out / "steps" / "ok_step.done").exists() and "hello" in (out / "logs" / "ok_step.log").read_text()
    for n in ("fail_step", "fail_out", "partial_step", "explo_fail"):
        assert not (out / "steps" / f"{n}.done").exists(), n
    failed = (out / "FAILED.txt").read_text()
    assert "FAILED fail_step exit 5" in failed and "FAILED fail_out exit 1" in failed and "explo_fail" not in failed
    assert "explo_fail" in (out / "FAILED_EXPLORATORY.txt").read_text()
    assert not (out / "x.json").exists() and list(out.glob("x.json.failed.*"))
    skipped = (out / "SKIPPED.txt").read_text()
    assert "partial_step partial" in skipped and "long_step SKIPPED" in skipped and "ok_step" not in skipped
    assert "Qwen/Qwen2.5-0.5B-Instruct (hub) key=llama8 TEST_MODE" in (out / "REVISIONS.txt").read_text()
    assert "1 passed" in (out / "logs" / "pytest.log").read_text() and len((out / "PYTEST_OK.txt").read_text().split()) == 3
    # MANIFEST.sha256 covers every file and checks; the archive holds it
    man = (out / "MANIFEST.sha256").read_text()
    assert "./STAGE8B_SCORE.txt" in man and "./logs/score.log" in man
    assert subprocess.run(["sha256sum", "--quiet", "-c", "MANIFEST.sha256"], cwd=out).returncode == 0
    with tarfile.open(tmp_path / "t.tgz") as t:
        assert any(n.endswith("MANIFEST.sha256") for n in t.getnames())
    # a second session keeps the done step, reruns the rest; FORCE_STEPS redoes it; an unmatched entry is noted
    r = run()
    assert "ok_step kept" in r.stdout and "==================== fail_step" in r.stdout
    assert list(out.glob("FAILED.*.txt"))   # the earlier FAILED.txt was moved aside
    assert (out / "PYTEST_OK.txt").read_text().count("\n") == 1   # one line per HEAD, host and file list
    r = run(FORCE_STEPS="ok_*,nosuch", TESTS="0")
    assert "TESTS=0 (TEST_MODE)" in r.stdout
    assert "ok_step kept" not in r.stdout and list((out / "steps").glob("ok_step.done.redone.*"))
    assert "FORCE_STEPS entry 'nosuch' matched no step" in r.stdout


def test_s8_pytest_failure_dies(tmp_path):
    t = tmp_path / "test_fails.py"
    t.write_text("def test_no():\n    assert False\n")
    env = {k: v for k, v in os.environ.items() if k not in ("FORCE", "FORCE_STEPS", "TESTS")}
    env.update(PY=PY, TEST_MODE="1", OUT=str(tmp_path / "out"), TGZ=str(tmp_path / "t.tgz"))
    cmd = f"PART=c; S8_DEADLINE_H_DEFAULT=1; source scripts/stage8_common.sh; s8_init; s8_pytest {t}; echo AFTER"
    r = subprocess.run(["bash", "-c", cmd], cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 1 and "AFTER" not in r.stdout and "FAILED FP32 unit tests" in r.stdout
    assert not (tmp_path / "out" / "PYTEST_OK.txt").exists() and (tmp_path / "t.tgz").exists()
    assert "1 failed" in (tmp_path / "out" / "logs" / "pytest.log").read_text()


# --------------------------------------------------------------------------- retries, disk, fatal fetch errors
class Flaky(Exception):
    pass


def test_transient_errors_retried_permanent_not(tmp_path):
    root, mp, man = fake_world(tmp_path)
    good = fv.local_root_download(root)
    calls = []

    def dl(repo, rev, fname, tmpdir, token):
        calls.append((repo, fname))
        if repo == "mirror/one" and fname == "config.json" and calls.count((repo, fname)) < 3:
            raise Flaky("connection reset")                     # transient: retried on the same source
        if repo == "mirror/one" and fname == "model.safetensors":
            e = Flaky("404 Client Error")
            e.code = 404                                         # permanent: the next source at once
            raise e
        return good(repo, rev, fname, tmpdir, token)
    out = fv.fetch("gated", tmp_path / "d", mp, download=dl, use_cache=False, log=lambda *a: None, wait=0)
    assert calls.count(("mirror/one", "config.json")) == 3 and out["files"]["config.json"]["source"] == f"mirror/one@{REV['m1']}"
    assert calls.count(("mirror/one", "model.safetensors")) == 1
    assert out["files"]["model.safetensors"]["source"] == f"mirror/two@{REV['m2']}"
    # every try fails with a transient error (network): not a verification failure but status 5, and the fetch stops at
    # the first such file (no fallback may follow a network outage)
    with pytest.raises(SystemExit) as ex:
        fv.fetch("plain", tmp_path / "p", mp, download=lambda *a: (_ for _ in ()).throw(Flaky("timeout")),
                 use_cache=False, log=lambda *a: None, wait=0, tries=2)
    assert ex.value.code == 5
    f = json.loads((tmp_path / "p" / "VERIFY_FAILED.json").read_text())
    assert f["failed"] == ["config.json"] and len(f["files"]["config.json"]["attempts"]) == 2
    assert "retry could fix" in f["files"]["config.json"]["reason"] and f["files"]["config.json"]["transient"] == [f"org/plain@{REV['plain']}"]
    # every source answers 404 (a retry cannot fix it): a refusal, status 1
    def gone(*a):
        e = Flaky("404 Client Error")
        e.code = 404
        raise e
    with pytest.raises(SystemExit) as ex:
        fv.fetch("plain", tmp_path / "q", mp, download=gone, use_cache=False, log=lambda *a: None, wait=0)
    assert ex.value.code == 1
    f = json.loads((tmp_path / "q" / "VERIFY_FAILED.json").read_text())
    assert f["failed"] == list(FILES) and f["files"]["config.json"]["reason"].startswith("unavailable from every listed source")
    # one mirror gives wrong bytes, the other times out: not refused for certain (the other might serve the right
    # bytes), status 5; a file refused for certain elsewhere makes it a refusal (1) even with a transient failure later
    root2, mp2, _ = fake_world(tmp_path / "w2", corrupt_m1=("config.json",))
    good2 = fv.local_root_download(root2)
    def m2_down(repo, rev, fname, tmpdir, token):
        if repo == "mirror/two":
            raise Flaky("timeout")
        return good2(repo, rev, fname, tmpdir, token)
    with pytest.raises(SystemExit) as ex:
        fv.fetch("gated", tmp_path / "r", mp2, download=m2_down, use_cache=False, log=lambda *a: None, wait=0, tries=2)
    assert ex.value.code == 5
    (root2 / "mirror/two" / REV["m2"] / "config.json").write_bytes(b'{"model_type": "toY"}\n')   # now wrong there too
    def tok_down(repo, rev, fname, tmpdir, token):
        if fname == "tokenizer.json":
            raise Flaky("timeout")
        return good2(repo, rev, fname, tmpdir, token)
    with pytest.raises(SystemExit) as ex:
        fv.fetch("gated", tmp_path / "s", mp2, download=tok_down, use_cache=False, log=lambda *a: None, wait=0, tries=2)
    assert ex.value.code == 1


def test_not_enough_disk_is_status_4(tmp_path, monkeypatch):
    root, mp, man = fake_world(tmp_path)
    monkeypatch.setattr(fv.shutil, "disk_usage", lambda p: type("U", (), {"free": 10 ** 6})())
    with pytest.raises(SystemExit) as ex:
        run_fetch("gated", tmp_path / "d", mp, root)
    assert ex.value.code == 4 and not (tmp_path / "d" / "config.json").exists()


FAKE_FETCHER = r'''import argparse, json, os, pathlib, sys
a = argparse.ArgumentParser(); a.add_argument("--key"); a.add_argument("--dest"); a = a.parse_args()
rc = int(os.environ.get("FAKE_RC_" + a.key.replace(".", "_"), "0"))
d = pathlib.Path(a.dest); d.mkdir(parents=True, exist_ok=True)
if rc == 0:
    (d / "model.safetensors").write_bytes(b"w")
    json.dump({"key": a.key, "repo": "org/" + a.key, "revision": "a" * 40, "attn": "sdpa", "token_used": False,
               "files": {"model.safetensors": {"source": "mirror/x@" + "b" * 40}}}, open(d / "VERIFIED.json", "w"))
else:
    json.dump({"failed": ["model.safetensors"]}, open(d / "VERIFY_FAILED.json", "w"))
sys.exit(rc)
'''

FETCH_SH = r'''
source scripts/stage8_common.sh
OUT=$PWD/out; TGZ=$PWD/out.tgz; S8_MODELS=$PWD/models; TEST_MODE=0; PART=b; PART_UC=B; S8_STATE=$(mktemp -d)
mkdir -p "$OUT/logs" "$OUT/steps"; : > "$OUT/REVISIONS.txt"
mkdir -p "$S8_MODELS/pre"; echo '{}' > "$S8_MODELS/pre/VERIFIED.json"     # verified before this session
DIR=$(s8_fetch good); echo "GOOD=$DIR rc=$?"
DIR=$(s8_fetch pre); echo "PRE=$DIR rc=$?"
DIR=$(s8_fetch bad); echo "BAD=$DIR rc=$?"
s8_drop good; s8_drop pre; echo "DROPPED"
s8_step after_refusal true && echo "STEP_OK"
s8_prefetch bad2; s8_prefetch disk2; wait; [ -s "$S8_STATE/fatal" ] && echo PREFETCH_FATAL || echo PREFETCH_QUIET
DIR=$(s8_fetch disk); echo "DISK=$DIR rc=$?"
DIR=$(s8_fetch good); echo "AFTER_FATAL=$DIR rc=$?"
s8_step after_disk true; echo "NOT REACHED"
'''


def test_s8_fetch_refusal_and_fatal_errors(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(LIB, repo / "scripts")
    (repo / "scripts" / "fetch_verified.py").write_text(FAKE_FETCHER)
    env = {k: v for k, v in os.environ.items() if k not in ("KEEP_CACHE", "TEST_MODE")}
    env.update(PY=PY, FAKE_RC_bad="1", FAKE_RC_disk="4", FAKE_RC_bad2="1", FAKE_RC_disk2="5")
    r = subprocess.run(["bash", "-c", FETCH_SH], cwd=repo, capture_output=True, text=True, env=env, timeout=120)
    s, out = r.stdout, repo / "out"
    assert f"GOOD={repo}/models/good rc=0" in s and f"PRE={repo}/models/pre rc=0" in s and "BAD= rc=1" in s, s + r.stderr
    assert "org/good " + "a" * 40 + " key=good attn=sdpa token=no sources=mirror/x@bbbbbbbbbb" in (out / "REVISIONS.txt").read_text()
    assert (out / "verified" / "good.json").exists() and list((out / "verified").glob("bad.FAILED.*.json"))
    assert "FETCH REFUSED bad" in (out / "FETCH_FAILED.txt").read_text()       # a refusal: the fallback rule applies
    assert not (repo / "models" / "good").exists() and (repo / "models" / "pre").exists()   # dropped / kept
    assert "STEP_OK" in s and "DISK= rc=4" in s and "NOT REACHED" not in s and r.returncode == 1
    assert "not enough disk" in (out / "FAILED.txt").read_text() and "disk" not in (out / "FETCH_FAILED.txt").read_text()
    # a background prefetch only logs: neither its refusal nor its environment error is recorded or stops the run
    assert "PREFETCH_QUIET" in s and "bad2" not in (out / "FETCH_FAILED.txt").read_text()
    assert "prefetch of disk2: exit 5" in (out / "logs" / "fetch_disk2.log").read_text()
    # after a fatal error, a fetch in a command substitution returns 5, never 1 (which a script would take for a refusal
    # and answer with the fallback), and prints no directory
    assert "AFTER_FATAL= rc=5" in s
