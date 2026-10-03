"""Stage 4 injection and tolerant_verify (experiments/refit_remap.py); CPU, no model weights.
Needs Paper 1's reviewer repository at $P1R (default ~/paper1/v5.5-reviewer-repository) for the release tests."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True  # Paper 1's tree is read-only
from experiments import refit_remap as rr  # noqa: E402

P1R = Path(os.environ.get("P1R", Path.home() / "paper1/v5.5-reviewer-repository"))
need_p1 = pytest.mark.skipif(not (P1R / "RELEASE.json").exists(), reason="set P1R to Paper 1's reviewer repository")
FILES = {"README.md": b"readme", "gpu/train.py": b"train", "gpu/behavior_engine.py": b"engine",
         "gpu/runtime/load.py": b"load", "gpu/behavior_data/datasets.json.gz": b"data", "docs/notes.md": b"notes"}


@pytest.fixture(scope="module")
def p1():
    return rr.import_paper1(P1R)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    for repo, rev in (rr.MISTRAL, ("Qwen/Qwen2.5-0.5B-Instruct", None)):
        try:
            return AutoTokenizer.from_pretrained(repo, revision=rev, use_fast=True, fix_mistral_regex=True,
                                                 local_files_only=True)
        except Exception:
            pass
    pytest.skip("no cached tokenizer")


def release(root, changed=None, missing=None, regenerate=None):
    """A synthetic release; `regenerate` changes a file AND its manifest entry (a consistent but different release)."""
    man = {}
    for name, data in FILES.items():
        data = b"regenerated" if name == regenerate else data
        man[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if name != missing:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(b"changed" if name == changed else data)
    (root / "RELEASE.json").write_text(json.dumps({"files": man}))


PINS = {n: hashlib.sha256(d).hexdigest() for n, d in FILES.items() if rr.must_match(n)}


@pytest.mark.parametrize("changed,missing,status", [
    (None, None, "PASS"), ("README.md", None, "PASS_EXCEPT_README"), (None, "README.md", "PASS_EXCEPT_README"),
    ("gpu/train.py", None, None), ("gpu/behavior_engine.py", None, None), ("gpu/runtime/load.py", None, None),
    ("gpu/behavior_data/datasets.json.gz", None, None), ("docs/notes.md", None, None), (None, "gpu/train.py", None)])
def test_tolerant_verify(tmp_path, changed, missing, status):
    release(tmp_path / "r", changed, missing)
    out = tmp_path / "out"
    if status is None:
        with pytest.raises(ValueError, match="beyond README.md"):
            rr.tolerant_verify(tmp_path / "r", out, PINS)
        assert json.loads((out / "INTEGRITY.json").read_text())["status"] == "FAIL"
    else:
        r = rr.tolerant_verify(tmp_path / "r", out, PINS)
        assert r["status"] == status and json.loads((out / "INTEGRITY.json").read_text())["status"] == status
        assert set(r["must_match_sha256"]) == {n for n in FILES if rr.must_match(n)}


@pytest.mark.parametrize("name", sorted(PINS))
def test_tolerant_verify_pins(tmp_path, name):
    """A release whose RELEASE.json was regenerated around a changed file is refused: it is not the reviewed copy."""
    release(tmp_path / "r", regenerate=name)
    with pytest.raises(ValueError, match="reviewed copy"):
        rr.tolerant_verify(tmp_path / "r", tmp_path / "out", PINS)
    r = json.loads((tmp_path / "out/INTEGRITY.json").read_text())
    assert r["status"] == "FAIL" and r["pinned_mismatches"] == [name]
    with pytest.raises(ValueError, match="lacks gpu/extra.py"):
        rr.tolerant_verify(tmp_path / "r", None, dict(PINS, **{"gpu/extra.py": "0" * 64}))


def test_resolved_commit(tmp_path):
    snap = tmp_path / "snapshots" / rr.MISTRAL[1]
    snap.mkdir(parents=True)
    cfg = argparse.Namespace()
    assert rr.resolved_commit(str(snap), rr.MISTRAL[1], cfg) == rr.MISTRAL[1]
    assert rr.resolved_commit("x", None, argparse.Namespace(_commit_hash="abc")) == "abc"
    assert rr.resolved_commit("x", None, cfg) is None


@need_p1
def test_prompt_identity(p1):
    be = p1[0]
    rec = be.datasets()["training"][0]["base"]
    none = rr.prompt_identity(rr.make_framed(be, "NONE"), "NONE", rec)
    assert none["example_prompt"].endswith("\nAnswer with one word.\nAnswer:") and set(none["ckeys_sha256"]) == set(rr.CKEYS)
    assert rr.prompt_identity(be.prompt, "P1", rec)["example_prompt"] == be.prompt(rec)
    with pytest.raises(AssertionError):
        rr.prompt_identity(be.prompt, "NONE", rec)


@need_p1
def test_tolerant_verify_on_release(p1, tmp_path):
    r = rr.tolerant_verify(P1R, tmp_path)
    assert {m["file"] for m in r["mismatches"]} <= {"README.md"} and r["files_checked"] == r["files_in_manifest"]
    assert set(r["pinned_sha256"]) == set(rr.PINNED) and not r["pinned_mismatches"]
    assert {"gpu/train.py", "gpu/behavior_engine.py", "gpu/behavior_data/datasets.json.gz"} <= set(r["must_match_sha256"])
    if r["mismatches"]:  # the anonymous zip's README: Paper 1's own verify refuses, which is why it is replaced
        with pytest.raises(ValueError, match="README.md"):
            p1[3].verify()


@need_p1
def test_p1_prompt_equal_on_all_training_records(p1):
    be = p1[0]
    pairs = be.datasets()["training"]
    assert len(pairs) == 1000 and rr.check_p1_equality(be, pairs) == 2000


@need_p1
def test_none_prompt(p1):
    be = p1[0]
    rec = be.datasets()["training"][0]["base"]
    framed = rr.make_framed(be, "NONE")
    text, k = framed(rec), len(be.PREFIX + rec["story"] + "\nQuestion: " + rec["query"]["text"])
    assert text[:k] == be.prompt(rec)[:k] and text[k:] == "\nAnswer with one word.\nAnswer:"
    with pytest.raises(ValueError, match="Location support changed"):
        framed(dict(rec, choices=rec["choices"][:5]))


@need_p1
def test_none_shares_token_prefix_through_event_span(p1, tok):
    be = p1[0]
    pairs = be.datasets()["training"]
    original = be.prompt
    assert rr.prefix_check(be, tok, pairs, "NONE") == 1000 and be.prompt is original
    b, s = rr.encode_as(be, rr.make_framed(be, "NONE"), tok, pairs[0], "mistral", 1024)
    shown = tok.decode(b["input_ids"])
    assert "Answer with one word." in shown and "Choices:" not in shown and b["span"] == s["span"]


@need_p1
def test_injection_points_reach_unmodified_train(p1, tmp_path, monkeypatch):
    """train() looks verify and load_engine up at call time, so module-attribute injection takes effect."""
    be, tr, ld, integ = p1

    class Reached(Exception):
        pass

    def stop(*a, **k):
        raise Reached
    monkeypatch.setattr(integ, "verify", lambda root=integ.ROOT: rr.tolerant_verify(root, tmp_path))
    monkeypatch.setattr(ld, "load_engine", stop)
    with pytest.raises(Reached):
        tr.train(argparse.Namespace(model="mistral", cohort="original_1000", model_path=None, output=tmp_path / "run"))
    assert (tmp_path / "INTEGRITY.json").exists() and json.loads((tmp_path / "run/RUN.json").read_text())["interface"] == "original"
    assert json.loads((tmp_path / "run/FAILED.json").read_text())["exception"] == "Reached"
