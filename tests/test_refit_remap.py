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


def fake_run(tmp, width=32, seed=0):
    """A finished refit_remap.py run directory: FRAME.json, run/COMPLETE.json inventory and 9 orthonormal bases."""
    import numpy as np
    run, rng, inv, val = tmp / "fit" / "run", np.random.default_rng(seed), {}, {}
    (run / "bases").mkdir(parents=True)
    for o in rr.OBJECTIVES:
        for s in rr.SEEDS:
            f = run / "bases" / f"{o}_ts{s}.npz"
            np.savez(f, rank_16=np.linalg.qr(rng.standard_normal((width, 16)))[0].T.astype(np.float32))
            inv[f"bases/{f.name}"], val[f"{o}_ts{s}"] = {"sha256": rr.sha(f)}, {"sha256": rr.sha(f)}
    (run / "COMPLETE.json").write_text(json.dumps({"status": "COMPLETE", "artifacts": inv}))
    (run.parent / "FRAME.json").write_text(json.dumps({"frame": "NONE", "status": "COMPLETE", "validation": {"bases": val},
                                                       "complete_sha256": rr.sha(run / "COMPLETE.json")}))
    return run


def test_load_refit_and_scorer_check_basis_inventory(tmp_path):
    """A basis file replaced after the fit is refused by the frames step and by the scorer."""
    from analysis import stage4_score as sc
    from experiments.paper1_frames import load_refit
    run, other = fake_run(tmp_path / "a"), fake_run(tmp_path / "b", seed=1)
    _, prov, meta = load_refit(run, "none", 32)
    assert meta["bases_match_inventory"] and prov["none/m3_101"]["sha256"] == rr.sha(run / "bases/m3_ts101.npz")
    frames = {"provenance": {"prefill": "Answer:", "bases": prov, "refits": {"none": meta}}}
    assert sc.provenance({"none/": run}, {"frames": (frames, "Answer:")}, None) == []
    (run / "bases/m3_ts101.npz").write_bytes((other / "bases/m3_ts101.npz").read_bytes())
    with pytest.raises(AssertionError, match="not the basis this run wrote"):
        load_refit(run, "none", 32)
    assert any("inventory" in b for b in sc.provenance({"none/": run}, {"frames": (frames, "Answer:")}, None))
    (run.parent / "FRAME.json").unlink()
    assert sc.provenance({"none/": run}, {}, None) == [f"{run}: missing FRAME.json"]


def test_scorer_population():
    """The scorer refuses a frames file over fewer cores, other formats or unequal cells (TEST_: equal cells only)."""
    from analysis import stage4_score as sc
    full = {"provenance": {"args": {"n": 0, "arms": ",".join(sc.ARMS)}}}
    rows = {(f, a): dict.fromkeys(range(96)) for f in ("", *sc.FITS) for a in sc.ARMS}
    assert sc.population({"frames": (full, rows), "frames_noprefill": (full, rows)}, False) == []
    short = {k: dict.fromkeys(range(95)) if k == ("none/", "LETTER") else v for k, v in rows.items()}
    assert len(sc.population({"frames": (full, short)}, False)) == 1 and len(sc.population({"frames": (full, short)}, True)) == 1
    few = {k: dict.fromkeys(range(2)) for k in rows}
    cut = {"provenance": {"args": {"n": 2, "arms": ",".join(sc.ARMS)}}}
    assert len(sc.population({"frames": (cut, few)}, False)) == 2 and sc.population({"frames": (cut, few)}, True) == []


def test_git_state_excludes_output_root():
    g = rr.git_state(rr.REPO / "results/gpu_stage4/fit_none")
    assert g["status_excluded"] == ["results/gpu_stage4"] and rr.git_state("/tmp/x/fit_none")["status_excluded"] == []


def test_fixed_tokenizer_detects_skipped_fix(monkeypatch, tmp_path):
    """transformers swallows a failed Hub lookup and returns the unfixed tokenizer; the retry from the snapshot
    directory must apply the fix, also in a cache made only by commit-hash downloads (no refs/main, as on a GPU box)."""
    import shutil
    import huggingface_hub
    import transformers.utils.hub as hub
    from huggingface_hub import snapshot_download
    from experiments.paper1_frames import fixed_tokenizer
    try:
        snap = Path(snapshot_download(*rr.MISTRAL[:1], revision=rr.MISTRAL[1], local_files_only=True,
                                      allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json"]))
        _, rec = fixed_tokenizer(*rr.MISTRAL, local_files_only=True)
    except Exception:
        pytest.skip("no cached Mistral-Small-24B tokenizer")
    assert rec["fix_applied"]
    dst = tmp_path / "hub" / snap.parent.parent.name / "snapshots" / snap.name
    shutil.copytree(snap, dst)  # files only, no refs

    def down(*a, **k):
        raise RuntimeError("429")

    class Down:  # the Hub lookup fails; every other Hub call goes through
        def __getattr__(self, n):
            return down if n == "model_info" else getattr(huggingface_hub.HfApi(), n)
    monkeypatch.setattr(huggingface_hub, "model_info", down)  # transformers 5.9.0
    monkeypatch.setattr(hub, "hf_api", lambda *a, **k: Down(), raising=False)  # later versions
    tok, rec = fixed_tokenizer(*rr.MISTRAL, cache_dir=str(tmp_path / "hub"))
    assert rec["fix_applied"] and rec["local_retry"] == str(dst) and rec["pre_tokenizer_sha256"]["fixed"].startswith("3b27505c")
    assert not (dst.parent.parent / "refs").exists()
    with pytest.raises(AssertionError, match="not applied"):
        fixed_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    assert fixed_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", strict=False, local_files_only=True)[1] == {"strict": False}


@need_p1
def test_post_run_failure_recorded(p1, tmp_path, monkeypatch):
    """A failure after train() returns leaves FRAME.json at POST_RUN_FAILED with the error and complete_sha256."""
    be, tr, ld, integ = p1

    def fake_train(args):
        integ.verify()
        args.output.mkdir(parents=True)
        (args.output / "COMPLETE.json").write_text(json.dumps({"status": "COMPLETE"}))

    def bad_validate(*a, **k):
        raise ValueError("Trained basis is not orthonormal")
    for mod, name, new in ((tr, "train", fake_train), (rr, "validate_run", bad_validate), (sys, "stdout", sys.stdout)):
        monkeypatch.setattr(mod, name, new)
    for mod, name in ((be, "prompt"), (integ, "verify"), (ld, "load_engine"), (tr, "datasets"), (tr, "schedule"),
                      (tr, "require")):
        monkeypatch.setattr(mod, name, getattr(mod, name))  # restored after the injections
    monkeypatch.setattr(sys, "argv", ["refit_remap.py", "--frame", "NONE", "--p1-root", str(P1R), "--out",
                                      str(tmp_path / "fit"), "--test-model", "Qwen/Qwen2.5-0.5B-Instruct"])
    with pytest.raises(ValueError, match="orthonormal"):
        rr.main()
    f = json.loads((tmp_path / "fit/FRAME.json").read_text())
    assert f["status"] == "POST_RUN_FAILED" and "orthonormal" in f["error"]
    assert f["complete_sha256"] == rr.sha(tmp_path / "fit/run/COMPLETE.json")


@need_p1
def test_tokenizer_check_records_fix(tmp_path):
    from transformers import AutoTokenizer
    from experiments.paper1_frames import tokenizer_check
    cores = json.loads((P1R / "gpu/component_data/native_story_120.json").read_text())["stories"][:2]
    repo = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except Exception:
        pytest.skip("no cached Qwen2.5-0.5B tokenizer")
    r = tokenizer_check(repo, None, tok, rr.ARMS5, cores, ["", "Answer:"], strict=False)
    assert r["prompts"] == 60 and r["fix_mistral_regex"] == {"strict": False}
