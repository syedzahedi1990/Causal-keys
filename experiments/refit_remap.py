"""Stage 4 (P-2026-10-05-F): refit Paper 1's remap M with its released, UNMODIFIED gpu/train.py under a prompt frame.

train.train(...) runs for cohort original_1000, model "mistral" (3 seeds x {f_star, m3} plus each seed's PCA basis).
Every change is injected at import time by replacing module attributes; nothing in Paper 1's tree is edited:
  * NONE: gpu.behavior_engine.prompt -> raw_prompt("NONE", story, query), after Paper 1's six-choice support check.
    P1: raw_prompt("P1", ...) == behavior_engine.prompt(record) is asserted on all 2000 training records and
    nothing is injected.
  * reproduce.integrity.verify -> tolerant_verify: the anonymous zip ships a README.md that differs from the
    manifest; only README.md may mismatch (train.py, behavior_engine.py, gpu/runtime/*.py and behavior_data must
    match), and RELEASE.json and every file train.py runs must equal the reviewed copy (PINNED). The outcome is
    written to OUT/INTEGRITY.json.
  * --loader ours: gpu.runtime.load.load_engine -> a loader that reproduces mistral_engine.py L192-202 (BF16, sdpa,
    device_map {"": device}, frozen, eval, use_cache False, pad = eos, use_fast, fix_mistral_regex) at a pinned
    revision without Paper 1's exact Python/package pins (the waived pins and actual versions are recorded).
  * TEST_MODE (--test-model): a tiny model on CPU, the first K training pairs, and exactly one waived require
    ("The original training pool must contain ..."); the run is still labelled "mistral" (system turn, padding).
Every injection is listed in OUT/FRAME.json with sha256 of the unmodified Paper 1 files and of the injected source.
FRAME.json also records the sha256 of experiments/paper1_frames.py (fixed_tokenizer) and ckeys/*.py (raw_prompt's
text), one framed prompt, and the git commit and dirty state. It is written before the run; the COMPLETE.json sha256
and the basis validation are added after it (status COMPLETE, else FAILED or POST_RUN_FAILED with the error).
--preflight runs the cheap checks (release, P1 equality, NONE prefix, fix_mistral_regex installed and ids unchanged)
before the fits.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # Paper 1's tree is read-only

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import time
import types
from pathlib import Path

import numpy as np

from ckeys.encoding import raw_prompt

MISTRAL = ("mistralai/Mistral-Small-24B-Instruct-2501", "9527884be6e5616bdd54de542f9ae13384489724")
PINNED = {  # the reviewed copy of Paper 1's v5.5 reviewer repository: RELEASE.json and every file train.py runs
    "RELEASE.json": "2dea297d508e51f07b927e9eb0571f40e99d25d996ccd7ad3342cb46942d0416",
    "gpu/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "gpu/behavior_data/datasets.json.gz": "c3c3027bd57a5ec4a28df7975e0d9820fd5f67ff4c1739aefdb34540524002b1",
    "gpu/behavior_engine.py": "90327f1269a4ebc169df7ccd85aa52ab966728aa2433f1add19054f50032c2cb",
    "gpu/runtime/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "gpu/runtime/color_task.py": "38253949129e403a29f54d2b83061ac15a45c73491c2774f4b864c4b3644a3d0",
    "gpu/runtime/encoding.py": "6f19193267b7ff09d329f8e8d290862c66b7bd302e6237dd5282c94e79264785",
    "gpu/runtime/hybrid_refs.py": "684d626b3230e6c05ed52f66771eb8ada0df6ae66cafed34f4a0acc8c7531827",
    "gpu/runtime/key_engine.py": "d123a5556eec495ff303d1a2d32eccd0513ddd5f69e45d3c8925847e540277a5",
    "gpu/runtime/load.py": "6bf86df6d1173b7d2e701c3d030a7f2651c546ab696a556d214ffc6531da99a7",
    "gpu/runtime/mistral_engine.py": "1ff1027a9fc2ddc840cf38c679920dbc0942d8f2f5c921557243ffde71493df8",
    "gpu/runtime/mistral_relay.py": "8d06406fbc0ab5bedb8336bb44301d3f924534e55c6e834799d36b716537c2ad",
    "gpu/runtime/null_control.py": "e5d276c652e06240b6ff53150f669ac9e1b1c88ff1640dfb12d357ef3f15d5b7",
    "gpu/runtime/qwen_engine.py": "a1cf60bd8ae774314146d334dc2531a1ba3b99bccfaca18d8617d9dba480ab86",
    "gpu/runtime/qwen_geometry.py": "e5fcbe40cea881161dbab00f82221cf89c5c81ebaf9eb8a52d71726ddc1e5528",
    "gpu/runtime/qwen_relay.py": "6a50eef9eb2fe25185495f6c0d86f49b4ee26aa38032b4ff8d6bfe9a1cabc84f",
    "gpu/runtime/story_task.py": "0f537050bd183ea15f60eef0a4e482a4984ae2a9be1f338c4c734c42596f5cb4",
    "gpu/train.py": "23600ac49a932724ee743f950b4f35bac1d06c06b29fbdc22fcd25dde89d798d",
    "reproduce/__init__.py": "cc54c795d06647945e882dee499f159193848abb6cabf97a41b509611047953b",
    "reproduce/integrity.py": "c6da1d455387ef1ab615a5a86f9b25f9a0d6810b3f367a30113841d31011b3f1"}
P1_FILES = tuple(PINNED)
REPO = Path(__file__).resolve().parent.parent
CKEYS = ("ckeys/__init__.py", "ckeys/encoding.py", "ckeys/story.py")
DEPS = ("experiments/paper1_frames.py", *CKEYS)  # imported by the injected code (fixed_tokenizer, raw_prompt)
ENDS = {"NONE": "\nAnswer with one word.\nAnswer:", "P1": "\nAnswer with exactly one choice.\nAnswer:"}
ARMS5 = ("P1", "NONE", "BEFORE", "POST", "LETTER")
TOLERATED = {"README.md"}
WAIVED = "The original training pool must contain"
PINS = ("python", "torch", "transformers", "tokenizers", "accelerate", "numpy", "huggingface_hub", "safetensors",
        "regex", "Jinja2", "MarkupSafe")
OBJECTIVES, SEEDS = ("pca", "f_star", "m3"), (101, 102, 103)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SELF_SHA256 = sha(__file__)  # the injected source as imported


def src_sha(obj):
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def stamp(msg, **kw):
    print(msg, json.dumps(kw) if kw else "", flush=True)  # stdout is Stamped


class Stamped:
    """Prefix every stdout line (train.py's JSON progress lines included) with a UTC timestamp."""
    def __init__(self, f):
        self.f, self.bol = f, True

    def write(self, x):
        for part in x.splitlines(True):
            if self.bol:
                self.f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()))
            self.f.write(part)
            self.bol = part.endswith("\n")
        return len(x)

    def flush(self):
        self.f.flush()


def must_match(name):
    return (name in ("gpu/train.py", "gpu/behavior_engine.py") or name.startswith("gpu/behavior_data/")
            or (name.startswith("gpu/runtime/") and name.endswith(".py")))


def tolerant_verify(root, out=None, pins=None):
    """reproduce/integrity.verify (L15-32) recomputed entry by entry; only README.md may mismatch; every file in
    `pins` (default PINNED, the reviewed copy) must be listed in the manifest and have the pinned sha256."""
    pins = PINNED if pins is None else pins
    root = Path(root).resolve()
    manifest = json.loads((root / "RELEASE.json").read_text())["files"]
    mismatches, checked, total, code = [], 0, 0, {}
    for name, expected in manifest.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe release path: " + name)
        path = root / rel
        if not path.is_file() or path.is_symlink():
            mismatches.append({"file": name, "reason": "missing or symlink", "expected": expected})
            continue
        actual = {"bytes": path.stat().st_size, "sha256": sha(path)}
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            mismatches.append({"file": name, "reason": "differs", "expected": expected, "actual": actual})
        if must_match(name):
            code[name] = actual["sha256"]
        checked += 1
        total += expected["bytes"]
    bad = sorted({m["file"] for m in mismatches} - TOLERATED)
    lacks = [n for n in ("gpu/train.py", "gpu/behavior_engine.py", "gpu/behavior_data/datasets.json.gz", *pins)
             if n != "RELEASE.json" and n not in manifest]
    if lacks or not any(n.startswith("gpu/runtime/") for n in manifest):
        raise ValueError("Release manifest lacks " + (", ".join(lacks) or "gpu/runtime"))
    pinned = {n: {"expected": h, "actual": sha(root / n) if (root / n).is_file() else None} for n, h in pins.items()}
    unpinned = sorted(n for n, v in pinned.items() if v["expected"] != v["actual"])
    result = {"status": "PASS_EXCEPT_README" if mismatches else "PASS", "files_checked": checked,
              "bytes_checked": total, "files_in_manifest": len(manifest), "mismatches": mismatches,
              "tolerated": sorted(TOLERATED), "must_match_sha256": code, "pinned_sha256": pinned,
              "pinned_mismatches": unpinned, "release_sha256": sha(root / "RELEASE.json"), "root": str(root)}
    if bad or unpinned:
        result["status"] = "FAIL"
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "INTEGRITY.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if bad:
        raise ValueError("Release integrity mismatch beyond README.md: " + ", ".join(bad))
    if unpinned:
        raise ValueError("Paper 1 release differs from the reviewed copy (PINNED): " + ", ".join(unpinned))
    return result


def import_paper1(p1_root):
    root = str(Path(p1_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import gpu.behavior_engine as be
    import gpu.runtime.load as ld
    import gpu.train as tr
    import reproduce.integrity as integ
    for mod, rel in ((be, "gpu/behavior_engine.py"), (tr, "gpu/train.py"), (ld, "gpu/runtime/load.py"),
                     (integ, "reproduce/integrity.py")):
        assert Path(mod.__file__).resolve() == Path(root) / rel, (mod.__file__, rel)
    return be, tr, ld, integ


def make_framed(be, frame):
    def framed(record):
        be.require(len(record["choices"]) == 6 and set(record["choices"]) == set(be.CHOICES), "Location support changed")
        return raw_prompt(frame, record["story"], record["query"]["text"])
    return framed


def check_p1_equality(be, pairs):
    from ckeys.story import LOCATIONS, PREFIX
    assert PREFIX == be.PREFIX and tuple(LOCATIONS) == tuple(be.CHOICES)
    p1 = make_framed(be, "P1")
    recs = [p[k] for p in pairs for k in ("base", "source")]
    bad = [i for i, r in enumerate(recs) if p1(r) != be.prompt(r)]
    assert not bad, f"raw_prompt('P1') differs from Paper 1's prompt on {len(bad)} records, first {bad[:5]}"
    return len(recs)


def encode_as(be, prompt_fn, tok, pair, model, padding):
    saved = be.prompt
    be.prompt = prompt_fn
    try:
        return be.encode_pair(tok, pair, "original", model, padding)
    finally:
        be.prompt = saved


def prefix_check(be, tok, pairs, frame, model="mistral", padding=1024):
    """Under `frame`, Paper 1's encoder must give the same event span and the same token ids through its end as P1."""
    framed, n = make_framed(be, frame), 0
    for pair in pairs:
        p1 = encode_as(be, be.prompt, tok, pair, model, padding)
        fr = encode_as(be, framed, tok, pair, model, padding)
        for a, b in zip(p1, fr):
            z = a["span"][1]
            assert a["span"] == b["span"] and a["input_ids"][:z] == b["input_ids"][:z], (frame, pair["base"]["story"][:80])
            assert a["choice_ids"] == b["choice_ids"]
        n += 1
    return n


def git_state(out=None):
    """HEAD, untracked/modified paths (excluding the stage-4 output root, the parent of --out, when inside the
    repository) and the sha256 of `git diff HEAD`."""
    root = Path(out).resolve().parent if out else None
    ex = [f":(exclude){root.relative_to(REPO)}"] if root and root != REPO and root.is_relative_to(REPO) else []
    try:
        g = lambda *c: subprocess.run(["git", "-C", str(REPO), *c], capture_output=True, text=True, check=True).stdout
        return {"head": g("rev-parse", "HEAD").strip(), "status_excluded": [e[10:] for e in ex],
                "status_porcelain": g("status", "--porcelain", "--", ".", *ex).splitlines(),
                "diff_head_sha256": hashlib.sha256(g("diff", "HEAD").encode()).hexdigest()}
    except (OSError, subprocess.CalledProcessError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def prompt_identity(framed, frame, record):
    """What fixes the injected text: raw_prompt's source files, and one framed prompt in full."""
    text = framed(record)
    assert text.endswith(ENDS[frame]), (frame, text[-60:])
    return {"ckeys_sha256": {f: sha(REPO / f) for f in CKEYS}, "raw_prompt_source_sha256": src_sha(raw_prompt),
            "example_record": "training[0].base", "example_prompt": text,
            "example_prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}


def resolved_commit(repo, revision, config):
    """The commit the weights were loaded from: config._commit_hash, else the HF snapshot directory's name."""
    c = getattr(config, "_commit_hash", None)
    if c is None and Path(repo).is_dir():
        c = Path(repo).resolve().name
    if c is None and revision is not None:
        from huggingface_hub import snapshot_download
        c = Path(snapshot_download(repo, revision=revision, local_files_only=True)).name
    return c


def versions():
    out = {"python": platform.python_version()}
    for p in PINS[1:]:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def ours_loader(repo, revision, device, test):
    """Reproduces Paper 1 mistral_engine.Engine L192-202 without its Python/package/device-count pins."""
    def load_engine(model, *, model_path, model_receipt=None, native_call_limit, journal_path, deadline_seconds):
        import torch
        from transformers import AutoModelForCausalLM
        from experiments.paper1_frames import fixed_tokenizer
        kw = dict(revision=revision) if revision else {}
        tok, tok_fix = fixed_tokenizer(repo, revision, strict=not test)
        mdl = AutoModelForCausalLM.from_pretrained(repo, **kw, trust_remote_code=False, torch_dtype=torch.bfloat16,
                                                   device_map={"": device}, low_cpu_mem_usage=True,
                                                   attn_implementation="sdpa")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        mdl.requires_grad_(False)
        mdl.eval()
        mdl.config.use_cache = False
        dev = torch.device(device)
        commit = resolved_commit(repo, revision, mdl.config)
        assert revision is None or commit == revision, (commit, revision)
        assert test or (mdl.config.hidden_size == 5120 and len(mdl.model.layers) == 40), "Mistral-Small-24B expected"
        assert not getattr(mdl, "is_quantized", False) and all(
            p.dtype == torch.bfloat16 and p.device == dev and not p.requires_grad and p.grad is None
            for p in mdl.parameters()), "Model must be frozen single-device unquantized BF16"
        assert mdl.config._attn_implementation == "sdpa"
        env = dict(model=repo, revision=revision, commit_hash=commit, loader="refit_remap.ours",
                   device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                   gpu_bytes=torch.cuda.get_device_properties(dev).total_memory if dev.type == "cuda" else None,
                   packages=versions(), cuda_runtime=torch.version.cuda, padding=1024, attention="sdpa",
                   tokenizer=dict(use_fast=True, fix_mistral_regex=tok_fix, pad_token_id=tok.pad_token_id),
                   native_call_limit=native_call_limit, deadline_seconds=deadline_seconds, test_mode=test)
        if journal_path is not None:
            with Path(journal_path).open("x") as f:
                f.write(json.dumps(dict(event="engine_ready", environment=env), sort_keys=True) + "\n")
        return types.SimpleNamespace(torch=torch, np=np, tok=tok, model=mdl, layers=mdl.model.layers, device=dev,
                                     environment=env, deadline=time.monotonic() + deadline_seconds,
                                     native_call_limit=native_call_limit)
    return load_engine


def validate_run(run, p1_root=None, model="mistral"):
    """9 bases {pca, f_star, m3} x seeds, float32 (16, hidden) with orthonormal rows; Paper 1's
    load_training_bases as well when the width is Mistral's 5120."""
    run = Path(run)
    complete = json.loads((run / "COMPLETE.json").read_text())
    assert complete["status"] == "COMPLETE" and not (run / "FAILED.json").exists()
    out, widths = {}, set()
    for o in OBJECTIVES:
        for s in SEEDS:
            f = run / "bases" / f"{o}_ts{s}.npz"
            with np.load(f, allow_pickle=False) as z:
                assert z.files == ["rank_16"], z.files
                U = z["rank_16"]
            assert U.dtype == np.float32 and U.ndim == 2 and U.shape[0] == 16 and np.isfinite(U).all(), U.shape
            err = float(np.abs(U.astype(float) @ U.astype(float).T - np.eye(16)).max())
            assert err <= 1e-5, (f, err)
            widths.add(U.shape[1])
            out[f"{o}_ts{s}"] = {"sha256": sha(f), "shape": list(U.shape), "orthonormality_error": err}
    assert len(widths) == 1
    res = {"bases": out, "hidden": widths.pop(), "complete_sha256": sha(run / "COMPLETE.json")}
    if res["hidden"] == 5120 and p1_root is not None:
        _, _, ld, _ = import_paper1(p1_root)
        got = ld.load_training_bases(run, model, "original_1000")
        res["paper1_load_training_bases"] = {"status": "PASS", "n": len(got)}
    else:
        res["paper1_load_training_bases"] = {"status": "SKIPPED", "reason": "width is not 5120 (test model)"}
    return res


def preflight(a, repo, revision, test):
    """Before any fit: release integrity + pins, P1 equality, NONE event prefix, and Paper 1's tokenizer identity
    with and without fix_mistral_regex on every prompt the frames step evaluates (paper1_frames.tokenizer_check)."""
    from transformers import AutoTokenizer
    from experiments.paper1_frames import fixed_tokenizer, tokenizer_check
    out = Path(a.out)
    be, _, _, integ = import_paper1(a.p1_root)
    res = {"integrity": tolerant_verify(integ.ROOT, out)["status"]}
    pairs = be.datasets()["training"]
    res["p1_equality_records"] = check_p1_equality(be, pairs)
    res["none_prompt"] = prompt_identity(make_framed(be, "NONE"), "NONE", pairs[0]["base"])
    tok_fix, _ = fixed_tokenizer(repo, revision, strict=not test)
    res["none_prefix_pairs"] = prefix_check(be, tok_fix, pairs, "NONE")
    cores = json.loads((Path(a.p1_root) / "gpu/component_data/native_story_120.json").read_text())["stories"]
    cores = cores[: a.cores] if a.cores else cores
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)  # as paper1_frames.main loads it
    res["tokenizer_check"] = tokenizer_check(repo, revision, tok, ARMS5, cores, ["", "Answer:"], strict=not test)
    res.update(repo=repo, revision=revision, cores=len(cores), versions=versions(), git=git_state(a.out))
    (out / "PREFLIGHT.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    stamp("preflight PASS", integrity=res["integrity"], tokenizer_prompts=res["tokenizer_check"]["prompts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", choices=("NONE", "P1"))
    ap.add_argument("--p1-root", required=True, help="unpacked Paper 1 reviewer repository (read-only)")
    ap.add_argument("--out")
    ap.add_argument("--loader", choices=("ours", "paper1"), default="ours")
    ap.add_argument("--model-path", default=None, help="HF repo id or local snapshot (default: pinned Mistral-Small-24B)")
    ap.add_argument("--revision", default=None, help=f"default {MISTRAL[1]} for the Mistral repo")
    ap.add_argument("--test-model", default=None, help="TEST_MODE: a tiny HF model id, run on CPU")
    ap.add_argument("--test-pairs", type=int, default=8, help="TEST_MODE: first K training pairs (>= 3)")
    ap.add_argument("--device", default=None, help="cuda:N or cpu (default cuda:0, cpu in TEST_MODE)")
    ap.add_argument("--validate", action="append", default=[], help="only validate these finished run directories")
    ap.add_argument("--preflight", action="store_true", help="only the pre-fit checks, written to OUT/PREFLIGHT.json")
    ap.add_argument("--cores", type=int, default=0, help="preflight: first N native cores (0 = all 120)")
    a = ap.parse_args()
    sys.stdout = Stamped(sys.stdout)
    if a.validate:
        for run in a.validate:
            stamp("validate", run=run, result=validate_run(run, a.p1_root))
        return
    test = a.test_model is not None
    repo = a.test_model or a.model_path or MISTRAL[0]
    revision = a.revision or (None if test else MISTRAL[1])  # outside TEST_MODE, also for a local --model-path
    assert test or revision == MISTRAL[1], f"stage 4 is preregistered at revision {MISTRAL[1]}"
    if a.preflight:
        assert a.out, "--out is required"
        Path(a.out).mkdir(parents=True, exist_ok=True)
        return preflight(a, repo, revision, test)
    assert a.frame and a.out, "--frame and --out are required"
    assert sys.version_info >= (3, 11), "Paper 1's train.py needs Python >= 3.11 (hashlib.file_digest)"
    assert not test or (a.loader == "ours" and a.test_pairs >= 3), "TEST_MODE needs --loader ours and K >= 3"
    device = a.device or ("cpu" if test else "cuda:0")
    out = Path(a.out)
    run = out / "run"
    assert not run.exists(), f"{run} exists; fits are never overwritten"
    be, tr, ld, integ = import_paper1(a.p1_root)
    root = Path(integ.ROOT).resolve()
    stamp("start", frame=a.frame, loader=a.loader, repo=repo, revision=revision, device=device, test=test)
    unmodified = {f: sha(root / f) for f in P1_FILES}
    integrity = tolerant_verify(root, out)
    stamp("integrity", status=integrity["status"], mismatches=[m["file"] for m in integrity["mismatches"]])
    all_pairs = be.datasets()["training"]
    n_eq = check_p1_equality(be, all_pairs)
    stamp("p1_equality", records=n_eq)
    from experiments.paper1_frames import fixed_tokenizer
    local = a.loader == "paper1" and a.model_path
    tok, tok_fix = fixed_tokenizer(repo, None if local else revision, strict=not test,
                                   **(dict(local_files_only=True) if local else {}))
    n_pref = prefix_check(be, tok, all_pairs, a.frame) if a.frame == "NONE" else None
    stamp("prefix_check", frame=a.frame, pairs=n_pref)
    del tok

    injections = []
    def inject(mod, name, new, note, source=None):
        setattr(mod, name, new)
        injections.append({"target": f"{mod.__name__}.{name}", "source_sha256": src_sha(source or new), "note": note})

    if a.frame == "NONE":
        inject(be, "prompt", make_framed(be, "NONE"),
               "raw_prompt('NONE', story, query) after Paper 1's six-choice support check", make_framed)
    inject(integ, "verify", lambda root=integ.ROOT: tolerant_verify(root, out),
           "only README.md may mismatch RELEASE.json; see INTEGRITY.json", tolerant_verify)
    if a.loader == "ours":
        inject(ld, "load_engine", ours_loader(repo, revision, device, test),
               "mistral_engine.py L192-202 reproduced; Python/package/device-count pins waived", ours_loader)
    prompt_id = prompt_identity(be.prompt, a.frame, all_pairs[0]["base"])  # the prompt train.py will call
    if test:
        real_ds, real_sched, real_req = be.datasets, be.schedule, tr.require
        inject(tr, "datasets", lambda: {"training": real_ds()["training"][: a.test_pairs]},
               f"TEST_MODE: first {a.test_pairs} training pairs")
        inject(tr, "schedule", lambda seed, cohort: [i for i in real_sched(seed, cohort) if i < a.test_pairs],
               "TEST_MODE: Paper 1 schedule restricted to the first K pairs")
        inject(tr, "require", lambda c, m: None if m.startswith(WAIVED) else real_req(c, m),
               f"TEST_MODE: waives only '{WAIVED} ...'")
    frame = {"preregistration": "P-2026-10-05-F", "frame": a.frame, "interface": "original",
             "interface_note": "RUN.json's 'interface': 'original' is literal (no assistant prefill, readout at the "
                               "reply start); the prompt text is set by 'frame'",
             "model_label": "mistral", "cohort": "original_1000", "repo": repo, "revision": revision,
             "loader": a.loader, "device": device, "test_mode": test,
             "test_pairs": a.test_pairs if test else None, "p1_root": str(root),
             "paper1_unmodified_sha256": unmodified, "injected_file": "experiments/refit_remap.py",
             "injected_file_sha256": SELF_SHA256, "injections": injections, "prompt_identity": prompt_id,
             "injected_dependencies_sha256": {f: sha(REPO / f) for f in DEPS},
             "git": git_state(out),
             "p1_prompt_equality_records": n_eq, "event_prefix_identical_to_p1_pairs": n_pref,
             "integrity_status": integrity["status"], "tokenizer_fix_mistral_regex": tok_fix,
             "pins_waived": ["python==3.12.14 and exact package versions", "exactly one CUDA device",
                             ">= 75 GiB", "Paper 1 asset receipt"] if a.loader == "ours" else [],
             "versions": versions(), "argv": sys.argv, "status": "RUNNING"}
    write = lambda: (out / "FRAME.json").write_text(json.dumps(frame, indent=2, sort_keys=True) + "\n")
    write()
    stamp("frame_written", injections=[i["target"] for i in injections])
    t0 = time.time()
    try:
        tr.train(argparse.Namespace(model="mistral", cohort="original_1000",
                                    model_path=Path(a.model_path) if a.loader == "paper1" and a.model_path else None,
                                    output=run))
    except BaseException as e:
        frame.update(status="FAILED", error=f"{type(e).__name__}: {e}", seconds=time.time() - t0)
        write()
        raise
    c = run / "COMPLETE.json"
    frame.update(seconds=time.time() - t0, complete_sha256=sha(c) if c.exists() else None)
    try:  # train() returned: a failure here is recorded as POST_RUN_FAILED (the fit is not evaluated)
        assert all(sha(root / f) == h for f, h in unmodified.items()), "Paper 1 files changed during the run"
        frame.update(validation=validate_run(run, root), status="COMPLETE")
    except BaseException as e:
        frame.update(status="POST_RUN_FAILED", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        write()
    stamp("done", seconds=round(time.time() - t0), complete_sha256=frame["complete_sha256"],
          load_training_bases=frame["validation"]["paper1_load_training_bases"]["status"])


if __name__ == "__main__":
    main()
