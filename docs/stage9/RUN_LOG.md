# Stage 9: run log

Each GPU session of `scripts/gpu_stage9.sh` is listed here, including sessions that stopped before any model loaded. The
prediction note (`docs/STAGE9_PREDICTIONS.md`) and the spec (`docs/stage9/SPEC.md`) have not changed since the first
freeze. Their sha256 values below are the ones every session's guard recorded in `COMMIT.txt`.

| File | sha256 |
|---|---|
| `docs/STAGE9_PREDICTIONS.md` | `3e97d03198de057cc50ec6f9147487c19125d7cfbb05e601ee79cd4151a8f056` |
| `docs/stage9/SPEC.md` | `ca9efe813d81deb8102baa8cf2751474e1f22e96457e11cd77419d08867315b0` |

## Session 1, 2026-10-21: stopped at the test gate, no model loaded

- **Freeze and box.** Freeze commit `45f5d8a` (2026-10-15T07:54:47Z). One A100-SXM4-80GB, Python 3.12.15, torch
  2.11.0+cu128, transformers 5.18.0 (installed by the script), no HF token. The guard passed at 09:17:02Z.
- **The gate.** The FP32 test gate ran 284 tests in 22 min 17 s: 280 passed, 3 skipped and 1 failed. The skipped
  tests are the study-tokenizer tests in `tests/test_stage9_passages.py`, which skip when a tokenizer is not cached
  yet. The script stopped at 09:40:03Z with `FAILED pytest exit 1`.
- **What it produced.** No model was fetched or loaded and no results file was written, so this session produced no
  stage-9 number. The archive `gpu_stage9_results.tgz` (326,496 bytes, sha256
  `b968c57fc1e16ff85b1514fe30e32eab0339d6a653c2a5190f42b0eddcf6c77d`) holds only the logs, `COMMIT.txt`, `ENV.txt`,
  `PIP_FREEZE.txt`, `REVISIONS.txt` (header only) and `FAILED.txt`.
- **The failure.** `tests/test_stage8_edits.py::test_prakash_caa_with_natural_difference_equals_id` failed with "release
  checkout unusable (release file /home/user/nix07/mind/data/story_templates.json missing ...)". That test reads Prakash
  et al.'s release (`https://github.com/Nix07/mind` at `0579347e3cf963d13d55edf041c7b595b0dcd88b`). No stage-9 step
  reads the release. `scripts/gpu_stage8c.sh` fetched it before its gates, but `scripts/gpu_stage9.sh` did not. Every
  earlier test of the stage-9 build ran where a checkout sits at the module's default path.
- **The fix.** Commit `1daa050` fetches the release before the gates, with the block from `scripts/gpu_stage8c.sh`: a
  sparse checkout at the pinned commit into `~/.cache/causal-keys/mind-<sha>` unless `PRAKASH_REPO` is set, with HEAD
  and the file hashes checked.
- **Found by the audit.** A read-only audit of the real-run path then looked at five things: the files the run reads,
  the code paths only the GPU takes, the gate tests on a fresh box, the fetches, and the fix itself. It found no other
  dependency that a fresh box lacks. It found one trap for the next session, which commit `753fb1b` closes:
  - **The trap.** Session 1 wrote no results file, but its guard left a freeze record in `COMMIT.txt`. A session at a
    new freeze reusing that `OUT` (on a kept box, or after unpacking session 1's archive) would pass the guard and leave
    two freeze records. The scorer would then mark every verdict NOT EVALUABLE after the full run.
  - **The new check.** The guard now refuses such an `OUT` before anything runs (check 4c, `s9_other_freeze`).
  - **A second fix.** The same commit makes the default release cache repair itself after an interrupted fetch.
- **A flaky gate test.** A clean-room gate run before the refreeze turned up one, which commit `54dc372` fixes.
  - **The failure.** `tests/test_fresh_factorial.py::test_paper1_frames_default_path_byte_identical` failed once. It
    requires two processes, the old and the current `experiments/paper1_frames.py`, to write byte-identical JSON. One
    clean-B log-prob differed by a few float32 ulps (-6.54779815... against -6.5478...).
  - **Not a code fault.** It passed in 19 isolated reruns, on the GPU box in session 1 and in the other gate runs.
    MKL's FP32 CPU kernels are not bitwise reproducible across processes.
  - **The fix.** The test still passes on byte equality. Otherwise it now requires the same structure, strings, ints
    and bools exactly, and every float to 1e-4, the tolerance of the sibling regression test.
  - **One more clean-room failure, not a fault either.** `tests/test_clamp.py::test_row_restricted_regression`
    starts its subprocess with only `PATH`, `HOME` and `PYTHONPATH`, so it needs the test model in the Hugging Face
    cache under `HOME`. That run used an empty fake `HOME`, so the cache was missing. On the box `HOME` is `/root`,
    where the gate's earlier tests have already downloaded the model, and the test passed there in session 1.
- **What did not change.** The note, the spec, the populations, the steps and the scorer. The next commit titled
  "Freeze stage-9 predictions" carries these fixes, and the guard takes the latest such commit. A session at the new
  freeze starts from a fresh clone, never from session 1's `OUT`.
- **The check before refreezing.** The CPU plumbing run used a fresh clone with the release fetched into a fresh path.
  The release check passed, and so did the gate test that stopped session 1. The steps and the scorer have not changed
  since the complete plumbing run made before the first freeze.
