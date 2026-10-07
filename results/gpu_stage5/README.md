# GPU stage 5 (preregistration P-2026-10-05-G)

Run of `scripts/gpu_stage5.sh` at commit 51e105e (the commit "Finalise preregistration G") on one A100-SXM4-80GB,
2026-10-06, BF16, transformers 5.18.0, torch 2.11.0+cu128 (`COMMIT.txt`, `ENV.txt`, `REVISIONS.txt`, `PIP_FREEZE.txt`).
Score: `STAGE5_SCORE.txt`, written on the box by `analysis/stage5_score.py`; re-scoring the archive off the box with the
committed scorer reproduces it byte for byte.

- The full archive `gpu_stage5_results.tgz` has sha256 7faf9b99c69bdc1497670ac101fa382d08b5fac2b840fe6f323f9b0fe2fce617;
  `MANIFEST.sha256` lists the sha256 of every file in it.
- The four `attention/*.npz` sidecars of part (a) (232 MB, one file over GitHub's 100 MB limit) are not in git; they are
  in the archive, with their hashes in `MANIFEST.sha256`. Part (a) of the scorer reads them, so re-scoring part (a)
  needs the archive.
- One step failed (`FAILED.txt`): the exploratory GPT-2 small attention probe `experiments/ioi_attention.py`, from a
  CPU/GPU device mismatch in that script. No prediction depends on it. It was re-run afterwards on CPU with the same
  command, cores, GPT-2 revision and the code of 51e105e; its output is in `results/gpu_stage5_cpu_probe/`, outside this
  directory, so that this directory still re-scores to the official `STAGE5_SCORE.txt`.
- The first attempt (eb0b5fd), which stopped in the unit tests before any model, is in `results/gpu_stage5_attempt1/`.
