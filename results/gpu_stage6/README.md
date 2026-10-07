# GPU stage 6 (preregistration P-2026-10-05-H)

Run of `scripts/gpu_stage6.sh` (PART=all) at commit cc3a3e0 (the commit "Finalise preregistration H") on one
A100-SXM4-80GB, 2026-10-07, BF16, transformers 5.18.0, torch 2.11.0+cu128 (`COMMIT.txt`, `ENV.txt`, `REVISIONS.txt`,
`PIP_FREEZE.txt`); Prakash et al.'s release fetched at 0579347 with every file hash and the pool hash checked
(`RELEASE.txt`). No step failed. The optional Llama-3-70B run (H12) was not made.
Score: `STAGE6_SCORE.txt`, written on the box by `analysis/stage6_score.py`; re-scoring this directory off the box with
the committed scorer reproduces it byte for byte. The archive `gpu_stage6_results.tgz` has sha256
d1abb1274d62b8f434f86e235317b9ffcb92b27760f1226f09e455986d372abb; `MANIFEST.sha256` lists every file in it.
