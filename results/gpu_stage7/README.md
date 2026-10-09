# GPU stage 7 (preregistration P-2026-10-08-I)

Run of `scripts/gpu_stage7.sh` at commit 35df52f (the commit "Finalise preregistration I") on one A100 80GB PCIe,
2026-10-09, BF16, transformers 5.18.0, torch 2.11.0+cu128 (`COMMIT.txt`, `ENV.txt`, `REVISIONS.txt`, `PIP_FREEZE.txt`).
Two sessions on the same box: the first (11:24 UTC) stopped before the unit tests because the predecessor's release
could not be downloaded there (HTTP 403; `FAILED.20261009T113909Z.txt`, moved aside by the second session, and
`results/gpu_stage7_attempt1/`). The second (11:39 UTC) used a copy of the same release uploaded to the box at the
script's default path; the run's release check found it equal to the pins (`RELEASE.txt`, `log_release.txt`). In the
second session no step failed and no exploratory part was skipped at the deadline.
Score: `STAGE7_SCORE.txt`, written on the box by `analysis/stage7_score.py`; re-scoring this directory off the box with
the committed scorer reproduces it byte for byte after its first line (which names local paths). The archive
`gpu_stage7_results.tgz` has sha256 ed17673574a564c1b8269ee739fca00d0bf6d5e103d58aa6eaba567afbaff746;
`MANIFEST.sha256` lists every file in it.
