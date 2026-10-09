# Stage 7, first GPU attempt (2026-10-09): stopped before any test or model

Run of `scripts/gpu_stage7.sh` at 35df52f ("Finalise preregistration I") on one A100 80GB PCIe. The script
stopped about a minute in, after the environment setup and before the unit tests, the release check or any
model: downloading the predecessor's released reviewer repository from anonymous.4open.science returned
HTTP 403 on that host (`FAILED.txt`), so `~/paper1/v5.5-reviewer-repository/RELEASE.json` did not exist.
The same URL served the release from another network the same day; that copy passes the run's
`release_check` (release sha 2dea297d…0416 as pinned, the nine bases and the native file as in the stage-4
provenance). No stage-7 output of any model exists from this attempt.

Archive `gpu_stage7_results.tgz`, sha256 50077d5516493f98cea6a4c30e5dc55c6cdb30892c93cad76277bc05b3f52110;
all of its files are here.
