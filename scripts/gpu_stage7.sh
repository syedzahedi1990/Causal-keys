#!/usr/bin/env bash
# GPU stage 7 (preregistered as P-2026-10-08-I in docs/PREREGISTRATION.md; paper v3): find the reader heads at
# Mistral-Small-24B-Instruct-2501 and block them at the option words while applying the released remap of our
# predecessor, Anonymous (2026). One model, five steps (experiments/stage7_link.py unless stated), each kept or redone:
#   preflight  tokenizer only: the predecessor's release (manifest, RELEASE.json pin, the nine bases, the stories file),
#              the fix_mistral_regex tokenizer check on every prompt, every encoding, R and E disjoint
#   rank       step 1, phase 1 (eager): a3 on the 60 ranking stories, the head sets, the means MU, duplicate scores
#   family     experiments/paper1_frames.py --model mistral, unchanged, five formats (the stage-3b reproduction, I-G1 (a))
#   gate       step 1, phase 2 (sdpa): HeadSplice sufficiency / knockout curves on the 96 cores, the NO-MENTION clamp
#   link       step 2 (sdpa): B_x, the curve and LIST-BEFORE batches, the family batches under the unblocked, A, A+ and N
#              conditions, five formats
#   remaprank  exploratory (eager), last: the remap's own a3 ranking and KO_x of its top set; a failure is recorded in
#              FAILED_EXPLORATORY.txt, not FAILED.txt
# BF16, use_cache=False, revision 9527884be6e5616bdd54de542f9ae13384489724 (that of stages 2-4), transformers 5.18.0.
# Needs one 80 GB GPU (A100/H100), >= 64 GB host RAM and >= 100 GB disk (47 GB of weights, downloaded once; removed from
# the HF cache afterwards unless KEEP_CACHE=1 or the model was cached before the run). Runtime about 2-2.5 h (cap 4 h:
# STAGE7_DEADLINE, DEADLINE_H hours after the start (default 3.5), is checked by link at the start of each format and by
# remaprank at its start; once it has passed, link skips that format's exploratory parts (the curve batch and the
# exploratory conditions) for every core and remaprank writes a stub; each skip is recorded in the provenance and printed
# by the scorer. A later session fills them: a remaprank stub counts as incomplete while the session's own deadline has
# not passed, and FORCE_STEPS=link redoes link alone).
# Usage:
#   git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
#   I=$(git log --format=%H -1 --grep='^Finalise preregistration I') && [ -n "$I" ] && git checkout "$I"
#   bash scripts/gpu_stage7.sh
#   TEST_MODE=1 bash scripts/gpu_stage7.sh      # CPU plumbing test: pytest, every step at Qwen2.5-0.5B (FP32, random
#                                               # bases) into a scratch OUT, then the scorer; about 20 min of pytest plus
#                                               # about 15 min of runs on a 4-core CPU (peak host RAM about 10 GB)
# Optional: OUT=<dir>, KEEP_CACHE=1, FORCE=1, FORCE_STEPS=<steps> (comma-separated, from preflight rank family gate link
# remaprank: redo these steps only), TESTS=0, PY=<python>, MINGIB=<GiB> (GPU memory floor, default 75), P1R=<the
# predecessor's unpacked reviewer repository> (must exist when given; unset: fetched into $P1_ROOT, default $HOME/paper1),
# DEADLINE_H=<hours, a positive number>.
# The switches TEST_MODE, KEEP_CACHE, FORCE, TESTS take 0 or 1 only (unset = 0 except TESTS = 1); outside TEST_MODE,
# TESTS=0 needs OUT/PYTEST_OK.txt from a passing pytest of the same HEAD on the same host. keep/FORCE as in
# scripts/gpu_stage6.sh: a step whose results file exists, is readable JSON and complete is kept unless FORCE=1 or
# FORCE_STEPS names it; a file written by a FAILED step, or unreadable or incomplete, is moved aside to
# <file>.failed.<UTC> and the step is run again, so a rerun with the same OUT resumes where it failed (at step
# granularity: a link step that failed in a late format is rerun for all five formats).
# Pytest: a fixed set, before any model, no reruns: the stage-7 tests (Gate I-G0: tests/test_stage7_link.py; the HeadSplice,
# knockout and release tests stage 7 builds on; tests/test_stage7_score.py) into log_pytest_stage7.txt, whose last run the
# scorer reads for I-G0; then the other shared modules it calls (interventions, encoding, row_restricted_keys, clamp) into
# log_pytest.txt. Any failure stops the script.
# Outside TEST_MODE it pins transformers 5.18.0, refuses a DRAFT P-2026-10-08-I, modified tracked files, or code
# (ckeys experiments analysis scripts tests and the entry) that differs from the commit 'Finalise preregistration I'.
# A failed step is recorded in $OUT/FAILED.txt and the pipeline goes on; exit status 1 if any step FAILED (the score
# step FAILS with exit 2 when the provenance or population check reports MISMATCH outside TEST_MODE).
# Output: gpu_stage7_results.tgz (TEST_MODE: $OUT.tgz), no weights; the score is $OUT/STAGE7_SCORE.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
for v in TEST_MODE KEEP_CACHE FORCE TESTS; do case ${!v:-} in ""|0|1) ;; *) echo "bad $v='${!v}' (0 or 1)"; exit 2;; esac; done
[[ "${DEADLINE_H:-3.5}" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v h="${DEADLINE_H:-3.5}" 'BEGIN{exit !(h > 0)}' \
  || { echo "bad DEADLINE_H='${DEADLINE_H:-}' (a positive number of hours)"; exit 2; }
STEPNAMES=" preflight rank family gate link remaprank "
FORCE_STEPS=${FORCE_STEPS:-}; for s in ${FORCE_STEPS//,/ }; do [[ "$STEPNAMES" == *" $s "* ]] || { echo "bad FORCE_STEPS entry '$s' (one of:$STEPNAMES)"; exit 2; }; done
forced() { on FORCE || [[ ",${FORCE_STEPS:-}," == *",$1,"* ]]; }   # FORCE=1 redoes every step, FORCE_STEPS the named ones
on() { [ "${!1:-0}" = 1 ]; }   # a switch is on only when it is 1
on TEST_MODE || unset TEST_MODE   # the experiments read TEST_MODE as set / unset
export GIT_TERMINAL_PROMPT=0 HF_HUB_DISABLE_PROGRESS_BARS=1
MODEL=mistralai/Mistral-Small-24B-Instruct-2501
REV=9527884be6e5616bdd54de542f9ae13384489724
if on TEST_MODE; then
  OUT=${OUT:-$(mktemp -d)/gpu_stage7}; TGZ=${TGZ:-$OUT.tgz}
  TINY=Qwen/Qwen2.5-0.5B-Instruct; TAG=TEST_${TINY##*/}
  export TEST_MODE=1 KEEP_CACHE=1   # the experiments then force Qwen2.5-0.5B, FP32, CPU, random bases, reduced sizes
  FAM=(--model-override "$TINY" --bases-override 896 --n 3); REVARG=(); NFAM=0
else
  OUT=${OUT:-results/gpu_stage7}; TGZ=${TGZ:-gpu_stage7_results.tgz}; TAG=mistral
  FAM=(); REVARG=(--revision "$REV"); NFAM=600
fi
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STAGE7_DEADLINE=$(( $(date +%s) + $(awk -v h="${DEADLINE_H:-3.5}" 'BEGIN{printf "%d", h * 3600}') ))
run() {
  local name=$1 rc=0 ff="$OUT/${FAILFILE:-FAILED.txt}"; shift
  echo "==================== $name  $(date -u +%H:%M:%S)"
  echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$OUT/log_$name.txt"
  PYTHONPATH=. "$@" >> "$OUT/log_$name.txt" 2>&1 || {
    rc=$?; local why="exit $rc"; [ $rc -gt 128 ] && why+=" (killed by signal $((rc - 128)); 9 = SIGKILL, often the host OOM killer)"
    echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED $why" >> "$OUT/log_$name.txt"
    echo "FAILED $name $why (see $OUT/log_$name.txt)" | tee -a "$ff"; }
  tail -n 2 "$OUT/log_$name.txt"
  return $rc
}
aside() { local old="$1.failed.$(date -u +%Y%m%dT%H%M%SZ)"; mv "$1" "$old"; echo "moved aside $old: $2" | tee -a "$OUT/COMMIT.txt"; }
complete() {  # the results file is readable JSON and holds everything its step writes
  $PY - "$1" "$NFAM" > /dev/null 2>&1 <<'EOF'
import json, sys
f, nfam = sys.argv[1], int(sys.argv[2]); j = json.load(open(f))
if f.endswith("/heads/rank.json"):
    assert {"P1", "LETTER", "POST"} <= set(j["arms"]) and j["sets_sha256"]
    import hashlib, pathlib
    assert hashlib.sha256((pathlib.Path(f).parent / j["mu_file"]).read_bytes()).hexdigest() == j["mu_sha256"]
elif f.endswith("/heads/gate.json"):
    n = len(j["none_clamp"]["eval"]); assert n and all(len(j["arms"][a]["eval"]) == n for a in ("P1", "LETTER", "POST"))
elif "/link/" in f:
    assert {"P1", "LETTER", "POST", "NONE", "BEFORE"} <= set(j["arms"])
elif f.endswith("/heads/remaprank.json"):   # a stub written after a deadline is incomplete while this session's deadline has not passed
    import os, time
    if j["provenance"].get("skipped"):
        assert time.time() > float(os.environ["STAGE7_DEADLINE"])
    else:
        assert {"P1", "LETTER", "POST"} <= set(j["arms"])
elif "/frames/" in f:
    assert j["results"] and (not nfam or len(j["results"]) == nfam)
EOF
}
STEPS=0
keep() {  # keep <results file> <name> <command...>: as in scripts/gpu_stage6.sh
  local f=$1 name=$2 rc=0 stamp; shift
  STEPS=$((STEPS + 1))
  if [ -e "$f" ]; then
    if ! complete "$f"; then aside "$f" "not readable JSON or incomplete (a step interrupted while writing); $name is run again"
    elif ! forced "$name"; then
      echo "==================== $name kept: $f exists (FORCE=1 or FORCE_STEPS=$name to redo)"
      [ "$name" = link ] && $PY -c "import json, sys; n = len(json.load(open(sys.argv[1]))['provenance'].get('explo_skipped') or []); n and print(f'  link kept with {n} exploratory parts skipped at an earlier deadline (FORCE_STEPS=link redoes link)')" "$f"
      return 0; fi
  fi
  stamp="$OUT/.stamp_$name"; touch "$stamp"; sleep 1   # a file older than the stamp was not written by this step
  run "$@" || { rc=$?; if [ -e "$f" ] && [ "$f" -nt "$stamp" ]; then aside "$f" "written by the FAILED step $name (exit $rc); the scorer does not read it"
                elif [ -e "$f" ]; then echo "kept $f: the FAILED step $name (exit $rc, FORCE=1) wrote no new file, the earlier one stays and is scored" | tee -a "$OUT/COMMIT.txt"; fi; }
  rm -f "$stamp"
  return $rc
}
die() {  # fatal before the models: record, archive the logs, stop
  echo "FAILED $1" | tee -a "$OUT/FAILED.txt"
  tar czf "$TGZ" "$OUT"; echo "Results archive (logs): $TGZ"; exit 1
}
for f in FAILED FAILED_EXPLORATORY; do [ -f "$OUT/$f.txt" ] && mv "$OUT/$f.txt" "$OUT/$f.$(date -u +%Y%m%dT%H%M%SZ).txt"; done
{ echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) TEST_MODE=${TEST_MODE:-0} FORCE=${FORCE:-0} FORCE_STEPS=${FORCE_STEPS:-} DEADLINE_H=${DEADLINE_H:-3.5}"; git rev-parse HEAD; git status --short; } | tee -a "$OUT/COMMIT.txt"
if ! on TEST_MODE; then  # the preregistered code only: the finalised entry, no local changes, the code of the finalising commit
  awk '/^## /{f = ($0 ~ /P-2026-10-08-I/)} f && /DRAFT, not yet final/{d = 1} END{exit !d}' docs/PREREGISTRATION.md \
    && die "preregistration P-2026-10-08-I is still a DRAFT: check out the commit 'Finalise preregistration I' (docs/GPU_RUNBOOK.md)"
  # (awk decides alone: an awk | grep -q pipeline under pipefail gets status 141 when grep exits first, and the refusal is skipped)
  git rev-parse HEAD > /dev/null 2>&1 && [ -z "$(git status --porcelain --untracked-files=no)" ] \
    || die "not a clean git checkout (modified tracked files above): run from a clean checkout of 'Finalise preregistration I'"
  I=$(git log --format=%H -1 --grep='^Finalise preregistration I')
  [ -n "$I" ] || die "no commit 'Finalise preregistration I' in the history of HEAD (docs/GPU_RUNBOOK.md)"
  git diff --quiet "$I" HEAD -- ckeys experiments analysis scripts tests docs/PREREGISTRATION.md \
    || die "the code at HEAD differs from the commit 'Finalise preregistration I' ($I): check that commit out (docs/GPU_RUNBOOK.md)"
fi

# ---- software environment of stages 1, 3b, 5 and 6
LOGENV="$OUT/log_env.txt"
if ! on TEST_MODE; then
  $PY -m pip install 'transformers==5.18.0' accelerate numpy scipy pytest >> "$LOGENV" 2>&1; tail -n 1 "$LOGENV"
  $PY -c "import transformers as t; assert t.__version__ == '5.18.0', t.__version__" >> "$LOGENV" 2>&1 || die "transformers is not 5.18.0 (see $LOGENV)"
fi
{ echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ)"; $PY -m pip freeze 2>/dev/null; } >> "$OUT/PIP_FREEZE.txt"
HUB=$($PY -c "from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)" 2>/dev/null) || HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
PRE=; [ -d "$HUB/models--${MODEL//\//--}" ] && PRE=1   # cached before this run: never removed
echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) TEST_MODE=${TEST_MODE:-} HF hub cache $HUB; deadline for the exploratory parts $(date -u -d @"$STAGE7_DEADLINE" +%H:%M:%SZ 2>/dev/null || echo "$STAGE7_DEADLINE")" >> "$OUT/ENV.txt"
$PY -c "import sys, torch, transformers, numpy; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" | tee -a "$OUT/ENV.txt"
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> "$OUT/ENV.txt"
MINGIB=${MINGIB:-75}
on TEST_MODE || $PY -c "import torch; assert torch.cuda.device_count() >= 1 and torch.cuda.get_device_properties(0).total_memory / 2**30 >= $MINGIB" \
  || die "no CUDA device with >= $MINGIB GiB visible to torch"

# ---- the predecessor's released reviewer repository (read-only), fetched as in scripts/gpu_stage4.sh, checked here
P1=${P1_ROOT:-$HOME/paper1}
if [ -n "${P1R:-}" ]; then   # given explicitly: used as given, never replaced by a fetched copy
  [ -d "$P1R" ] || die "P1R=$P1R does not exist (unset P1R to fetch the release into $P1)"
else
  P1R=$P1/v5.5-reviewer-repository
fi
if [ ! -d "$P1R" ]; then
  mkdir -p "$P1"
  curl -fsSL --retry 3 --retry-delay 30 -o "$P1/p1.zip" "https://anonymous.4open.science/api/repo/Beyond-the-fitted-Scope---Causal-keys-230B/zip" \
    && $PY -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$P1/p1.zip" "$P1"
fi
[ -f "$P1R/RELEASE.json" ] || die "could not fetch the predecessor's release into $P1R (no RELEASE.json)"
export P1R
run release $PY -c "import json, os, sys; from experiments.stage7_link import release_check; r = release_check(sys.argv[1], bool(os.environ.get('TEST_MODE'))); print(json.dumps(r))" "$P1R" \
  && { echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) the predecessor's release at $P1R"; tail -n 1 "$OUT/log_release.txt"; } >> "$OUT/RELEASE.txt" \
  || die "the predecessor's release does not match the pins (see $OUT/log_release.txt)"

# ---- unit tests (FP32 at Qwen2.5-0.5B on the CPU; Gate I-G0 = tests/test_stage7_link.py), before any model
S7T="tests/test_stage7_link.py tests/test_head_splice.py tests/test_knockout.py tests/test_refit_remap.py tests/test_stage7_score.py"
SHT="tests/test_interventions.py tests/test_encoding.py tests/test_row_restricted.py tests/test_clamp.py"
OKID="$(git rev-parse HEAD 2>/dev/null) $(hostname 2>/dev/null || uname -n)"
if [ "${TESTS:-1}" = 1 ]; then
  rm -f "$OUT/PYTEST_OK.txt"
  run pytest_stage7 $PY -m pytest $S7T -v -rA -p no:cacheprovider || die "stage-7 unit tests (see $OUT/log_pytest_stage7.txt)"
  run pytest $PY -m pytest $SHT -v -rA -p no:cacheprovider || die "shared-module unit tests (see $OUT/log_pytest.txt)"
  echo "$OKID" > "$OUT/PYTEST_OK.txt"
elif ! on TEST_MODE; then   # the entry requires the FP32 tests before any model: skip them only after a pass of this HEAD here
  [ "$(cat "$OUT/PYTEST_OK.txt" 2>/dev/null)" = "$OKID" ] || die "TESTS=0 needs a passing pytest of this HEAD on this host in $OUT (PYTEST_OK.txt); run with TESTS=1"
  echo "TESTS=0: pytest passed earlier in $OUT at $OKID" | tee -a "$OUT/COMMIT.txt"
fi

# ---- revision (pinned), preflight (tokenizer only), weights
if ! on TEST_MODE; then
  grep -q "^$MODEL $REV$" "$OUT/REVISIONS.txt" 2>/dev/null || echo "$MODEL $REV" >> "$OUT/REVISIONS.txt"
  echo "$MODEL revision $REV (pinned)" | tee -a "$OUT/ENV.txt"
fi
S7=(experiments/stage7_link.py --p1-root "$P1R" --out "$OUT" "${REVARG[@]}")
keep "$OUT/preflight.json" preflight $PY "${S7[@]}" --stage preflight || die "preflight (see $OUT/log_preflight.txt)"
if ! on TEST_MODE; then
  run download $PY -c "from huggingface_hub import snapshot_download; p = snapshot_download('$MODEL', revision='$REV', ignore_patterns=['consolidated*']); print(p); assert p.rstrip('/').endswith('$REV'), p" \
    || die "download of $MODEL at $REV (see $OUT/log_download.txt)"
fi

# ---- the steps (each kept or redone; the order lets a second session finish after a cap)
keep "$OUT/heads/rank.json" rank $PY "${S7[@]}" --stage rank
keep "$OUT/frames/$TAG.json" family $PY experiments/paper1_frames.py --model mistral --p1-root "$P1R" --out "$OUT/frames" "${FAM[@]}"
keep "$OUT/heads/gate.json" gate $PY "${S7[@]}" --stage gate
keep "$OUT/link/$TAG.json" link $PY "${S7[@]}" --stage link
FAILFILE=FAILED_EXPLORATORY.txt keep "$OUT/heads/remaprank.json" remaprank $PY "${S7[@]}" --stage remaprank
echo "$MODEL loaded snapshot(s): $(ls "$HUB/models--${MODEL//\//--}/snapshots" 2>/dev/null | tr '\n' ' ')" >> "$OUT/ENV.txt"
d="$HUB/models--${MODEL//\//--}"
on KEEP_CACHE || [ -n "$PRE" ] || [ ! -d "$d" ] || { rm -rf "$d"; echo "$MODEL removed from the HF cache ($d)" >> "$OUT/ENV.txt"; }

# ---- score (provenance, population, gates, one verdict line per prediction I1-I7, reported lines, exploratory report)
run score $PY analysis/stage7_score.py --root "$OUT" --tag "$TAG"
awk '/^######## EXPLORATORY/{exit} /^GATES/{g = 1} /^REPORTED/{g = 0} g || /^(SUMMARY|  commits:|  population:)/' "$OUT/STAGE7_SCORE.txt" 2>/dev/null | cut -c1-400

tar czf "$TGZ" --exclude='*.safetensors' --exclude='*.bin' "$OUT"
echo "==================== DONE  $(date -u +%H:%M:%S)"
echo "Results archive: $(cd "$(dirname "$TGZ")" && pwd)/$(basename "$TGZ")"
[ -f "$OUT/FAILED_EXPLORATORY.txt" ] && { echo "exploratory step failed (not a pipeline failure):"; cat "$OUT/FAILED_EXPLORATORY.txt"; }
if [ -f "$OUT/FAILED.txt" ]; then cat "$OUT/FAILED.txt"; exit 1; fi
echo "no failures (no step FAILED; gate and prediction verdicts above are results, see $OUT/STAGE7_SCORE.txt)"
