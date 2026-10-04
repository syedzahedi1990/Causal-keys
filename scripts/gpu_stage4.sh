#!/usr/bin/env bash
# GPU stage 4 (preregistered as P-2026-10-05-F in docs/PREREGISTRATION.md; paper v2 only): refit Paper 1's remap M
# with its released, unmodified gpu/train.py under NO-MENTION (fit_none) and under Paper 1's own format (fit_p1,
# same-code control), then evaluate both refits next to the released bases in Paper 1's frames and score.
# Needs 1-2 x 80 GB GPUs (each fit needs one card with >= 75 GiB; with 2 GPUs the two fits run in parallel)
# and >= 150 GB disk (47 GB weights, downloaded once). Runtime: about 2-3 h on 2 x H100, 3-5 h on 2 x A100
# (one fit run is 0.6-1 h on an H100, 1.3-2.3 h on an A100; sequential with one GPU); evaluation about 0.5 h.
# Usage:
#   git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
#   F=$(git log --format=%H -1 --grep='^Finalise preregistration F') && [ -n "$F" ] && git checkout "$F"
#   bash scripts/gpu_stage4.sh
#   TEST_MODE=1 OUT=/tmp/s4test bash scripts/gpu_stage4.sh   # CPU plumbing test, Qwen2.5-0.5B, about 15 minutes
# Optional: P1R=<unpacked Paper 1 repo> (else downloaded), LOADER=paper1 (Paper 1's own load_engine; needs its exact
# Python 3.12.14 / package pins), PY=<python>. Needs Python >= 3.11 (else a Python 3.12 venv with torch 2.11.0+cu128
# is made with uv, and reused on a rerun). Outside TEST_MODE it refuses a DRAFT P-2026-10-05-F or modified tracked
# files. Fit run directories are never overwritten: rerunning with the same OUT reuses finished fits
# (FRAME.json COMPLETE), moves an unfinished one aside to fit_X.failed.<UTC> (noted in COMMIT.txt) and refits it, then
# redoes validation, evaluation and scoring; logs and COMMIT.txt are appended to. Exit status 1 if any step FAILED.
# Output: gpu_stage4_results.tgz (TEST_MODE: $OUT.tgz)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOADER=${LOADER:-ours}
MODEL=mistralai/Mistral-Small-24B-Instruct-2501
REV=9527884be6e5616bdd54de542f9ae13384489724
if [ -n "${TEST_MODE:-}" ]; then
  OUT=${OUT:-$(mktemp -d)/gpu_stage4}; TGZ=${TGZ:-$OUT.tgz}
  TINY=Qwen/Qwen2.5-0.5B-Instruct; K=${TEST_K:-6}; N=${TEST_N:-3}
  FIT=(--test-model "$TINY" --test-pairs "$K" --device cpu)
  EVAL=(--model-override "$TINY" --bases-override 896 --n "$N"); PRE=(--test-model "$TINY" --cores "$N")
  TAG=TEST_${TINY##*/}
else
  OUT=${OUT:-results/gpu_stage4}; TGZ=${TGZ:-gpu_stage4_results.tgz}
  FIT=(--revision "$REV"); EVAL=(); PRE=(--revision "$REV"); TAG=mistral
fi
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {
  local name=$1 rc=0; shift
  echo "==================== $name  $(date -u +%H:%M:%S)"
  echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$OUT/log_$name.txt"
  PYTHONPATH=. "$@" >> "$OUT/log_$name.txt" 2>&1 || {
    rc=$?; local why="exit $rc"; [ $rc -gt 128 ] && why+=" (killed by signal $((rc - 128)); 9 = SIGKILL, often the host OOM killer)"
    echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED $why" >> "$OUT/log_$name.txt"
    echo "FAILED $name $why (see $OUT/log_$name.txt)" | tee -a "$OUT/FAILED.txt"; }
  tail -n 2 "$OUT/log_$name.txt"
  return $rc
}
die() {  # fatal before the fits: record, archive the logs, stop
  echo "FAILED $1" | tee -a "$OUT/FAILED.txt"
  tar czf "$TGZ" "$OUT"; echo "Results archive (logs): $TGZ"; exit 1
}
[ -f "$OUT/FAILED.txt" ] && mv "$OUT/FAILED.txt" "$OUT/FAILED.$(date -u +%Y%m%dT%H%M%SZ).txt"
{ echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ)"; git rev-parse HEAD; git status --short; } | tee -a "$OUT/COMMIT.txt"
if [ -z "${TEST_MODE:-}" ]; then  # the preregistered code only: the finalised entry, no local changes
  awk '/^## /{f = ($0 ~ /P-2026-10-05-F/)} f' docs/PREREGISTRATION.md | grep -q "DRAFT, not yet final" \
    && die "preregistration P-2026-10-05-F is still a DRAFT: check out the commit 'Finalise preregistration F' (docs/GPU_RUNBOOK.md)"
  git rev-parse HEAD > /dev/null 2>&1 && [ -z "$(git status --porcelain --untracked-files=no)" ] \
    || die "not a clean git checkout (modified tracked files above): run from a clean checkout of 'Finalise preregistration F'"
fi

# ---- software environment (stage 2/3b: transformers 5.9.0, Paper 1's pin); Paper 1's train.py needs Python >= 3.11
LOGENV="$OUT/log_env.txt"
if [ -z "${TEST_MODE:-}" ]; then
  if ! $PY -c "import sys; sys.exit(sys.version_info < (3, 11))" && .venv-s4/bin/python -c "import torch, pip" 2> /dev/null; then
    PY=$(pwd)/.venv-s4/bin/python; echo "Python < 3.11: reusing .venv-s4" | tee -a "$LOGENV"
  elif ! $PY -c "import sys; sys.exit(sys.version_info < (3, 11))"; then
    echo "Python < 3.11: creating .venv-s4 (Python 3.12) with uv" | tee -a "$LOGENV"
    { $PY -m pip install uv && $PY -m uv venv --clear -p 3.12 .venv-s4 && \
      $PY -m uv pip install --python .venv-s4/bin/python 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128 && \
      $PY -m uv pip install --python .venv-s4/bin/python 'numpy==1.26.4' pip; } >> "$LOGENV" 2>&1 || die "python_env (see $LOGENV)"
    PY=$(pwd)/.venv-s4/bin/python
  fi
  $PY -m pip install 'transformers==5.9.0' 'accelerate==1.13.0' numpy >> "$LOGENV" 2>&1; tail -n 1 "$LOGENV"
  $PY -c "import transformers as t; assert t.__version__ == '5.9.0', t.__version__" >> "$LOGENV" 2>&1 \
    || die "transformers is not 5.9.0 (see $LOGENV)"
fi
$PY -m pip freeze 2>/dev/null > "$OUT/PIP_FREEZE.txt"
$PY -c "import sys, torch, transformers, accelerate, numpy; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'accelerate', accelerate.__version__, 'numpy', numpy.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" | tee "$OUT/ENV.txt"
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> "$OUT/ENV.txt"
NGPU=$($PY -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
[ -n "${TEST_MODE:-}" ] || [ "$NGPU" -ge 1 ] || die "no CUDA device visible to torch (NGPU=$NGPU; driver too old for the torch build?)"
[ -n "${TEST_MODE:-}" ] || $PY -c "import torch; g = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(min(2, torch.cuda.device_count()))]; print('GPU GiB', [round(x, 1) for x in g]); assert min(g) >= 75" \
  || die "each fit needs one GPU with >= 75 GiB (GPU 0, and GPU 1 when two are visible)"

# ---- Paper 1's released repository (read-only)
P1=${P1_ROOT:-$HOME/paper1}; P1R=${P1R:-$P1/v5.5-reviewer-repository}
if [ ! -d "$P1R" ]; then
  mkdir -p "$P1"; P1R=$P1/v5.5-reviewer-repository
  curl -fsSL --retry 3 --retry-delay 30 -o "$P1/p1.zip" "https://anonymous.4open.science/api/repo/Beyond-the-fitted-Scope---Causal-keys-230B/zip" \
    && $PY -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$P1/p1.zip" "$P1"
fi
[ -f "$P1R/gpu/train.py" ] || die "p1_fetch (no $P1R/gpu/train.py)"

# ---- preflight before the weights and fits (tokenizer only): release pins, P1 equality, NONE prefix,
# fix_mistral_regex installed and ids unchanged on every frames prompt
run preflight $PY experiments/refit_remap.py --preflight --p1-root "$P1R" --out "$OUT/preflight" \
    "${PRE[@]}" || die "preflight (see $OUT/log_preflight.txt)"

# ---- weights once, before the fits split over GPUs
if [ -z "${TEST_MODE:-}" ]; then
  run download $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', revision='$REV', ignore_patterns=['consolidated*']))" || die "download"
fi

# ---- fits: fit_none and fit_p1 (in parallel on GPUs 0 and 1 when two are visible); finished fits are reused
finished() {  # fit_$1's FRAME.json says COMPLETE under frame $2
  $PY -c "import json, sys; f = json.load(open(sys.argv[1])); sys.exit(not (f['status'] == 'COMPLETE' and f['frame'] == sys.argv[2]))" \
    "$OUT/fit_$1/FRAME.json" "$2" 2>/dev/null
}
fit() {
  if finished "$1" "$3"; then echo "reusing finished $OUT/fit_$1"; return; fi
  if [ -e "$OUT/fit_$1" ]; then
    local old="$OUT/fit_$1.failed.$(date -u +%Y%m%dT%H%M%SZ)" st
    st=$($PY -c "import json, sys; print(json.load(open(sys.argv[1])).get('status'))" "$OUT/fit_$1/FRAME.json" 2>/dev/null || echo "no FRAME.json")
    mv "$OUT/fit_$1" "$old" && echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) unfinished fit_$1 (status $st) moved to $old; refitting" | tee -a "$OUT/COMMIT.txt"
  fi
  run "fit_$1" env ${2:+CUDA_VISIBLE_DEVICES=$2} $PY experiments/refit_remap.py --frame "$3" --p1-root "$P1R" \
    --out "$OUT/fit_$1" --loader "$LOADER" "${FIT[@]}"
}
if [ -z "${TEST_MODE:-}" ] && [ "$NGPU" -ge 2 ]; then
  fit none 0 NONE & fit p1 1 P1 & wait
else
  fit none "" NONE; fit p1 "" P1
fi
grep -h "fit_complete\|done" "$OUT"/log_fit_*.txt | tail -n 14
if ! { finished none NONE && finished p1 P1; }; then
  echo "FAILED fits not COMPLETE: validation, evaluation and scoring skipped (rerun with the same OUT to refit)" | tee -a "$OUT/FAILED.txt"
  tar czf "$TGZ" --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' "$OUT"; echo "Results archive (logs): $TGZ"; exit 1
fi

# ---- validation (bases; Paper 1's load_training_bases at width 5120)
run validate $PY experiments/refit_remap.py --p1-root "$P1R" --validate "$OUT/fit_none/run" --validate "$OUT/fit_p1/run"

# ---- evaluation in Paper 1's frames: released bases (anchor) + both refit families
REFITS=(--refit "none=$OUT/fit_none/run" --refit "p1=$OUT/fit_p1/run")
run frames $PY experiments/paper1_frames.py --model mistral --p1-root "$P1R" "${REFITS[@]}" --prefill "Answer:" \
    "${EVAL[@]}" --out "$OUT/frames"
run frames_noprefill $PY experiments/paper1_frames.py --model mistral --p1-root "$P1R" "${REFITS[@]}" --prefill "" \
    "${EVAL[@]}" --out "$OUT/frames_noprefill"

# ---- score
echo "==================== score  $(date -u +%H:%M:%S)"
if ! PYTHONPATH=. $PY analysis/stage4_score.py --root "$OUT" --tag "$TAG" --p1-root "$P1R" > "$OUT/STAGE4_SCORE.txt" 2> "$OUT/log_score.txt"; then
  echo "FAILED score (see $OUT/log_score.txt)" | tee -a "$OUT/FAILED.txt"
fi
cat "$OUT/STAGE4_SCORE.txt" "$OUT/log_score.txt" | grep -E "^( *G1 ->|G[123] |F[1-4]|GATES|ALL GATES|H_fit|  provenance|  population|PROVENANCE)"

tar czf "$TGZ" --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' "$OUT"
echo "==================== DONE  $(date -u +%H:%M:%S)"
echo "Results archive: $(cd "$(dirname "$TGZ")" && pwd)/$(basename "$TGZ")"
if [ -f "$OUT/FAILED.txt" ]; then cat "$OUT/FAILED.txt"; exit 1; fi
echo "no failures (no step FAILED; gate and prediction verdicts above are results, see STAGE4_SCORE.txt)"
