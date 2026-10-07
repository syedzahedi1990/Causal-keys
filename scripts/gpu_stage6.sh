#!/usr/bin/env bash
# GPU stage 6 (preregistered as P-2026-10-05-H in docs/PREREGISTRATION.md; paper v3): the reader heads and the second hop,
# and the exchange on Prakash et al.'s intervention with the out-of-sample law. Two parts, one per-model loop:
#   heads    experiments/stage6_heads.py at Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (BF16, eager attention,
#            use_cache=False; R = make_cores(60, Random(0)), E = make_cores(60, Random(1)); OPTIONS-AFTER and SENTENCE-AFTER)
#   prakash  experiments/prakash_swap.py at Qwen2.5-14B-Instruct (BF16, sdpa): filter, sweeps (l*, l*_ID), exchange, clamp,
#            and the exploratory FP32 re-check of the NO-MENTION sweeps at l* +- 2 / l*_ID +- 2 (sweep_*_fp32.json, not scored),
#            on the release of https://github.com/Nix07/mind at the commit pinned in ckeys/causaltom.py (RELEASE_SHA),
#            fetched sparsely into PRAKASH_REPO (default ~/.cache/causal-keys/mind-<sha>); every file used and the seed-10
#            pool are sha256-asserted (python -m ckeys.causaltom) before any test or model; nothing of it enters the archive
# Needs one 80 GB GPU (A100/H100), >= 64 GB host RAM and >= 100 GB disk (after its steps, each model this run downloaded
# is removed from the HF cache; models cached before the run are left alone; KEEP_CACHE=1 keeps everything).
# Runtime about 6 h (part (a) about 2.5 h, part (b) 2.5-3.5 h, setup and pytest 0.5 h).
# MODEL=llama70 is a separate invocation (PART=prakash only): Meta-Llama-3-70B-Instruct (gated) on 2 x 80 GB GPUs,
# >= 200 GB disk; only what H12 needs (formats NO-MENTION, QNAMES, OPTIONS-AFTER; no exploratory sweeps, no onset sweep;
# about 56k row-forwards; device_map=auto runs the two GPUs one after the other): about 4-5 h, 8-10 GPU-hours.
# It needs HF_TOKEN in the environment (an account that accepted the licence) and stops at once without it; nothing
# ever asks for a token. Its results go to the same OUT (H12 is scored against the 14B files there: unpack
# gpu_stage6_results.tgz into OUT on a 2-GPU box first, or run it on the main box afterwards).
# Usage:
#   git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
#   H=$(git log --format=%H -1 --grep='^Finalise preregistration H') && [ -n "$H" ] && git checkout "$H"
#   bash scripts/gpu_stage6.sh                      # PART=all
#   MODEL=llama70 bash scripts/gpu_stage6.sh        # optional, 2 GPUs, HF_TOKEN set in the environment beforehand
#   TEST_MODE=1 bash scripts/gpu_stage6.sh          # CPU plumbing test: pytest, both parts at Qwen2.5-0.5B (FP32) into a
#                                                   # scratch OUT, then the scorer; about 8 min of pytest plus
#                                                   # about 11 min of runs on a 4-core CPU
# Optional: PART=heads|prakash|all (default all), OUT=<dir>, KEEP_CACHE=1, FORCE=1, TESTS=0, PY=<python>, MINGIB=<GiB>
# (per-GPU memory floor, default 75), PRAKASH_REPO=<checkout of the release> (its HEAD must be the pinned commit);
# the switches TEST_MODE, KEEP_CACHE, FORCE, TESTS take 0 or 1 only (unset = 0 except TESTS = 1); outside TEST_MODE,
# TESTS=0 needs OUT/PYTEST_OK.txt from a passing pytest of the same HEAD on the same host. keep/FORCE as in
# scripts/gpu_stage5.sh: a step whose results file exists, is readable JSON and complete (a heads file holds both arms) is
# kept unless FORCE=1; a file written by a FAILED step, or unreadable or incomplete, is moved aside to <file>.failed.<UTC>
# and the step is run again, so a rerun with the same OUT resumes where it failed. Each model's Hub revision is resolved
# once, pinned in $OUT/REVISIONS.txt (reused by a rerun), passed to every step as --revision and recorded in ENV.txt.
# Pytest: a fixed set, chosen by the code stage 6 runs, no reruns. First the stage-6 tests (Gate a1:
# tests/test_head_splice.py; tests/test_prakash.py; tests/test_clamp.py, which also holds the format_factorial.run_item and
# RowSplice regressions; tests/test_stage6_score.py), verbose, into log_pytest_stage6.txt, whose last run the scorer reads
# for Gate a1; then the tests of the other shared modules stage 6 calls (ckeys.interventions, ckeys.encoding and
# ckeys.story, row_restricted_keys: tests/test_interventions.py, test_encoding.py, test_row_restricted.py) into
# log_pytest.txt. Any failure stops the script. The other test files cover stage 1-5 code only (run them locally).
# Outside TEST_MODE it pins transformers 5.18.0, refuses a DRAFT P-2026-10-05-H, modified tracked files, or code
# (ckeys experiments analysis scripts tests and the entry) that differs from the commit 'Finalise preregistration H',
# then runs pytest and the preflights (release and pool hashes; prakash_swap.py --stage preflight, tokenizer only) before
# any model. A failed step is recorded in $OUT/FAILED.txt and the pipeline goes on; exit status 1 if any step FAILED (the
# score step FAILS with exit 2 when the provenance or population check reports MISMATCH outside TEST_MODE).
# Output: gpu_stage6_results.tgz (MODEL=llama70: gpu_stage6_llama70_results.tgz; TEST_MODE: $OUT.tgz), no weights;
# the score is $OUT/STAGE6_SCORE.txt (analysis/stage6_score.py).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
for v in TEST_MODE KEEP_CACHE FORCE TESTS; do case ${!v:-} in ""|0|1) ;; *) echo "bad $v='${!v}' (0 or 1)"; exit 2;; esac; done
on() { [ "${!1:-0}" = 1 ]; }   # a switch is on only when it is 1
on TEST_MODE || unset TEST_MODE   # the experiments read TEST_MODE as set / unset
PART=${PART:-all}; MODEL=${MODEL:-}
case $PART in heads|prakash|all) ;; *) echo "bad PART '$PART' (heads, prakash or all)"; exit 2;; esac
case $MODEL in "") ;; llama70) [ "$PART" = heads ] && { echo "MODEL=llama70 runs part prakash only (PART=heads given)"; exit 2; }; PART=prakash
  on TEST_MODE && { echo "MODEL=llama70 has no TEST_MODE (the plumbing test runs at Qwen2.5-0.5B)"; exit 2; }
  [ -n "${HF_TOKEN:-}" ] || { echo "MODEL=llama70 needs HF_TOKEN in the environment (export it in the box's terminal; the script never asks for it)"; exit 2; };;
  *) echo "bad MODEL '$MODEL' (unset, or llama70)"; exit 2;; esac
has() { [ "$PART" = all ] || [ "$PART" = "$1" ]; }
export GIT_TERMINAL_PROMPT=0 HF_HUB_DISABLE_PROGRESS_BARS=1   # nothing interactive; no progress bars in the logs
Q7=Qwen/Qwen2.5-7B-Instruct; MI=mistralai/Mistral-7B-Instruct-v0.3; Q14=Qwen/Qwen2.5-14B-Instruct; L70=meta-llama/Meta-Llama-3-70B-Instruct
if on TEST_MODE; then
  OUT=${OUT:-$(mktemp -d)/gpu_stage6}; TGZ=${TGZ:-$OUT.tgz}
  TINY=Qwen/Qwen2.5-0.5B-Instruct; DT=float32; TAG=TEST_${TINY##*/}
  export TEST_MODE=1 KEEP_CACHE=1   # the experiments then force Qwen2.5-0.5B, FP32, CPU (prakash_swap.py keeps --n)
  NP=${TEST_NP:-2}; HEADS=("$TINY"); PRAK=("$TINY"); NGPU=0
  REL=${PRAKASH_REPO:-}; [ -z "$REL" ] && [ -d /home/user/nix07/mind/.git ] && REL=/home/user/nix07/mind   # the local clone, else fetched
else
  OUT=${OUT:-results/gpu_stage6}; DT=bfloat16; TAG=stage6; NP=150; REL=${PRAKASH_REPO:-}
  if [ "$MODEL" = llama70 ]; then TGZ=${TGZ:-gpu_stage6_llama70_results.tgz}; HEADS=(); PRAK=("$L70"); NGPU=2
  else TGZ=${TGZ:-gpu_stage6_results.tgz}; HEADS=("$Q7" "$MI"); PRAK=("$Q14"); NGPU=1; fi
fi
has heads || HEADS=(); has prakash || PRAK=()
MODELS=("${HEADS[@]}"); for m in "${PRAK[@]}"; do case " ${MODELS[*]} " in *" $m "*) ;; *) MODELS+=("$m");; esac; done
among() { local x=$1; shift; for y in "$@"; do [ "$x" = "$y" ] && return 0; done; return 1; }
mkdir -p "$OUT"
echo "PART=$PART MODEL=${MODEL:-} FORCE=${FORCE:-0} models: ${MODELS[*]}"
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
aside() { local old="$1.failed.$(date -u +%Y%m%dT%H%M%SZ)"; mv "$1" "$old"; echo "moved aside $old: $2" | tee -a "$OUT/COMMIT.txt"; }
complete() {  # the results file is readable JSON and, for a heads file, holds both arms with their evaluation stories
  $PY - "$1" > /dev/null 2>&1 <<'EOF'
import json, sys
f = sys.argv[1]; j = json.load(open(f))
if "/heads/" in f:
    assert {"P1", "POST"} <= set(j["arms"]) and all(a.get("eval") for a in j["arms"].values())
EOF
}
STEPS=0
keep() {  # keep <results file> <name> <command...>: as in scripts/gpu_stage5.sh, with the completeness check above
  local f=$1 name=$2 rc=0 stamp; shift
  STEPS=$((STEPS + 1))
  if [ -e "$f" ]; then
    if ! complete "$f"; then aside "$f" "not readable JSON or incomplete (a step interrupted while writing); $name is run again"
    elif ! on FORCE; then echo "==================== $name kept: $f exists (FORCE=1 to redo)"; return 0; fi
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
[ -f "$OUT/FAILED.txt" ] && mv "$OUT/FAILED.txt" "$OUT/FAILED.$(date -u +%Y%m%dT%H%M%SZ).txt"
{ echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) PART=$PART MODEL=${MODEL:-} TEST_MODE=${TEST_MODE:-0} FORCE=${FORCE:-0}"; git rev-parse HEAD; git status --short; } | tee -a "$OUT/COMMIT.txt"
if ! on TEST_MODE; then  # the preregistered code only: the finalised entry, no local changes, the code of the finalising commit
  awk '/^## /{f = ($0 ~ /P-2026-10-05-H/)} f && /DRAFT, not yet final/{d = 1} END{exit !d}' docs/PREREGISTRATION.md \
    && die "preregistration P-2026-10-05-H is still a DRAFT: check out the commit 'Finalise preregistration H' (docs/GPU_RUNBOOK.md)"
  # (awk decides alone: an awk | grep -q pipeline under pipefail gets status 141 when grep exits first, and the refusal is skipped)
  git rev-parse HEAD > /dev/null 2>&1 && [ -z "$(git status --porcelain --untracked-files=no)" ] \
    || die "not a clean git checkout (modified tracked files above): run from a clean checkout of 'Finalise preregistration H'"
  H=$(git log --format=%H -1 --grep='^Finalise preregistration H')
  [ -n "$H" ] || die "no commit 'Finalise preregistration H' in the history of HEAD (docs/GPU_RUNBOOK.md)"
  git diff --quiet "$H" HEAD -- ckeys experiments analysis scripts tests docs/PREREGISTRATION.md \
    || die "the code at HEAD differs from the commit 'Finalise preregistration H' ($H): check that commit out (docs/GPU_RUNBOOK.md)"
fi

# ---- software environment of stages 1, 3b and 5
LOGENV="$OUT/log_env.txt"
if ! on TEST_MODE; then
  $PY -m pip install 'transformers==5.18.0' accelerate numpy scipy pytest >> "$LOGENV" 2>&1; tail -n 1 "$LOGENV"
  $PY -c "import transformers as t; assert t.__version__ == '5.18.0', t.__version__" >> "$LOGENV" 2>&1 || die "transformers is not 5.18.0 (see $LOGENV)"
fi
{ echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ)"; $PY -m pip freeze 2>/dev/null; } >> "$OUT/PIP_FREEZE.txt"
HUB=$($PY -c "from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)" 2>/dev/null) || HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
declare -A PRE; for m in "${MODELS[@]}"; do [ -d "$HUB/models--${m//\//--}" ] && PRE[$m]=1; done   # cached before this run: never removed
clean() {  # removes only the directory of the model just processed, only if this run created it (not the whole hub cache)
  local d="$HUB/models--${1//\//--}"
  on KEEP_CACHE || [ -n "${PRE[$1]:-}" ] || [ ! -d "$d" ] || { rm -rf "$d"; echo "$1 removed from the HF cache ($d)" >> "$OUT/ENV.txt"; }
}
echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) PART=$PART MODEL=${MODEL:-} TEST_MODE=${TEST_MODE:-} HF hub cache $HUB HF_TOKEN $([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)" >> "$OUT/ENV.txt"
$PY -c "import sys, torch, transformers, numpy; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" | tee -a "$OUT/ENV.txt"
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> "$OUT/ENV.txt"
MINGIB=${MINGIB:-75}
on TEST_MODE || $PY -c "import torch; n = torch.cuda.device_count(); assert n >= $NGPU and all(torch.cuda.get_device_properties(i).total_memory / 2**30 >= $MINGIB for i in range($NGPU))" \
  || die "fewer than $NGPU CUDA device(s) with >= $MINGIB GiB visible to torch"

# ---- Prakash et al.'s release at the pinned commit (sparse, data and the two ported code files), hashes asserted
SHA=$(PYTHONPATH=. $PY -c "from ckeys.causaltom import RELEASE_SHA, RELEASE_URL; print(RELEASE_SHA)") || die "ckeys.causaltom not importable"
URL=$(PYTHONPATH=. $PY -c "from ckeys.causaltom import RELEASE_URL; print(RELEASE_URL)")
DEFREL=$HOME/.cache/causal-keys/mind-$SHA; REL=${REL:-$DEFREL}
if [ ! -d "$REL/.git" ]; then   # fetched only into a directory of the script's own (the default cache path, or a new one)
  if [ -e "$REL" ]; then [ "$REL" = "$DEFREL" ] && rm -rf "$REL" || die "PRAKASH_REPO=$REL exists but is not a git checkout of $SHA (left untouched)"; fi
  { mkdir -p "$REL" && git init -q "$REL" && git -C "$REL" remote add origin "$URL" \
      && git -C "$REL" sparse-checkout set --no-cone /data/ /src/dataset.py /notebooks/causalToM_novis/utils.py \
      && git -C "$REL" fetch -q --depth 1 --filter=blob:none origin "$SHA" && git -C "$REL" checkout -q FETCH_HEAD; } >> "$OUT/log_release.txt" 2>&1 \
    || { rm -rf "$REL"; die "could not fetch $URL at $SHA into $REL (see $OUT/log_release.txt)"; }
fi
[ "$(git -C "$REL" rev-parse HEAD 2>/dev/null)" = "$SHA" ] || die "the release checkout $REL is not at $SHA"
export PRAKASH_REPO=$REL
{ echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) $URL at $(git -C "$REL" rev-parse HEAD) in $REL"; } >> "$OUT/RELEASE.txt"
run release $PY -m ckeys.causaltom && tail -n 2 "$OUT/log_release.txt" >> "$OUT/RELEASE.txt" || die "release or pool hash check (see $OUT/log_release.txt)"
if [ "$MODEL" = llama70 ]; then   # the gated weights are reachable with this token, before any test or download
  run access $PY -c "from experiments.prakash_swap import access; ok, why = access('$L70'); print('access', ok, why); assert ok" \
    || die "no access to $L70 with the HF_TOKEN set here: accept the licence on its Hub page with that account (see $OUT/log_access.txt)"
fi

# ---- unit tests (FP32 at Qwen2.5-0.5B on the CPU; stage 6 first, Gate a1 = tests/test_head_splice.py), before any model
S6T="tests/test_head_splice.py tests/test_prakash.py tests/test_clamp.py tests/test_stage6_score.py"
SHT="tests/test_interventions.py tests/test_encoding.py tests/test_row_restricted.py"   # the other shared modules stage 6 calls
OKID="$(git rev-parse HEAD 2>/dev/null) $(hostname 2>/dev/null || uname -n)"
if [ "${TESTS:-1}" = 1 ]; then
  rm -f "$OUT/PYTEST_OK.txt"
  run pytest_stage6 $PY -m pytest $S6T -v -rA -p no:cacheprovider || die "stage-6 unit tests (see $OUT/log_pytest_stage6.txt)"
  run pytest $PY -m pytest $SHT -v -rA -p no:cacheprovider || die "shared-module unit tests (see $OUT/log_pytest.txt)"
  echo "$OKID" > "$OUT/PYTEST_OK.txt"
elif ! on TEST_MODE; then   # the entry requires the FP32 tests before any 7B model: skip them only after a pass of this HEAD here
  [ "$(cat "$OUT/PYTEST_OK.txt" 2>/dev/null)" = "$OKID" ] || die "TESTS=0 needs a passing pytest of this HEAD on this host in $OUT (PYTEST_OK.txt); run with TESTS=1"
  echo "TESTS=0: pytest passed earlier in $OUT at $OKID" | tee -a "$OUT/COMMIT.txt"
fi

# ---- revisions (resolved once, pinned in REVISIONS.txt) and the tokenizer-only preflight of part (b)
declare -A RV
for m in "${MODELS[@]}"; do
  REV=$(awk -v m="$m" '$1 == m {r = $2} END {print r}' "$OUT/REVISIONS.txt" 2>/dev/null); PIN=pinned
  [ -n "$REV" ] || { PIN=resolved; REV=$($PY -c "from huggingface_hub import model_info; print(model_info('$m').sha)" 2>/dev/null) && [ -n "$REV" ] && echo "$m $REV" >> "$OUT/REVISIONS.txt"; }
  [ -n "$REV" ] && RV[$m]="--revision $REV" || { REV=""; RV[$m]=""; }
  echo "$m revision $([ -n "$REV" ] && echo "$REV ($PIN)" || echo "unavailable (offline: the steps load the cached snapshot, recorded below)")" | tee -a "$OUT/ENV.txt"
done
for m in "${PRAK[@]}"; do
  s=${m##*/}
  keep "$OUT/prakash/$s/preflight.json" "preflight_$s" $PY experiments/prakash_swap.py --model "$m" --stage preflight --n "$NP" --out "$OUT/prakash" ${RV[$m]} \
    || die "preflight of $m (see $OUT/log_preflight_$s.txt)"
done

# ---- per-model loop (each model downloaded once, then the cache is cleaned)
SWEEPLAST=sweep_ID_LETTERS-AFTER.json; { on TEST_MODE || [ "$MODEL" = llama70 ]; } && SWEEPLAST=lstar.json   # the sweep step's last file (no exploratory sweeps)
for m in "${MODELS[@]}"; do
  s=${m##*/}; echo "######## $m  $(date -u +%H:%M:%S)"
  if among "$m" "${HEADS[@]}"; then
    keep "$OUT/heads/$s.json" "heads_$s" $PY experiments/stage6_heads.py --model "$m" --dtype "$DT" --out "$OUT/heads" ${RV[$m]}
  fi
  if among "$m" "${PRAK[@]}"; then
    P=(experiments/prakash_swap.py --model "$m" --dtype "$DT" --n "$NP" --out "$OUT/prakash" ${RV[$m]})
    [ "$MODEL" = llama70 ] && P+=(--formats NO-MENTION,QNAMES,OPTIONS-AFTER --explore-sweeps "" --onset-sweep 0)   # H12 only (0 is a primary onset)
    on TEST_MODE || keep "$OUT/prakash/$s/filter.json" "filter_$s" $PY "${P[@]}" --stage filter
    keep "$OUT/prakash/$s/$SWEEPLAST" "sweep_$s" $PY "${P[@]}" --stage sweep
    keep "$OUT/prakash/$s/exchange.json" "exchange_$s" $PY "${P[@]}" --stage exchange
    keep "$OUT/prakash/$s/clamp.json" "clamp_$s" $PY "${P[@]}" --stage clamp
    if ! on TEST_MODE && [ "$MODEL" != llama70 ]; then   # exploratory: the NO-MENTION sweeps re-run in FP32 at l* +- 2 and l*_ID +- 2 (not scored)
      LL=$($PY -c "import json, sys; d = sys.argv[1]; j = json.load(open(d + '/lstar.json')); L = json.load(open(d + '/sweep_BIND_NO-MENTION.json'))['layers']
f = lambda l: ','.join(str(x) for x in L if abs(x - l) <= 2); print(f(j['lstar']), f(j['lstar_ID']))" "$OUT/prakash/$s" 2>/dev/null)
      if [ -n "$LL" ]; then keep "$OUT/prakash/$s/sweep_ID_NO-MENTION_fp32.json" "fp32_$s" $PY "${P[@]}" --stage sweep --dtype float32 --label _fp32 \
        --sweep-layers "${LL% *}" --sweep-layers-id "${LL#* }"
      else echo "fp32_$s skipped: no lstar.json or BIND sweep in $OUT/prakash/$s" | tee -a "$OUT/COMMIT.txt"; fi
    fi
  fi
  echo "$m loaded snapshot(s): $(ls "$HUB/models--${m//\//--}/snapshots" 2>/dev/null | tr '\n' ' ')" >> "$OUT/ENV.txt"
  clean "$m"
done
[ "$STEPS" -gt 0 ] || die "no step selected"

# ---- score (provenance, population, gates, one verdict line per prediction H1-H12, then each part's report)
run score $PY analysis/stage6_score.py --root "$OUT" --tag "$TAG"
awk '/^######## PART/{exit} /^GATES/{g = 1} /^VERDICTS/{g = 0} g || /^(VERDICTS|  H[0-9]+ |         |       Gate a1|SUMMARY|  commits:|  population:)/' "$OUT/STAGE6_SCORE.txt" 2>/dev/null   # gates, verdicts with reasons, summary

tar czf "$TGZ" --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' "$OUT"
echo "==================== DONE  $(date -u +%H:%M:%S)"
echo "Results archive: $(cd "$(dirname "$TGZ")" && pwd)/$(basename "$TGZ")"
if [ -f "$OUT/FAILED.txt" ]; then cat "$OUT/FAILED.txt"; exit 1; fi
echo "no failures (no step FAILED; gate and prediction verdicts above are results, see $OUT/STAGE6_SCORE.txt)"
