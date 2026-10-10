# shellcheck shell=bash
# Shared library of the stage-8 GPU scripts (preregistered as P-2026-10-10-J in docs/PREREGISTRATION.md, parts A-D:
# scripts/gpu_stage8a.sh ... gpu_stage8d.sh). Sourced, never run. A part script does:
#   PART=<a|b|c|d>; S8_DEADLINE_H_DEFAULT=<hours>; source scripts/stage8_common.sh; s8_init
#   s8_pytest tests/....py ...        # FP32 gates before any model; dies on failure (TESTS=0 skips, logged)
#   DIR=$(s8_fetch <model-key>)       # local dir of verified files for a key of scripts/stage8_models.json;
#                                     # in TEST_MODE it prints Qwen/Qwen2.5-0.5B-Instruct
#   s8_step <name> <cmd...>           # keep/FORCE_STEPS semantics, log to $OUT/logs/<name>.log
#   s8_time_left <minutes> || s8_skip <name> "<reason>"
#   s8_drop <model-key>               # delete weights unless KEEP_CACHE=1
#   s8_finish analysis/stage8<x>_score.py   # scorer, MANIFEST.sha256, tgz; its status is the script's exit status
#   die "<message>"                   # fatal: record in FAILED.txt, archive the logs, exit 1
# Variables after s8_init: OUT (results dir), PY, TEST_MODE (0/1; exported to the steps only when 1, so a step may test
# os.environ.get("TEST_MODE")), STAGE8_DEADLINE (epoch s, exported), DEV ("cuda" or "cpu"), DTYPE ("bfloat16" or
# "float32"); also S8_MODELS (the model store), S8_J (the finalising commit; empty in TEST_MODE), TGZ.
#
# s8_init. Switches TEST_MODE, KEEP_CACHE, FORCE, TESTS take 0 or 1 only (unset = 0, except TESTS = 1); DEADLINE_H is a
# positive number of hours (default S8_DEADLINE_H_DEFAULT); FORCE_STEPS is a comma-separated list of step names or
# shell globs (r2_*). Outside TEST_MODE it refuses (exit 1, logs archived): a J entry (the section of
# docs/PREREGISTRATION.md whose heading contains P-2026-10-10-J) that still says "DRAFT, not yet final" (awk alone
# decides: an awk | grep -q pipeline under pipefail gets status 141 when grep exits first, and the refusal is skipped);
# modified tracked files; a history without a commit whose subject is exactly "Finalise preregistration J"; code
# (ckeys experiments analysis scripts tests data) that differs between HEAD and that commit; a J section that differs
# from its text at that commit; no CUDA device with >= MINGIB GiB (default 75). It pins transformers 5.18.0 (pip
# install only when the version or a needed package differs, as gpu_stage7.sh) and writes COMMIT.txt, ENV.txt,
# PIP_FREEZE.txt and REVISIONS.txt (appended to on a rerun). OUT=results/gpu_stage8<part>, DEV=cuda, DTYPE=bfloat16.
# TEST_MODE=1: no guards, no pip, OUT=results/gpu_stage8<part>_test, DEV=cpu, DTYPE=float32, KEEP_CACHE=1, the archive
# is gpu_stage8<part>_test_results.tgz; the part's steps run at Qwen2.5-0.5B-Instruct (FP32, CPU, n = 2-3, outputs
# tagged TEST_). OUT=, TGZ=, PY= (default python), S8_MODELS= (default $HOME/stage8_models) override.
#
# s8_pytest <files...> runs "$PY -m pytest <files> -v -rA" into $OUT/logs/pytest.log (S8_PYTEST_LOG=<name> for another
# log; the scorer reads its last run for the part's G0 gate) before any model; a failure dies. A pass is recorded in
# PYTEST_OK.txt (HEAD, host, file list); outside TEST_MODE, TESTS=0 skips the run only after such a pass in the same OUT.
# s8_step <name> <cmd...> runs cmd with PYTHONPATH=. into $OUT/logs/<name>.log. Success writes $OUT/steps/<name>.done
# (time, HEAD, command); a later session keeps a done step unless FORCE=1 or FORCE_STEPS names it. A failure is
# recorded in $OUT/FAILED.txt ("FAILED <name> exit <rc>") and the pipeline goes on; the step runs again next time.
# Optional prefixes: S8_OUTPUTS="<file> ..." (the step's results files: a file the FAILED step wrote is moved aside to
# <file>.failed.<UTC>, noted in COMMIT.txt, so the scorer does not read it); S8_EXPLORATORY=1 (a failure goes to
# FAILED_EXPLORATORY.txt and does not fail the run). Exit status 3 of a step means "complete up to the deadline, parts
# skipped": no done file, recorded in SKIPPED.txt, so the next session runs the step again; it is not a failure.
# s8_time_left <m> is true when at least m minutes remain before STAGE8_DEADLINE. s8_skip <name> <reason> records the
# skip in $OUT/SKIPPED.txt (a step done earlier is reported as kept instead). The scorer reads SKIPPED.txt.
# s8_fetch <key> runs scripts/fetch_verified.py into $S8_MODELS/<key> (log $OUT/logs/fetch_<key>.log), copies
# VERIFIED.json to $OUT/verified/<key>.json, appends "<repo> <revision> key=... attn=... token=... sources=..." to
# REVISIONS.txt and prints the directory. On a refusal (a file not verified) it prints nothing, records the key in
# FETCH_FAILED.txt and returns 1, so the part script applies the entry's fallback rule (01-ai/Yi-1.5-9B-Chat, key yi9)
# before any output of the refused model exists. Any other failure (not enough disk, a bad manifest) is not a
# verification failure: s8_fetch returns its status and the next s8_step, s8_fetch or s8_finish dies with the reason,
# so no fallback runs. A background prefetch (s8_fetch <key> > /dev/null &) is safe: the fetcher locks the directory.
# s8_attn <key> prints the key's attention implementation (eager for Gemma-2, sdpa otherwise). s8_done <name> is true
# when the step is done and not forced, so a rerun need not fetch the weights of a model whose steps are all kept.
# s8_drop <key> deletes $S8_MODELS/<key> unless KEEP_CACHE=1, TEST_MODE, or the key was verified there before this
# session's first fetch of it (as stage 7 kept a model cached before the run).
# s8_finish <scorer> runs "$PY <scorer> --results $OUT" (plus --test in TEST_MODE) into logs/score.log; a scorer exit
# status other than 0 is a FAILED step (the score file is still archived); prints the GATES and SUMMARY blocks of
# $OUT/STAGE8<PART>_SCORE.txt; writes $OUT/MANIFEST.sha256 (every file under OUT); writes $TGZ in the repo root
# (gpu_stage8<part>_results.tgz, no weights); returns 1 if $OUT/FAILED.txt exists, else 0.
set -uo pipefail
S8_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
S8_ENTRY=P-2026-10-10-J
S8_FINAL_SUBJECT="Finalise preregistration J"
S8_TINY=Qwen/Qwen2.5-0.5B-Instruct
S8_CODE_DIRS="ckeys experiments analysis scripts tests data"

on() { [ "${!1:-0}" = 1 ]; }   # a switch is on only when it is 1
s8_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
s8_stamp() { date -u +%Y%m%dT%H%M%SZ; }

die() {  # fatal: record, archive what there is, stop
  if [ -n "${OUT:-}" ]; then
    mkdir -p "$OUT"; echo "FAILED $1" | tee -a "$OUT/FAILED.txt"
    if [ -n "${TGZ:-}" ]; then s8_tar; echo "Results archive (logs): $TGZ"; fi
  else
    echo "FAILED $1"
  fi
  exit 1
}

s8_patterns() { local IFS=,; S8_PATS=(); read -ra S8_PATS <<< "${FORCE_STEPS:-}"; }   # FORCE_STEPS split, never globbed

s8_forced() {  # FORCE=1 redoes every step; FORCE_STEPS the named ones (shell globs allowed, matched against step names)
  on FORCE && return 0
  local p; s8_patterns
  for p in "${S8_PATS[@]}"; do
    # shellcheck disable=SC2053
    [[ "$1" == $p ]] && { echo "$p" >> "$S8_STATE/forced_patterns_matched"; return 0; }
  done
  return 1
}

s8_init() {
  cd "$S8_ROOT" || { echo "cannot cd to $S8_ROOT"; exit 2; }
  case ${PART:-} in a|b|c|d) ;; *) echo "bad PART='${PART:-}' (a, b, c or d)"; exit 2;; esac
  [[ "${S8_DEADLINE_H_DEFAULT:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "S8_DEADLINE_H_DEFAULT (hours) must be set before s8_init"; exit 2; }
  local v; for v in TEST_MODE KEEP_CACHE FORCE TESTS; do case ${!v:-} in ""|0|1) ;; *) echo "bad $v='${!v}' (0 or 1)"; exit 2;; esac; done
  DEADLINE_H=${DEADLINE_H:-$S8_DEADLINE_H_DEFAULT}
  [[ "$DEADLINE_H" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v h="$DEADLINE_H" 'BEGIN{exit !(h > 0)}' \
    || { echo "bad DEADLINE_H='$DEADLINE_H' (a positive number of hours)"; exit 2; }
  FORCE_STEPS=${FORCE_STEPS:-}
  local s; s8_patterns; for s in "${S8_PATS[@]}"; do [[ "$s" =~ ^[A-Za-z0-9_.*?-]+$ ]] || { echo "bad FORCE_STEPS entry '$s'"; exit 2; }; done
  PY=${PY:-python}
  $PY -c "import sys; assert sys.version_info >= (3, 10)" 2> /dev/null || { echo "PY=$PY is not a Python >= 3.10"; exit 2; }
  PART_UC=$(printf '%s' "$PART" | tr '[:lower:]' '[:upper:]')
  export STAGE8_DEADLINE=$(( $(date +%s) + $(awk -v h="$DEADLINE_H" 'BEGIN{printf "%d", h * 3600}') ))   # from the start, setup included
  TEST_MODE=${TEST_MODE:-0}; [ -n "$TEST_MODE" ] || TEST_MODE=0
  if on TEST_MODE; then
    export TEST_MODE=1 KEEP_CACHE=1
    OUT=${OUT:-results/gpu_stage8${PART}_test}; TGZ=${TGZ:-gpu_stage8${PART}_test_results.tgz}; DEV=cpu; DTYPE=float32
  else
    export -n TEST_MODE   # the steps see TEST_MODE only when it is 1
    OUT=${OUT:-results/gpu_stage8${PART}}; TGZ=${TGZ:-gpu_stage8${PART}_results.tgz}; DEV=cuda; DTYPE=bfloat16
  fi
  S8_MODELS=${S8_MODELS:-$HOME/stage8_models}
  S8_STATE=$(mktemp -d "${TMPDIR:-/tmp}/s8state.XXXXXX") || exit 2   # per-session marks (subshell-safe)
  mkdir -p "$OUT/logs" "$OUT/steps" || exit 2
  export GIT_TERMINAL_PROMPT=0 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
  export HF_XET_CHUNK_CACHE_SIZE_BYTES=${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}   # no second copy of the weights in a download cache
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  local f; for f in FAILED FAILED_EXPLORATORY FETCH_FAILED; do [ -f "$OUT/$f.txt" ] && mv "$OUT/$f.txt" "$OUT/$f.$(s8_stamp).txt"; done
  { echo "==== $(s8_utc) stage 8 part $PART TEST_MODE=$TEST_MODE FORCE=${FORCE:-0} FORCE_STEPS=$FORCE_STEPS DEADLINE_H=$DEADLINE_H TESTS=${TESTS:-1} KEEP_CACHE=${KEEP_CACHE:-0}"
    git rev-parse HEAD; git status --short; } 2>&1 | tee -a "$OUT/COMMIT.txt"

  S8_J=
  if ! on TEST_MODE; then  # the preregistered code only: the finalised entry, no local changes, the code of the finalising commit
    awk '/^## / && /P-2026-10-10-J/{e = 1} END{exit !e}' docs/PREREGISTRATION.md \
      || die "no entry $S8_ENTRY in docs/PREREGISTRATION.md: check out the commit '$S8_FINAL_SUBJECT' (docs/GPU_RUNBOOK.md)"
    awk '/^## /{f = ($0 ~ /P-2026-10-10-J/)} f && /DRAFT, not yet final/{d = 1} END{exit !d}' docs/PREREGISTRATION.md \
      && die "preregistration $S8_ENTRY is still a DRAFT: check out the commit '$S8_FINAL_SUBJECT' (docs/GPU_RUNBOOK.md)"
    # (awk decides alone: an awk | grep -q pipeline under pipefail gets status 141 when grep exits first, and the refusal is skipped)
    git rev-parse HEAD > /dev/null 2>&1 && [ -z "$(git status --porcelain --untracked-files=no)" ] \
      || die "not a clean git checkout (modified tracked files above): run from a clean checkout of '$S8_FINAL_SUBJECT'"
    S8_J=$(git log --format='%H%x09%s' | awk -F'\t' '$2 == "Finalise preregistration J" && h == "" {h = $1} END {print h}')
    [ -n "$S8_J" ] || die "no commit with the subject '$S8_FINAL_SUBJECT' in the history of HEAD (docs/GPU_RUNBOOK.md)"
    # shellcheck disable=SC2086
    git diff --quiet "$S8_J" HEAD -- $S8_CODE_DIRS \
      || die "the code at HEAD ($S8_CODE_DIRS) differs from the commit '$S8_FINAL_SUBJECT' ($S8_J): check that commit out (docs/GPU_RUNBOOK.md)"
    [ "$(git show "$S8_J:docs/PREREGISTRATION.md" | s8_jsection)" = "$(s8_jsection < docs/PREREGISTRATION.md)" ] \
      || die "the $S8_ENTRY section of docs/PREREGISTRATION.md differs from its text at '$S8_FINAL_SUBJECT' ($S8_J)"
    echo "preregistration $S8_ENTRY final at $S8_J; HEAD $(git rev-parse HEAD)" | tee -a "$OUT/COMMIT.txt"
  fi

  # ---- software environment of stages 1, 3b and 5-7
  local logenv="$OUT/logs/env.log"
  if ! on TEST_MODE; then
    if ! $PY -c "import transformers as t, accelerate, numpy, scipy, pytest, huggingface_hub; assert t.__version__ == '5.18.0'" >> "$logenv" 2>&1; then
      $PY -m pip install 'transformers==5.18.0' accelerate numpy scipy pytest >> "$logenv" 2>&1; tail -n 1 "$logenv"
    fi
    $PY -c "import transformers as t; assert t.__version__ == '5.18.0', t.__version__" >> "$logenv" 2>&1 || die "transformers is not 5.18.0 (see $logenv)"
  fi
  { echo "==== $(s8_utc)"; $PY -m pip freeze 2>/dev/null; } >> "$OUT/PIP_FREEZE.txt"
  local man=scripts/stage8_models.json mansha
  mansha=$(sha256sum "$man" 2>/dev/null | cut -c1-64)
  { echo "==== $(s8_utc) part $PART TEST_MODE=$TEST_MODE OUT=$OUT DEV=$DEV DTYPE=$DTYPE deadline $(date -u -d @"$STAGE8_DEADLINE" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "$STAGE8_DEADLINE") (DEADLINE_H=$DEADLINE_H)"
    echo "HEAD $(git rev-parse HEAD 2>/dev/null) finalising commit ${S8_J:-none (TEST_MODE)}"
    echo "manifest $man sha256 ${mansha:-missing}; model store $S8_MODELS; HF_TOKEN set: $([ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ] && echo yes || echo no) (the token itself is never recorded)"
    on TEST_MODE || { mkdir -p "$S8_MODELS" && df -h "$S8_MODELS" | tail -n 1 | awk '{print "disk at the model store: " $4 " free of " $2}'; }
  } >> "$OUT/ENV.txt"
  $PY -c "import sys, torch, transformers, numpy, huggingface_hub as h; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__, 'huggingface_hub', h.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" 2>&1 | tee -a "$OUT/ENV.txt"
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> "$OUT/ENV.txt" 2>&1
  [ -f "$OUT/REVISIONS.txt" ] || echo "# <repo> <revision> key=<key> attn=<impl> token=<used> sources=<where the verified bytes came from>; pins: $man sha256 ${mansha:-missing}" > "$OUT/REVISIONS.txt"
  MINGIB=${MINGIB:-75}
  on TEST_MODE || $PY -c "import sys, torch; sys.exit(0 if torch.cuda.device_count() >= 1 and torch.cuda.get_device_properties(0).total_memory / 2**30 >= float(sys.argv[1]) else 1)" "$MINGIB" \
    || die "no CUDA device with >= $MINGIB GiB visible to torch"
  echo "==================== stage 8 part $PART: OUT=$OUT DEV=$DEV DTYPE=$DTYPE TEST_MODE=$TEST_MODE deadline in $DEADLINE_H h"
}

s8_pytest() {  # s8_pytest <test files...>: the FP32 gates before any model; dies on a failure; TESTS=0 skips (logged)
  local name=${S8_PYTEST_LOG:-pytest} okid
  [ $# -gt 0 ] || die "s8_pytest: no test files given"
  okid="$(git rev-parse HEAD 2>/dev/null) $(hostname 2>/dev/null || uname -n) $(printf '%s\n' "$@" | sha256sum | cut -c1-16)"
  if [ "${TESTS:-1}" = 1 ]; then
    if [ -f "$OUT/PYTEST_OK.txt" ]; then grep -vxF "$okid" "$OUT/PYTEST_OK.txt" > "$OUT/PYTEST_OK.txt.tmp"; mv "$OUT/PYTEST_OK.txt.tmp" "$OUT/PYTEST_OK.txt"; fi
    s8_run "$name" $PY -m pytest "$@" -v -rA -p no:cacheprovider || die "FP32 unit tests ($*): see $OUT/logs/$name.log"
    echo "$okid" >> "$OUT/PYTEST_OK.txt"
  elif ! on TEST_MODE; then   # the entry requires the FP32 tests before any model: skip them only after a pass of this HEAD here
    grep -qxF "$okid" "$OUT/PYTEST_OK.txt" 2>/dev/null \
      || die "TESTS=0 needs a passing pytest of these files at this HEAD on this host in $OUT (PYTEST_OK.txt); run with TESTS=1"
    echo "$(s8_utc) TESTS=0: pytest of $* passed earlier in $OUT ($okid)" | tee -a "$OUT/COMMIT.txt"
  else
    echo "$(s8_utc) TESTS=0 (TEST_MODE): pytest of $* not run" | tee -a "$OUT/COMMIT.txt"
  fi
}

s8_tar() {  # the archive of OUT, no weights (an absolute OUT is stored without its leading /, as tar does, silently)
  if [[ "$OUT" = /* ]]; then tar czf "$TGZ" --exclude='*.safetensors' --exclude='*.bin' -C / "${OUT#/}"
  else tar czf "$TGZ" --exclude='*.safetensors' --exclude='*.bin' "$OUT"; fi
}

s8_jsection() { awk '/^## /{f = ($0 ~ /^## P-2026-10-10-J/)} f'; }   # the J entry's own section (stdin)

s8_run() {  # s8_run <name> <cmd...>: run into $OUT/logs/<name>.log; a failure goes to FAILED.txt (FAILED_EXPLORATORY.txt)
  local name=$1 rc=0 ff="$OUT/FAILED.txt" log="$OUT/logs/$1.log"; shift
  on S8_EXPLORATORY && ff="$OUT/FAILED_EXPLORATORY.txt"
  echo "==================== $name  $(date -u +%H:%M:%S)"
  echo "==== $(s8_utc) $*" >> "$log"
  PYTHONPATH=. "$@" >> "$log" 2>&1 || rc=$?
  if [ $rc = 3 ] && [ -n "${S8_IN_STEP:-}" ]; then
    echo "==== $(s8_utc) exit 3: complete up to the deadline, parts skipped" >> "$log"
  elif [ $rc != 0 ]; then
    local why="exit $rc"; [ $rc -gt 128 ] && why+=" (killed by signal $((rc - 128)); 9 = SIGKILL, often the host OOM killer)"
    echo "==== $(s8_utc) FAILED $why" >> "$log"
    echo "FAILED $name $why (see $log)" | tee -a "$ff"
  fi
  tail -n 2 "$log"
  return $rc
}

s8_check() { [ ! -s "${S8_STATE:-/nonexistent}/fatal" ] || die "$(cat "$S8_STATE/fatal")"; }   # a fatal fetch error stops the run here

s8_step() {  # s8_step <name> <cmd...>: keep a step done earlier, else run it (see the header)
  local name=$1 rc=0 done_f stamp f; shift
  s8_check
  [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || die "bad step name '$name'"
  [ $# -gt 0 ] || die "s8_step $name: no command"
  done_f="$OUT/steps/$name.done"
  echo "$name" >> "$S8_STATE/steps_seen"
  if [ -f "$done_f" ]; then
    if ! s8_forced "$name"; then
      echo "==================== $name kept: done $(head -n 1 "$done_f") (FORCE=1 or FORCE_STEPS=$name to redo)"
      return 0
    fi
    mv "$done_f" "$done_f.redone.$(s8_stamp)"; echo "$(s8_utc) $name redone (FORCE=${FORCE:-0} FORCE_STEPS=$FORCE_STEPS)" >> "$OUT/COMMIT.txt"
  fi
  stamp="$S8_STATE/stamp_$name"; touch "$stamp"; sleep 1   # a file older than the stamp was not written by this step
  S8_IN_STEP=1 s8_run "$name" "$@" || rc=$?
  if [ $rc = 0 ]; then
    { echo "$(s8_utc) HEAD $(git rev-parse HEAD 2>/dev/null)"; printf '%q ' "$@"; echo; } > "$done_f"
  elif [ $rc = 3 ]; then
    echo "$(s8_utc) $name partial: complete up to the deadline, parts skipped (the step runs again next session)" | tee -a "$OUT/SKIPPED.txt"
    rc=0
  else
    for f in ${S8_OUTPUTS:-}; do
      if [ -e "$f" ] && [ "$f" -nt "$stamp" ]; then
        local old; old="$f.failed.$(s8_stamp)"; mv "$f" "$old"
        echo "moved aside $old: written by the FAILED step $name (exit $rc); the scorer does not read it" | tee -a "$OUT/COMMIT.txt"
      fi
    done
  fi
  rm -f "$stamp"
  return $rc
}

s8_done() {  # s8_done <name>: true when the step is done and neither FORCE nor FORCE_STEPS redoes it (e.g. skip a fetch)
  [ -f "$OUT/steps/$1.done" ] && ! s8_forced "$1"
}

s8_time_left() {  # s8_time_left <minutes>: true when at least that many minutes remain before STAGE8_DEADLINE
  [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "s8_time_left needs a number of minutes, got '${1:-}'"
  awk -v now="$(date +%s)" -v m="$1" -v d="$STAGE8_DEADLINE" 'BEGIN{exit !(now + m * 60 <= d)}'
}

s8_skip() {  # s8_skip <name> <reason>: record a step not run (a step done earlier stays done and is not a skip)
  local name=$1 why=${2:-no reason given}
  if [ -f "$OUT/steps/$name.done" ]; then
    echo "==================== $name already done $(head -n 1 "$OUT/steps/$name.done"); not run again ($why)"
    echo "$(s8_utc) $name done earlier, not run again: $why" >> "$OUT/COMMIT.txt"; return 0
  fi
  echo "==================== $name SKIPPED: $why"
  echo "$(s8_utc) $name SKIPPED: $why" >> "$OUT/SKIPPED.txt"
  return 0
}

s8_attn() {  # s8_attn <key>: the key's attention implementation from the manifest
  $PY -c "import json, sys; print(json.load(open('scripts/stage8_models.json'))['models'][sys.argv[1]]['attn'])" "$1"
}

s8_fetch() {  # s8_fetch <key>: print the local directory of the key's verified files (TEST_MODE: the 0.5B hub id)
  local key=$1 dir log rc=0 line
  s8_check
  log="$OUT/logs/fetch_$key.log"
  if on TEST_MODE; then
    echo "==== $(s8_utc) TEST_MODE: $key -> $S8_TINY (from the HF cache or the hub; not verified)" >> "$log"
    grep -q " key=$key TEST_MODE" "$OUT/REVISIONS.txt" 2>/dev/null || echo "$S8_TINY (hub) key=$key TEST_MODE" >> "$OUT/REVISIONS.txt"
    echo "$S8_TINY"; return 0
  fi
  dir="$S8_MODELS/$key"
  # verified there before this session's first fetch of the key: s8_drop keeps it
  [ -e "$S8_STATE/seen_$key" ] || { touch "$S8_STATE/seen_$key"; [ -f "$dir/VERIFIED.json" ] && touch "$S8_STATE/pre_$key"; }
  echo "==== $(s8_utc) fetch $key -> $dir" >> "$log"
  PYTHONPATH=. $PY scripts/fetch_verified.py --key "$key" --dest "$dir" >> "$log" 2>&1 || rc=$?
  mkdir -p "$OUT/verified"
  if [ $rc = 1 ]; then   # a verification failure: the entry's fallback rule applies
    [ -f "$dir/VERIFY_FAILED.json" ] && cp "$dir/VERIFY_FAILED.json" "$OUT/verified/$key.FAILED.$(s8_stamp).json"
    echo "$(s8_utc) FETCH REFUSED $key (exit 1; see $log)" | tee -a "$OUT/FETCH_FAILED.txt" >&2
    return 1
  elif [ $rc != 0 ]; then   # disk, manifest or setup: not a verification failure; the next library call stops the run
    echo "fetch of $key: exit $rc ($([ $rc = 4 ] && echo 'not enough disk at '"$dir" || echo 'manifest or setup error')); not a verification failure, no fallback (see $log)" > "$S8_STATE/fatal"
    cat "$S8_STATE/fatal" >&2
    return $rc
  fi
  cp "$dir/VERIFIED.json" "$OUT/verified/$key.json"
  line=$($PY -c "import json, re, sys; j = json.load(open(sys.argv[1])); src = sorted({m for r in j['files'].values() for m in re.findall(r'[\w.-]+/[\w.-]+@[0-9a-f]{10}', r['source'])}); print(j['repo'], j['revision'], 'key=' + j['key'], 'attn=' + j['attn'], 'token=' + ('yes' if j['token_used'] else 'no'), 'sources=' + ','.join(src))" "$dir/VERIFIED.json")
  grep -qxF "$line" "$OUT/REVISIONS.txt" 2>/dev/null || echo "$line" >> "$OUT/REVISIONS.txt"
  echo "$dir"
}

s8_drop() {  # s8_drop <key>: delete the key's verified files unless KEEP_CACHE=1 (or TEST_MODE, or verified before this session)
  local key=$1 dir="$S8_MODELS/$1"
  on TEST_MODE && return 0
  if on KEEP_CACHE; then echo "$(s8_utc) $key kept in $dir (KEEP_CACHE=1)" >> "$OUT/ENV.txt"; return 0; fi
  if [ -e "$S8_STATE/pre_$key" ]; then echo "$(s8_utc) $key kept in $dir (verified there before this session)" >> "$OUT/ENV.txt"; return 0; fi
  if [ -d "$dir" ]; then rm -rf "$dir" && echo "$(s8_utc) $key removed from $dir" >> "$OUT/ENV.txt"; fi
  return 0
}

s8_finish() {  # s8_finish <scorer>: score, print GATES and SUMMARY, MANIFEST.sha256, archive; status 1 if any step FAILED
  local scorer=$1 score_f="$OUT/STAGE8${PART_UC}_SCORE.txt" p
  s8_check
  local args=(--results "$OUT"); on TEST_MODE && args+=(--test)
  s8_run score $PY "$scorer" "${args[@]}"
  if [ -f "$score_f" ]; then
    awk '{ s = $0; sub(/^[#= \t-]+/, "", s) }
         s ~ /^(PROVENANCE|POPULATION|GATES|PREDICTIONS|VERDICTS|REPORTED|SUMMARY|EXPLORATORY)([^A-Z_]|$)/ { sec = s; sub(/[^A-Z_].*$/, "", sec) }
         sec == "GATES" || sec == "SUMMARY"' "$score_f" | cut -c1-400
  else
    echo "no score file $score_f (see $OUT/logs/score.log)"
  fi
  s8_patterns
  for p in "${S8_PATS[@]}"; do
    grep -qxF "$p" "$S8_STATE/forced_patterns_matched" 2>/dev/null || echo "note: FORCE_STEPS entry '$p' matched no step of this session" | tee -a "$OUT/COMMIT.txt"
  done
  ( cd "$OUT" && find . -type f ! -name MANIFEST.sha256 ! -name 'MANIFEST.sha256.tmp' -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum ) > "$OUT/MANIFEST.sha256.tmp" \
    && mv "$OUT/MANIFEST.sha256.tmp" "$OUT/MANIFEST.sha256"
  s8_tar
  echo "==================== DONE  $(date -u +%H:%M:%S)"
  echo "Results archive: $(cd "$(dirname "$TGZ")" && pwd)/$(basename "$TGZ")"
  [ -f "$OUT/SKIPPED.txt" ] && { echo "skipped or partial at a deadline (a later session may fill them; not failures):"; cat "$OUT/SKIPPED.txt"; }
  [ -f "$OUT/FETCH_FAILED.txt" ] && { echo "model files refused by the verification (see REVISIONS.txt for what ran instead):"; cat "$OUT/FETCH_FAILED.txt"; }
  [ -f "$OUT/FAILED_EXPLORATORY.txt" ] && { echo "exploratory step failed (not a pipeline failure):"; cat "$OUT/FAILED_EXPLORATORY.txt"; }
  if [ -f "$OUT/FAILED.txt" ]; then cat "$OUT/FAILED.txt"; return 1; fi
  echo "no failures (no step FAILED; gate and prediction verdicts above are results, see $score_f)"
  return 0
}
