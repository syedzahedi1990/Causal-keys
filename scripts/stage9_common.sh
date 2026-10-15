# shellcheck shell=bash
# Stage-9 library of scripts/gpu_stage9.sh (docs/stage9/SPEC.md section 12; predictions docs/STAGE9_PREDICTIONS.md).
# Sourced after scripts/stage8_common.sh, whose helpers it uses (s8_step, s8_run, s8_fetch, s8_prefetch, s8_drop,
# s8_pytest, s8_time_left, s8_skip, s8_done, s8_check, die, s8_tar); s8_init and s8_finish are not called. The script does:
#   source scripts/stage8_common.sh; source scripts/stage9_common.sh; s9_init
#   ... s8_pytest / s8_fetch / s8_step ... (as stage 8)
#   s9_finish analysis/stage9_score.py
#
# s9_init replaces s8_init (which hard-codes the stage-8 parts and the guard of preregistration J). It checks the switches
# (TEST_MODE, KEEP_CACHE, FORCE, TESTS, TEST_ALL: 0 or 1; DEADLINE_H a positive number of hours, default 4.5; FORCE_STEPS
# step names or shell globs) and sets PART=9, PART_UC=9, OUT (results/gpu_stage9; TEST_MODE results/gpu_stage9_test), TGZ
# (gpu_stage9_results.tgz / gpu_stage9_test_results.tgz), PY (default python), DEV and DTYPE (cuda bfloat16; TEST_MODE cpu
# float32), S8_MODELS (default $HOME/stage8_models), S8_STATE, STAGE8_DEADLINE (epoch s from the start, exported: the
# steps stop before work that would end inside it minus their --reserve-min), KEEP_CACHE (1 in TEST_MODE), FORCE,
# FORCE_STEPS, TESTS. It moves an earlier session's FAILED.txt, FAILED_EXPLORATORY.txt, FETCH_FAILED.txt and SKIPPED.txt
# aside, appends the session header to COMMIT.txt and writes ENV.txt, PIP_FREEZE.txt and the REVISIONS.txt header.
# Outside TEST_MODE it calls s9_guard first; TEST_MODE: no guard, no pip, and STAGE9_FREEZE is unset (the results files
# then record freeze_commit null).
#
# s9_guard refuses (die: FAILED.txt, logs archived, exit 1):
#   0. results under OUT tracked by git (results are never committed between sessions: the scorer wants one commit and
#      a clean tree across the results files);
#   1. a tree with modified tracked files, or untracked (not ignored) files in the code directories (code a run would
#      import that is not in F);
#   2. a history of HEAD without a commit whose subject is exactly "Freeze stage-9 predictions" (the most recent such
#      commit is F, the freeze commit);
#   3. docs/STAGE9_PREDICTIONS.md or docs/stage9/SPEC.md absent at HEAD, or different from its text at F;
#   4. code (ckeys experiments analysis scripts tests data) that differs between F and HEAD;
#   4b. results files already in OUT (an earlier session's) written at a commit other than HEAD (s9_other_commits): a
#      later session resumes on the same HEAD, or the scorer would see two commits and score nothing;
#   5. transformers other than 5.18.0 (pip install only when the version or a needed package differs, as s8_init), or no
#      CUDA device with >= MINGIB GiB (default 75).
# Checks 1-4 are s9_git_checks, which needs no CUDA (tests/test_stage9_guard.py runs it in temporary repositories). On
# a pass s9_guard appends "stage-9 predictions frozen at F (commit date); note sha256 ...; spec sha256 ...; HEAD ..." to
# COMMIT.txt (the scorer compares every results file's freeze_commit, predictions_sha256 and spec_sha256 with it) and
# exports STAGE9_FREEZE=F, STAGE9_NOTE_SHA256 and STAGE9_SPEC_SHA256 (ckeys.stage9.run_meta records them).
#
# s9_lb_stale <tag>: true when OUT/stories/<tag>.json exists and records another lb sha256 than OUT/lb/<tag>.json has
# (the cal step reran after the eval, e.g. FORCE_STEPS=cal_<key>): gpu_stage9.sh then redoes st_<key> (the eval reruns
# every format against the new lb file) instead of keeping a stories file the scorer would refuse.
#
# s9_finish <scorer> = s8_finish with the score file STAGE9_SCORE.txt: runs "$PY <scorer> --results $OUT" (--test in
# TEST_MODE) into logs/score.log (a scorer exit status other than 0 is a FAILED step; the score file is still archived),
# prints the CHECKS, PREDICTIONS, FALSIFIERS and SUMMARY sections, writes MANIFEST.sha256 (every file under OUT) and the
# archive $TGZ (no weights); returns 1 if FAILED.txt exists, else 0.
set -uo pipefail
S9_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
S9_FREEZE_SUBJECT="Freeze stage-9 predictions"
S9_NOTE=docs/STAGE9_PREDICTIONS.md
S9_SPEC=docs/stage9/SPEC.md
S9_CODE_DIRS="ckeys experiments analysis scripts tests data"

s9_git_checks() {  # s9_git_checks [dir]: checks 1-4 in the repository at dir (default the repository of this file)
  # Success: prints "F<TAB>commit date of F<TAB>note sha256<TAB>spec sha256<TAB>HEAD" and returns 0.
  # Refusal: prints the reason on stderr and returns 1.
  local dir=${1:-$S9_ROOT} F head f fdate
  head=$(git -C "$dir" rev-parse HEAD 2> /dev/null) || { echo "not a git checkout with a commit: $dir" >&2; return 1; }
  if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
    echo "modified tracked files (git status --porcelain --untracked-files=no is not empty): run from a clean checkout" >&2
    return 1
  fi
  # untracked, not ignored files in the code directories: a run would import code that is not in the freeze commit
  # shellcheck disable=SC2086
  f=$(git -C "$dir" status --porcelain --untracked-files=all -- $S9_CODE_DIRS)
  if [ -n "$f" ]; then
    echo "untracked files in the code directories ($S9_CODE_DIRS): $(echo "$f" | head -n 5 | cut -c4- | tr '\n' ' ')- commit them before the freeze or remove them" >&2
    return 1
  fi
  # the most recent commit in the history of HEAD whose subject is exactly the freeze subject (awk alone decides)
  F=$(git -C "$dir" log --format='%H%x09%s' HEAD | awk -F'\t' -v s="$S9_FREEZE_SUBJECT" '$2 == s && h == "" {h = $1} END {print h}')
  if [ -z "$F" ]; then
    echo "no commit with the subject '$S9_FREEZE_SUBJECT' in the history of HEAD: the predictions are not frozen" >&2
    return 1
  fi
  for f in "$S9_NOTE" "$S9_SPEC"; do
    if ! git -C "$dir" cat-file -e "HEAD:$f" 2> /dev/null; then
      echo "$f is absent at HEAD" >&2; return 1
    fi
    if ! git -C "$dir" cat-file -e "$F:$f" 2> /dev/null || ! git -C "$dir" diff --quiet "$F" HEAD -- "$f"; then
      echo "$f differs from its text at the freeze commit $F" >&2; return 1
    fi
  done
  # shellcheck disable=SC2086
  if ! git -C "$dir" diff --quiet "$F" HEAD -- $S9_CODE_DIRS; then
    echo "the code at HEAD ($S9_CODE_DIRS) differs from the freeze commit $F:" >&2
    # shellcheck disable=SC2086
    git -C "$dir" diff --stat "$F" HEAD -- $S9_CODE_DIRS | tail -n 5 >&2
    return 1
  fi
  fdate=$(git -C "$dir" log -1 --format=%cI "$F")
  printf '%s\t%s\t%s\t%s\t%s\n' "$F" "$fdate" "$(sha256sum "$dir/$S9_NOTE" | cut -c1-64)" \
    "$(sha256sum "$dir/$S9_SPEC" | cut -c1-64)" "$head"
}

# the git_commit of every results file under OUT (the files the scorer reads) that differs from argv[2], one per line
S9_PY_COMMITS='
import json, sys
from pathlib import Path
out, head = Path(sys.argv[1]), sys.argv[2]
for d in ("check", "lb", "stories", "edits", "passages"):
    for f in sorted((out / d).glob("*.json")) if (out / d).is_dir() else []:
        try:
            c = json.loads(f.read_text())["provenance"].get("git_commit")
        except Exception:
            continue
        if c and c != head:
            print(c, f"{d}/{f.name}")
'

s9_other_commits() {  # s9_other_commits <head>: the results files in OUT written at another commit ("<commit> <file>")
  $PY -c "$S9_PY_COMMITS" "$OUT" "$1"
}

s9_lb_stale() {  # s9_lb_stale <tag>: see the header
  local st="$OUT/stories/$1.json" lb="$OUT/lb/$1.json" rec
  [ -f "$st" ] && [ -f "$lb" ] || return 1
  rec=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['provenance'].get('lb_sha256') or '')" "$st" 2> /dev/null) \
    || return 0   # an unreadable stories file is redone too
  [ "$rec" != "$(sha256sum "$lb" | cut -c1-64)" ]
}

s9_guard() {  # checks 0-5; records the freeze in COMMIT.txt and exports STAGE9_FREEZE, STAGE9_NOTE_SHA256, STAGE9_SPEC_SHA256
  local info err fdate head other tracked logenv="$OUT/logs/env.log"
  err="$S8_STATE/s9_guard.err"
  tracked=$(git -C "$S9_ROOT" ls-files -- "$OUT" 2> /dev/null | head -n 3 | tr '\n' ' ')
  [ -z "$tracked" ] || die "stage-9 guard: results under $OUT are tracked by git (${tracked}...): never commit results between sessions; resume on the same checkout with the earlier results unpacked and untracked (docs/GPU_RUNBOOK.md, Stage 9)"
  info=$(s9_git_checks "$S9_ROOT" 2> "$err") || die "stage-9 guard: $(cat "$err") (docs/GPU_RUNBOOK.md, Stage 9)"
  IFS=$'\t' read -r STAGE9_FREEZE fdate STAGE9_NOTE_SHA256 STAGE9_SPEC_SHA256 head <<< "$info"
  other=$(s9_other_commits "$head") || die "stage-9 guard: cannot read the results files in $OUT"
  [ -z "$other" ] || die "stage-9 guard: results files in $OUT were written at another commit than HEAD $head ($(echo "$other" | head -n 3 | tr '\n' ';')): the scorer wants one commit across the results files; check out that commit (code and note are frozen, so it passed this guard), or start a new OUT (docs/GPU_RUNBOOK.md, Stage 9)"
  export STAGE9_FREEZE STAGE9_NOTE_SHA256 STAGE9_SPEC_SHA256
  echo "stage-9 predictions frozen at $STAGE9_FREEZE ($fdate); note sha256 $STAGE9_NOTE_SHA256; spec sha256 $STAGE9_SPEC_SHA256; HEAD $head" \
    | tee -a "$OUT/COMMIT.txt"
  if ! $PY -c "import transformers as t, accelerate, numpy, scipy, pytest, huggingface_hub; assert t.__version__ == '5.18.0'" >> "$logenv" 2>&1; then
    $PY -m pip install 'transformers==5.18.0' accelerate numpy scipy pytest >> "$logenv" 2>&1; tail -n 1 "$logenv"
  fi
  $PY -c "import transformers as t; assert t.__version__ == '5.18.0', t.__version__" >> "$logenv" 2>&1 || die "transformers is not 5.18.0 (see $logenv)"
  MINGIB=${MINGIB:-75}
  $PY -c "import sys, torch; sys.exit(0 if torch.cuda.device_count() >= 1 and torch.cuda.get_device_properties(0).total_memory / 2**30 >= float(sys.argv[1]) else 1)" "$MINGIB" \
    || die "no CUDA device with >= $MINGIB GiB visible to torch"
}

s9_init() {
  cd "$S9_ROOT" || { echo "cannot cd to $S9_ROOT"; exit 2; }
  PART=9; PART_UC=9
  local v s f
  for v in TEST_MODE KEEP_CACHE FORCE TESTS TEST_ALL; do case ${!v:-} in ""|0|1) ;; *) echo "bad $v='${!v}' (0 or 1)"; exit 2;; esac; done
  DEADLINE_H=${DEADLINE_H:-4.5}
  [[ "$DEADLINE_H" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v h="$DEADLINE_H" 'BEGIN{exit !(h > 0)}' \
    || { echo "bad DEADLINE_H='$DEADLINE_H' (a positive number of hours)"; exit 2; }
  FORCE_STEPS=${FORCE_STEPS:-}
  s8_patterns; for s in "${S8_PATS[@]}"; do [[ "$s" =~ ^[A-Za-z0-9_.*?-]+$ ]] || { echo "bad FORCE_STEPS entry '$s'"; exit 2; }; done
  PY=${PY:-python}
  $PY -c "import sys; assert sys.version_info >= (3, 10)" 2> /dev/null || { echo "PY=$PY is not a Python >= 3.10"; exit 2; }
  export STAGE8_DEADLINE=$(( $(date +%s) + $(awk -v h="$DEADLINE_H" 'BEGIN{printf "%d", h * 3600}') ))   # from the start, setup included
  TEST_MODE=${TEST_MODE:-0}; [ -n "$TEST_MODE" ] || TEST_MODE=0
  if on TEST_MODE; then
    export TEST_MODE=1 KEEP_CACHE=1
    unset STAGE9_FREEZE STAGE9_NOTE_SHA256 STAGE9_SPEC_SHA256
    OUT=${OUT:-results/gpu_stage9_test}; TGZ=${TGZ:-gpu_stage9_test_results.tgz}; DEV=cpu; DTYPE=float32
  else
    export -n TEST_MODE   # the steps see TEST_MODE only when it is 1
    OUT=${OUT:-results/gpu_stage9}; TGZ=${TGZ:-gpu_stage9_results.tgz}; DEV=cuda; DTYPE=bfloat16
  fi
  S8_MODELS=${S8_MODELS:-$HOME/stage8_models}
  S8_STATE=$(mktemp -d "${TMPDIR:-/tmp}/s9state.XXXXXX") || exit 2   # per-session marks (subshell-safe)
  mkdir -p "$OUT/logs" "$OUT/steps" || exit 2
  export GIT_TERMINAL_PROMPT=0 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
  export HF_XET_CHUNK_CACHE_SIZE_BYTES=${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  for f in FAILED FAILED_EXPLORATORY FETCH_FAILED SKIPPED; do [ -f "$OUT/$f.txt" ] && mv "$OUT/$f.txt" "$OUT/$f.$(s8_stamp).txt"; done
  { echo "==== $(s8_utc) stage 9 TEST_MODE=$TEST_MODE FORCE=${FORCE:-0} FORCE_STEPS=$FORCE_STEPS DEADLINE_H=$DEADLINE_H TESTS=${TESTS:-1} KEEP_CACHE=${KEEP_CACHE:-0} TEST_ALL=${TEST_ALL:-0}"
    git rev-parse HEAD; git status --short; } 2>&1 | tee -a "$OUT/COMMIT.txt"
  on TEST_MODE || s9_guard
  { echo "==== $(s8_utc)"; $PY -m pip freeze 2> /dev/null; } >> "$OUT/PIP_FREEZE.txt"
  local man=scripts/stage8_models.json mansha
  mansha=$(sha256sum "$man" 2> /dev/null | cut -c1-64)
  { echo "==== $(s8_utc) stage 9 TEST_MODE=$TEST_MODE OUT=$OUT DEV=$DEV DTYPE=$DTYPE deadline $(date -u -d @"$STAGE8_DEADLINE" +%Y-%m-%dT%H:%M:%SZ 2> /dev/null || echo "$STAGE8_DEADLINE") (DEADLINE_H=$DEADLINE_H)"
    echo "HEAD $(git rev-parse HEAD 2> /dev/null) freeze commit ${STAGE9_FREEZE:-none (TEST_MODE)}"
    echo "manifest $man sha256 ${mansha:-missing}; model store $S8_MODELS; HF_TOKEN set: $([ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ] && echo yes || echo no) (the token itself is never recorded)"
    on TEST_MODE || { mkdir -p "$S8_MODELS" && df -h "$S8_MODELS" | tail -n 1 | awk '{print "disk at the model store: " $4 " free of " $2}'; }
  } >> "$OUT/ENV.txt"
  $PY -c "import sys, torch, transformers, numpy, huggingface_hub as h; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__, 'huggingface_hub', h.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" 2>&1 | tee -a "$OUT/ENV.txt"
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> "$OUT/ENV.txt" 2>&1
  [ -f "$OUT/REVISIONS.txt" ] || echo "# <repo> <revision> key=<key> attn=<impl> token=<used> sources=<where the verified bytes came from>; pins: $man sha256 ${mansha:-missing}" > "$OUT/REVISIONS.txt"
  echo "==================== stage 9: OUT=$OUT DEV=$DEV DTYPE=$DTYPE TEST_MODE=$TEST_MODE deadline in $DEADLINE_H h"
}

s9_finish() {  # s9_finish <scorer>: score, print CHECKS, PREDICTIONS, FALSIFIERS, SUMMARY, MANIFEST.sha256, archive
  local scorer=$1 score_f="$OUT/STAGE9_SCORE.txt" p
  s8_check
  local args=(--results "$OUT"); on TEST_MODE && args+=(--test)
  s8_run score $PY "$scorer" "${args[@]}"
  if [ -f "$score_f" ]; then
    awk '/^#### /{p = ($2 == "CHECKS" || $2 == "PREDICTIONS" || $2 == "FALSIFIERS" || $2 == "SUMMARY")} p' "$score_f" | cut -c1-400
  else
    echo "no score file $score_f (see $OUT/logs/score.log)"
  fi
  s8_patterns
  for p in "${S8_PATS[@]}"; do
    grep -qxF "$p" "$S8_STATE/forced_patterns_matched" 2> /dev/null || echo "note: FORCE_STEPS entry '$p' matched no step of this session" | tee -a "$OUT/COMMIT.txt"
  done
  ( cd "$OUT" && find . -type f ! -name MANIFEST.sha256 ! -name 'MANIFEST.sha256.tmp' -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum ) > "$OUT/MANIFEST.sha256.tmp" \
    && mv "$OUT/MANIFEST.sha256.tmp" "$OUT/MANIFEST.sha256"
  s8_tar
  echo "==================== DONE  $(date -u +%H:%M:%S)"
  echo "Results archive: $(cd "$(dirname "$TGZ")" && pwd)/$(basename "$TGZ")"
  [ -f "$OUT/SKIPPED.txt" ] && { echo "skipped or partial at a deadline (a later session may fill them; not failures):"; cat "$OUT/SKIPPED.txt"; }
  [ -f "$OUT/FETCH_FAILED.txt" ] && { echo "model files refused by the verification (no fallback: the model's predictions are NOT EVALUABLE):"; cat "$OUT/FETCH_FAILED.txt"; }
  [ -f "$OUT/FAILED_EXPLORATORY.txt" ] && { echo "exploratory step failed (not a pipeline failure):"; cat "$OUT/FAILED_EXPLORATORY.txt"; }
  if [ -f "$OUT/FAILED.txt" ]; then cat "$OUT/FAILED.txt"; return 1; fi
  echo "no failures (no step FAILED; the verdicts above are results, see $score_f)"
  return 0
}
