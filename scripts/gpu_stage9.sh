#!/usr/bin/env bash
# GPU stage 9 (docs/stage9/SPEC.md; predictions docs/STAGE9_PREDICTIONS.md, frozen in the commit "Freeze stage-9
# predictions"): the listed candidates set the edit window at the writing token (format x depth), and the key threshold.
# Arms: stories (150 new fresh cores N9; l_b fixed first on the 30 stage-8b calibration cores C), edits and dose (80 new
# canonical-lexicon cores E9), passages (the 185 stage-8a E items, decisive, and the 71 R items, a held-out check).
# Models (official weights, sha256-verified by scripts/fetch_verified.py through s8_fetch; HF_TOKEN optional, Llama from
# the manifest's verified mirrors without one), in this order: mistral7, llama8, qwen7. No fallback model: a key whose
# files are refused is listed in FETCH_FAILED.txt and its predictions are NOT EVALUABLE.
# Guard (scripts/stage9_common.sh, s9_guard): refuses a tree with modified tracked files or untracked files in the code
# directories, a history without the commit "Freeze stage-9 predictions", a note, spec or code (ckeys experiments analysis
# scripts tests data) that differs from that commit, results under OUT that git tracks or that an earlier session wrote at
# another commit than HEAD, transformers other than 5.18.0 and a GPU under MINGIB GiB (default 75); it records the freeze
# commit and the sha256 of the note and the spec in COMMIT.txt and exports them to the steps (every results file records
# them).
# Steps (scripts/stage8_common.sh: each kept once done, FORCE=1 / FORCE_STEPS=<names or globs> redo, logs in
# $OUT/logs/<step>.log, a failed step goes to FAILED.txt and the pipeline goes on; exit 3 of a step = complete up to the
# deadline, recorded in SKIPPED.txt and redone next session):
#   pytest          the FP32 exactness tests of every stage-9 builder and the shared modules they call, before any model
#   per model key:
#     chk_st_<key>  experiments/stage9_stories.py --stage check  (tokenizer only)        -> check/<tag>_stories.json
#     chk_ed_<key>  experiments/stage9_edits.py --stage check    (tokenizer and config)  -> check/<tag>_edits.json
#     chk_ps_<key>  experiments/stage9_passages.py --stage check (tokenizer, SQuAD)      -> check/<tag>_passages.json
#     cal_<key>     stories --stage cal  (C; if chk_st passed)                           -> lb/<tag>.json (l_b)
#     st_<key>      stories --stage eval (N9; if the lb file exists; redone, even when done, if the stories file
#                   records another lb sha256 than the lb file has, i.e. cal reran: s9_lb_stale)  -> stories/<tag>.json
#     ed_<key>      edits --stage eval   (E9; if chk_ed passed)                           -> edits/<tag>.json
#     ps_<key>      passages --stage eval --formats NOM,OPTB,OPTA --splits E,R (if chk_ps passed) -> passages/<tag>.json
#     leta_<key>    passages --stage eval --formats LETA --splits E (exploratory; only if LETA_MIN minutes remain after
#                   the reserve)                                                          -> passages/<tag>_LETA.json
#   score           analysis/stage9_score.py -> $OUT/STAGE9_SCORE.txt, stage9_summary.json, fig_*.csv; MANIFEST.sha256;
#                   the archive gpu_stage9_results.tgz
# Deadline (DEADLINE_H, default 6.0 h; STAGE8_DEADLINE): every step gets --reserve-min = the minutes of the later steps of
# its model (stories ST_MIN 25, edits ED_MIN 25, passages PS_MIN 30) plus the CORE_MIN of the later models (mistral7 88,
# llama8 83, qwen7 78: the upper ends of SPEC 16's per-step minutes) plus 10 for the score; it stops before work that
# would end inside the reserve (exit 3: partial, redone next session). The next model's weights are prefetched in the
# background.
# Compute (SPEC section 16, A100-80GB, BF16): about 70-90 min per model; expected wall time about 4.3-5 h with the pytest,
# the LETA steps and the score (the 6.0 h default deadline holds the reserves' upper ends).
# Disk: >= 50 GB (one model at a time plus a prefetch).
# Usage:
#   F=$(git log --format=%H -1 --grep='^Freeze stage-9 predictions$') && [ -n "$F" ] && git checkout "$F"
#   bash scripts/gpu_stage9.sh                  # HF_TOKEN optional
#   TEST_MODE=1 bash scripts/gpu_stage9.sh   # CPU plumbing run at Qwen2.5-0.5B-Instruct (FP32, n = 2 per population,
#                          # outputs TEST_<key>) for qwen7 and llama8 (TEST_ALL=1: all three keys); measured 1 h 51 min on
#                          # a 4-core CPU (pytest 52 min, the steps of the two keys 58 min), inside the 6.0 h default
# Optional: OUT, SQUAD (default results/gpu_stage8a/squad/dev-v1.1.json), DEADLINE_H, FORCE, FORCE_STEPS, KEEP_CACHE,
# TESTS=0, PY, S8_MODELS, MINGIB, TEST_ALL.
cd "$(dirname "$0")/.." || exit 2
# shellcheck source=scripts/stage8_common.sh
source scripts/stage8_common.sh
# shellcheck source=scripts/stage9_common.sh
source scripts/stage9_common.sh
s9_init

# ---- FP32 gates before any model: every stage-9 test file and the tests of the shared modules stage 9 calls
s8_pytest tests/test_stage9_pops.py tests/test_stage9_stories.py tests/test_stage9_passages.py tests/test_stage9_edits.py \
  tests/test_stage9_score.py tests/test_stage9_guard.py tests/test_stage9_pre.py \
  tests/test_fresh.py tests/test_fresh_factorial.py tests/test_generate.py tests/test_surface.py tests/test_clamp.py \
  tests/test_natural_clamp.py tests/test_stage8_edits.py tests/test_stage8_populations.py

TAGP=; on TEST_MODE && TAGP=TEST_
SQ=${SQUAD:-results/gpu_stage8a/squad/dev-v1.1.json}
ST=(experiments/stage9_stories.py)
ED=(experiments/stage9_edits.py)
PS=(experiments/stage9_passages.py)
KEYS=(mistral7 llama8 qwen7)
if on TEST_MODE && ! on TEST_ALL; then   # every key runs the same 0.5B model: two keys (one with E4b, one without)
  KEYS=(qwen7 llama8)
  echo "$(s8_utc) TEST_MODE: keys ${KEYS[*]} (TEST_ALL=1 runs mistral7 too)" | tee -a "$OUT/COMMIT.txt"
fi
# CORE_MIN: fetch, loads, checks, cal, stories, edits, passages of a model (SPEC 16: the upper ends of the per-step
# minutes; llama8 at L = 32 with a prefetched fetch, mistral7 + 5 for its own fetch, qwen7 - 5 for 28 blocks)
declare -A CORE_MIN=([mistral7]=88 [llama8]=83 [qwen7]=78)
ST_MIN=25; ED_MIN=25; PS_MIN=30; LETA_MIN=10; SCORE_MIN=10

later_models() {  # later_models <index>: the CORE_MIN of the models after KEYS[index], plus the score
  local i s=$SCORE_MIN
  for ((i = $1 + 1; i < ${#KEYS[@]}; i++)); do s=$((s + CORE_MIN[${KEYS[$i]}])); done
  echo "$s"
}
core_done() {  # every core step of the key done (and its stories file read the current lb file)
  local s; for s in chk_st chk_ed chk_ps cal st ed ps; do s8_done "${s}_$1" || return 1; done
  ! s9_lb_stale "${TAGP}$1"
}

run_key() {  # run_key <index>: every step of KEYS[index]; 1 when its files were refused (no fallback)
  local i=$1 key=${KEYS[$1]} t="${TAGP}${KEYS[$1]}" later dir okst=0 oked=0 okps=0
  later=$(later_models "$i")
  if core_done "$key" && { s8_done "leta_$key" || ! s8_time_left $((later + LETA_MIN)); }; then
    echo "==================== $key: every step kept"; return 0
  fi
  if ! dir=$(s8_fetch "$key"); then
    s8_check   # dies on a fetch failure that is not a refusal (disk, network, manifest)
    echo "$(s8_utc) $key: files refused (FETCH_FAILED.txt); no fallback, its predictions are NOT EVALUABLE" | tee -a "$OUT/COMMIT.txt"
    return 1
  fi
  local M=(--model "$dir" --key "$key" --out "$OUT")
  S8_OUTPUTS="$OUT/check/${t}_stories.json" s8_step "chk_st_$key" $PY "${ST[@]}" --stage check "${M[@]}" \
    --reserve-min $((later + ST_MIN + ED_MIN + PS_MIN)) && okst=1
  S8_OUTPUTS="$OUT/check/${t}_edits.json" s8_step "chk_ed_$key" $PY "${ED[@]}" --stage check "${M[@]}" \
    --reserve-min $((later + ST_MIN + ED_MIN + PS_MIN)) && oked=1
  S8_OUTPUTS="$OUT/check/${t}_passages.json" s8_step "chk_ps_$key" $PY "${PS[@]}" --stage check --squad "$SQ" "${M[@]}" \
    --reserve-min $((later + ST_MIN + ED_MIN + PS_MIN)) && okps=1
  # the decisive stories first (l_b on C before any N9 item), then the edits, then the passages
  if [ $okst = 1 ]; then
    S8_OUTPUTS="$OUT/lb/$t.json" s8_step "cal_$key" $PY "${ST[@]}" --stage cal "${M[@]}" \
      --reserve-min $((later + ST_MIN + ED_MIN + PS_MIN))
    if [ -f "$OUT/lb/$t.json" ]; then
      # cal reran after the eval (e.g. FORCE_STEPS=cal_<key>): a kept st step would leave a stories file that read
      # another lb file (the scorer refuses it); redo it (the eval sees the new lb sha256 and reruns every format)
      if [ -f "$OUT/steps/st_$key.done" ] && s9_lb_stale "$t"; then
        mv "$OUT/steps/st_$key.done" "$OUT/steps/st_$key.done.redone.$(s8_stamp)"
        echo "$(s8_utc) st_$key redone: stories/$t.json read another lb file than lb/$t.json (the cal step reran)" | tee -a "$OUT/COMMIT.txt"
      fi
      S8_OUTPUTS="$OUT/stories/$t.json" s8_step "st_$key" $PY "${ST[@]}" --stage eval "${M[@]}" \
        --reserve-min $((later + ED_MIN + PS_MIN))
    else
      s8_skip "st_$key" "no lb file $OUT/lb/$t.json (the cal step has not finished)"
    fi
  else
    s8_skip "cal_$key" "the stories check failed (chk_st_$key)"; s8_skip "st_$key" "the stories check failed (chk_st_$key)"
  fi
  if [ $oked = 1 ]; then
    S8_OUTPUTS="$OUT/edits/$t.json" s8_step "ed_$key" $PY "${ED[@]}" --stage eval "${M[@]}" --reserve-min $((later + PS_MIN))
  else
    s8_skip "ed_$key" "the edits check failed (chk_ed_$key)"
  fi
  if [ $okps = 1 ]; then
    S8_OUTPUTS="$OUT/passages/$t.json" s8_step "ps_$key" $PY "${PS[@]}" --stage eval --formats NOM,OPTB,OPTA --splits E,R \
      --squad "$SQ" "${M[@]}" --reserve-min "$later"
    if s8_done "leta_$key" || s8_time_left $((later + LETA_MIN)); then
      S8_EXPLORATORY=1 S8_OUTPUTS="$OUT/passages/${t}_LETA.json" s8_step "leta_$key" $PY "${PS[@]}" --stage eval --formats LETA \
        --splits E --squad "$SQ" "${M[@]}" --reserve-min "$later"
    else
      s8_skip "leta_$key" "exploratory: less than $((later + LETA_MIN)) min to the deadline"
    fi
  else
    s8_skip "ps_$key" "the passages check failed (chk_ps_$key)"; s8_skip "leta_$key" "the passages check failed (chk_ps_$key)"
  fi
  s8_drop "$key"
  return 0
}

for i in "${!KEYS[@]}"; do
  next=${KEYS[$((i + 1))]:-}
  if [ -n "$next" ] && ! on TEST_MODE && ! core_done "$next"; then s8_prefetch "$next"; fi   # background (the fetcher locks)
  run_key "$i" || true
done
wait   # a prefetch still running

# ---- score, manifest, archive
s9_finish analysis/stage9_score.py
