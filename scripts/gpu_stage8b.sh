#!/usr/bin/env bash
# GPU stage 8, part B (preregistered as P-2026-10-10-J part B in docs/PREREGISTRATION.md): the fresh-sample replication of
# the key/value format factorial in new model families, scored on the forms the models emit. Populations F (150 fresh
# cores, two lexicons, eight sentences), C (30 calibration cores) and S0 (the cores of stages 1 and 3b), ckeys/fresh.py.
# Models (official weights, sha256-verified by scripts/fetch_verified.py through s8_fetch): P4 = qwen7, qwen14, mistral7,
# olmo7 (F and S0); N4 = llama8, gemma9, phi4, falcon7 (F; yi9 takes the slot of one whose files fail verification or
# whose tokenizer check fails, before any output of it exists; at most one replacement); gemma2b (F, J-B-SMALL).
# Steps per model (scripts/stage8_common.sh: each kept once done, FORCE=1 / FORCE_STEPS=<names or globs> redo, logs in
# $OUT/logs/<step>.log, a failed step goes to FAILED.txt and the pipeline goes on), all experiments/fresh_factorial.py:
#   tok_<key>    J-B-G0b, tokenizer only: lexicons, FormSets, every arm of F, C, S0 valid, sentence lengths
#   calib_<key>  frame discovery on C (greedy generations of the clean runs); frames/<key>.json (its sha256 is recorded by
#                every later file of the model and checked by the scorer)
#   g3_<key>     J-B-G3: the trie against the plain path on the first 30 F cores (NONE, POST, AFTER); picks the scorer
#   evalF_<key>  F, arms AFTER BEFORE NONE POST PRE POST-NULL (trie pass, 13 rows + clean runs, generation)
#   evalS0_<key> P4 only: S0, arms P1 AFTER BEFORE POST PRE NONE, plus format_factorial.run_item (the published path)
# Then, deadline-guarded and in this order: jb8 (J-B8: the release of Anonymous (2026) checked as in stage 7, calibration at
# Mistral-Small-24B, then experiments/paper1_frames.py --model mistral --score E on the release's native cores, five
# formats), x2 (exploratory: Qwen2.5-1.5B and 3B on S0 under E, arms P1 AFTER POST NONE), x1 (exploratory: Qwen2.5-7B on
# the first 60 S0 cores, AFTER POST NONE, FP32 sdpa and BF16 eager); then the scorer (analysis/stage8b_score.py ->
# $OUT/STAGE8B_SCORE.txt and the paper tables), MANIFEST.sha256 and the archive.
# Deadline (DEADLINE_H, default 3.5 h; STAGE8_DEADLINE): every eval step gets --reserve-min = the core minutes of the models
# after it plus 10, and stops before an arm that would start inside that reserve (exit 3: partial, redone next session);
# jb8 runs only if 45 min remain, x2 if 15, x1 if 12. The next model's weights are prefetched in the background.
# Batch sizes (A100-80GB, prompts <= 160 tokens plus <= ~400 trie nodes): one 3-row and one 13-row scoring forward and one
# 8-row generation batch (<= 16 new tokens) per item and arm; the JB-G3 fallback (score_cached) chunks node continuations
# into 256 rows; under 10 GB beyond the weights at 14B. Compute estimate (entry, Compute): about 2.6 GPU-h for the core,
# about 3.4 h with jb8, x2 and x1. Disk: >= 120 GB (two models at a time; 47 GB for jb8).
# Usage:
#   J=$(git log --format=%H -1 --grep='^Finalise preregistration J$') && git checkout "$J"
#   bash scripts/gpu_stage8b.sh                  # HF_TOKEN optional
#   TEST_MODE=1 bash scripts/gpu_stage8b.sh      # CPU plumbing run at Qwen2.5-0.5B (FP32, n = 2, outputs TEST_<key>):
#                                                # every step for qwen7, llama8, gemma2b, jb8, qwen1.5 and x1 (TEST_ALL=1:
#                                                # every key), about 1 h of runs plus the pytest on a 4-core CPU
# Optional: OUT, P1R=<the unpacked release of Anonymous (2026)> (unset: fetched into $P1_ROOT, default $HOME/paper1), DEADLINE_H,
# FORCE, FORCE_STEPS, KEEP_CACHE, TESTS=0, PY, S8_MODELS, MINGIB.
cd "$(dirname "$0")/.." || exit 2
PART=b
S8_DEADLINE_H_DEFAULT=3.5
# shellcheck source=scripts/stage8_common.sh
source scripts/stage8_common.sh
s8_init

# ---- FP32 gates before any model (J-B-G0): part B's tests, the scorer's tests and the shared modules part B calls
s8_pytest tests/test_fresh.py tests/test_fresh_factorial.py tests/test_stage8b_score.py tests/test_surface.py \
  tests/test_generate.py tests/test_clamp_families.py

TAGP=; on TEST_MODE && TAGP=TEST_
FF=(experiments/fresh_factorial.py --out "$OUT")
P4=" qwen7 qwen14 mistral7 olmo7 "
KEYS=(qwen7 qwen14 mistral7 olmo7 llama8 gemma9 phi4 falcon7 gemma2b)
X2KEYS=(qwen1.5 qwen3b)
if on TEST_MODE && ! on TEST_ALL; then   # every key would run the same 0.5B model: one per role unless TEST_ALL=1
  KEYS=(qwen7 llama8 gemma2b); X2KEYS=(qwen1.5)
  echo "$(s8_utc) TEST_MODE: keys ${KEYS[*]} and ${X2KEYS[*]} (one per role; TEST_ALL=1 runs every key)" | tee -a "$OUT/COMMIT.txt"
fi
declare -A CORE_MIN=([qwen7]=22 [qwen14]=40 [mistral7]=22 [olmo7]=22 [llama8]=12 [gemma9]=19 [phi4]=19 [falcon7]=12
                     [gemma2b]=7 [yi9]=14)
JB8_MIN=45; X2_MIN=15; X1_MIN=12

reserve_after() {  # reserve_after <index>: core minutes of the models after KEYS[index], plus the score
  local i s=10
  for ((i = $1 + 1; i < ${#KEYS[@]}; i++)); do s=$((s + CORE_MIN[${KEYS[$i]}])); done
  echo "$s"
}
has_output() { compgen -G "$OUT/frames/${TAGP}$1.json" > /dev/null || compgen -G "$OUT/eval/${TAGP}$1_*.json" > /dev/null; }
all_done() { local k=$1 s; for s in tok calib g3 evalF; do s8_done "${s}_$k" || return 1; done
             [[ "$P4" != *" $k "* ]] || s8_done "evalS0_$k"; }

run_key() {  # run_key <key> <reserve minutes>: 0 done, 1 files refused, 2 tokenizer check failed (both before any output)
  local key=$1 reserve=$2 dir t="${TAGP}$1"
  if all_done "$key"; then echo "==================== $key: every step kept"; return 0; fi
  dir=$(s8_fetch "$key") || return 1
  local M=(--model "$dir" --key "$key")
  S8_OUTPUTS="$OUT/tokcheck/$t.json" s8_step "tok_$key" $PY "${FF[@]}" --stage tokcheck "${M[@]}" || { s8_drop "$key"; return 2; }
  S8_OUTPUTS="$OUT/frames/$t.json" s8_step "calib_$key" $PY "${FF[@]}" --stage calib "${M[@]}"
  S8_OUTPUTS="$OUT/g3/$t.json" s8_step "g3_$key" $PY "${FF[@]}" --stage g3 "${M[@]}"
  if [ -f "$OUT/frames/$t.json" ] && [ -f "$OUT/g3/$t.json" ]; then
    S8_OUTPUTS="$OUT/eval/${t}_F.json" s8_step "evalF_$key" $PY "${FF[@]}" --stage eval --population F "${M[@]}" --reserve-min "$reserve"
    if [[ "$P4" == *" $key "* ]]; then
      S8_OUTPUTS="$OUT/eval/${t}_S0.json" s8_step "evalS0_$key" $PY "${FF[@]}" --stage eval --population S0 "${M[@]}" --reserve-min "$reserve"
    fi
  else
    echo "FAILED eval of $key not run: no frames or G3 file" | tee -a "$OUT/FAILED.txt"
  fi
  s8_drop "$key"
  return 0
}

FALLBACK=
for i in "${!KEYS[@]}"; do
  key=${KEYS[$i]}
  next=${KEYS[$((i + 1))]:-mistral24}
  if ! on TEST_MODE && ! all_done "$next"; then s8_fetch "$next" > /dev/null 2>&1 & fi   # prefetch (the fetcher locks)
  rc=0; run_key "$key" "$(reserve_after "$i")" || rc=$?
  [ $rc = 0 ] && continue
  why=$([ $rc = 1 ] && echo "its files failed verification" || echo "its tokenizer check (J-B-G0b) failed")
  if [[ " llama8 gemma9 phi4 falcon7 " == *" $key "* ]] && [ -z "$FALLBACK" ] && ! has_output "$key"; then
    FALLBACK=$key   # entry rule G2: the one technical fallback, decided before any output of $key exists
    echo "$(s8_utc) fallback: yi9 replaces $key ($why; no output of $key exists)" | tee -a "$OUT/COMMIT.txt"
    run_key yi9 "$(reserve_after "$i")" || echo "FAILED the fallback yi9 (exit $?)" | tee -a "$OUT/FAILED.txt"
  else
    echo "FAILED $key: $why (no fallback applies)" | tee -a "$OUT/FAILED.txt"
  fi
done
wait   # a prefetch still running

# ---- J-B8: the intervention frames at Mistral-Small-24B re-scored under E (deadline-guarded)
if s8_done jb8 || s8_time_left "$JB8_MIN"; then
  P1=${P1_ROOT:-$HOME/paper1}
  if [ -n "${P1R:-}" ]; then
    [ -d "$P1R" ] || die "P1R=$P1R does not exist (unset P1R to fetch the release into $P1)"
  else
    P1R=$P1/v5.5-reviewer-repository
  fi
  if [ ! -d "$P1R" ]; then
    mkdir -p "$P1"
    curl -fsSL --retry 3 --retry-delay 30 -o "$P1/p1.zip" "https://anonymous.4open.science/api/repo/Beyond-the-fitted-Scope---Causal-keys-230B/zip" \
      && $PY -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$P1/p1.zip" "$P1"
  fi
  if [ -f "$P1R/RELEASE.json" ] && s8_step jb8_release $PY -c "import json, os, sys; from experiments.stage7_link import release_check; print(json.dumps(release_check(sys.argv[1], bool(os.environ.get('TEST_MODE')))))" "$P1R"; then
    { echo "==== $(s8_utc) the release of Anonymous (2026) at $P1R"; tail -n 1 "$OUT/logs/jb8_release.log"; } >> "$OUT/RELEASE.txt"
    if dir=$(s8_fetch mistral24); then
      t="${TAGP}mistral24"
      S8_OUTPUTS="$OUT/tokcheck/$t.json" s8_step tok_mistral24 $PY "${FF[@]}" --stage tokcheck --model "$dir" --key mistral24
      S8_OUTPUTS="$OUT/frames/$t.json" s8_step calib_mistral24 $PY "${FF[@]}" --stage calib --model "$dir" --key mistral24
      if on TEST_MODE; then J8=(--model-override "$dir" --bases-override 896 --n 3); J8OUT="$OUT/jb8/TEST_${dir##*/}.json"
      else J8=(--model-dir "$dir"); J8OUT="$OUT/jb8/mistral.json"; fi
      mkdir -p "$OUT/jb8"
      [ -f "$OUT/frames/$t.json" ] && S8_OUTPUTS="$J8OUT" s8_step jb8 $PY experiments/paper1_frames.py --model mistral \
        --p1-root "$P1R" --out "$OUT/jb8" --score E --frames "$OUT/frames/$t.json" "${J8[@]}"
      s8_drop mistral24
    else
      s8_skip jb8 "the files of mistral24 were refused (FETCH_FAILED.txt)"
    fi
  else
    s8_skip jb8 "the release of Anonymous (2026) is absent or does not match the pins (see $OUT/logs/jb8_release.log)"
  fi
else
  s8_skip jb8 "less than $JB8_MIN min to the deadline"
fi

# ---- exploratory X2 (Qwen2.5-1.5B and 3B on S0) and X1 (FP32 and eager anchors at Qwen2.5-7B), deadline-guarded
for key in "${X2KEYS[@]}"; do
  if s8_done "x2_$key" || s8_time_left "$X2_MIN"; then
    if dir=$(s8_fetch "$key"); then
      M=(--model "$dir" --key "$key"); t="${TAGP}$key"
      S8_EXPLORATORY=1 s8_step "tok_$key" $PY "${FF[@]}" --stage tokcheck "${M[@]}" \
        && S8_EXPLORATORY=1 S8_OUTPUTS="$OUT/frames/$t.json" s8_step "calib_$key" $PY "${FF[@]}" --stage calib "${M[@]}" \
        && S8_EXPLORATORY=1 S8_OUTPUTS="$OUT/g3/$t.json" s8_step "g3_$key" $PY "${FF[@]}" --stage g3 "${M[@]}" \
        && S8_EXPLORATORY=1 S8_OUTPUTS="$OUT/eval/${t}_S0.json" s8_step "x2_$key" $PY "${FF[@]}" --stage eval --population S0 \
             "${M[@]}" --arms P1,AFTER,POST,NONE --no-plain
      s8_drop "$key"
    fi
  else
    s8_skip "x2_$key" "less than $X2_MIN min to the deadline"
  fi
done
if { s8_done x1_fp32 && s8_done x1_eager; } || s8_time_left "$X1_MIN"; then
  if dir=$(s8_fetch qwen7); then
    X1=(--stage eval --population S0 --model "$dir" --key qwen7 --n 60 --arms AFTER,POST,NONE --no-plain)
    on TEST_MODE && X1=(--stage eval --population S0 --model "$dir" --key qwen7 --arms AFTER,POST,NONE --no-plain)
    S8_EXPLORATORY=1 S8_OUTPUTS="$OUT/eval/${TAGP}qwen7_x1fp32_S0.json" s8_step x1_fp32 $PY "${FF[@]}" "${X1[@]}" --dtype float32 --suffix _x1fp32
    S8_EXPLORATORY=1 S8_OUTPUTS="$OUT/eval/${TAGP}qwen7_x1eager_S0.json" s8_step x1_eager $PY "${FF[@]}" "${X1[@]}" --attn eager --suffix _x1eager
    s8_drop qwen7
  fi
else
  s8_skip x1 "less than $X1_MIN min to the deadline"
fi

# ---- score, manifest, archive
s8_finish analysis/stage8b_score.py
