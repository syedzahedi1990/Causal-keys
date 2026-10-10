LABEL design:replication
### part
B: fresh-sample replication on new model families, and scoring on the forms the models actually emit (preregistration J, part B; predictions JB1-JB9, gates JB-G0 to JB-G5)

### objections_answered
[
 "O4, measurement validity: the no-mention and sentence formats were scored on lowercase ' w' tokens that carry about 0.00-0.03 of the probability at Qwen2.5-7B/14B and Mistral-7B. Fix: every candidate is scored exactly (chain rule) as the logsumexp over its surface forms. Sigma is the 12-form set the reviewers named (case, leading space, after The/the). The primary score E adds locative and markdown frames ('In the', 'On the', 'At the', 'Inside the', '**'), plus model-specific frames found on held-out calibration stories. The candidate mass is reported per cell. A behavioural measure on greedy generations (beta_K, beta_V) is added. Invariance is tested on the original seed-0 cores at Qwen2.5-7B, Qwen2.5-14B, Mistral-7B and OLMo-2-7B (JB6). Optionally the same check is run on the intervention frames at Mistral-Small-24B (JB8).",
 "O5, the sentence effect was 'confirmed' on the same cores and models where it was found: the confirmatory 2x2 and natural factorial run on 150 never-used story cores (seed 20261013, filtered against every earlier population). This is done at the same four primaries, labelled as a low-risk replication, and in four model families never examined. Every verdict line carries a preregistered risk class (L = implied by data in hand, R = not implied), and met rates are summarised separately for each class.",
 "Stats reviewer, nats are not commensurable across scale: every threshold is scale-free. These are r_f = ID_K(f)/ID_K(LIST-AFTER) on the same cores, s_ID, and the bounded behavioural beta in [-1, 1]. The old 0.5-nat and \u00b11-nat margins are no longer used in any Part-B verdict.",
 "Stats reviewer, intervals ignore clustering and multiplicity: the primary intervals come from a two-stage cluster bootstrap. It first resamples the 30 ordered (base, source) location pairs, then cores within each pair. One index set is shared by all arms and scorings, so every contrast is paired. Verdicts use Bonferroni-level 98.75% intervals across the four models of each set.",
 "'7 of 10 models are Qwen': four non-Qwen families are added: Llama-3.1-8B-Instruct, Gemma-2-9B-it, Phi-4 (14B) and Falcon3-7B-Instruct. That gives 14 models in 8 families, 7 of them Qwen. The weights of the two gated models are provably the official ones without a token, because ungated copies have byte-identical files, checked against the official repos' public sha256 and git-blob ids.",
 "Reproducibility (stats reviewer, minor): every model revision is pinned with per-file hashes. The published lowercase numbers are reproduced on the same stack as a gate before they are compared (JB-G2). The BF16 numerical floor of the new scorer is measured (JB-G3), and an FP32 anchor at Qwen2.5-7B bounds the eager/sdpa gap the skeptic asked about (exploratory X1).",
 "Generalist, the 1.5B/3B dissociation is confounded with competence and scored on non-emitted tokens: exploratory X2 re-scores Qwen2.5-1.5B/3B on the seed-0 cores under E, with a competent-only breakdown."
]

### design

# Part B: fresh-sample replication on new families, scored on what the models emit

## B0. Logic
1. **Objection 5 and family breadth.** The paper's descriptive core is replicated on fresh stories. These are (i) a re-measurement at the four models where it was found (low risk) and (ii) an out-of-sample test in four new families (risky).
2. **Objection 4.** The measurement is changed to score the forms the models emit, and the result is shown not to depend on the scoring, on the original data.
3. **Primary statistics.** Every primary statistic is scale-free. A behavioural measure on generated answers is added, which needs no choice of scored token at all.

## B1. Stimuli and populations
- **Generator and arms.** `ckeys/story.py`, `ckeys/encoding.py`, the wrapper and the "Answer:" prefill are unchanged, with the direct view only. Arms (code name = paper name): P1 = OPTIONS-AFTER, LETTER = LETTERS-AFTER, AFTER = LIST-AFTER, BEFORE = LIST-BEFORE, POST = SENTENCE-AFTER, PRE = SENTENCE-BEFORE, NONE = NO-MENTION.
- **U, the cores used before.** U = ∪ make_cores(1000, Random(s)) for s ∈ {0,1,2,3}. Stages 1–7 used only seeds 0 and 1 (grep of experiments/ and scripts/), so U is a superset of them.
- **F, the fresh evaluation cores.** The first 150 cores of the stream make_cores(·, Random(20261013)) that are not in U.
  - 4 of the first 400 stream cores are excluded.
  - F covers all 30 ordered (B,S) pairs and all 20 objects.
  - sha256(json(F)) begins e1f28077. Pin the full hash at the finalising commit.
  - No pilot touched seed 20261013.
- **C, the calibration cores** (frame discovery only, never in a statistic). The first 30 of the stream Random(20261014) not in U ∪ F.
- **S0.** make_cores(150, Random(0)): the cores of stages 1 and 3b, Table 1 and Tables 14–16.
- **Size and populations.** n = 150 per model × arm. The primary population is all items. The secondary population is competent items: clean B and clean S answered correctly by the E-argmax over the six candidates.

## B2. Models
All run in BF16 with use_cache=False, transformers 5.18.0 (pinned), sdpa attention (Gemma-2 eager, as in format_factorial), revision pinned, every file hash-verified.

| Set | Model | Revision | Source of bytes |
|---|---|---|---|
| P4 | Qwen/Qwen2.5-7B-Instruct | a09a3545… | official |
| P4 | Qwen/Qwen2.5-14B-Instruct | cf98f3b3… | official |
| P4 | mistralai/Mistral-7B-Instruct-v0.3 | c170c708… | official (HF-format shards only) |
| P4 | allenai/OLMo-2-1124-7B-Instruct | 470b1fba… | official |
| N4 | meta-llama/Llama-3.1-8B-Instruct (gated) | 0e9e39f2… | HF_TOKEN → official; else weights, config, generation_config, tokenizer.json and special_tokens_map from NousResearch/Meta-Llama-3.1-8B-Instruct@d10aef79, and tokenizer_config.json from modularai/Llama-3.1-8B-Instruct-GGUF@96669450 |
| N4 | google/gemma-2-9b-it (gated) | 11c9b309… | HF_TOKEN → official; else weights, tokenizer.json, tokenizer.model and special_tokens_map from unsloth/gemma-2-9b-it@fc7d4737, config.json from thr3a/gemma-2-9b-it@e99c393f, and tokenizer_config.json and generation_config from dnhkng/RYS-Gemma-2-9b-it@dd19021a |
| N4 | microsoft/phi-4 (Phi3 architecture, fused qkv_proj) | 2db69c1c… | official, non-gated |
| N4 | tiiuae/Falcon3-7B-Instruct (Llama architecture) | 1e57a0ec… | official, non-gated |

- **Hash checks.** Every file is checked against the official repo's sha256 (LFS files) or git-blob SHA-1 (small files), all captured on 2026-10-10 in `hf_manifest.json`. The assembled directory is therefore byte-identical to the official one, and the paper says "official weights".
- **Fallback.** If a gated model fails verification, or if Phi-4 fails the fused-qkv exactness test, the replacement is 01-ai/Yi-1.5-9B-Chat: non-gated, Llama architecture, all six candidates single tokens, 0 of 150 fresh cores skipped in every arm. The fallback is used only for such technical failures, decided before any output of that model exists. It is never used after a competence or verdict failure.
- **Checked and infeasible.**
  - granite-3.1-8b: " shelf" is two tokens, and 145/150 items fail the single-position check.
  - deepseek-llm-7b-chat: " cabinet" is two tokens.
  - internlm2.5: needs remote code.

## B3. Scoring
### B3.1 Three scorings of candidate c in run Z
- **L (published).** L(c|Z) = log p(" c").
- **Σ (reviewer-named).** Σ(c|Z) = logsumexp over the 12 forms φ+w and φ+W, for φ ∈ Φ_Σ = {" ", "", " The ", " the ", "The ", "the "}, with W = w.capitalize().
- **E (primary, emitted-form score).** The same over Φ_E = Φ_Σ ∪ {" In the ", " in the ", " On the ", " on the ", " At the ", " at the ", " Inside the ", " inside the ", " **", "**"} ∪ Φ_disc(model). That is 32 fixed forms per candidate plus discovered ones.
- **Exact chain rule.** log P(form) = Σ_t log p(f_t | prompt, f_<t, Z).
  - The continuation ids are ids(prefill+form) − ids(prefill) when the boundary is stable. Otherwise they are the form's standalone ids: byte-level BPE merges ":The", but a model that has already read ":" emits "The" as its own token.
  - Identical token sequences are counted once.
  - It is asserted that no form's sequence is a proper prefix of another's, so the summed events are disjoint.
- **One forward per batch row.** The prompt is followed by the trie of all proper prefixes of all forms.
  - The 4D tree-attention mask lets each node attend to the prompt and its own ancestors. It is boolean for sdpa and additive for eager.
  - Explicit position ids place the node at depth d at position T−1+d.
  - logits_to_keep = N+1 returns every chain-rule term.
  - Trie nodes for 6 candidates: Σ has 9 (Qwen, Llama, OLMo, Phi-4), 5 (Gemma) or 44 (Mistral). The fixed E set has 33–37 at Qwen and Llama.
- L, Σ and E always come from the same pass, so their comparison has no numerical confound.

### B3.2 Frame discovery (calibration only; rule preregistered, frames determined before any evaluation item)
- **Generations.** On C, 30 cores × 7 arms × clean B, S and X runs, which gives 630 greedy generations per model of up to 16 tokens.
- **Frame extraction.** The frame is the text between "Answer:" and the first candidate word. The item's names are replaced by {a}, {b}, {o} and {d}. Frames that contain a newline or are longer than 60 characters are dropped.
- **Admission.** A frame not in Φ_E is admitted if it occurs in at least 2% of some arm's generations (at least 2 of 90). At most 16 frames are admitted per model, most frequent first, and they are instantiated per item with its names.
- **Freezing.** Φ_disc is written to frames_{model}.json, and its sha256 is logged before the model's first evaluation item. The scorer checks the hash.

### B3.3 Measures per model × arm × population (σ ∈ {L, Σ, E})
- **Rows.** The 13 rows of format_factorial.row_specs (self-clamp ID; K_S, V_S and KV_S at ℓ0 ∈ {0, round(0.0625L), round(0.3L)}; K_X, V_X and KV_X at 0) and the clean B, S and X runs, all unchanged.
- **Change against the self-clamp row.** Δ^σ_Y(Z) = σ(Y|Z) − σ(Y|ID).
- **Key identity.** ID_K^σ = ½[(Δ_S(K_S) − Δ_S(K_X)) + (Δ_X(K_X) − Δ_X(K_S))], at ℓ0 = 0.
- **Value identity.** ID_V^σ is the same with V_S and V_X. A bar denotes the mean over cores.
- **Key read relative to the list.** r^σ_f = mean ID_K^σ(f) / mean ID_K^σ(AFTER), on the same cores.
- **Identity key share.** s^σ_ID(f) = mean ID_K / (mean ID_K + mean ID_V). It is defined when the denominator is at least 2 nats and mean ID_V ≥ −0.5. Every in-hand value of the denominator is at least 10 nats.
- **Paired contrast.** δ^σ_{a−b} = [mean ID_K(a) − mean ID_K(b)] / mean ID_K(AFTER), paired over cores.
- **Candidate mass.** M^σ(f) = mean over cores of Σ_{6 candidates} exp σ(c | clean B). Also reported:
  - the first-token gap: the mass of first tokens that begin no form;
  - the top 10 first tokens.
- **Generated answers.** Greedy argmax decoding of up to 12 new tokens, with the clamps active, use_cache=False, stopping early at the first candidate match, a newline or EOS.
  - It runs in rows ID, K_S, K_X, V_S and V_X (ℓ0 = 0) and in the clean B, S and X runs.
  - a(Z) is the first match of (?i)\b(box|basket|shelf|drawer|cabinet|closet) in the continuation, or "other" if there is none.
- **Behavioural identity.**
  - β_K = ½[(P̂(a=S|K_S) − P̂(a=S|K_X)) + (P̂(a=X|K_X) − P̂(a=X|K_S))] ∈ [−1, 1].
  - β_V is the same with V_S and V_X.
  - b_ID = β_K / (β_K + β_V), reported when the denominator is at least 0.2.
  - Also reported: acc_B and acc_S of the clean generations, and the "other" rate.
- **Agreement.** A = P̂(argmax_c E(c|Z) = a(Z) | a(Z) ≠ other), over the 8 generated rows.

## B4. Runs
- **R0 (calibration).** All 8 models on C: clean runs and generation only.
- **R1 (S0 re-score).** P4 × {P1, AFTER, BEFORE, POST, PRE, NONE} on S0, with the trie pass and generation. It also runs, on all items, a plain pass byte-identical to format_factorial.run_item (the published code path) for JB-G2 and JB-G3.
- **R2.** P4 × 7 arms on F.
- **R3.** N4 × 7 arms on F.
- **Deadline-guarded runs, in this order:**
  - R4 = JB8: experiments/paper1_frames.py at Mistral-Small-24B with E scoring (release checks as in stage 7);
  - X1: FP32 anchor at Qwen2.5-7B on the first 60 S0 cores, AFTER/POST/NONE, comparing FP32 with BF16 sdpa and BF16 eager;
  - X2: Qwen2.5-1.5B and 3B on S0 (POST, NONE, AFTER, P1) under E, with a competent-only breakdown;
  - X3: gemma-2-27b-it on F, only if all files verify (weights byte-identical in unsloth/gemma-2-27b-it; config and tokenizer_config need verified copies or a token).

## B5. Statistics
- **Bootstrap.** Primary intervals come from a two-stage cluster bootstrap: resample the 30 ordered (B,S) pairs with replacement, then cores within each drawn pair.
  - 10,000 resamples, seed 20261013.
  - One index set per (model, population) is shared by all arms, scorings and β, so contrasts are paired.
  - Ratios are recomputed in every resample.
  - The core bootstrap is printed beside.
  - In-hand check at P4 (stage 3b, lowercase): the cluster SDs are 1.5–2× the core SDs, for example r(POST) at Mistral 0.506 [0.441, 0.569] against [0.471, 0.543].
- **Verdict level.** Two-sided percentile intervals at 98.75% (Bonferroni over the four models of a set); 95% intervals are printed.
  - A ratio whose definedness rule fails in more than 5% of resamples makes that cell NOT EVALUABLE.
  - A quantity made undefined by the arm itself counts as NOT MET, never as NOT EVALUABLE.
- **Aggregation.**
  - P4f lines are MET iff every evaluable model meets the line and at least 3 models are evaluable.
  - N4 lines are MET iff at least 3 of the 4 models meet the line. A NOT EVALUABLE model counts as not meeting it.
  - A line is NOT EVALUABLE iff fewer than 3 models are evaluable.
- **Risk class.** Each verdict line carries a risk class: L means the seed-0 data imply the line, R means nothing in hand does. The summary gives met rates separately for L and R.
- **Reading the output.** Verdicts use the point estimates and bounds named below and nothing else.

## B6. Gates
- **JB-G0, exactness.** FP32 on CPU, before any model, none skipped, tolerance 1e-4 nats.
  1. The trie score equals separate full passes for every form, at Qwen2.5-0.5B (sdpa and eager), clean and under K_S/V_S clamps.
  2. The same on tiny random-weight Qwen2, Mistral, Llama, OLMo2, Gemma2 (eager, softcapping) and Phi3.
  3. The trie answer-position log-probs equal a plain forward.
  4. Phi3 fused qkv_proj: C_KV(S) from layer 0 reproduces the S run, and C_K and C_V write only their own slice.
  5. Greedy generation under clamps equals the step-by-step argmax of full forwards.
  6. The lowercase fields of the new run_item equal format_factorial.run_item on 2 items.
  7. F is disjoint from U, deterministic, of size 150, and its hash matches the pin; C is disjoint from U ∪ F.
  8. The cluster-bootstrap indices are deterministic.
  9. The scorer returns the known verdicts (MET, NOT MET, NOT EVALUABLE) on synthetic data.

  After the files are fetched, a tokenizer-only check (JB-G0b) runs on each model before it loads:
  - every form decodes back to its string;
  - no sequence is a proper prefix of another;
  - the 7 arms give 0 skipped items on F.
- **JB-G1, files.** Every file must match the pinned hash, otherwise the model is not run.
- **JB-G2, reproduction (R1 plain pass vs the committed stage-1/3b files).** Per P4 model and arm, |Δs_ID^L| ≤ 0.03 and |Δr^L| ≤ 0.03 (E3's tolerance). A failure is flagged and JB6 is still scored, since it is a same-pass comparison.
- **JB-G3, trie floor (R1).** Per cell, |s_ID^L(trie) − s_ID^L(plain)| ≤ 0.01 and the mean |Δ L| ≤ 0.1 nats. If it fails, that model reruns in per-node fallback mode (separate passes, exact) and this is logged.
- **JB-G4, per cell.**
  - The E coverage M^E(clean B) must be at least 0.5. Otherwise the E-based criteria of that cell are NOT EVALUABLE, and β still counts.
  - The BF16 batch floor, mean |m(ID) − m(clean B)|, must be at most 0.5 nats.
- **JB-G5, per model.**
  - Competence: generated acc_B ≥ 0.8 under NONE and under AFTER. Otherwise JB1–JB5 are NOT EVALUABLE for that model, and the model is not replaced.
  - Anchor: ID_K^E(AFTER) ≥ 1 nat with lower bound > 0. Otherwise the r-based criteria are NOT EVALUABLE.

## B7. Confirmatory predictions
All use E scoring, all items, and F unless stated. Bounds are 98.75% cluster-bootstrap bounds.

**JB1, the list after the story opens a key read that carries most of the identity** (N4: R; P4f: L).
- Criterion, per model: s_ID^E(AFTER) ≥ 0.50 with lower bound ≥ 0.40, and the lower bound of ID_K^E(AFTER) > 0.
- In hand: 0.59–0.86 at P4, and OPTIONS-AFTER 0.67–0.88 in all 9 models from 3B up.
- Why it is risky: a new family could keep value-dominant identity even with the list.

**JB2, position: the same list or sentence before the writing token opens no key read** (N4: R; P4f: L).
- Criterion, per model: the upper bounds of r^E(BEFORE), r^E(PRE), s^E_ID(BEFORE) and s^E_ID(PRE) are all ≤ 0.10, i.e. at most a tenth of the list-after read.
- In hand: −0.035 to 0.000.
- Why it is risky: the causal mask blocks a direct read by the list, but the question and answer tokens come after p and could still read the key.

**JB3, a neutral sentence after the story opens an intermediate key read in 7–14B models** (N4: R; P4f: L). This is the post hoc finding of stage 1, now tested out of sample.
- Criterion, per model:
  - r^E(POST) ≥ 0.15 with lower bound ≥ 0.10 (three times the typical NO-MENTION level);
  - r^E(POST) ≤ 0.75 with upper bound < 1.0 (it stays below the list);
  - the lower bounds of δ^E_{POST−PRE} and of δ^E_{POST−NONE} are both > 0.
- In hand: r = 0.27, 0.34, 0.51 and 0.23.
- This is the riskiest line: the claim "from 7B even a neutral sentence opens the key" has never been tested outside Qwen, Mistral and OLMo.

**JB4, without a later mention the identity is copied through the value** (N4: R; P4f: L).
- Criterion, per model: s^E_ID(NONE) ≤ 0.10 with upper bound ≤ 0.15, and the lower bound of ID_V^E(NONE) > 0.
- In hand: 0.03–0.06 in all 10 models.

**JB5, the channel that decides the generated answer flips with later mentions** (N4: R; P4f: L*, since in hand only as the argmax over 4 candidates, not as generation).
- Criterion, per model:
  - β_K(AFTER) ≥ 0.50 with lower bound ≥ 0.40;
  - β_V(NONE) ≥ 0.50 with lower bound ≥ 0.40;
  - the upper bounds of β_K(NONE), β_K(BEFORE) and β_K(PRE) are ≤ 0.10;
  - the lower bound of [β_K(AFTER) − β_V(AFTER)] > 0.
- In hand (proxy from the 4-candidate argmax, stage 3b):

| Quantity | Range at P4 |
|---|---|
| β_K(AFTER) | 0.82–1.00 |
| β_V(AFTER) | 0.00–0.19 |
| β_V(NONE) | 0.92–1.00 |
| β_K(NONE, BEFORE, PRE) | 0.00 |
| β_K(POST), not predicted | 0.02–0.30 |

**JB6, emitted-form scoring leaves the published scale-free results unchanged** (S0, P4; R). This is the direct answer to O4, on the original data.
- (a) Coverage: M^E(clean B) ≥ 0.80 in all 24 cells (4 models × P1, AFTER, BEFORE, POST, PRE, NONE).
- (b) Invariance: in every cell, |s^E_ID − s^L_ID| ≤ 0.05 and |r^E − r^L| ≤ 0.05 (point estimates; the paired-difference CI is printed). The same is reported for Σ against L.
- (c) The verdicts of E1a–c, B2 and C4's BEFORE bound, recomputed under E with their original rules, are unchanged (reported line by line).
- In hand: under SENTENCE-AFTER the capitalised-id ID_K equals the lowercase one within 5–12% at Qwen2.5-7B/14B (stage 5). ID_V and s_ID under the emitted forms have never been measured.

**JB7, the emitted-form score is behaviourally valid** (F, all 8 models; R).
- Criterion, in every evaluable model × arm cell:
  - agreement A ≥ 0.95;
  - the clean-B "other" rate ≤ 0.10;
  - M^E(clean B) ≥ 0.80.
- MET iff all P4 cells and at least 3 of 4 N4 models meet it.

**JB8 (if run), the intervention frames at Mistral-Small-24B are invariant to the scoring** (R).
- Criterion, in each of the five formats: |φ^E − φ^L|, |ψ_K^E − ψ_K^L| and |ψ_V^E − ψ_V^L| ≤ 0.05, and M^E ≥ 0.80. φ, ψ_K and ψ_V are defined as in stage 3b, with m = E(T) − E(S).
- If it is not run (the deadline passes), the verdict is NOT RUN.

**JB9, the fresh sample agrees with the original sample** (P4; L).
- Criterion: |s^E_ID(F) − s^E_ID(S0)| ≤ 0.05 and |r^E(F) − r^E(S0)| ≤ 0.05 for every model and arm of the 2×2. This bounds the variation between story samples that a story bootstrap cannot see.

**Expected values** (the author's, not thresholds):
- **N4.** s_ID(AFTER) 0.6–0.9; r(POST) 0.2–0.5; r(BEFORE) and r(PRE) ≤ 0.02; s_ID(NONE) 0.03–0.07; β_K(AFTER) ≥ 0.8; β_V(NONE) ≥ 0.8. The subjective probability that JB3 is met in at least 3 of the 4 new families is about 0.6.
- **P4f.** The seed-0 values within 0.03.

## B8. Power
Monte-Carlo simulation (`power_sim.py` and `power_jb2.py`, scratchpad) on the committed stage-3b per-core rows:
- n = 150; the two-stage cluster bootstrap at the stricter 99.375% level; 100 or 60 replicates × 600 or 400 resamples.
- Templates: Qwen2.5-7B (tight) and OLMo-2-7B (noisiest).
- Noise: ×1 and ×2 per-core deviations.

| Criterion | True value | P(met), ×1 noise | P(met), ×2 noise |
|---|---|---|---|
| JB3 r(POST) | 0.12 | 0.00 | 0.00 |
| JB3 r(POST) | 0.15 (the bound) | 0.50 | 0.14–0.50 |
| JB3 r(POST) | 0.18 | 1.00 | 0.60–1.00 |
| JB3 r(POST) | 0.22 | 1.00 | 0.86–1.00 |
| JB2 r(BEFORE), r(PRE) | 0 or 0.05 | 1.00 | 0.95–1.00 |
| JB2 r(BEFORE) | 0.08 | 0.08–0.55 | 0.08–0.55 |
| JB4 s_ID(NONE) | ≤ 0.09 | ≥ 0.91 | ≥ 0.91 |
| JB4 s_ID(NONE) | 0.11 | ≤ 0.06 | ≤ 0.06 |
| JB1 s_ID(AFTER) | 0.50 | 0.18–0.54 | 0.18–0.54 |
| JB1 s_ID(AFTER) | ≥ 0.55 | 0.87–1.00 | 0.87–1.00 |

- **Per model.** For a new family inside the in-hand range, the power per model is at least 0.86, so the "at least 3 of 4" rule is met with probability at least 0.90. The false-positive rate at the bound is 0.5 by construction, and 0 one SE below it.
- **β.** SE(P̂) ≤ 0.041 at n = 150 and z(98.75%) = 2.50, so β_K(AFTER) and β_V(NONE) are met with probability ≈ 1 at true values ≥ 0.65, and the upper bound ≤ 0.10 is met at true values ≤ 0.03.
- **P4f.** Every P4f line sits more than 10 cluster-SEs from its bound in hand, so P(met | seed-0 values) ≈ 1.00. These lines are a replication and are labelled L.

## B9. Exploratory (labelled; no verdict)
- Σ and L versions of every statistic.
- The competent-only population.
- The P1 and LETTER natural factorial in N4 (ID_K, s_K, s_ID, the interaction).
- Depth onsets 0.0625L and 0.3L, as s_ID per onset in the new families.
- The sign of LIST-BEFORE (ID_K(BEFORE) − ID_K(NONE)) across all 8 models.
- The calibration frame census and the uncovered first-token mass.
- b_ID and β under POST.
- X1, X2 and X3.

## B10. Compute (one A100-80GB, BF16)
- **Basis.** Stage 3b ran Qwen2.5-7B at 24 s per 150-item arm (16 row-forwards per item, T ≈ 100).
- **Per item.** The trie adds about 30–40% of tokens, and generation over 8 rows with early stop averages about 3 steps, so an item costs about 2.5× a stage-3b item. That is about 60 s per arm at 7–9B and about 120 s at 14B; Gemma-2 eager is about 1.6×.
- **Run times.**
  - R1 (S0, 6 arms, plus the plain pass): 7B 7 min, 14B 14 min, Mistral 7 min, OLMo 7 min, about 35 min in all.
  - R2 (7 arms): about 35 min.
  - R3: Llama 8 min, Gemma 13 min, Phi-4 14 min, Falcon 7 min, about 42 min.
  - R0: about 8 min.
- **Overhead.**
  - Downloads: about 140 GB, prefetching the next model while the current one runs, 10–15 min.
  - Hashing: 3 min. Model loads: 12 min.
  - pytest for JB-G0: about 10 min. Scoring: 5 min.
- **Total.** Core about 2.5 GPU-hours (cap 3.5 h through STAGE8B_DEADLINE).
- **Optional.** R4/JB8 adds 0.5 h (47 GB download), X1 0.2 h, X2 0.15 h, X3 0.8 h.
- **Lean option.** LEAN=1 drops LETTER and P1 from R3 and P1 from R1, saving about 20 min.
- **Disk.** At least 120 GB, with weights deleted after each model unless KEEP_CACHE=1.

## B11. What would count against the account
| Outcome | Consequence |
|---|---|
| JB2 not met in any model | The position account is false for that model. |
| JB4 not met | Keys carry identity without later mentions, and "copied" is false. |
| JB1 not met in 2 or more new families | The list lookup is not a general property. |
| JB3 not met in 2 or more new families | The paper must restate the sentence effect as specific to Qwen, Mistral and OLMo, and keep only the list result as general. |
| JB5 not met | The log-probability account does not translate into generated behaviour. |
| JB6 not met | The published sentence and no-mention numbers depend on the scoring; the paper switches to E-scored numbers as primary and says so. |
| JB9 not met | The S0 numbers do not generalise across story samples. |

Each outcome has a pre-written sentence in the entry, as in preregistration I's alternatives table.


### code_plan

**New and changed files** (the builder writes them; the prototypes are in the scratchpad):

1. **ckeys/surface.py** (new; prototype `scratchpad/s8design/partB/surface.py`, pilot-verified).
   - `FRAMES_SIGMA`, `FRAMES_E_FIXED`, `form_ids(tok, form, prefill="Answer:")`.
   - `class FormSet(tok, words, frames, names=None)`:
     - per-word token sequences, deduplicated;
     - trie nodes with parents before children;
     - asserts: decode(seq) == form, and no sequence is a proper prefix of another;
     - instantiation of discovered frames with an item's {a, b, o, d}.
   - `trie_inputs(ids, fs, impl, dtype)` → (ids2, position_ids, 4D mask). The mask is bool for sdpa and finfo.min-additive for eager, following ckeys/knockout.py.
   - `score(model, ids, fs, topk=10)` → lower, Σ, E, per-form log-probs, masses, first-token top-k and first-token gap. It uses logits_to_keep = N+1 and a float32 log-softmax chunked over rows.
   - `score_reference` (tests only).
   - `per_node_fallback` (separate passes, used if JB-G3 fails).
2. **ckeys/generate.py** (new).
   - `greedy_answers(model, tok, ids, ctx, max_new=12)`: use_cache=False, clamps active through `ctx`, per-row early stop on the candidate regex, newline or EOS.
   - `parse_answer(text)`; `CAND_RE`.
   - `extract_frame(text, core)`, which substitutes {a, b, o, d}; `discover_frames(gens, rule)` implements B3.2.
3. **ckeys/clamp.py** (edit).
   - `kv_sites`: if `blk.self_attn` has `qkv_proj` (Phi-3/Phi-4), return (qkv_proj, slice(q, q+kv)) and (qkv_proj, slice(q+kv, q+2kv)), with q = num_attention_heads·head_dim and kv = num_key_value_heads·head_dim. The slicing was checked against transformers 5.18 modeling_phi3.py, lines 229–233.
   - Nothing else changes. The existing GPT-2 fused path shows that sequential hooks on one module compose.
4. **ckeys/story.py** (edit). Add `fresh_cores(n, seed, exclude_seeds=(0,1,2,3), exclude_n=1000)` and `POPULATIONS = {"F": (150, 20261013), "C": (30, 20261014), "S0": (150, 0)}`. `make_cores` is unchanged.
5. **experiments/fresh_factorial.py** (new; reuses format_factorial.row_specs/LABEL, encoding.encode/build_prompt, clamp.capture_kv/clamp_kv/stack_rows and story.pick_x/record).
   - `run_item(model, tok, core, arm, fs, gen=True, plain=False)` keeps format_factorial's item schema, with 'clean' and 'm' computed from the trie pass's lowercase. It adds:
     - 'sig' and 'E' per row for S, B, X and init;
     - all six candidates in the clean runs;
     - 'mass' {L, Σ, E} per clean run;
     - 'first_top10' and 'gap';
     - 'gen' (raw strings) and 'ans' (parsed) for the 8 generated rows;
     - with `plain=True`, 'plain_m', the published-path pass.
   - CLI: `--model --revision --local-dir --population {F,S0,C} --arms --n --dtype bfloat16 --attn auto --max-new 12 --frames frames_{model}.json --plain --out`.
   - Mode C (calibration) writes the generations and frames_{model}.json with its sha256.
   - Provenance: git commit, versions, revision, VERIFIED.json hash list, attention implementation, system_merged, FormSet node counts and sequences, population hash, frames hash.
6. **experiments/paper1_frames.py** (edit behind a flag, JB8). `--score E` routes `lp_rows` through `surface.score` and stores the L and E candidate vectors. The default path is untouched, and a unit test checks that L under the flag equals the old `lp_rows`.
7. **scripts/stage8b_models.json** and **scripts/fetch_verified.py** (new).
   - The manifest holds, per model: the official repo and revision; per-file expected sha256 or git-blob id (from `scratchpad/s8design/partB/hf_manifest.json`); an ordered list of byte-identical sources; allow_patterns (Mistral: the HF shards only).
   - `fetch_verified.py` downloads, verifies (sha256 for LFS; SHA-1 of "blob {size}\0"+content for small files), assembles a local directory and writes VERIFIED.json. It refuses on any mismatch.
8. **scripts/gpu_stage8b.sh** (new; stage-7 pattern).
   - Guards: refuses a DRAFT J entry, a modified tree, or code that differs from the commit 'Finalise preregistration J'; pins transformers 5.18.0; GPU memory floor; TEST_MODE runs on Qwen2.5-0.5B FP32 CPU with n = 3.
   - Steps (keep/FORCE semantics as in stage 7): pytest (JB-G0) → for each model {fetch+verify, tokenizer check JB-G0b, R0 calib, R1 or R2/R3}, prefetching the next model in the background and deleting weights afterwards → deadline-guarded R4, X1, X2, X3 → score.
   - Output: gpu_stage8b_results.tgz.
9. **analysis/stage8b_score.py** with **analysis/stage8_parts/b_stats.py**, **b_verdicts.py** and **b_tables.py** (new).
   - `cluster_index(keys, B=10000, seed=20261013)` returns two-stage indices, one set per (model, population).
   - Statistic functions for ID^σ, r, s_ID, δ, M, β, b_ID and A, with definedness rules.
   - Gates JB-G2 to G5; verdicts JB1–JB9 with risk class; a summary of met rates by risk class.
   - Paper tables: tab_fresh (8 models × 2×2: r, s_ID and β), tab_mass (L/Σ/E mass per cell), tab_invariance (S0: L against E), tab_behaviour; numbers.tex macros.
10. **Tests (JB-G0).**
    - tests/test_surface.py: the pilot checks at 1e-4, plus FormSet invariants on the cached tokenizers.
    - tests/test_clamp_fused.py: tiny Phi3; exact C_KV(S) reproduction; channel slices only.
    - tests/test_generate.py: greedy_answers equals the stepwise argmax under clamps; parse cases " The shelf", " Boxes", "**Shelf**", " On the shelf", " I don't know" → other; frame extraction with placeholders.
    - tests/test_fresh_factorial.py: the lowercase fields equal format_factorial.run_item on 2 items; the populations (F disjoint from U, size, hash; C disjoint).
    - tests/test_stage8b_score.py: synthetic items with known ID and β give known verdicts, including NOT EVALUABLE and κ-style undefined paths; the cluster bootstrap is deterministic.
11. **docs/PREREGISTRATION.md, entry J part B.** The text of B1–B11, the "seen before finalisation" list (stage 1/3b/5 committed numbers, tokenizers and HF metadata, the CPU pilots below), the pinned F hash and the manifest hash. It is committed with the scorer and code before any GPU run.


### pilot_results

All runs were on CPU (4 cores, shared with another process), FP32. The scripts and logs are in `/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partB/`. No study model was run on fresh cores, and seed 20261013 (F) was never touched. These pilots must be disclosed in the entry.

**(1) Tokenizer and template check** (`tok_check.py`, `tok_check.json`; tokenizers only).
- Llama-3.1-8B (unsloth tokenizer), Gemma-2-9B, Phi-4 and Falcon3-7B:
  - all six lowercase " w" and capitalised " W" forms are single tokens;
  - 0 of 150 fresh cores (Random(8)) are skipped in any of the 7 arms;
  - Gemma's template rejects the system role, so it is merged (already handled).
- Mistral-7B: " Shelf", " Drawer", " Cabinet" and " Closet" are multi-token, which is why the chain rule is needed (44 trie nodes).
- Infeasible: granite-3.1-8b (" shelf" is two tokens; 145/150 items skipped) and deepseek-llm-7b (" cabinet" is two tokens).
- Fallback: Yi-1.5-9B-Chat passes the same check.

**(2) Hugging Face metadata** (`hf_manifest.py`, `hf_manifest.json`).
- The official gated repos' safetensors sha256 equal those of the ungated copies:
  - Llama: NousResearch (weights, config, generation_config, tokenizer.json, special_tokens_map) and modularai (tokenizer_config.json);
  - Gemma-2-9B: unsloth (weights and tokenizers), thr3a (config.json) and dnhkng (tokenizer_config.json);
  - Gemma-2-27B: weights in unsloth.
- All revisions and file hashes were captured, including the P4 revisions used in stages 5 and 6.

**(3) Fresh populations.** Seed 20261013 gives 4 of 400 stream cores in U. The first 150 cover all 30 ordered pairs and all 20 objects. C (seed 20261014) has 30 cores and is disjoint.

**(4) Exactness of the trie scorer** (`surface.py`, `test_surface_pilot.py`, `exact_pilot.log`; 1280 s).
- Tiny random-weight Qwen2, Mistral, Llama, OLMo2 (sdpa and eager), Gemma2 (eager, softcapping) and Phi3 (sdpa and eager): the trie equals separate passes to ≤ 3.8e-6 nats, and the answer-position log-probs equal a plain forward to ≤ 1.9e-6.
- Qwen2.5-0.5B (trained), sdpa POST with E forms (24 nodes) and eager NONE with Σ, clean and under K_S/V_S clamps: ≤ 4.7e-5. All pass at 1e-4.

**(5) Mini factorial at Qwen2.5-0.5B** (`mini_pilot.py`, `mini_qwen05.log`; n = 6 cores from seed 20261011, not F; about 100 s per item on the contended CPU). The AFTER arm was cut by the timeout.

| Quantity | NONE | POST |
|---|---|---|
| Candidate mass L / Σ / E | 0.135 / 0.694 / 0.873 | 0.238 / 0.800 / 0.912 |
| s_ID under L / Σ / E | +0.020 / −0.002 / −0.001 | −0.146 / −0.171 / −0.181 |
| β_K / β_V | 0.00 / 0.75 | 0.00 / 0.58 |
| Agreement of the Σ-argmax with generation | 0.97 (n = 30) | 0.97 |

- Top first tokens under NONE include ' In' 0.11 and ' On' 0.09, and generations include ' On the shelf'. The locative frames are needed.

**(6) Coverage at Llama-3.2-1B-Instruct** (`coverage_pilot.py`, `cov_llama1b.log`; ≤ 1.5B, clean runs only, n = 8 and 12; unsloth mirror). This previews the Llama family.

| Arm | L | Σ | E (fixed frames) |
|---|---|---|---|
| NONE | 0.017–0.025 | 0.25–0.29 | 0.43–0.47 |
| POST | 0.017–0.025 | 0.23–0.28 | 0.49–0.52 |
| PRE | 0.023 | 0.29 | 0.54 |
| AFTER | 0.85 | 0.93 | 0.93 |
| BEFORE | 0.32 | 0.85 | 0.86 |

- The first tokens are spread over ' On' 0.29, ' In', ' Under', ' Behind' and ' Top', and generations read ' On the shelf' and ' In the basket'.
- Clean accuracy is weak (0.25–0.75).
- Conclusion: the reviewer-named Σ set alone would cover under 30% in the Llama family at this size. This is why E (locative frames plus calibration-discovered frames) is primary, why coverage is gated per cell (JB-G4) and predicted (JB7), and why the generation-based β is a parallel measure.

**(7) In-hand scale-free statistics** (`power_inhand.py`; committed stage-3b rows, lowercase, n = 150; core bootstrap vs location-pair cluster bootstrap).

| Statistic | Qwen2.5-7B | Qwen2.5-14B | Mistral-7B | OLMo-2-7B |
|---|---|---|---|---|
| r(POST) | 0.268 | 0.337 | 0.506 | 0.233 |
| s_ID(AFTER) | 0.777 | 0.863 | 0.819 | 0.593 |
| s_ID(NONE) | 0.059 | 0.052 | 0.039 | 0.033 |

- r(BEFORE) and r(PRE) are −0.035 to +0.000.
- The cluster-bootstrap SDs are 1.5–2× the core SDs.

**(8) Behavioural proxy** (argmax over the 4 stored candidates, stage 3b).

| Quantity | Range at P4 |
|---|---|
| β_K(AFTER) | 0.82–1.00 |
| β_V(AFTER) | 0.00–0.19 |
| β_V(NONE) | 0.92–1.00 |
| β_K(NONE, BEFORE, PRE) | 0.00 |
| β_K(POST) | 0.02–0.30 |

**(9) Power simulation** (`power_sim.py`, `power_jb2.py`). See B8.


### paper_payoff

**Main text.** Table 1 is replaced by "Fresh stories, eight families, scored on what the models say":
- rows: 4 original models and 4 new ones (Llama-3.1-8B, Gemma-2-9B, Phi-4, Falcon3-7B);
- columns: r and s_ID for LIST-AFTER, SENTENCE-AFTER, LIST-BEFORE, SENTENCE-BEFORE and NO-MENTION, plus β_K and β_V;
- each line marked confirmatory and risky, or replication.

One figure shows the behavioural crossover: the share of generated answers changed by the key alone versus the value alone, per format, per family.

**Headline sentence** (wording depends on the outcome). "On 150 new stories in eight model families, four of them never examined, clamping only the writing token's key changes the generated answer to the clamped value in X–Y% of stories when the candidates are listed after it and in at most Z% when they are not, while clamping only its value does the reverse. Scored on the forms the models actually emit (candidate mass ≥ M in every cell), the identity key share moves by at most 0.0x from the published lowercase numbers."

**Appendix.**
- A mass-per-cell table (L, Σ, E).
- An invariance table on the original cores.
- The frame census.
- The FP32 anchor.
- Hash-verified weights for the gated models.

**Expected effect on the panel.**
- **Stats reviewer (5).** It listed exactly these as +1 (fresh-sample replication with scale-free margins in Llama/Gemma) and +0.5 (re-scoring with mass per cell). It also asked for cluster-aware CIs, risk-separated met rates and pinned revisions, all delivered. Expected 6 and soundness 3 → 4.
- **Skeptic and mechinterp reviewers.** Both named the non-emitted-token scoring as a moderate weakness and asked for re-scoring of Tables 14–16 and 22 (+0.5). JB6 does this for Tables 14–16 and Table 1, and JB8 for Table 22 at 24B. The skeptic's eager/sdpa and FP32 question is answered by X1. Soundness → 4.
- **Generalist.** "7 of 10 are Qwen" and "measurement validity in the free-form formats" are both resolved. A behavioural, generation-level statement also gives the broad ICML reader something concrete.
- **Contribution and generality.** These move moderately: Part B keeps the templated task (natural data is Part A's job), but shows the regularity is not a Qwen or template-sample artefact.
- **If JB3 fails in new families.** That is reported as a sharp, preregistered boundary (lists general, sentences family-dependent), which reviewers usually reward over an untested claim.


### risks

1. **The sentence effect may not transfer.** If JB3 is not met in at least 2 new families, the main-text claim about sentences narrows. The list result (JB1, JB2, JB4, JB5) carries the paper, and the narrowing is pre-written.
2. **Coverage in verbose families.**
   - Llama and Gemma may answer "Alice believes the candle is in the shelf" or "**Shelf**".
   - Mitigations: calibration frame discovery with name placeholders, the "**" frames, the per-cell coverage gate, and β, which is immune.
   - Residual risk: a cell with E-mass below 0.5 becomes NOT EVALUABLE for the E-based lines. The Llama-3.2-1B pilot reached only about 0.5 with fixed frames, so this is a real risk at 8B. Discovery should lift it.
3. **Numerics of the 4D tree mask.** An explicit mask disables flash sdpa, so BF16 numerics differ from the published path. Stage 5 already saw eager vs sdpa move ID_K by 10–20% at Qwen2.5-7B. JB-G3 measures the trie floor against a plain pass. JB6 compares L and E within the same pass. X1 gives an FP32 anchor. If the floor gate fails, the exact per-node fallback costs about 3× more for that model.
4. **Gated models.** The byte-identical mirrors could disappear before the run; in that case use HF_TOKEN or the Yi-1.5-9B fallback. The manifest pins hashes, so a changed mirror is refused, not silently used.
5. **Phi-4 needs the fused-qkv clamp site** (a new code path); it is gated by an FP32 exactness test, with the Yi fallback.
6. **Gemma-2.** It needs eager attention with softcapping (verified on a tiny config in the pilot), and its system turn is merged into the user turn. That changes the wrapper relative to the other models, so it is recorded and disclosed.
7. **Generation.**
   - Generation under clamps with use_cache=False is exact, but greedy decoding can differ from the E-argmax at near-ties (agreement was 0.97 at 0.5B).
   - Answers naming no candidate ("I don't know") count as "other" and shrink β. JB-G5's competence gate and the reported "other" rate cover this.
8. **Scale-free ratios.** If a new family's LIST-AFTER key read is small (under 1 nat), r is unstable. The JB-G5 anchor makes those lines NOT EVALUABLE instead of noisy.
9. **Budget.** The core needs about 2.5 GPU-hours of the 8–12 h shared by all of stage 8. JB8 and the exploratory runs are deadline-guarded, and LEAN=1 saves about 20 min.
10. **Overlap with other parts.** Part A may also add new families or a natural-data task. The Part-B scorer and FormSet are reusable there (E scoring on multi-token values needs the chain rule, which is already exact).
11. **Optimism of the P4 lines.** The P4f lines are close to certain by construction (L class). The entry and the paper must not count them as risky confirmations; the summary prints met rates by risk class.


