LABEL design:mechanism
### part
PART D: Mechanism beyond "duplicate-token heads": the identity flag the readers write, the sign of the key read, and what QK/OV plus the causal mask do not imply (preregistration J, part D; stage 8)

### objections_answered
[
 "Obj. 1 (novelty: QK/OV plus duplicate-token heads already imply it; the position effect follows from the causal mask). Answered by (i) characterising the intermediate variable: the readers write a per-layer rank-1 flag into the matched word's own row. Tests: sufficiency by injection, necessity by directional ablation, and the hop-2 route by key. (ii) A polarity account for the sign: the same frozen readers and the same flag give positive or negative identity effects depending on the downstream reader. (iii) A precise statement, part C of the design, of what the architecture fixes and what it leaves open.",
 "Reviewer 1's '+1' request: identify a low-rank 'matched' direction, inject it into an option row with no key clamp, show the answer moves to that option, and use it to explain the 1.5B/3B dissociation (D1-D4, D6).",
 "Reviewer 1's '+0.5 to +1' request and reviewer 2's question on the sign: are the negative key reads (IOI INLINE, IOI BEFORE, LIST-BEFORE) carried by the same duplicate-token readers feeding suppression-type downstream readers? Addressed by D7-D9, which include a preregistered sign prediction on a new configuration (the 'mentioned'/'not mentioned' question pair). The structural point on LIST-BEFORE is stated, and IOI BEFORE gets an exploratory localisation.",
 "Obj. 2, in part (generality and anomalies): the IOI in-sentence negative read becomes a tested prediction of the same readers, not an anomaly. 'Later mentions decide' is restated as: later mentions decide whether the key is read, and the reader decides the sign.",
 "The 1.5B/3B dissociation ('attention present without the read', G2) is left unexplained. D6 decides between 'never written', 'written but not read' and 'written but read with a scale-dependent polarity', using the competent-core population and case-marginalised scoring, as reviewers 2 and 3 asked.",
 "Circularity of the a3 ranking (H1/H2 'partly expected' because heads are ranked by the attention change under the same key clamp). The injection and directional-ablation tests use no key clamp at evaluation, and the flag is fit on R and evaluated on fresh stories.",
 "H5 'answer reads the option words mainly by key' was an unscored secondary contrast. D4 scores it for the injected flag (r_ans^inj(K) against r_ans^inj(V)).",
 "The 'canonical duplicate-token core' depends on set size. D8 uses the frozen full top-k* set (40 / 52 heads) from stage 6 and tests it out of task: on IOI and on the new question pair.",
 "Reviewer 2: 'the identity is already in the option word; p's key only serves as a match target, so key carries identity over-reads'. Conceded and made precise. The flag is identity-free (a pointer): injected into the row of an option absent from the story, it moves the answer to that option. D5 (secondary) tests whether the flag carries the role of the matched occurrence (event location against initial location).",
 "Obj. 4 (scoring on near-zero-mass lowercase tokens) as it bears on part D. Every list-format outcome is reported on candidate-renormalised log-probs (pilot candidate mass 0.98-1.00). The sentence-format dissociation (D6) uses case-marginalised candidates on competent cores."
]

### design

# PART D: what the reader heads write, why the key read can be negative, and what QK/OV does not imply

## 0. Summary of the account under test

Hop 1 is a duplicate-token match: a later copy of a candidate word attends to the writing token p. The heads that make the match (H*) write a flag into the matched word's own row. Per layer, the flag is a single vector, the same in every story. It is identity-free: it points to the row it sits in. The answer then finds the flagged row by key (hop 2) and outputs that row's own token.

Under a causal mask this flag is written wherever a candidate is re-mentioned after p. What it does is set by the downstream reader:
- positive when the question asks for the event location or 'which is mentioned';
- negative when the question asks 'which is not mentioned';
- negative for IOI's in-sentence re-mentions;
- (pilot) negative for sentence re-mentions at small scale.

Two things change across format and scale: whether the flag is read, and with which sign. Whether it is written does not change.

## 1. Models, material, sets

**Models** (BF16, A100-80GB; all non-gated):
- Primary: Qwen2.5-7B-Instruct (rev a09a354) and Mistral-7B-Instruct-v0.3 (rev c170c70), the stage-6 models.
- Scale series for D6: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct.
- No HF token is needed.

**Reader set H\*.**
- At 7B it is frozen from `results/gpu_stage6/heads/<model>.json`: `arms.P1.rankings.a3[:kstar]`, with k* = 40 (Qwen-7B) and 52 (Mistral-7B). It spans 18 / 19 layers.
- The random sets are the stage-6 `sets.rand` (the first k* heads of the three stored permutations).
- The sha256 of the source file and of the canonical JSON of H* are asserted.
- At 1.5B and 3B, H* is ranked here with the stage-6 formula a3 (eager attention; `Stage6.base_runs`, phase 1 only) on R under OPTIONS-AFTER (P1), with k* = ceil(0.05 × n_heads): 17 at 1.5B, 29 at 3B.
- L* = the layers containing an H* head.

**Stories.**
- R = make_cores(60, Random(0)): the stage-6 ranking set. Flags are fit here.
- E8 = make_cores(100, Random(81)): fresh. It is asserted disjoint, by the full core tuple, from the seed-0 150 cores (of which R is the first 60), from the stage-6 E (Random(1), 60) and from every other stage-8 seed.
- R' and E8' = the cores of R and E8 with distractor_location ≠ initial (for D5; n is reported).
- IOI cores (ckeys.ioi):
  - F_ioi = the first 60 valid cores of ioi.make_cores(90, Random(82)).
  - E_ioi = the first 100 cores of ioi.make_cores(150, Random(83)) that are valid (`encode_runs` not None and `check_occurrences` passing) in every IOI arm run for that model.
  - Skipped counts are recorded.

**Formats.**
- P1 (OPTIONS-AFTER) and POST (SENTENCE-AFTER), as in the paper.
- New arms Q_IN and Q_OUT (`ckeys/questions.py`, registered with `register_arm`), identical to P1 except for the question line:
  - `PREFIX + story + "\nQuestion: Which of the choices is mentioned in the story?\n" + LISTING + "\nAnswer with exactly one choice.\nAnswer:"`
  - and `"... is not mentioned in the story?"`.
  - Rows G = the six list words, as in P1.
- IOI INLINE (raw) and IOI AFTER (chat), as in stage 5.
- Queries for D5: `direct` ('Where does {a} believe…', answer B) and `other_agent` ('Where does {b} believe…', answer = initial). Both use the P1 format.

**Eager or sdpa.** sdpa everywhere, except the a3 ranking (1.5B, 3B) and the exploratory hop-2 attention probe, which use eager. use_cache=False on every pass. Every scored quantity is a difference against an in-batch reference row.

## 2. Definitions (exact)

Notation:
- y_l = the o_proj output (attention output) of layer l.
- z_{l,h}[r] = head h's slice of the o_proj input at row r.
- W_O^{l,h} = the o_proj columns of head h (o_proj has no bias in Qwen2 and Mistral).
- K_W = the full key clamp at p, from layer 0: the pre-RoPE k_proj output at p is set to that of the run in which W is written. This is the stage-6 natural clamp.
- r_w = the option row of candidate w.

**Flag, per story s and layer l ∈ L*:**

δ_l(s) = ½ Σ_{h∈H*_l} W_O^{l,h} ( [z^{K_S}_{l,h}(r_S) − z^{B}_{l,h}(r_S)] + [z^{B}_{l,h}(r_B) − z^{K_S}_{l,h}(r_B)] ).

This is the gain of the readers' write at the source row plus its loss at the base row: the matched-minus-unmatched write, averaged over two words.

**Flag:** Δ_l = mean_{s∈R} δ_l(s). Per layer it is a single vector, i.e. rank 1.

**Descriptive statistics:**
- consistency c_l = mean_s cos(δ_l(s), Δ_l);
- energy share φ_l = ‖Δ_l‖² / mean_s ‖δ_l(s)‖²;
- the cross-layer cosine matrix of Δ;
- the projection of y_l at the initial- and distractor-location rows (story duplicates) onto Δ̂_l, against absent-word rows. This says whether story-mention duplicates carry the same flag.

**Other flags.** The same formula gives:
- Δ^POST (POST rows, on R);
- Δ^in, the initial-location flag: the clamp is at the initial-location token p_i of the queried object, with the key of the story whose initial location is I'. I' is the location absent from {init, dloc, B, S, X}, which exists on R'. The rows are r_init and r_I';
- Δ^IOI: IOI INLINE, F_ioi, the listed IO_S and IO_B rows.

**Injection.** I(row, α, v) adds α·v_l to y_l[row] for every l ∈ L*, in the clean base run. There is no key clamp.

**Outcomes** (last position, full-vocabulary log-softmax):
- ℓ_w = log p(w) − log Σ_{c∈C} p(c): candidate-renormalised, over the six candidates (four names in IOI).
- Under POST, p(w) is case-marginalised: the sum over the single-token forms {' w', ' W', 'w', 'W'} that exist in the tokenizer, recorded per word. This is harmonised with the stage-8 rescoring part.
- m_X = log p(X) − log p(B), the paper's unrenormalised margin, is reported beside it.
- X = pick_x(core): absent from the story and ≠ B, S.

**Natural reference:** N_X = ℓ_X(K_X) − ℓ_X(none).

**Injection ratio:** ι(cond) = mean_s[ℓ_X(cond) − ℓ_X(none)] / mean_s N_X, a ratio of means recomputed in every bootstrap resample.

**Behaviour:** π_X(cond) = the fraction of stories whose argmax over C is X.

**Directional ablation** A(u, μ): for l ∈ L* and r ∈ G,

y_l[r] ← y_l[r] − (⟨y_l[r], u_l⟩ − μ_l) u_l,

where μ_l is the mean over R and the absent-word rows (words not in {B, init, dloc}) of ⟨y_l[r], u_l⟩ in the clean run. It is applied in every row of `format_factorial.run_item` (13-row batch and its three clean passes, as H3 did).

ρ_K(A) = mean ID_K(A) / mean ID_K(none), with ID_K as in the paper (lowercase candidates, l0 = 0).

**Hop-2 path** (HopSplice, rows G; clean-run K and V of G captured in-batch):

r_ans^inj(C) = 1 − [ℓ_X(move + ans_C) − ℓ_X(none)] / [ℓ_X(move) − ℓ_X(none)], for C ∈ {K, V, KV}.

Here only the answer row reads the clean run's C at the option rows, while the injection is in place.

**Identity contrast of a configuration:** ID_K(cfg) = ½[c(K_S) − c(K_X)], where c = lp(S) − lp(X) relative to the self-clamp row (run_item; IOI: ioi_factorial.run_item / identity_measures).

**Head transfer** (HeadSplice, ks per batch row = K_S or K_X; rows G of the configuration):

id(Set) = ½[(c(Set; K_S) − c(∅; K_S)) − (c(Set; K_X) − c(∅; K_X))].

- d_G = id(allG): every head in the rows G.
- R(H*) = mean id(H*) / mean d_G.
- KO(H*) = 1 − mean id(allG∖H*) / mean d_G.

These are signed ratios of means, so they are valid for negative reads.

**Dissociation statistics:**
- W(m,f) = Σ_{l∈L*} mean_{s∈E8} ⟨δ^f_l(s), Δ̂^{P1}_l⟩ (the write along the model's own list flag, on held-out stories);
- ω(m) = W(m, POST) / W(m, P1);
- κ(m) = the norm-weighted mean over l of cos(Δ^POST_l, Δ^P1_l);
- ρ(m) = mean Δℓ_X(POST; +Δ^{P1} at the sentence row of X) / mean Δℓ_X(P1; +Δ^{P1} at the list row of X), computed on competent cores: argmax over C, case-marginalised, correct in both clean B and S runs in both formats.

**Statistics.** 95% percentile intervals from 10,000 story bootstrap resamples (seed 20261010, one fixed index set per n), with ratios recomputed in every resample. A two-model prediction is MET if met in both evaluable models, NOT MET if not met in one, and NOT EVALUABLE otherwise.

## 3. Gates

- **Gate D0 (exactness, FP32, CPU, Qwen2.5-0.5B, `tests/test_flag.py`, runs before any model loads):**
  - zero injection equals clean (1e-5 in logits);
  - injection at the last layer changes the final hidden state at the row by exactly the added vector and leaves the other rows unchanged (1e-5);
  - batched rows equal single runs (1e-4);
  - ablation with μ set to the row's own projection equals clean, and after ablation the projection equals μ (1e-5);
  - Σ_h head_out(z_h) equals the o_proj output (1e-4);
  - Inject composes with HopSplice and HeadSplice: a zero add leaves their outputs unchanged (1e-4);
  - HeadSplice with per-row ks in {K_S, K_X}: an all-heads, all-rows row equals the full K_S or K_X clamp of that row (1e-4);
  - the Q_IN, Q_OUT and IOI-row encodings are checked on 50 cores × all tokenizers: B, S and X differ only at p, and the rows G lie after p.
  - Failure stops the run.
- **Gate D1 (the natural effect exists, per model):** mean N_X ≥ 3 nats with the CI excluding 0 under P1 on E8. Otherwise D1-D4 are not evaluable in that model.
- **Gate D2 (batch floor):** in every batch a duplicated none row agrees with the first to ≤ 0.05 nats (mean |Δ|).
  - It does not compare against single passes: stage 6's Gate a2 failed on batch-shape offsets that cancel in in-batch differences.
- **Gate D3 (configuration competence, per model × arm, clean B run over C):**
  - Q_IN: the argmax is a mentioned location in ≥ 0.9 of stories;
  - Q_OUT: the argmax is an unmentioned location in ≥ 0.9;
  - candidate mass ≥ 0.5;
  - IOI arms: stage 5's Gate e on E_ioi.
  - A failing cell is not evaluable.
- **Gate D4 (transfer evaluable, per cfg):** |mean d_G| ≥ 1 nat with the CI excluding 0, the same sign as ID_K(cfg), and |d_G| / |ID_K(cfg)| ≥ 0.5 (the re-mention rows carry the read).
- **Gate D5 (role test evaluable):** under `other_agent`:
  - accuracy (argmax = init) ≥ 0.8 on E8';
  - the natural initial-location read Δℓ_{I'}(K_{I'} at p_i) ≥ 1 nat with the CI excluding 0.
- **Gate D6 (dissociation population):**
  - ≥ 40 competent cores per model in POST;
  - the P1 injection Δℓ_X > 0 with the CI excluding 0 in each model (the flag is read in lists);
  - at 1.5B and 3B, the in-run H* passes R(k*) ≥ 0.6 under P1 on E8, which shows the ranked heads are the readers.

## 4. Confirmatory predictions

Each prediction is stated with what makes it risky.

### (i) The identity flag (Qwen2.5-7B, Mistral-7B; P1; flag fit on R, evaluated on E8, n = 100)

**D1. Sufficiency: a flag placed on an absent option moves the answer there.**
- "move" (+Δ at r_X, −Δ at r_B, α = 1): ι ≥ 0.5 with lower bound ≥ 0.35, and π_X(move) ≥ 0.5.
- Add-only (+Δ at r_X): ι ≥ 0.25 with lower bound > 0.
- Dose response monotone on point estimates: ι(0.5) < ι(1) < ι(2).
- Risk: nothing in hand shows that a mean vector, rather than a story-specific or non-linear write, suffices at 7B. Stage 6 only ablated whole heads.

**D2. Specificity.** Each of the following must hold:
- isotropic random directions, norm-matched per layer (3 draws): |mean ι| ≤ 0.10, and each draw |ι| ≤ 0.20;
- head-span random vectors, Σ_{h∈H*_l} W_O^{l,h} z_h with z ~ N(0, I) rescaled to ‖Δ_l‖ (3 draws; directions the readers could write): the same bounds;
- the layer-permuted flag (a fixed derangement of L*, rescaled): ι ≤ 0.25;
- +Δ at the non-option rows 'Choices' and 'Question': |ι| ≤ 0.10 and π_X unchanged within 0.05.

**D3. Necessity: one direction per layer carries the key read.**
- A(Δ̂, μ) at the six option rows: ρ_K ≤ 0.5 with upper bound ≤ 0.6.
- Three isotropic unit directions with their own μ: ρ_K ≥ 0.85 each.
- Run on the first 60 cores of E8.
- Risk: stage 6's whole-head mean-ablation removed 0.76 / 0.94. A single direction per layer is a far smaller intervention.

**D4. Route: the injected flag is read at hop 2, by key.**
- r_ans^inj(KV) ≥ 0.6 with lower bound ≥ 0.45.
- r_ans^inj(K) − r_ans^inj(V) > 0 with the CI excluding 0.
- This also scores H5's unscored secondary contrast.

**D5. (secondary confirmatory; Gate D5) The flag carries the role of the matched occurrence.** On E8', with + at r_X:

Ψ = [Δℓ_X(Δ^ev | direct) − Δℓ_X(Δ^ev | other_agent)] − [Δℓ_X(Δ^in | direct) − Δℓ_X(Δ^in | other_agent)] > 0 with the CI excluding 0.

Both simple contrasts must also go the predicted way with CIs excluding 0:
- Δ^ev beats Δ^in under `direct`;
- Δ^in beats Δ^ev under `other_agent`.

The required outcome holds in 2/2 models. cos(Δ^ev, Δ^in) per layer is reported. The rival, a generic duplicate bit, predicts Ψ ≈ 0 and cos ≈ 1.

### (i, cont.) The 1.5B/3B dissociation (Qwen2.5-1.5B, 3B, 7B; P1 and POST; competent cores; case-marginalised)

**D6a. Written at every scale.** ω ≥ 0.6 with lower bound ≥ 0.4, and κ ≥ 0.7, in 3/3 models.

**D6b. One flag across formats.** Δ^POST injected at the P1 list row r_X gives ≥ 0.6 × the effect of Δ^P1 there, in 3/3.

**D6c. The sentence reader's polarity changes with scale.**
- ρ(1.5B) < 0 with upper bound < 0.
- ρ(7B) ≥ 0.2 with lower bound > 0.
- ρ(3B) lies strictly between the two point estimates.

**Decision table (fixed now):**

| Outcome | Condition |
|---|---|
| Written; sentence read with scale-dependent polarity (author's prediction) | D6a and D6c met |
| Written, not read | D6a met; \|ρ\| ≤ 0.1 with CI within ±0.2 at 1.5B and 3B; ρ(7B) ≥ 0.2 |
| Never written | ω ≤ 0.3 at 1.5B and 3B, and ≥ 0.6 at 7B |
| Mixed | anything else, reported as such |

Risk: the 7B line and the 1.5B/3B lines are untested. The prediction rests on a disclosed 0.5B pilot (the 1.5B pilot touched only Q_IN/Q_OUT; 3B is untouched).

### (ii) The sign of the key read

**D7. Sign set by the question, new configuration (Gate D3).**
- ID_K(Q_OUT) < 0 with upper bound < 0, and mean ≤ −1.0 nat.
- ID_K(Q_IN) > 0 with lower bound > 0, and mean ≥ 1.0.
- Paired ID_K(Q_IN) − ID_K(Q_OUT) > 0 with the CI excluding 0.
- All in 2/2 models.
- Behavioural correlate (secondary): under K_S in Q_OUT, Δℓ_B > 0 with the CI excluding 0. The written word B becomes a 'not mentioned' answer.
- Rivals: 'salience' (the key read raises S under any question) predicts a positive Q_OUT read; 'Q_OUT is solved without the option lookup' predicts ≈ 0.

**D8. The same frozen readers carry reads of both signs (Gate D4 per cfg).**
- For cfg ∈ {Q_IN, Q_OUT, IOI-INLINE (both models), IOI-AFTER (Qwen-7B only; Mistral AFTER failed stage-5 Gate e)}:
  - R(H*) ≥ 0.6 with lower bound ≥ 0.45;
  - KO(H*) ≥ 0.6 with lower bound ≥ 0.45;
  - stage-6 random sets: mean R_rand ≤ 0.15 and mean KO_rand ≤ 0.15.
- Met in a model only if met in every evaluable cfg. Overall needs 2/2.
- IOI-INLINE (−) and IOI-AFTER (+) at Qwen-7B give a within-task sign flip with the same heads.
- Risk: H* was ranked on belief P1 only. Transfer to a different task and to negative reads is untested at 7B.

**D9. The same flag, opposite effect.** Δ^P1 (fit on belief P1) is injected at r_X in Q_IN and in Q_OUT.
- Δℓ_X > 0 in Q_IN and < 0 in Q_OUT, each with the CI excluding 0.
- The ratio Δℓ_X(inject) / N_X(cfg) ≥ 0.3 in each. This is positive in both, because N_X is negative in Q_OUT.
- Isotropic random (3 draws): |Δℓ_X| ≤ 0.25 |N_X|.
- 2/2 models.

**LIST-BEFORE (stated, not predicted).** No candidate is mentioned after p, so the re-mention readers cannot carry it. At 7B its ID_K is about −0.1 nats, near the BF16 floor (Table 2x2). The paper states this structurally and does not claim to explain it.

## 5. Exploratory (labelled; run after the confirmatory parts, deadline-gated)

- **E1.** Flag geometry: c_l, φ_l, the cross-layer cosines (pilot: no single residual direction), and the story-duplicate projection. Also cos between Δ^P1, Δ^POST, Δ^in and Δ^IOI.
- **E2.** IOI flag injection at the listed IO_X row:
  - Δ^IOI in INLINE (predicted negative) and in AFTER at Qwen-7B (positive);
  - the belief Δ^P1 cross-task.
- **E3.** Where the polarity is implemented (eager, 60 stories), for Q_IN, Q_OUT and P1, at the top-10 hop-2 heads (ranked on R by the attention change from END to r_S under K_S, P1):
  - the attention change END → r_X under +Δ at r_X;
  - the direct logit attribution of each head to X.
  - Readout: 'avoid' (attention to the flagged row falls in Q_OUT) against 'attend-and-suppress' (attention rises, DLA negative). Two-sided, no verdict.
- **E4.** A per-head answer-row HopSplice under the injection: which hop-2 heads carry it.
- **E5.** IOI BEFORE at Qwen-7B (ID_K −2.0): RowSplice localisation with `experiments/row_restricted_keys.py --task ioi` (sentence tail including S2, and the tail/answer rows; n = 60), plus HeadSplice of H* at the localised rows. This asks where a negative read without a later re-mention lives.
- **E6.** All D1-D4 statistics on all cores and on m_X; Holm correction within part D (reported beside the uncorrected verdicts).

## 6. Power (from the disclosed CPU pilots; FP32, Qwen2.5-0.5B unless stated)

- **D1.** Per-story sd of Δm_X(move) is 1.57 against N_X = 3.25. At n = 100 the SE of ι is about 0.05, so the CI half-width is about 0.1. If the true ι is 0.8 (as at 0.5B), P(lower bound ≥ 0.35) > 0.99; at a true ι of 0.5 it is about 0.5. 0.5B is a weak model (K_X moved its argmax to X in 6% of stories), so π_X is untestable there; at 7B, N_X ≈ d_full ≈ 36 nats.
- **D3.** ρ_K was 0.08 for the flag against 0.99 for random directions.
- **D7.** At 1.5B, ID_K(Q_OUT) = −4.17 (sd 1.42; 12/12 negative), so at n = 100 the SE is about 0.14.
- **D8.** At 0.5B on IOI INLINE, R = 0.95 and KO = 0.97 (n = 24).
- **D6.** At 1.5B, natural ID_K(POST) = −0.53 [−0.64, −0.41] (n = 150, stage 5). An injection effect of 0.3-0.5 nats with sd about 0.5 has SE about 0.06 at n = 80 competent cores.
- **Cell sizes.** n = 100 per cell; 60 for D3 and E3.

## 7. Compute (one A100-80GB, BF16)

Calibrated on stage 6 (about 0.3 s per batched forward with hooks; 21 s per story for about 70 calls).

| Block | Forwards per story | GPU time |
|---|---|---|
| Per 7B model: fits on R (P1, POST, initial-location flag, IOI) | ~720 forwards in total | ~3 min |
| Injection battery (≈ 20 rows) + path batch | 5 calls × 100 | ~4 min |
| Ablation | 20 calls × 60 | ~6 min |
| Q_IN / Q_OUT: run_item + transfer batch (≈ 22 rows) + injection | ~12 calls × 2 × 100 | ~12 min |
| IOI INLINE (+ AFTER at Qwen) | ~10 calls × 100 × 1-2 | ~8 min |
| D5 role test | ~6 calls × 2 queries × 100 | ~6 min |
| Loading and overhead | | ~6 min |
| **Per 7B model** | | **≈ 45 min** |
| Two 7B models | | **1.5 h** |
| 1.5B + 3B: ranking, fits, two-format injection | | **≈ 0.3 h** |
| Exploratory E3-E5 | | **≈ 0.3 h** |
| **Part D total** | | **≈ 2.1 GPU-h (budget 2.5 h)** |

Cut order if the budget is short: E3-E5, then D5, then the Mistral line of D6.

## 8. What would count against the account

| Outcome | What it would mean |
|---|---|
| D1 or D2 fails (ι < 0.35, or random or head-span vectors as effective) | The key read is not mediated by a low-rank additive flag. The paper keeps the routing result only. |
| D3 fails (ρ_K > 0.6) | Other carriers exist. The flag is sufficient but not necessary. |
| D4 fails | The flag acts by a route other than hop-2 keys. |
| D7: Q_OUT positive | 'Salience', not polarity. The sign account is refuted. |
| D7: Q_OUT ≈ 0 | The lookup is task-specific. |
| D8 fails for IOI-INLINE | The negative IOI read is a separate mechanism, and the paper must say so. |
| D9: same sign in Q_IN and Q_OUT | The flag is a generic salience bump. |
| D6: 'never written' or 'written, not read' | The scale dependence is not a polarity change. The unification is dropped. |
| D5: Ψ ≈ 0 | The flag is a role-free duplicate bit. Role selection happens elsewhere, e.g. through values. |

## 9. Claim (iii): what QK/OV plus the causal mask imply, and what they do not

Let p be the writing token. The architecture fixes three things:
- (a) Every format that shares the prefix through p writes the same K and V at p.
- (b) A change of k_p acts on the output only through the attention weights a_{t,p} of later tokens t > p.
- (c) For that change to depend on which value was written, q_t must depend on a candidate's identity. Unless identity is first copied into t, t must be a later mention of a candidate.

From (a)-(c) it follows that tokens before p cannot read k_p, so a list before p opens no read at the list rows, and that without a later mention any key read is second-order.

They do not imply:
- **N1, realisation.** A permitted, realised, key-following match need not change the answer. At 1.5B and 3B the sentence re-mentions attend to p and follow the clamped key one for one (G2), and D6 locates what happens downstream. The mask permits the read; it does not make it.
- **N2, sign.** The same frozen readers and the same flag vector raise the matched candidate for 'where does Alice believe' or 'which is mentioned'. They lower it for 'which is not mentioned' and for IOI's in-sentence re-mentions (D7-D9). Existence is decided by later mentions; sign is decided by the reader.
- **N3, route and content.**
  - The read is a two-hop pointer: the readers write an identity-free, per-layer rank-1 flag into the matched word's own row, and the answer selects that row by key (D1-D4).
  - The identity comes from the row's own token. Under K_S the answer is S, although p's value still encodes B.
  - The key therefore carries identity only as an address. Neither the two hops nor the dimensionality of the write follows from QK/OV.
- **N4, preference.** When both channels are available, the model routes 70-90% of the identity through the lookup (s_ID). Removing the readers raises the value identity (H3d). The architecture does not say which channel is used.
- **N5, no second-order lookup.** The question and answer tokens follow p in every format and could query k_p with an identity copied from an earlier list. Under LIST-BEFORE they do not (≈ 0 at 7B).

Together these make the channel a property of the context, not of the token or the head. The same K, V and readers give a positive, zero or negative identity effect depending on what comes after p and what is asked.


### code_plan
Everything is new code; no tested module is modified. Reuse is noted for each file.

1. **`ckeys/flag.py`** (new)
   - `OCap`: o_proj forward-pre-hook capturing the o_proj inputs at given rows, per batch row, at the layers asked for.
   - `Inject`: o_proj forward hook with two modes, both broadcasting over batch dimension 1 or B:
     - `add = {l: [B|1, T, D]}`, additive;
     - `proj = {l: (u[D], mu, rowmask[B|1, T])}`, directional mean-ablation.
     - It passes through while inactive and has a `_once` guard, as headsplice does.
     - It composes with HeadSplice and HopSplice: they call o_proj inside their wrapped forward, so the injection applies in both passes. This is documented and tested.
   - `head_out(model, l, z, heads)`.
   - `fit_flag(model, tok, cases, Hs, rows_fn, clamp_fn)` → per-story δ [n, |L*|, D], Δ, c_l, φ_l and the cross-layer cosine matrix. The K_S clamp reuses `ckeys.interventions.edits` exactly as `stage6_heads.base_runs` does.
   - Controls: `iso_random(Delta, n, seed)`, `headspan_random(model, Hs, Delta, n, seed)`, `layer_perm(Delta, seed)` (a fixed derangement), `fit_mu(model, cases, dirs)` (absent-word rows).
   - `case_forms(tok, word)` for the case-marginalised scorer. It is shared with the stage-8 rescoring part; if that part ships one, import it.

2. **`ckeys/questions.py`** (new): `register_arm('Q_IN' | 'Q_OUT', build, span=lambda *_: LISTING)`, so that `format_factorial.run_item`, `Stage6.prep` (via `ARM_SPAN`, extended locally) and `row_restricted_keys` work unchanged. Use `--arm-modules ckeys.questions`.

3. **`experiments/stage8_flag.py`** (new; pattern of `stage7_link.py`: `--stage` subcommands, atomic JSON writes, provenance, `TEST_MODE` at Qwen2.5-0.5B FP32 with n = 2, deadline env for the exploratory parts). Stages:
   - **`sets`:** load H* and the random sets from the stage-6 JSON at 7B, asserting the sha256 of the file and of the canonical sets. At 1.5B and 3B, rank a3 on R with `Stage6(model, tok, 'P1', hs, hop).base_runs(d, ks, phase1=False)`, eager, phase 1 only, and record it.
   - **`fit`:** Δ^P1 and Δ^POST on R; Δ^in on R'; Δ^IOI on F_ioi (IOI rows from `ckeys.ioi.RowTask.groups(...)['options']`, which handles BOS offsets); μ for the ablation directions; the controls. Saved to `flags.pt` with a sha256 that every later file records.
   - **`inject`** (E8, P1). One batch per story of 20 rows:
     - none ×2 (Gate D2);
     - add at r_X with α ∈ {0.5, 1, 2};
     - move;
     - −Δ at r_B;
     - iso ×3, head-span ×3, layer-perm ×1;
     - +Δ at 'Choices' and at 'Question';
     - K_X and K_S full clamps, via `clamp_kv(..., per_row=...)`.
     - Then a capture pass (`capture_kv` at G) and a HopSplice batch [none, move, move+ans_K, move+ans_V, move+ans_KV]. Reuse `stage6_heads.configure_hop` with a custom spec, or set `hop.tabs`/`which` directly.
     - Store the six-candidate log-probs, the case-marginalised log-probs and the argmax.
   - **`ablate`** (first 60 of E8): `ff.run_item` under {none, flag-dir, iso-dir ×3}, with `Inject.proj` at the rows G and layers L*. The ID_K and ID_V formulas are copied from `Stage6.ablation`.
   - **`role`** (E8'): queries direct and other_agent. Batch: [none, +Δ^ev at r_X, +Δ^in at r_X, iso, K_{I'} at p_i (gate), K_X at p (reference)]. p_i is found as the single position that differs between the base run and the initial-swap run.
   - **`diss`** (models 1.5B, 3B, 7B; P1 and POST on E8):
     - the δ passes for W and κ (3 passes per story);
     - an injection batch [none, +Δ^P1 at r_X, +Δ^POST at r_X, iso, K_X];
     - competence flags from case-marginalised clean B and S runs;
     - a gate batch at 1.5B and 3B: HeadSplice R(k*) under P1.
   - **`sign`:**
     - Q_IN and Q_OUT: `ff.run_item` (ID_K), a transfer batch (HeadSplice with `ks[l]` stacked per row from K_S or K_X; sets ∅, H*, rand0-2, allG, allG∖H*, allG∖rand0-2, allT; masks from `headsplice.cells_dense`/`head_masks` at G) and an injection batch [none, +Δ^P1 at r_X, iso ×3, K_X].
     - IOI INLINE (both models) and AFTER (Qwen): `experiments.ioi_factorial.run_item` with `ckeys.ioi.identity_measures` for ID_K, and the same transfer batch at the listed-name rows. The exploratory injection batch [none, Δ^IOI and Δ^P1 at the listed IO_X row, iso, K_X] is renormalised over the four names.
   - **`hop2`** (exploratory, eager): attentions at the top-10 hop-2 heads, plus DLA via the head outputs at END times W_U[X] (unembedding after the final norm, frozen at the clean run's norm scale).
   - **`before`** (exploratory): call `experiments/row_restricted_keys.py --task ioi --arms BEFORE` at Qwen-7B, n = 60.

4. **`analysis/stage8_partD_score.py`** (new; pattern of `stage6_score.py`):
   - Gates D0-D6, then D1-D9 verdicts with the bootstrap rules above.
   - Writes `results/gpu_stage8/partD/STAGE8_PARTD_SCORE.txt`.
   - Exploratory lines are printed but enter no verdict.
   - Scorer unit tests on synthetic inputs go in `tests/test_stage8_partD_score.py`: signed ratios, negative denominators, decision-table branches.

5. **`tests/test_flag.py`** (FP32, Qwen2.5-0.5B, CPU, 16-thread cap from conftest): the eight Gate-D0 checks listed in the design, plus `Q_IN`/`Q_OUT`/IOI encoding checks over 50 cores for the Qwen and Mistral tokenizers. The 1.5B and 3B models share the Qwen tokenizer.

6. **`scripts/gpu_stage8.sh`**, part-D block (pattern of `gpu_stage7.sh`):
   - guards: finalised entry J, clean tree, commit match;
   - FP32 tests first;
   - model loop: Qwen-7B → Mistral-7B → 1.5B → 3B;
   - HF cache cleaned between models; `REVISIONS.txt` pinned on first use;
   - `STAGE8_DEADLINE` for the exploratory parts; manifest sha256.

7. **Paper figure:** `make_figures.py` gets `fig_flag` with four panels: (a) dose response with controls; (b) ρ_K for the flag against random directions; (c) the sign panel (Q_IN, Q_OUT, IOI-AFTER, IOI-INLINE: ID_K, R(H*), injection); (d) the scale panel (ω and ρ at 1.5B, 3B, 7B).

**Pilot code** that the builder can lift is in `/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partD/`:
- `flagkit.py`: OCap, Inject, head_out, prep for P1, POST, Q_IN and Q_OUT;
- `pilot_flag.py`, `pilot_ablate.py`, `pilot_sign.py`, `pilot_diss.py`, `pilot_ioi.py`, `pilot_ioi_inject.py`;
- logs `*.log`.

### pilot_results
All runs were on CPU in FP32 with transformers 5.18.0 and torch 2.14.1+cpu, while other parts' pilots were using the same CPU (wall times inflated). Scripts and logs are in `/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partD/`. All must be disclosed in entry J.

**(1) Qwen2.5-0.5B, P1, flag fit and injection** (`pilot_flag.py`; `pilot05_P1.log`, `pilot05b.log`).
- Setup: a3 ranking on 16 stories of R (Random(0)), giving k* = 17 heads in layers 2-16. The flag was fit on the same 16 stories.
- Flag geometry:
  - per-story consistency cos(δ, Δ) is 0.84-0.97 at the main layers (3, 7, 11, 13) and 0.50-0.65 at minor ones;
  - the mean cross-layer cosine is 0.07, so the flag is layer-specific, not one residual direction.
- Injection, on 16 held-out stories (Random(1)), on m_X against the natural K_X effect (+3.25 nats):

| Condition | ι |
|---|---|
| add-only | 0.45 |
| move | 0.80 |
| α = 0.5 / 1 / 2 | 0.19 / 0.45 / 0.69 (monotone) |
| isotropic random (3 draws) | +0.01, −0.05, −0.16 |
| +Δ at 'Choices' | −0.04 |
| +Δ at 'Question' | +0.04 |

- 0.5B is weak behaviourally: clean accuracy is 0.62, and even K_X makes X the argmax in only 6% of stories, so π_X is uninformative here.

**(2) Qwen2.5-0.5B, directional ablation** (`pilot_ablate.py`; 16 fit, 16 eval).
- Ablating one direction per layer at the six option rows: ID_K 1.29 → 0.10 (ρ_K = 0.08).
- Two random unit directions: ρ_K = 0.99 and 0.99.
- ID_V 3.26 → 3.00.

**(3) Qwen2.5-0.5B, Q_IN / Q_OUT** (n = 10).
- Q_IN: competence 1.00, ID_K +2.21 (se 0.37). The P1 flag at r_X gives Δℓ_X +2.26 (move +2.80; K_X +3.31); random −0.41 to +0.08.
- Q_OUT: 0.5B is weakly competent (30% of argmaxes are mentioned words). ID_K −0.28 (se 0.22). Injection −0.15 / −0.38 against K_X −0.60. Noisy.

**(4) Qwen2.5-1.5B, Q_IN / Q_OUT sign and competence, no injection** (`pilot_sign.py`, n = 12, Random(1)).
- Q_OUT: competence 12/12; ID_K −4.17 (se 0.41), negative in 12/12. Under K_S, Δlp(B) = +6.45 and the argmax becomes the mentioned word B in 3/12.
- Q_IN: competence 12/12; ID_K +2.12 (se 0.55), positive in 12/12.
- This is the only 1.5B pilot. No sentence or dissociation measure was run at 1.5B, and 3B is untouched.

**(5) Qwen2.5-0.5B, dissociation measures** (`pilot_diss.py`; 16 fit, 16 eval, case-marginalised).
- The flag is written in the sentence as in the list: ω = 0.99, per-layer cos(Δ^POST, Δ^P1) 0.72-1.00.
- Under POST, natural ID_K is −0.74. The P1 flag injected at the sentence row of X gives Δℓ_X −0.46 (move −0.75), against K_X −1.06; random +0.10.
- The POST-fit flag injected at the P1 list row gives +1.63, against the P1 flag's +1.47.
- So at 0.5B the flag is written and is read negatively in sentences and positively in lists. This motivates D6's polarity prediction.

**(6) Qwen2.5-0.5B, IOI INLINE reader transfer** (`pilot_ioi.py`, n = 24, IOI Random(5)).
- Full ID_K −2.75; the listed names carry 0.81 of it.
- The belief-ranked H* recovers R = 0.95 and KO = 0.97 of the negative listed-name read; random sets 0.00.

**(7) Qwen2.5-0.5B, IOI flag injection** (`pilot_ioi_inject.py`; 20 fit, 20 eval).
- The IOI-fit flag at the listed IO_X row: Δℓ(IO_X, four-way) −2.20 (se 0.25), against K_X −4.63.
- The belief flag: −0.58 (se 0.13). Random: −0.02.
- cos(Δ^IOI, Δ^belief) per layer ranges from −0.03 to 0.86.

**What this means for riskiness.** The 7B predictions D1-D4 and D8-D9 are informed by 0.5B pilots and D7 by the 1.5B sign pilot. All stay confirmatory at 7B, where nothing was run. D6's 1.5B, 3B and 7B lines are untested, and the 0.5B pilot only motivates them.

### paper_payoff
**Section.** A new subsection, "What the readers write", replaces the sentence "we claim no new head type" with a mechanism, plus one four-panel figure (`fig_flag`).

**Candidate sentences** (numbers to be filled):
- "The reader heads write one vector per layer into the re-mentioned word's own row. Injected, with no key clamp, into the row of an option absent from the story, it moves the answer there (ι = …, π_X = …). Removing that one direction removes the key read (ρ_K = …), and the answer finds the flag by key (r_ans^inj(K) = …)."
- "The same frozen readers and the same flag raise the matched candidate when asked what was mentioned and lower it when asked what was not, or in IOI's in-sentence re-mentions. Later mentions decide whether the key is read; the reader decides the sign."
- "At 1.5B the flag is written as at 7B (ω = …), but sentences read it with the opposite polarity. This explains the 'attention without the read' dissociation."

**Proposed title and abstract change:** "Later mentions decide whether a written value is looked up; the reader decides how."

**The (iii) paragraph** (N1-N5) goes into the main text as the answer to "isn't this just QK/OV?".

**Expected score movement:**
- Contribution, 2 → 3 for reviewers 1, 2 and 4. This is the explicit '+1' (flag) and '+0.5 to +1' (sign and polarity) request of reviewer 1, and questions 1 and 2 of reviewer 2. It turns the IOI negative reads from counterevidence into a tested prediction of the same mechanism.
- Soundness, +0.5: evaluation without the key clamp removes the a3-ranking circularity, H5's secondary contrast becomes scored, and the frozen full k* set is tested out of task.
- Overall: the novelty objection (all three 4s cite it as major) is the main lever from 4 to 6 or 7, together with the generality parts (A/B) of stage 8.

**If parts fail:** the preregistered decision table states the narrower claim the paper would make, so the section survives as an honest boundary.

### risks
- **Scale.** At 7B the flag may be more story-specific or non-linear, giving ι < 0.35 although D3 holds ('necessary, not sufficient as a rank-1 vector'). The thresholds were set before any 7B run, and the decision table says what the paper then claims.
- **Large norms.** Per-layer norm-matched isotropic directions gave −0.16 at 0.5B. The D2 thresholds allow per-draw |ι| ≤ 0.20, and the head-span and layer-permuted controls are stricter; a failure there would be informative.
- **Directional ablation.** It acts on the total attention output, not only H*. If story-mention duplicates (the initial and distractor rows) share the flag direction, D3 removes those flags too. This is still valid for ID_K, but the base answer can change (0.62 → 0.44 at 0.5B). The projection is reported (E1).
- **D5 is the riskiest prediction.** The `other_agent` question may not be answered by the option lookup at all. Gate D5 makes this not evaluable rather than a false negative.
- **D6** rests on a 0.5B pilot. At 3B the sign may already be positive on competent cores (stage 5: +0.71), which could make 'between' fail. At 1.5B the injection effect is small (expected about −0.3 to −0.5 nats), so the case-marginalised scorer and the competent-core filter must be fixed exactly in the entry.
- **Mistral-7B** fails IOI AFTER (stage-5 Gate e), so the within-task IOI sign flip rests on Qwen-7B alone. Q_IN and Q_OUT give the 2/2 sign test.
- **Prompt wording.** The Q_IN/Q_OUT question may interact with the 'Answer with exactly one choice' instruction (several correct answers). Competence is gated, and ID_K uses S against X only (both unmentioned).
- **Hooks.**
  - Inject inside HopSplice and HeadSplice double passes must apply identically in both passes; this is covered by a unit test.
  - BF16 batch-shape offsets (the Qwen-7B POST Gate a2 failure) are avoided because every quantity is in-batch. Gate D2 checks duplicated reference rows only.
- **Pilot disclosure.** The 0.5B pilots used stories from the stage-6 R (Random(0)) and E (Random(1)) generators. The 1.5B pilot used Random(1). Entry J must list every pilot and its numbers, and state that the 7B and 3B cells and every POST cell above 0.5B are unseen.
- **Multiplicity.** Nine confirmatory predictions × 2 models. Holm-adjusted verdicts are reported beside the primary ones. Each verdict is also tied to a stated alternative, so a failure is interpretable rather than a lost bet.
- **Budget.** About 2.1 GPU-h of the 8-12 h. If stage 8 overruns, cut E3-E5, then D5, then 1.5B/3B to n = 60.
- **Anonymity.** Part D does not touch the Anonymous (2026) remap, so there is no anonymity exposure.

