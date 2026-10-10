LABEL design:intervention
### part
C: an out-of-sample test of the intervention lesson on independently obtained identity edits, at matched depth (preregistration J, part C; predictions JC1-JC6, gates JC-G0-JC-G7)

### objections_answered
[
 "(3) The intervention lesson rested on one remap family, and its one out-of-sample test failed. Part C tests the matched-depth law on three identity-edit families obtained independently of the natural clamp: (E1) in-task steering vectors from held-out stories; (E2) steering vectors from unrelated neutral sentences; (E3) features of a third-party sparse autoencoder (andyrdt/saes-qwen2.5-7b-instruct, BatchTopK, trained on chat and Pile text). A fourth family (E4) is a new instance of the remap family: a rank-16 DAS remap at the writing token, trained with the released recipe at a new position, new depths and new models. Coverage is 4 formats, 3-4 depths and 3 models, one of them a new family (Llama-3.1-8B). Preregistered: transfer, crossover, law (gap thresholds), depth tracking and reader identity.",
 "(3, the H11 tautology) The entry states and unit-tests a lemma. Any full-residual identity patch at p at block l is exactly the natural K/V clamp from l+1, so its key share equals s_ID identically. That patch is kept only as an equivalence control (T). A cell counts as out-of-sample only when the edit's cached keys and values at p differ from the natural ones by at least 30% of the natural displacement (nu >= 0.30).",
 "(3, edit kind confounded with depth; Gate b3 failed) JC6 revisits the H7 cell itself: Qwen2.5-14B, Prakash et al.'s material and population, block 28, positions [p, p+1]. A non-tautological identity edit (E1) is set against the binding swap, re-run in the same job, so the kind of edit varies while depth, positions and material are fixed. JC4 separately tests whether non-natural identity edits follow the natural read as it falls with depth.",
 "(3, the measure might be unable to tell edits apart) There are two controls. T is equivalent to the natural clamp by construction, so kappa_T equals sigma to within the BF16 floor. The re-run binding swap is known to be read differently (|kappa - sigma| = 0.60-0.93 in H), so it shows the exchange can report a large departure. Representational distance nu and story-level distance D are reported for every edit.",
 "(8) Reliance on an unreadable anonymous submission: Section 4's general claim no longer depends on the released remap. It rests on edits that anyone can rebuild: mean differences, a public third-party SAE, and a public training recipe applied to public models. The released remap stays only as the motivating case.",
 "(1) Novelty ('QK/OV already implies it'): the result becomes a falsifiable, predictive rule for attributing interventions. From the unpatched model's own read at matched depth, it predicts which cache channel any identity edit will appear to act through, before the edit is run. The rule has stated rivals: copy-only, depth-blind and key-flat. The binding swap shows the key-flat rival can occur, so the rule is not a corollary of QK/OV.",
 "(5, in part) New story seeds (8101-8103), held-out calibration sets, and a model family never used in stages A-I (Llama-3.1-8B).",
 "(4) All Part C quantities use Part B's surface-form score E with its coverage gate. The legacy lowercase score is reported as a robustness check."
]

### design
## C.0 The claim, and the tautology that H11 fell into

**Law under test (H_read at matched depth).** Let E write a vector into the residual stream at the writing token p, at the output of (0-based) block l, so that the model reports a different value. Its identity key share in format f is
kappa_E(f,l) = mean_s ID_K^E(s) / [mean_s ID_K^E(s) + mean_s ID_V^E(s)].
The unpatched model's own identity key share at the same depth is
sigma(f,l) = s_ID of the natural K/V clamp at p from block l+1 (onset l0 = l+1).
The law states kappa_E(f,l) = sigma(f,l): the channel an identity edit appears to act through is set by how that format's readers read the written value at that depth, not by the edit.

**Lemma (stated in the entry; unit-tested in JC-G0).** B and S differ only at p.
- (i) An edit confined to (p,l) reaches every other position only through p's K and V in blocks >= l+1. Under the causal mask p's residual reaches other tokens only through attention, and p's K/V in blocks <= l are computed from block inputs, which the edit does not change. So B with E's K/V written at p from block l+1 is exactly E, and the key-only and value-only exchanges are exact, with an interaction term.
- (ii) If E writes h_S,l(p), then its K/V at p are S's, so E = C_KV(S) from l+1 and kappa_E = sigma(f,l) identically. H11 (ID at block 0) and every full-residual identity patch at p are of this kind. They check the pipeline; they cannot test the law.
- (iii) An edit is evidence for the law only to the extent that its cached K/V at p differ from the natural ones while it still changes the reported value. That distance (nu) is measured, and only cells where it is large are counted.

**Pilot fact that shapes the design (disclosed).** At Qwen2.5-1.5B (FP32), a steering vector built from unrelated sentences changes the answer as much as the natural edit (identity effect 0.97-1.02x under NO-MENTION). Its K/V distance from the natural edit is nu = 0.12 at block 1, 0.32 at block 5, 0.62 at block 9 and 0.70 at block 13. An in-task steering vector stays closer: 0.04, 0.16, 0.26, 0.33. The residual at p in early blocks is mostly the token's own embedding, so every identity edit there is nearly natural. A non-natural edit at shallow depth therefore has to change few dimensions (SAE latents, a rank-16 subspace), whereas steering vectors become non-natural only from mid-depth. The design spans depth and covers both kinds of edit.

## C.1 Edit families

Every edit writes one vector at p at the output of block l. t is the target in {S, X}, where X is the third value absent from the story (as in ID_K) and B is the base value.

- **E1 CAA-in (in-task steering vector; ActAdd / CAA).**
  - mu_l(x) = mean of h_x,l(p) over the held-out set H: 200 cores, each with all six values written at p, using prefix passes through p.
  - Edit: h <- h_B,l(p) + mu_l(t) - mu_l(B), with alpha = 1 fixed (no tuning).
- **E2 CAA-out (lexical vector from unrelated text).**
  - mu'_l(x) = mean block-l output at the token " x" over 24 fixed neutral sentences. The sentences have no movement, belief, question or container semantics. Each is the user turn of the same chat template, with no prefill.
  - The sentence list is frozen in the entry (12 are in pilot_c.py and 12 more are listed with this design). All 144 sentence-value pairs are clean in the Qwen2.5, Mistral and Llama-3 tokenizers.
  - Edit: h <- h_B,l(p) + mu'_l(t) - mu'_l(B), with alpha = 1.
- **E3 SAE (third-party dictionary; Qwen2.5-7B only).**
  - Dictionary: andyrdt/saes-qwen2.5-7b-instruct, resid_post_layer_{3,7,11,15}, trainer_1 (BatchTopK, k = 64, 131,072 latents, published FVE 0.93 at layer 3). The revision is pinned and the sha256 of each ae.pt is asserted.
  - Encoding: a(h) = relu(W_enc(h - b_dec) + b_enc), masked by the stored threshold. d_j = decoder.weight[:, j].
  - Selection, on H only: abar_j(x) = mean a_j(h_x,l(p)) over H. Selectivity s_j(x) = abar_j(x) - max over y != x of abar_j(y). F_x = the top-k_F latents by s_j(x) with s_j(x) > 0.
  - Edit toward t: h <- h_B + sum over j in F_t of (beta * abar_j(t) - a_j(h_B)) d_j - sum over j in F_B \ F_t of a_j(h_B) d_j. Every other latent and the SAE error stay at B.
  - k_F per layer is the smallest of {4, 8, 16, 32, 64} whose flip-to-target rate is >= 0.7 on H_cal (the last 50 cores of H, NO-MENTION, targets S and X). If none qualifies, k_F = 64 and beta = the smallest of {2, 4} that qualifies. If still none, E3 is "ineffective at l".
  - k_F, beta, F_x and abar are written to calib.json, and its sha256 is logged before any E item.
- **E4 DAS-remap@p (secondary; the remap family at a new position, new depths and new models).**
  - Subspace: RotatedSubspace of rank 16 at (p,l), initialised with the PCA basis of h_S',l - h_B',l over the training pairs.
  - Training uses the recipe released by Anonymous (2026): pair-swap objective (target pi(S')), six-way cross-entropy over the candidate scores, AdamW lr 1e-3, batch 1, one shuffled epoch over 1,000 T-set pairs, seeds 101 and 102. Format: NO-MENTION with the "Answer:" prefill (a disclosed deviation from their reply-start interface).
  - Edit toward t: h <- h_B + U^T U (h_pi(t),l - h_B,l). The source is pi(t), so the remap writes t, and no natural run is in this state.
  - Models and depths: Qwen2.5-7B and Mistral-7B, l in {3, 7, 11}.
- **Controls.**
  - T (equivalence): h <- h_t,l(p), equal to the natural clamp by the lemma. It is computed through the same pipeline (prefix pass with a residual write, captured tables, clamp rows), so it measures the pipeline's BF16 floor.
  - R (specificity): h <- h_B + r_t, where r_t is a fixed Gaussian direction (Generator seed 8104, per model, layer and target) scaled to ||mu_l(t) - mu_l(B)||.
  - Discrepancy anchor: the binding swap re-run in JC6.
- **Exploratory only:**
  - BIND-p: h <- h_C,l at (p, p+1), where C is the story whose moved object is the distractor; target = the object's initial location. The pilot found it ineffective: Phi <= 0.07 nats at 0.5B, <= 0.6 at 1.5B.
  - SAE-lex vs SAE-task split.
  - LIST-BEFORE, using in-format means.
  - alpha-dose for E1/E2 (alpha in {0.5, 2} at l = 7).
  - Gemma-2-9B-it with Gemma Scope (google/gemma-scope-9b-it-res, layers 9 and 20; eager attention), if budget remains.

**Why E1-E4 are not the natural clamp.**
- E1 replaces the story-specific difference h_t - h_B by its average over other stories, removing everything that depends on this story's agent, object, initial location and distractor. It is added, not substituted.
- E2 carries no task information at all: it comes from contexts where the word is neither a destination nor followed by a question.
- E3 changes at most 64 of 131,072 latents of a dictionary trained elsewhere and leaves the SAE error and all other latents at B.
- E4 writes t through a rotated 16-dimensional projection of the difference toward a different value, pi(t).

In every case nu is measured (C.3), and only cells with nu >= 0.30 count.

## C.2 Models, depths, formats, populations, scoring

**Models.** BF16, sdpa, use_cache = False; revisions pinned in REVISIONS.txt.
- Qwen2.5-7B-Instruct: E1-E4; reader heads from stage 6; l in {3, 7, 11, 15}, matching the SAE layers.
- Mistral-7B-Instruct-v0.3: E1, E2, E4; reader heads from stage 6; l in {3, 7, 11}.
- Llama-3.1-8B-Instruct: E1, E2; l in {3, 7, 11}. This family was never used in stages A-I.
  - Source: meta-llama with HF_TOKEN. Fallback: unsloth/Meta-Llama-3.1-8B-Instruct (ungated mirror, shard sizes identical to the official listing; LFS sha256 2b1879f3..., 09d433f6..., fc1cdddd..., 92ecfe1a... recorded). Second fallback: OLMo-2-1124-7B-Instruct (non-gated; s_K under OPTIONS falls from 0.73 to 0.03 at l0 = 10).
- Qwen2.5-14B-Instruct, for JC6 only (shared with Part B).

**Formats.** LETTERS-AFTER (LETTER), OPTIONS-AFTER (P1), SENTENCE-AFTER (POST), NO-MENTION (NONE). Prompts are identical through p in all four formats (checked: 138/138 cores in the Qwen, Mistral, Llama-3 and Gemma tokenizers). Each edit's K/V tables are therefore literally the same tensors in all four formats, captured once from the prefix through p; only their readers differ.

**Scoring.** Part B's score E (ckeys/surface.py: exact chain-rule logsumexp over surface forms, using that model's frozen frames file). The lowercase score L is reported. JC5 is scored with L, because P1 candidate mass is about 1.0 and the trie mask is not combined with HeadSplice. JC6 uses H's single-token readout, for comparability with the BIND numbers in hand.

**Populations.** Disjointness is asserted.
- E (evaluation): the first 100 cores of make_cores(600, Random(8101)) with X = pick_x(core), pi(S) != B, pi(X) != B, and single-position differences in all four formats.
- H (calibration): make_cores(200, Random(8102)); H_cal = its last 50.
- T-set (DAS training): 1,000 pairs from make_cores(1500, Random(8103)) with S' != B' and pi(S') != B'.
- JC6: H's 150-pair population; means use the other 170 pool pairs.

## C.3 Measures (exact)

**Rows.** For donor Z (natural S or X, or edit E_t) and channel C in {K, V, KV}, row C(Z) is B with p's C replaced by Z's in blocks >= l+1. Every edit is applied as a clamp row from captured tables; this is exact by lemma (i). Delta_Y(row) = score_Y(row) - score_Y(self row in the same batch).

**Identity contrast, per story.** ID_C^E = 1/2 [Delta_S(C(E_S)) - Delta_S(C(E_X)) + Delta_X(C(E_X)) - Delta_X(C(E_S))].

**Shares and transfer.**
- kappa_E = mean ID_K^E / (mean ID_K^E + mean ID_V^E).
- sigma: the same, with natural donors (equal to kappa_T up to the floor).
- phi_E(f,l) = mean ID_KV^E / mean ID_KV^nat.
- Interaction share iota = 1 - (ID_K + ID_V) / ID_KV.

**Non-equivalence nu_E(l).** The mean over E stories, t in {S, X} and blocks q >= l+1 of
||[K;V]_q^{E_t} - [K;V]_q^{t}|| / ||[K;V]_q^{t} - [K;V]_q^{B}||
at p, with all KV heads concatenated.

**Story-level distance.** D_E = mean_s (|ID_K^E - ID_K^nat| + |ID_V^E - ID_V^nat|) / mean_s ID_KV^nat. Reported for every edit. D_T is the floor.

**Flip rate.** The fraction of (story, t) whose argmax over the six candidates in the row KV(E_t) is t.

**Reader knockout KO_E (JC5).**
KO_E = 1 - ID_K^E[blind] / ID_K^E.
- In the blinded row, every head in every row reads E's key at p from block l+1, except the cells (H*_{>l}, G), which read B's key. This is the stage-7 x_S construction with ks = K_E.
- H*_{>l} = the stage-6 top-k* a3 heads under OPTIONS-AFTER (k* = 40 at Qwen2.5-7B, 52 at Mistral-7B) lying in blocks > l. That is 33 heads at Qwen2.5-7B and 43 at Mistral-7B for l = 7.
- G = the six option-word rows.
- Controls: three size-matched random sets from blocks > l (numpy default_rng(2)).

**Intervals.** 95% percentile intervals from 10,000 story bootstraps (seed 20261012). Ratios are recomputed within each resample, and kappa and sigma come from the same resample. E4 seeds are averaged within story; seed-level values are reported. Verdicts use point estimates and the named bounds, as in A-I.

## C.4 Gates

- **JC-G0 (CPU FP32 unit tests on Qwen2.5-0.5B; 1e-4 in logits).** Any failure means Part C is not run.
  - (i) Writing h_S at (p,l) equals C_KV(S) from l+1, for l in {0, 3, 10}.
  - (ii) For random vectors, B + KV_E from l+1 equals E.
  - (iii) Prefix-captured tables equal full-run tables (to 1e-5).
  - (iv) Tables are identical across the four formats.
  - (v) BatchTopK encode/decode matches a direct reference on a random 64 x 16 dictionary, including threshold masking and decoder orientation; an SAE edit over all differing latents with zero error reproduces h_t.
  - (vi) ID, kappa, nu and D give known answers on synthetic rows.
  - (vii) The HeadSplice x_S row equals the plain K-clamp row with an empty set and clean B with every cell.
  - (viii) T rows equal natural rows through the full pipeline (kappa_T = sigma).
  - (ix) On Prakash et al.'s material, the CAA arm with v = h_S - h_B at both positions equals ID.
- **JC-G1 (equivalence control, per model, BF16).** In every cell: |kappa_T - sigma| <= 0.02, nu_T <= 0.02, and mean_s |ID_K^T - ID_K^nat| <= 0.25 nats. Failure means the model is not evaluable.
- **JC-G2 (natural format effect, per model).** sigma(LETTER, 3) - sigma(NONE, 3) >= 0.5 with the CI excluding 0. Failure means the model is not evaluable.
- **JC-G3 (SAE layer alignment).** The SAE's FVE on h_x,l(p) over H is >= 0.80 at each layer. Failure means E3 is not evaluable at that layer.
- **JC-G4 (natural cell evaluable).** All of:
  - mean ID_KV^nat >= 2 nats;
  - iota^nat <= 0.5, and ID_K^nat, ID_V^nat >= -0.1 ID_KV^nat;
  - Part B's coverage gate passes for that model and format.
- **JC-G5 (edit efficacy, per model, family and depth, under NONE).** Flip rate >= 0.5 and phi_E(NONE, l) >= 0.5. Otherwise the family is "ineffective at l" and is reported but not counted.
- **JC-G6 (specificity).** |mean ID_KV^R| <= 0.1 mean ID_KV^nat under NONE at every depth. A failure flags that depth.
- **JC-G7 (discrepancy anchor, JC6 job).** BIND@28 reproduces: IIA >= 0.95, and kappa_BIND is within 0.05 of 0.618 / 0.906 / 0.864. This checks that the exchange still reports a large departure when an edit is read differently.

**Cell status.** A cell is out-of-sample if nu_E(l) >= 0.30. Cells with nu < 0.30 are "near-natural": they are reported as consistency checks and never counted.

**Combo.** A combo is a (model, family) pair. The independent combos are E1 and E2 at three models plus E3 at Qwen2.5-7B, seven in all; E4 has two.

## C.5 Confirmatory predictions

Every prediction is computed on out-of-sample cells that pass JC-G4 and JC-G5.

**JC1 (transfer).** In every independent combo and depth: phi_E(LETTER, l) >= 0.6 phi_E(NONE, l).
- Why it is risky: a non-natural edit could write identity only into features that the value route reads (rival R1). It would then not act under LETTERS-AFTER, where values carry 0.01 of the natural identity.

**JC2 (crossover).** In every independent combo, at the shallowest depth whose LETTER and NONE cells are both out-of-sample: kappa_E(LETTER) - kappa_E(NONE) >= 0.5 with the paired CI excluding 0, and kappa_E(NONE) <= 0.25.
- Why it is risky: the rival R3 (key-flat) occurred for the binding swap (kappa = 0.62 without a later mention).

**JC3 (the law at matched depth; primary).** Per independent combo, over at least 4 cells in at least 2 formats:
- MAD = mean |kappa_E - sigma| <= 0.12, with bootstrap upper bound <= 0.17;
- max |kappa_E - sigma| <= 0.25;
- Pearson r(kappa_E, sigma) >= 0.85 when sigma spans >= 0.4 over those cells.

A cell whose edit fails the iota rule while the natural cell passes counts as a gap of 1.
- Verdict: MET if met in every evaluable combo with at least 3 evaluable; MET IN PART if met in at least half; NOT MET otherwise; NOT EVALUABLE with fewer than 3 evaluable combos.
- JC3-DAS: the same rule for the E4 combos, as a secondary prediction.
- The tolerance is tighter than the released remap's in-sample deviation (maximum 0.18).

**JC4 (depth tracking).** For every (combo, format) with Delta_sigma = sigma(f, l_min) - sigma(f, l_max) >= 0.25 and a deep cell that is out-of-sample: Delta_kappa >= 0.5 Delta_sigma and |Delta_kappa - Delta_sigma| <= 0.2.
- Verdict: MET if this holds in at least 80% of at least 3 qualifying pairs; NOT EVALUABLE with fewer than 3 qualifying pairs.
- Rival R2 (depth-blind): Delta_kappa <= 0.1.
- Why it is risky: sigma has never been measured with identity contrasts beyond l0 = 0. The one-sided s_K drops at about 0.3L (OPTIONS 0.82 to 0.66 at Qwen2.5-7B and 0.92 to 0.53 at Mistral-7B; SENTENCE 0.40 to 0.07 at Mistral-7B) are the only guide.

**JC5 (same readers).** Qwen2.5-7B and Mistral-7B, OPTIONS-AFTER, l = 7: KO_E >= 0.7 for each independent family with an out-of-sample cell there, and the mean over the three random sets <= 0.25.
- Gate: the natural KO at onset 8 is >= 0.7; otherwise JC5 is not evaluable in that model.
- Why it is risky: a non-natural edit could reach the option rows through heads outside the natural readers.

**JC6 (kind vs depth, at the H7 cell).** Qwen2.5-14B, Prakash et al.'s material, l* = 28, positions [p, p+1], H's one-sided kappa (m = log p(S) - log p(s_q)), exchange from block 29.
- Edit: E1 with means mu_28(x, pos) over the 170 held-out pool pairs, with each of the 23 drinks written at the queried state.
- Prediction: kappa_CAA(f) <= 0.30 in each of NO-MENTION, QNAMES and OPTIONS-AFTER, and kappa_BIND(f) - kappa_CAA(f) >= 0.30 with the paired CI excluding 0.
- Gates: Phi_CAA >= 3 nats; NONE flip rate >= 0.5; nu against the ID@28 tables >= 0.30; the iota rule.
- Readings fixed now:
  - MET: at the depth, positions and material where the law failed for the binding swap, a non-natural identity edit follows the natural read (sigma about 0). The kind of edit, not its depth, explains H7.
  - RIVAL (kappa_CAA >= 0.5 in at least 2 formats): at that depth even identity edits are key-carried. Depth or material explains H7, and the law fails for non-natural edits at depth.
  - Otherwise: intermediate, reported as such.

**Preregistered headline readings.**
- "Supported out of sample": JC3 MET, JC1 and JC2 MET, and at least one of JC4 or JC5 MET.
- "Restricted to near-natural edits", which is the definitional reading the panel suspected and which the paper would then state: JC3 NOT MET with the R1 or R3 pattern in at least half of the combos, or JC3 met only in near-natural cells.
- "Format-only": JC3 MET but JC4 NOT MET with the R2 pattern.

## C.6 Rivals, and what counts against the account

- **R1 copy-only (the content account).** Generic identity edits are copied, not looked up: phi(LETTER) < 0.3 phi(NONE), and kappa <= 0.25 where sigma >= 0.6.
- **R2 depth-blind.** kappa follows the format but not the depth.
- **R3 key-flat (address-like).** kappa(NONE) >= 0.6 for an identity edit.
- **Mixed by family.** For example, E2 meets the law and E3 fails it. That would mean the law holds for edits that write a word's lexical identity but not for feature-level edits, and the paper would report this boundary.
- **D far below nu** (representationally different but read identically, story by story) is the mechanism the law posits: readers extract a low-dimensional identity signal. The pilot shows exactly this at 0.5B, and the paper will say so rather than claim the edits differ functionally. The paper must not use D to argue non-equivalence.

## C.7 Scope boundary (stated in the paper)

- The law covers edits that change which value is written at the writing token.
- Prakash et al.'s binding swap exchanges which state is bound to which container: an address/payload swap that changes ordering, not identity. H7-H10 stand as failures and are not reinterpreted. JC6 only diagnoses whether that failure reflects the kind of edit or its depth; it does not extend the law to binding edits.
- Lemma (i) is exact for single-position edits only. The released remap patches the whole event span, so its exchange is not exact.
- The data are templated belief stories with single-token values; natural text is Part A's test.
- No binding edit at the writing token was effective in the pilot, so a shallow binding test is left exploratory.

## C.8 Power

- **Sampling noise.** From the committed stage-1 per-story rows (seen before finalisation; disclosed), the bootstrap SD of s_ID at n = 100-120 is <= 0.012 in every format at Qwen2.5-7B, Mistral-7B and OLMo-2-7B. At deep onsets with small effects (2-6 nats) the one-sided share has SD <= 0.05.
- **Monte Carlo (power_c.py).** 150 replicates of 12 cells, with edit and natural rows resampled independently (the worst case, with no within-story correlation). Probability that JC3 is met:
  - true kappa = sigma: 1.00 at both n = 60 and n = 120 (mean MAD 0.007-0.010, mean maximum gap 0.02-0.03);
  - a 0.15 systematic under-read in OPTIONS and SENTENCE: 1.00 (MAD 0.08; this is within tolerance);
  - a 0.25 under-read: 0.01-0.04;
  - R1: 0.00 (MAD 0.48);
  - R3: 0.00 (MAD 0.24).
  So JC3 is decided by systematic deviations of 0.25 or more, not by sampling noise.
- **Edit-specific noise (0.5B pilot).** Edit and natural kappa agree to within 0.03 at n = 6. D <= 0.022 wherever the effect is >= 1.6 nats.
- **JC2.** The natural gap sigma(LETTER) - sigma(NONE) is about 0.93-0.95 at 7-14B.
- **JC5.** Stage 6 found KO 0.97-0.98 against random sets at 0.01.
- **JC6.** BIND kappa is 0.62-0.91, with CI half-widths <= 0.05 at n = 150.

## C.9 Compute (one A100-80GB, BF16)

Per-row cost is about 0.010 s at 7-8B, from Part B's figure for stage 3b (24 s per 150-item arm at 16 rows per item), plus about 35% for the score-E trie, giving about 0.0135 s.

| Part | Work | Time |
|---|---|---|
| Eval rows, Qwen2.5-7B | 100 stories x 4 formats x (36 rows x 4 depths + 12 E4 rows x 3 depths) = 72,000 | |
| Eval rows, Mistral-7B | 42 rows x 3 depths x 4 formats x 100 = 50,400 | |
| Eval rows, Llama-3.1-8B | 30 rows x 3 depths x 4 formats x 100 = 36,000 | |
| Eval rows, total | 158,400 | about 36 min |
| Prefix captures and calibration | about 6k short passes and about 3k NONE rows per model | about 4 min |
| DAS fits | 6 (model, depth) x 2 seeds x 1,000 steps at about 0.07 s/step | about 14 min |
| JC5 | about 14.7k row-equivalents (second attention pass included) | about 3 min |
| JC6 at 14B | means about 3.7k prefix passes, plus 150 x 4 x 17 = 10,200 rows | about 7 min with load |
| Exploratory | | about 7 min |
| SAE downloads | 4 x 3.76 GB | about 4 min |
| Model loads | | about 4 min |

Rows per (story, format, depth): self 1, natural 6, T 6, R 2, E1 6, E2 6, E3 6 (Qwen2.5-7B only), E4 12 (two seeds), BIND-p 3.

Total: about 1.3 GPU-hours if the 7-14B weights are shared with Parts A/B, about 1.6 h otherwise, with a 2.25 h cap. The deadline drops exploratory parts first (Gemma, BIND-p, LIST-BEFORE, alpha-dose). The cost is about $3-5. Gemma with Gemma Scope, if run, adds about 0.4 h.

## C.10 Exploratory (labelled; no verdicts)

- LIST-BEFORE, with in-format means.
- SAE-lex vs SAE-task: latents selective on the neutral sentences vs latents selective on the stories but inactive on the neutral sentences; efficacy-gated; read two-sided.
- BIND-p.
- alpha-dose.
- One-sided kappa.
- L scoring.
- Per-seed E4.
- kappa against nu and D, per story.
- E2 at the JC6 cell, using drink words in neutral sentences.
- Gemma-2-9B-it with Gemma Scope at layers 9 and 20, as a second third-party dictionary.
- Projection of the edit's K/V displacement onto the natural one, per layer (the low-dimensional readout behind small D).

## C.11 Seen before finalisation (disclosed)

- The stage-1 per-story rows (used for power).
- The stage-6 head rankings (used as fixed sets in JC5).
- H's BIND@28, ID@28 and s_ID(f, l0) values on Prakash et al.'s material (used for the JC6 thresholds and anchor).
- The CPU pilots at 1.5B and 0.5B described in pilot_results. No 7-14B output of any Part C edit has been seen.

### code_plan
### New files

**1. ckeys/edits.py** (edit construction and table capture)
- `write_resid(model, layer, pos, vecs)`: a context manager that adds a block-output hook setting out[r, pos] = vecs[r] ([R, P, D]); rows given as None are left alone. This generalises prakash_swap.resid_patch to per-row vectors.
- `prefix_tables(model, ids, pos, layer, vecs)`: runs ids[:, :max(pos)+1] expanded to R rows under write_resid, with ckeys.clamp.capture_kv(pos, range(layer+1, nL)) and a capture of the block outputs. Returns ({(q, 'k'|'v'): [R, P, D]}, resid).
- `natural_tables(model, ids_by_value, pos)`.
- `class_means(model, tok, cores, layers, arm)`: E1, from prefix passes over all six values.
- `lexical_means(model, tok, SENTENCES, layers)`: E2.
- `vec_caa`, `vec_random`, `vec_das`, `vec_sae`.
- `clamp_rows(tabs_by_donor, layer0, spec)`: builds the per-row [R, 1, D] tables in a fixed row order (self; nat K/V/KV for S and X; T; R; E1; E2; E3; E4 seed 101 and 102; BIND-p), filling unwritten rows with B's own tables.
- `nu(tab_e, tab_t, tab_b, layers)` and `id_contrast(rows, iS, iX)`.

**2. ckeys/sae.py**
- `BatchTopK.load(path, sha256)`: reads the ae.pt state dict (encoder.weight [m, d], encoder.bias, decoder.weight [d, m], b_dec, threshold, k).
- `encode`, `decode`, `fve`.
- `select(acts_by_value, kF)`: selectivity ranking.
- `edit(hB, t, b, sel, abar, beta)`.
- An optional `JumpReLU.load` for Gemma Scope.

**3. ckeys/neutral.py**
- The 24 sentences, frozen; their sha256 goes into the entry.
- `token_pos(tok, ids, word)`, which asserts exactly one occurrence of the word.

**4. ckeys/das.py (extend)**
- `train_remap_at(model, pairs, layer, rank=16, target=PAIR_SWAP, lr=1e-3, epochs=1, batch=1, seed, init='pca', score_fn)`.
- Source activations come from prefix passes; PCA via interventions.pca_basis; the loss is CE over six candidate scores (logsumexp of the single-token " x" and " X" forms for training).
- Returns U (asserting U U^T = I) and the loss curve.
- Reuses RotatedSubspace.

**5. experiments/stage8_edits.py**
Steps preflight | calib | dasfit | eval | readers | prakash | explore. Each step writes one JSON under results/gpu_stage8/edits/{model}/, with keep/FORCE semantics as in stage 7.
- **preflight** (tokenizer only):
  - the populations E, H and T-set, and their disjointness;
  - single-position differences, and prefix identity across LETTER/P1/POST/NONE for every E core;
  - the neutral-token check (at least 20 of 24 sentences per tokenizer);
  - the Part B frames file and its hash;
  - SAE sha256 values via the HF API;
  - the Llama weight source and its safetensors sha256.
- **calib:**
  - E1 and E2 means;
  - SAE FVE (JC-G3), selection, and the k_F/beta rule on H_cal;
  - random vectors;
  - calib.json written and hashed before eval.
- **dasfit:** E4 fits, with bases saved and hashed; the NONE flip rate on 100 held-out T-set pairs.
- **eval**, per E core:
  - prefix passes for the natural B, S, X, C, pi(S) and pi(X) runs, and for every family, depth and target;
  - per format, one batch per depth of clamp rows, scored with Part B's surface.score (E) and with L;
  - stored per row: the six candidate scores, the E-mass and the argmax;
  - stored per story: nu for each family and depth.
- **readers:** HeadSplice x_S rows (reusing experiments/stage7_link.py's mask builders: readerblind.cell_masks / blind_masks) at Qwen2.5-7B and Mistral-7B, P1, l = 7, for the natural row, E1, E2, E3 and E4 (seed 101), targets S and X, over H*_{>7} and three random sets. Unblinded K rows and the self row are in the same batch. The H* sets are read from results/gpu_stage6/heads/*.json and their sha256 recorded.
- **prakash** (JC6): experiments/prakash_swap.py gains `--arm CAA`, a `means` stage and `--anchors BIND,ID`, with exchange rows r0-r4 for CAA toward S and X and for both anchors at l* = 28, in NO-MENTION, QNAMES and OPTIONS-AFTER (LETTERS-AFTER exploratory), plus nu against the ID@28 tables. The release is fetched at run time at 0579347 with the hashes asserted, as in stage 6.
- **explore:** the exploratory list, skipped first at the deadline.

**6. analysis/stage8_parts/c.py and analysis/stage8c_score.py**
- Gates JC-G0 to JC-G7.
- Cell status (evaluable, out-of-sample, near-natural).
- JC1 to JC6 with their combination rules and headline readings, CIs (seed 20261012; paired kappa and sigma), and the R1-R3 patterns.
- The exploratory report and the provenance and population-hash checks.
- Writes results/gpu_stage8/STAGE8C_SCORE.txt.

**7. tests/test_stage8_edits.py** (JC-G0, FP32, Qwen2.5-0.5B, CPU)
- (i) Writing h_S at (p,l) equals C_KV(S) from l+1, for l in {0, 3, 10} (1e-4).
- (ii) For random vectors, B + KV_E from l+1 equals E (1e-4).
- (iii) Prefix tables equal full-run tables (1e-5).
- (iv) Tables are identical across the four formats (exact).
- (v) BatchTopK matches a reference on a random 64 x 16 dictionary, including threshold, orientation and the full-latent edit reproducing h_t.
- (vi) ID, kappa, nu and D on synthetic rows with known answers.
- (vii) HeadSplice x_S with an empty set equals the K-clamp row; with all cells it equals clean B.
- (viii) T equals natural through the whole pipeline.
- (ix) The prakash CAA arm with v = h_S - h_B equals ID.
- A no_grad assertion on every forward of the eval path. The pilot script hit an autograd memory blow-up without it.

**8. scripts/gpu_stage8.sh (PART=C block)**
- Order: pytest (JC-G0); then per model preflight, calib, dasfit (Qwen2.5-7B and Mistral-7B), eval, readers (Qwen2.5-7B and Mistral-7B); then prakash at Qwen2.5-14B; then explore; then score.
- Guards as in stage 7: refuse a draft entry, a dirty tree or a commit mismatch; PARTC_DEADLINE_H = 2.25; models are removed from the cache unless shared with Parts A/B; TEST_MODE runs everything at 0.5B with random SAE and DAS bases and n = 2.

### Reused
- ckeys/clamp.py (capture_kv, clamp_kv, stack_rows)
- ckeys/encoding.py, ckeys/story.py (make_cores, pick_x, record, PAIR_SWAP)
- ckeys/headsplice.py and ckeys/readerblind.py
- ckeys/interventions.py (blocks, pca_basis, RotatedSubspace)
- ckeys/causaltom.py and experiments/prakash_swap.py
- Part B's ckeys/surface.py

### Exactness tests
JC-G0 (i)-(ix) above. The pilot already confirmed:
- (i) and (viii) in FP32: the T tables differ from the natural tables by 1.7e-6;
- (ii): the self-row difference in the exchange batch is 0.0.

### pilot_results
All runs were CPU, FP32, under /tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partC/ (pilot_c.py, analyse_a.py, analyse_b.py, power_c.py, plus the *_summary.txt and *.log outputs). Everything here is disclosed as seen before finalisation.

**Pilot A: Qwen2.5-1.5B-Instruct, 4 stories, NONE and LETTER, blocks 1/5/9/13.** It was planned for 8 stories and stopped at 4 after about 14 min, because other processes were competing for the 4 cores. Edits were T, E1 (CAA-in from 20 held-out cores), E2 (CAA-out from 12 neutral sentences), R and BIND-p.
- E1 and E2 at alpha = 1 reproduce the natural identity effect. Under NONE they reach 1.00-1.02x and 0.97-1.02x at every block. Under LETTER they reach 0.91-1.04x while the natural effect is at least 2 nats (blocks <= 9).
- nu_KV for E1 is 0.04 / 0.16 / 0.26 / 0.33 and for E2 is 0.12 / 0.32 / 0.62 / 0.70 at blocks 1 / 5 / 9 / 13. nu_T is 1.7e-6.
- The natural LETTER effect falls from 7.2 to 2.2 to 0.35 nats between blocks 5, 9 and 13: the key route closes with depth.
- R flips nothing. Its non-specific shift of m_SB is up to 1.5 nats, and the ID contrast cancels it.
- BIND-p is ineffective: Phi <= 0.6 nats at every block.

**Pilot B at 1.5B: no data.** It was killed twice by the memory cgroup. The first kill was mostly caused by a bug in the pilot script: a clamp forward was not under no_grad. The bug is fixed, and a no_grad assertion is added to the test plan.

**Pilot B at Qwen2.5-0.5B-Instruct: 6 stories, LETTER / P1 / NONE, blocks 1 and 9, about 9 min.** Key-only and value-only clamps of each edit's own K/V were run from block l+1. The exchange self-row check gave 0.0.

Identity key shares (score with case forms; the lowercase score gives the same values):

| Format, onset | Natural sigma (T) | E1 | E2 |
|---|---|---|---|
| LETTER, onset 2 | 0.77 | 0.77 | 0.78 |
| P1, onset 2 | 0.24 | 0.23 | 0.23 |
| P1, onset 10 | 0.09 | 0.07 | 0.06 |
| NONE, both onsets | 0.01 | 0.01 | 0.01 |

- The P1 depth drop (0.24 to 0.09) is tracked by both edits.
- LETTER at onset 10 is not evaluable: the natural identity effect there is 0.26 nats.
- Story-level D is <= 0.022 wherever the effect is >= 1.6 nats. So at 0.5B the steering edits are read like the natural edit story by story, although their K/V differ. This motivates reporting D as the mechanism (a low-dimensional readout) and not using it to argue non-equivalence.
- BIND-p again has Phi <= 0.07 nats.

**Tokenizer checks (tokenizer only).**
- Qwen2.5-7B, Mistral-7B, Llama-3.1-8B (mirror) and Gemma-2-9B: 138 of the 400 cores of make_cores(400, Random(8101)) meet the pi constraints (a stricter version), and all 138 have single-position differences and prefixes identical across the four formats.
- The 24 neutral sentences give 144/144 clean single-token occurrences in Qwen2.5-7B, Mistral-7B, Llama-3.1-8B and Qwen2.5-14B.

**Resource checks via the HF API.**
- andyrdt/saes-qwen2.5-7b-instruct: ungated, layers 3/7/11/15/19/23/27, k = 32-256, about 3.76 GB per SAE, FVE 0.917-0.951 at layer 3.
- google/gemma-scope-9b-it-res and Goodfire/Llama-3.1-8B-Instruct-SAE-l19: ungated.
- The unsloth Llama-3.1-8B mirror's safetensors sizes are identical to the official listing.

**Power simulation (power_c.py, on the committed stage-1 rows).** Probability that JC3 is met:
- 1.00 when the law holds (mean MAD 0.007-0.010);
- 1.00 with a 0.15 under-read;
- 0.01-0.04 with a 0.25 under-read;
- 0.00 under copy-only and under key-flat.

**Not piloted:** E3 (SAE) and E4 (DAS) at 7B, the reader knockout (JC5) and JC6. A 7B model does not fit this CPU box.

### paper_payoff
**Where it goes.** Section 4 is rebuilt around one figure and one claim. The released remap of Anonymous (2026) moves to a short motivating paragraph.

**New Figure 4.**
- (a) For every out-of-sample cell (4 formats x 3-4 depths x 3 model families x 4 edit families): the edit's identity key share kappa against the unpatched model's own share sigma at matched depth. The equivalence control T lies on the diagonal by construction, and the binding swap (H7, re-run) sits far off it.
- (b) |kappa - sigma| against the edit's distance from the natural cache, nu.
- (c) The reader knockout KO for each family, beside the natural readers.

**Main-text sentence, if confirmed:** "Which cache channel an edit that changes the written value appears to act through can be predicted before the edit is run, from how the unpatched model reads that value in the same prompt at the same depth. Steering vectors from held-out stories and from unrelated sentences, and features of a third-party sparse autoencoder, write cached keys and values 30-70% away from the natural ones. Yet their key share falls within 0.12 of the natural share across four formats, up to four depths and three model families, and is read by the same heads. At the depth where the binding swap failed, an identity edit follows the natural read and the binding swap does not, so the law's boundary is the kind of edit, not its depth."

**Scope sentence:** "The rule covers edits of the written identity, not binding or ordering edits."

**Reviewer scores this should move.**
- Soundness 3 to 4, for the stats reviewer and the generalist. Objection 3 is answered by preregistered, independently obtained edits with a tautology lemma and an equivalence control. The depth confound behind Gate b3 is broken at the H7 cell itself.
- Contribution 2 to 3, for the mechinterp reviewer and the skeptic. Section 4 changes from a case study of one anonymous remap into a predictive attribution rule with stated rivals and a scope boundary, which is a methodological result usable by anyone who patches activations. It also removes the anonymous-dependence concern (objection 8).
- If the law fails (copy-only, key-flat or depth-blind), the paper says preregistered that the format law holds only for near-natural edits, which is close to definitional. That is a clean negative that pre-empts the "definitional" critique instead of leaving it open.

### risks
1. **The SAE family (E3) may fail the efficacy gate.** Clamping a few BatchTopK latents may not flip the answer, especially at layers 11 and 15. The k_F/beta rule on H_cal mitigates this. If E3 is ineffective, JC3 rests on the E1 and E2 combos (6 of 7), and the entry says so in advance.

2. **E1 may be near-natural at most depths.** At 1.5B its nu is only 0.33 at block 13, so few E1 cells may be out-of-sample. E2, E3 and E4 then carry JC3. If fewer than 3 combos are evaluable, JC3 is NOT EVALUABLE rather than met.

3. **Functional identity.** At 0.5B the edits are read like the natural edit story by story (D <= 0.022). A reviewer may call a success "functionally tautological". The entry pre-commits to the reading: small D with large nu is the low-dimensional-readout mechanism the law posits; the tautology is nu = 0. E3, E4 and JC6 are the cells where the edit's content most plausibly differs.

4. **Deep cells may be too small to evaluate.** Under LETTERS and OPTIONS the natural effect collapses with depth (Mistral-7B LETTER dKV is 4.7 nats at l0 = 10; 0.5B LETTER is 0.26 nats at onset 10). Deep LETTER and P1 cells may fail JC-G4, so JC4 may have fewer than 3 qualifying pairs and be NOT EVALUABLE. Qwen2.5-7B adds l = 15 for this reason.

5. **Llama-3.1-8B is new.** Its competence, natural format effect (JC-G2) and Part B coverage of the score E under NONE and SENTENCE (Part B saw about 0.5 at 1B) are unknown. A failure removes that model's combos. Fallback order: official repo, then the unsloth mirror, then OLMo-2-7B.

6. **JC6 effect size.** CAA at block 28 may have Phi < 3 nats under QNAMES and OPTIONS. H's natural ID@28 had 9-11 nats there but flipped no answers. JC6 would then be NOT EVALUABLE, and the depth confound remains open; the paper would say so.

7. **BF16 numerics.** The trie mask disables flash sdpa, and stage 5 saw eager vs sdpa move ID_K by 10-20%. JC-G1 (T against natural) bounds the floor in every cell. kappa and sigma come from the same pass type.

8. **Reader sets.** The stage-6 H* sets were ranked from layer 0 under OPTIONS. Restricting them to blocks > 7 keeps 33 of 40 heads at Qwen2.5-7B and 43 of 52 at Mistral-7B. If the natural KO at onset 8 is below 0.7, JC5 is not evaluable.

9. **Pilot limitations.** The pilots ran at 0.5B and 1.5B, on 4-6 stories, under heavy CPU contention. One 1.5B run was lost to a memory bug in the pilot script, now fixed and covered by a test. Nothing at 7-14B was piloted.

10. **Budget.** About 1.3-1.6 GPU-hours against a 2.25 h cap. The deadline drops exploratory parts first. The SAE downloads (about 15 GB) and the Prakash release fetch need network access on the box.

11. **Scope.** Even if everything is met, the claim covers single-position identity edits on templated tasks. Binding and ordering edits stay outside the law, so H7-H10 remain failures. Natural-text generality is Part A's test.

