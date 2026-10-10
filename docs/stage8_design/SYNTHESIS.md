# Stage 8 (preregistration J, P-2026-10-10-J): synthesis of the designs and critiques

Sources: design files res0.md (Part A), res1.md (B), res2.md (C), res3.md (D), res4.md (E, rewrite); critiques res5.md
(B), res6.md (A), res7.md (C), res8.md (D), res9.md (E). A part's spec is its design file AS AMENDED BELOW. Where this file
and a design file disagree, THIS FILE WINS. Anything a critique required that is not listed here as accepted is
rejected or deferred for the stated reason.

## G. Global rules (all parts)

G1. One entry, four parts, four GPU scripts. Entry `## P-2026-10-10-J: GPU stage 8 ...` in docs/PREREGISTRATION.md with
    sections Part A-D. Scripts scripts/gpu_stage8a.sh, gpu_stage8b.sh, gpu_stage8c.sh, gpu_stage8d.sh share
    scripts/stage8_common.sh (guards copied from scripts/gpu_stage7.sh: refuse a DRAFT J entry (awk on the J section),
    a modified tracked tree, code differing from the commit titled "Finalise preregistration J"; pin transformers
    5.18.0; FP32 pytest of that part first; DEADLINE_H; FORCE_STEPS; KEEP_CACHE; TEST_MODE=1 runs the whole part at
    Qwen2.5-0.5B-Instruct FP32 on CPU with n = 2-3 and tags every output TEST_). Each script writes
    results/gpu_stage8{a,b,c,d}/ plus a tgz, MANIFEST.sha256, ENV.txt, REVISIONS.txt, and runs its scorer at the end.
    The parts are independent: the user may run them in any order or on different GPUs.

G2. Model files. scripts/stage8_models.json + scripts/fetch_verified.py (Part B code plan item 7) serve ALL parts.
    Official repo with HF_TOKEN if set; otherwise byte-identical sources verified file by file against the official
    hashes in the manifest (copy scratchpad/s8design/partB/hf_manifest.json into the repo). Llama-3.1-8B-Instruct and
    Gemma-2-9B-it are described as "official weights (sha256-verified)". The single technical fallback for a fresh
    family, in every part, is 01-ai/Yi-1.5-9B-Chat, used only when a file fails verification or a pre-output exactness
    test fails; never after any output of the replaced model exists. Part A's preflight re-checks item validity with the
    fallback tokenizer and drops failing items (counted).

G3. Intervals and combination. Per model: 95% percentile intervals (two-stage cluster bootstrap where clusters exist:
    articles in A, ordered location pairs in B; stories elsewhere), 10,000 resamples, ratios recomputed in every
    resample. A line that must hold "in every evaluable model" is an intersection-union test and uses the 95% interval
    without correction. A line that holds "in k of N models" uses 98.75% intervals. Every interval criterion is written
    as a one-sided test of a named null (e.g. "H0: s_ID <= 0.35 rejected: lower bound > 0.35"). Point floors are effect-size
    conditions and are described as such. Holm over each part's interval components is reported as a sensitivity
    analysis, flagging any verdict that changes.

G4. Risk classes and priors. Every confirmatory line gets a class and a recorded prior P(met) in the entry, before any
    run:
      L = implied by data in hand on the same models and material (a replication);
      M = extrapolates a regularity seen in every model in hand (prior >= 0.8);
      R = risky (prior < 0.8, or no data in hand bears on it).
    Measurement-validity lines (scoring invariance, coverage) are tallied separately from account lines. Each scorer
    prints met counts per class, and the sum of priors against the observed met count per class. The paper's abstract
    reports the met rate among R lines.

G5. Codes. Lines are numbered J-A1 ..., J-B1 ..., J-C1 ..., J-D1 ...; gates J-A-G0 ... Verdict words: MET, NOT MET,
    NOT EVALUABLE, MET IN PART (only where a part defines it).

G6. Populations. Every templated population in stage 8 is asserted disjoint (by full core tuple) from U = the union of
    make_cores(1000, Random(s)) for s in 0..3 (stages 1-7 used seeds 0 and 1 only), and from every other stage-8
    population, except where a part re-measures S0 = make_cores(150, Random(0)) on purpose (B's JB6, labelled).

G7. Pilots. Every CPU pilot of the designs and of the critiques (scratchpad/s8design/part*/ and critic_*/) is listed in
    the entry's "seen before finalisation" section with its numbers, including the critics' closed-book, cue-conflict,
    non-lexical-steering and core-overlap checks.

G8. Shared code already committed (43700ae, and the generation commit): ckeys/surface.py (emitted-form trie scorer),
    ckeys/generate.py (greedy under clamps, parse, frames), ckeys/squad_items.py + ckeys/natural_formats.py (items),
    span HeadSplice, fused-qkv clamp sites. Builders reuse them; changes to them need a test.

## A. Part A (natural data), res0.md amended by res6.md

Accepted (all of R1-R15 except where noted):
A-1  Items: already rebuilt in ckeys/squad_items.py with (i) the type-exclusion audit, (ii) no partial mentions (LEAK
     stratum: e_B content word elsewhere in passage/question; such items enter no primary population; LEAK split of up
     to 80 E-article items is kept as an exploratory stratum: the natural in-passage re-mention), (iii) PLACE place-class
     matching with the audited CLASS_FIX, 'where' questions without a place head noun dropped. Current build:
     R 71 items (12 articles), E 185 (34 articles; PERSON 80, NUMBER 77, PLACE 28), YEAR 80, LEAK 65. Regenerate all
     counts in the entry from scripts/build_stage8a_items.py; commit data/stage8a_items.json; the preflight rebuilds
     from the downloaded SQuAD file (sha256 asserted) and asserts byte equality.
A-2  Prior controls (R1): closed-book prompts CB-OPT (OPT-A wording, passage removed) and CB-LET; a KV_Z row in NOM, OPT-A,
     LET-A. Population "prior-free" = CB argmax over the four options != B AND KV_Z generation != B (per model).
     Every accuracy outcome is reported on all competent items and on prior-free items; accuracy-based verdicts use
     prior-free items.
A-3  Behaviour (R2) replaces A6/A7 as primary: cue-conflict rows (K_S,V_X) and (K_X,V_S) (key source vs value source),
     flag-only (K_S,V_Z), copy-fallback (K_Z,V_S). Generated answer matched to S, X, B, Z, D or other.
     J-A6 (R): OPT-A and LET-A: key-source rate >= 0.6 (lower bound > 0.5) and key-source minus value-source >= 0.3
     (lower bound > 0); NOM: value-source rate >= 0.6 and value-source minus key-source >= 0.3. Flag-only: OPT-A
     answers S at >= 0.6 (lower bound > 0.5), NOM at <= 0.2 (upper bound < 0.3). Copy-fallback (K_Z,V_S) in OPT-A:
     two-sided, preregistered expectation written in the entry (the account: the key matches no option, so the
     lookup is silent and the value copy gives S; predict S-rate >= 0.5), class R. The old A6 flips are secondary.
     Delete old A7.
A-4  A8 (R4): relabel as a low-risk (L) channel-level replication of induction-head K-composition (cite Elhage et al.
     2021; Olsson et al. 2022), restricted to leak-free PERSON/PLACE items; continuation measured on tokens where c_S
     differs from c_B; framed as a third regime (key = address, value = content). Remove it from every novelty claim
     and from the sign account. Keep A8(d) (hybrid rate) as class M on that population.
A-5  Heads (R5, R6 first option): keep HA1 (natural readers, M) and HA2 (template readers transfer, R). Rebuild HA3 as:
     (a, R) MCQ behavioural necessity: with N* mean-ablated over Q+, OPT-A faithful accuracy on prior-free competent
     items under the KV_S row (passage says S; teacher-forced argmax chain on c_S) drops by >= 0.3 relative to none,
     with each random set dropping <= 0.1; (b, L) NOM continuation and decision accuracy under N* within 0.05 of
     none (stated as expected by construction). A copy set C* (top-k* heads by direct logit attribution to dec_B at the
     answer row in NOM on R items, k matched to N*) is ablated in both formats as an exploratory double dissociation.
     Drop HA4's reconciliation claim (report N* ablation's ID_V rise as exploratory).
A-6  Quantization (R7) promoted: J-A7 (R): KIVI-style fake quantization at 2 bits (keys per-channel, values per-token,
     group 32, all passage tokens, every layer), on prior-free competent items with the S passage as base and S as the
     faithful answer. Primary: DiD = [drop_NOM(V) - drop_OPTA(V)] - [drop_NOM(K) - drop_OPTA(K)] >= 0.10 with lower
     bound > 0; secondary: drop_NOM(V) - drop_OPTA(V) >= 0.10. Report the relative reconstruction errors of K and V
     (no strength matching).
A-7  Risk labels (R8) by the written rule "risky = no pilot or in-hand estimate already satisfies the threshold":
     A1 M; A1b R; A1c M; A2 L; A3 derived (reported, not counted separately); A4 L; A5 (MEN-A vs NOM, R11) R; MEN-B
     arm L control; A6 per A-3; A7 per A-6; A8 per A-4; HA1 M; HA2 R; HA3(a) R, HA3(b) L.
A-8  Emitted form (R9): per-model answer frame fixed in the preflight from greedy ID generations on the first 30 R items in
     NOM and OPT-A: the most frequent text between "Answer:" and the entity among {"", " ", " **", "**", " The ",
     " the "}; scoring conditions on prompt + frame (teacher-forced) and the decision token after it. A-G3 extends to S
     and X: g1(KV_S) = dec_S and g1(KV_X) = dec_X in >= 95% of competent items.
A-9  A5 (R11): J-A5 is s_ID(MEN-A) - s_ID(NOM) >= 0.10 with lower bound > 0, and ID_K(MEN-A) lower bound > 0; MEN-B as the
     control. Delete the templated add-on (Part B owns objection 5).
A-10 Sourcing and fallback (R12) per G2; intervals per G3.
A-11 E4 (YEAR): keep exploratory with the corrected expectation (value-only decision digit, I_dec ~ 0, in every tokenizer).
A-12 Graded A1 consequence (R15): if A3 met and s_ID(OPT-A) in [0.35, 0.5) with lower bound > 0.2: "a substantial but not
     dominant key read on natural text"; the template-only consequence only for A3 not met or lower bound < 0.2.
A-13 Compute: re-estimate (~3-3.5 GPU-h); the preflight measures throughput and the DEADLINE drops exploratory passes
     first, then LET-A rows beyond the letter accuracy and cue-conflict rows, then MEN-B.
A-14 Optional improvements accepted: none as confirmatory; D_in moderator and onset 0.3L stay exploratory, reported
     two-sided.

## B. Part B (fresh-sample replication), res1.md amended by res5.md

B-1  Drop JB9.
B-2  New wording and lexicon for F: (i) 8 preregistered neutral after-sentences (and the same 8 as before-sentences), each
     naming all six candidates, within +-3 tokens of ROOM in every tokenizer, assigned by core hash, identical between
     the POST and PRE arms of a core; (ii) the candidate order shuffled per core (same order in list and sentence, and
     across the arms of a core); (iii) a second lexicon {bin, crate, tray, jar, bucket, chest} in half of F (by core
     hash), with its own PAIR_SWAP-free generator (story words, list, sentences, CAND_RE and E forms built per item from
     its lexicon). Wording and lexicon are reported as fixed effects with per-level estimates; the bootstrap clusters by
     (lexicon, ordered pair). Calibration C covers both lexicons. JB6 stays on S0 with the original wording.
     Lexicon-2 words must pass the single-token and stability checks in every Part-B tokenizer (critic: all pass).
B-3  New arm POST-NULL (same position, syntax and length as the core's after-sentence, six nouns from neither lexicon).
     New line J-B-NULL (R in N4): upper bounds of r and s_ID(POST-NULL) <= 0.10 and lower bound of
     delta(POST - POST-NULL) > 0. JB3's "re-mention" reading is conditional on it.
B-4  Behavioural sentence line J-B5b (R): lower bound of beta_K(POST) - beta_K(PRE) > 0 in every P4 model (IUT) and in
     >= 3 of 4 N4 models; pre-written flip-rate sentence.
B-5  JB3 restated: s_ID(POST) lower bound > 0.10 and s_ID(POST) - s_ID(NONE) lower bound > 0, together with the r-based
     criteria; competence gate under POST (generated acc_B >= 0.8); competent-only must agree in direction.
B-6  Risk classes per G4 with the critic's assignment (JB1, JB2, JB4, JB5(AFTER/NONE) in N4 = M; JB3 in N4, J-B5b,
     J-B-NULL, JB6, JB7 = R; P4f lines = L).
B-7  JB6 as an equivalence test on S0: the paired-difference interval of s^E - s^L and of r^E - r^L inside +-0.05; ID_K
     and ID_V relative changes reported with tolerance 15%; headline count over cells with M^L(clean B) < 0.5, list cells
     reported separately; (c) the original verdicts recomputed under E is primary.
B-8  Coverage on every scored row: an E-based criterion is evaluable in a cell only if the minimum over rows of M^E is
     >= 0.8; JB-G4 aligned to 0.8; below it the beta counterpart is used.
B-9  LETTER and P1 are dropped from R2/R3 (LEAN default). LETTER is out of JB5/JB7 and every "every cell" rule.
B-10 JB-G3 (trie vs plain) runs for every model on 30 cores x NONE, POST, AFTER, with a relative tolerance:
     |s_ID(trie) - s_ID(plain)| <= 0.02 and mean |dL| <= 0.05 x mean ID_KV; fallback to score_reference is budgeted.
B-11 Nat thresholds made relative where possible (JB-G4 floor <= 0.05 x ID_K(AFTER); definedness relative to
     ID_V(AFTER)); the claim "every threshold is scale-free" is deleted.
B-12 Per-family reporting and pre-written sentences for exactly one and exactly two failing new families.
B-13 Compute: drop X3; JB8 (re-score of the 24B frames) runs last under the deadline; re-estimate generation steps
     (8-12 for verbose frames). Keep the core <= 2.5 GPU-h.
B-14 Optional accepted: (a) a small new-family model gemma-2-2b-it (cached), line J-B-SMALL (R): upper bound of
     r(POST) <= 0.10 with competence gated; (b) J-B-LB (M): upper bound of ID_K(BEFORE) - ID_K(NONE) < 0 in >= 3 of 4 N4
     models; (c) JB2 restated relative to NONE: upper bound of r(BEFORE) - r(NONE) <= 0.05; (d) P4f at 95% (IUT), N4 at
     98.75%; (e) the behavioural crossover figure.

## C. Part C (interventions), res2.md amended by res7.md

C-1  The law is a channel-ratio law (RC1). psi_C^E(f,l) = mean ID_C^E / mean ID_C^nat for C in {K, V, KV}; a channel is
     used where its natural effect is >= 2 nats and >= 10% of ID_KV^nat. lambda_E = log(psi_K / psi_V), within P1 and POST
     cells and across formats (psi_K in LETTER or P1 against psi_V in NONE), paired story bootstrap. Verdicts:
     equivalence by TOST with 90% CI inside |lambda| <= log 1.25; R1 copy-only lambda <= log 0.5; R3 key-flat lambda
     >= log 2; otherwise "graded departure" with its size. kappa vs sigma is descriptive only (Fig. 4); Pearson r and the
     share MAD are not verdict criteria.
C-2  Sensitivity gate J-C-G8 (RC2): synthetic rows from captured tables (natural K table interpolated toward B with
     lambda_K in {0.5, 0.8} with natural V, and the reverse) must be classified correctly (c = 1 equivalent; 0.5 and 2
     departures); otherwise the law line is NOT EVALUABLE. The old JC3 rule's verdict on these rows is reported.
C-3  Decomposition (RC3): per cell report residual cosine, per-channel K/V cosines, pooled nu. Rows PAR and PERP per
     family/depth/target; LEX/NONLEX split on the 5-dimensional English lexical span L_l (from E2's frozen sentences).
     A component "carries identity" if its phi(NONE) >= 0.3; only such PERP/NONLEX components and Tier-2 families are a
     genuine out-of-sample test; a family whose effect is carried by PAR or LEX (phi >= 0.8 of the full edit) is reported
     as "acts through the natural lexical code" and is not evidence that attribution is independent of the edit.
C-4  Boundary family E5 (RC4): steering vectors from neutral sentences with the FR, DE or SYN form of each value
     (ckeys/variants.py _F), plus NONLEX(E1). Efficacy phi(NONE) >= 0.3 with CI excluding 0; alpha in {1, 2} chosen on
     H_cal under NONE only. J-C-BOUND (R): lambda_E5 <= log 0.5 where gated in, against lambda >= log 0.8 for English E2
     at the same cell. Disclose the critic's 1.5B pilot.
C-5  JC6 (RC5): relabel "a non-natural identity edit at block 28 is not key-flat" (rules out R3 at depth); delete the
     kind-vs-depth readings. Add an overlap screen on Prakash et al.'s material at Qwen2.5-7B and Llama-3.1-8B: LM filter,
     BIND NO-MENTION sweep over all blocks, s_ID(OPTIONS, l0) at ~10 onsets; window = BIND Phi >= 3 nats and IIA >= 0.5
     while s_ID(OPTIONS, l+1) >= 0.4. If a window exists, run H's H7/H9 exchange there with H's thresholds (J-C-WIN, R);
     else the preregistered finding "binding forms only after the identity key route closes in every model screened".
C-6  Tiers (RC6): Tier 0 = E1, E2 at shallow/mid depth (L); Tier 1 = E2 deep, Llama-3.1-8B (M); Tier 2 = E3, E4, PERP,
     NONLEX, E5 (R). "Supported out of sample" requires the law met in E3 and E4 (or identity-carrying PERP/NONLEX) plus
     J-C-BOUND met. Tier 0 alone supports at most "consistent for near-natural steering vectors".
C-7  E4 (RC7) co-primary: seed 101 PCA init, seed 102 random orthonormal init, both reported; the pi(t) diagnostic under
     LETTER (K-only row Delta_pi(t) vs Delta_t) preregistered.
C-8  SAE gates (RC8): FVE >= published - 0.05; FVE at l exceeds FVE at l-1 and l+1; TEST_MODE unit test that io='out' of
     model.layers[l] equals the block-l output hook; pin revision and hash ae.pt, config.json, eval_results.json.
C-9  Feasibility table (RC9) in the entry; add l = 15 at Mistral-7B and Llama-3.1-8B; the iota rule uses psi_KV against
     psi_K + psi_V.
C-10 JC4 demoted to a consistency check (RC10); not in the headline rule.
C-11 Relative efficacy gate (RC11): flip rate >= 0.8 x the natural flip rate in the cell and phi(NONE) >= 0.5 with lower
     bound > 0.3.
C-12 Pre-committed Section-5 text per outcome (RC12); delete "30-70% away" and the kind-vs-depth sentence.
C-13 Cuts (O6): drop Gemma-2/Gemma Scope, alpha-dose and BIND-p. Optional O1 (first-order predictor of psi from the
     edit's displacement in the H* key subspace) is accepted as exploratory; O3 hierarchical bootstrap accepted (stories
     within (base, source) pairs; E4 seeds as a level).

## D. Part D (mechanism), res3.md amended by res9.md

D-1  Headline rescoped: delete "existence is decided by later mentions; sign by the reader" and N5's "LIST-BEFORE ~ 0 at
     7B". Replacement text (critic RC1) goes in the entry. IOI BEFORE at Qwen-7B becomes a preregistered two-sided
     exploratory test: knock out p's attention to the listed-name rows ({p} x list rows, ckeys/knockout.py, in source
     and base runs) and report the fraction of |ID_K| removed (>= 0.5 "p-as-re-mention"; <= 0.2 "second-order read
     elsewhere"), plus the RowSplice localisation.
D-2  Sign block rebuilt around channel dissociation. D7 and D9 become consistency lines (class per G4; disclose the
     critic's 1.5B pilot: Q_OUT ID_K -5.13, ID_V -8.24). New lines:
     J-D-SIGN-Q (M): Q_OUT sign(ID_K) = sign(ID_V) < 0 (upper bounds < 0) in 2/2 models;
     J-D-SIGN-IOI (L, stage-5 replication on fresh IOI cores): IOI INLINE ID_K < 0 < ID_V, CIs excluding 0, 2/2 models;
     J-D-ROUTE-IOI (R): IOI INLINE HopSplice route test with answer row END: rows none, K_S, K_S+ans_K, K_S+ans_V,
     K_S+ans_KV, and the same with the IOI-fit flag injected instead of K_S; r_ans(KV) >= 0.6 and r_ans(K) - r_ans(V) >
     0 (CI excluding 0); two-sided alternative r_other >= 0.5 reported;
     J-D-HOP2 (R): per-head answer-row HopSplice under the injected flag in P1, Q_OUT and IOI INLINE, top-10 hop-2 heads
     ranked on R under P1; "same reader, opposite sign" if they carry >= 0.5 of the Q_OUT and IOI-INLINE effects,
     "different reader" if <= 0.2; else intermediate.
D-3  D5 replaced by the binding test on the two initial-state sentences (critic RC3): fit Delta^init at p_init (key of an
     absent I') and Delta^dloc at p_dloc (key of an absent I''), norm-matched per layer, injected at r_X under
     other_agent (answer init) and irrelevant_object (answer dloc); population ~140 cores from a fresh seed so ~100 have
     dloc != init, stratified by sentence order (object-first vs distractor-first). J-D5 (R): Psi_bind > 0 with CI
     excluding 0 pooled over strata, and the stratum difference in Psi_bind has a CI including 0 (an ordering-ID account
     predicts a sign flip between strata). Gate: accuracy >= 0.8 for both queries. Event-vs-initial stays a secondary
     contrast labelled "role confounded with order".
D-4  D1 in the paper's currency (RC4): ID_inj = 1/2[(l_S - l_X)(move B->S) - (l_S - l_X)(move B->X)] against the
     in-batch ID_K = 1/2[c(K_S) - c(K_X)] (six-way renormalised); leave-one-word-out flag Delta^{-w} fit on R stories where
     neither B nor S is w, injected at r_w. J-D1 (R): ID_inj^LOO / ID_K >= 0.5 with lower bound >= 0.35, and
     pi_X(move) >= 0.5. iota vs N_X and the dose response secondary.
D-5  Identity-freeness J-D-ADDR (M): Delta^KV fit from the natural S run; norm-weighted mean over L* of
     cos(Delta^K, Delta^KV) >= 0.9 and ID_inj(Delta^KV)/ID_inj(Delta^K) in [0.8, 1.25] with CI inside [0.7, 1.4];
     report the B-identity variance share and the logit-lens cosine with W_U[locations].
D-6  K_N test J-D-KN (R): p's key from a story whose event word is a single-token non-candidate (fixed list, checked
     single-token in Qwen and Mistral tokenizers); the six-candidate change under K_N matches the "-Delta at r_B"
     injection (per-story Pearson >= 0.8 over candidates; B-loss ratio in [0.7, 1.3]); |mean(dl_S - dl_X)| <= 0.1 x ID_K.
D-7  Structured controls (RC6): D2 = the story's own write orthogonal to Delta-hat (norm-matched), the mean clean-run H*
     output direction at the option rows, and a flag fit identically from the stage-6 most-active-at-G set; criterion
     iota(Delta) - iota(control) >= 0.4 and iota(control)/iota(Delta) <= 0.3. D3 adds the top principal direction of
     clean y_l at the option rows and the mean-H*-output direction, each with rho_K >= 0.7; isotropic as sanity rows.
     Report rank-1 capture.
D-8  D6 fixed (RC7): drop "never written"; J-D6 (R) mediation: sign(rho) = sign(rho_nat) and |rho - rho_nat| <=
     max(0.1, 0.5|rho_nat|) with rho_nat = N_X(POST)/N_X(P1) in-run on the same competent cores and scorer; POST route
     batch at 1.5B, 3B, 7B; J-D6-ROUTE (R): at 1.5B r_ans^inj(V) > r_ans^inj(K); branch "rho ~ 0 while rho_nat < 0" =
     the natural negative read is not mediated by the sentence-row flag. 3B at n = 60.
D-9  IOI cells (RC8): add the chat-wrapped INLINE arm (INLINE-CHAT) at Qwen-7B and Mistral-7B under stage 5's Gate e;
     report d_Gc / ID_K per D8 cell (Gc = the clamped names' rows), a cell counts as a re-mention read only if >= 0.5;
     D8 scored per cell with IOI-INLINE primary (R) and the Q cells consistency (L).
D-10 Cross-task flag (RC10): exploratory with a two-sided rule (ratio >= 0.6 shared pointer; <= 0.3 task-specific), paint
     and schedule P1 tasks (ckeys/tasks.py), each task's own flag fit on 60 stories.
D-11 Code (RC11): Gate D0 has NONZERO-add composition tests (Inject with a no-op HopSplice or an empty-set HeadSplice equals
     Inject alone to 1e-4; HeadSplice-ablate with mu = its own output plus Inject equals Inject alone); the add and the
     projection are computed in FP32 inside the hook and cast back; "2/2" needs two evaluable models; the Holm family is
     the R lines of Part D.
D-12 Cuts (optional list): drop D9 as a separate line (rows stay), drop E1's story-duplicate projection pass and E3's eager
     probe. Optional accepted as exploratory: the Q_OUT behavioural headline (fraction of stories answering B "not
     mentioned" under K_S and V_S); the H3/G7b reconciliation by directional ablation at r_init/r_dloc vs r_B vs all six
     rows (base-argmax and initial-location answer rates). Disclose the critic pilots (Random(0) indices 16-46, Random(7),
     Random(9)); E8 = Random(81), IOI seeds 82/83 and the D-3 seed must be disjoint from them.
D-13 Presentation for the paper (later): three numbers for the flag in the main text (ID_inj^LOO/ID_K, rho_K,
     r_ans^inj(K)); cite Wang et al. 2022 and Feng & Steinhardt 2023; novelty stated as in the critic's last item.

## E. Part E (rewrite), res4.md amended by res8.md (applies to entry J now and to v5 later)

E-1  One risk rule for entry J (= G4). A-I labels appear only in an appendix, computed by a script from each entry's
     "seen before" record, labelled post hoc, never pooled with J. Brier score / calibration summary in the appendix.
E-2  Entry J names the single primary line behind each planned abstract clause and Table 1 row, and gives, for every
     primary line, the pre-written abstract text, title consequence, Table 1 row and figure for MET / NOT MET (RC6). Each
     part's builder drafts these rows for its part.
E-3  For v5 (after the run): Table 1 rebuilt with three columns (architecture alone | known circuits, cited | what we find);
     novelty = (i) later list recruits selection under a free-answer instruction and on passages, value route nearly idle;
     (ii) no second-order lookup after an earlier list (small negative read stated as unexplained); (iii) cue-conflict;
     (iv) a question-blind flag sufficient and necessary for the read; (v) the key channel reads only the token-form part
     of a written value, natural or edited. Claim 3 per C-1..C-12. Related Work restores Wu & Shomali 2026, Ok & Lee
     2026, Steele et al. 2026, Wong et al. 2026, Feucht et al. 2025, Zhou 2026, MIB; fixes oh2026rebinding and
     prakash2025lookbacks; Tulchinskii "resembles". Title: "Looked Up or Copied? Later Mentions Decide Whether a Model
     Reads an In-Context Value Through Its Key or Its Value" with a templates-only fallback. Limitations per RC7. Number
     and symbol rules per RC8. Page budget redone (RC10). Anonymity per RC11 (EDITED_FILES.txt rename done; extend the
     scrub and leak grep to entry J, stage-8 code and result JSONs).
