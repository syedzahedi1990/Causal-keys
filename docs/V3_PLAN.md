# Paper v3: the rewrite plan after stages 5 (preregistration G) and 6 (preregistration H)

This plan builds on `docs/V3_STAGE5_IMPLICATIONS.md` (the stage-5 memo) and applies the actual stage-6 outcome.

**Verdicts are the scorers' records:**
- Stage 5: `results/gpu_stage5/STAGE5_SCORE.txt` at 51e105e. 12 MET, 10 NOT MET of 22.
- Stage 6: `results/gpu_stage6/STAGE6_SCORE.txt` at cc3a3e0. 6 MET (H1–H5, H11), 4 NOT MET (H7–H10), 1 NOT EVALUABLE (H6), 1 NOT RUN (H12) of 12.

Independent recomputations reproduced both scores, and nothing here changes a verdict.

**Where the numbers come from:**
- Unmarked numbers are from the two score files.
- *(recomputed)* means recomputed from the raw files and not printed by a scorer. Such a number must appear in the paper as exploratory or post hoc.
- *(arithmetic)* means a difference or ratio of printed values.

**Conventions:**
- In paper text the predecessor is only \citet{anonymous2026fitted} or "our predecessor".
- Model pairs are Qwen2.5-7B / Mistral-7B (Instruct) unless stated. Part (b) of stage 6 is Qwen2.5-14B-Instruct, with formats in the order NO-MENTION / QNAMES / OPTIONS-AFTER.

---

## 0. What changed, in five sentences

1. Stage 5 conceded that hop 1 is duplicate-token attention and showed that it is not sufficient: the re-mentions attend to the writing token at 1.5B and 3B, whose key identity is about zero (within the preregistered ±1-nat margin; G1–G3). It also removed "the lookup replaces the copy" as a causal lead (G7 partial, G11 and G20 not met).
2. Stage 6 (a) identified the readers and closed the second hop at the 7B pair. About 5 % of heads are sufficient (H1) and necessary (H2) for the option-row read; ablating them removes most of the key identity and raises the value copy (H3); their core is canonical duplicate-token heads (H4); and the answer position reads the option words directly (H5). That it reads them through their keys is H5's preregistered secondary contrast, reported but not scored.
3. Stage 6 (b) replicated Prakash et al.'s state-token binding swap at Qwen2.5-14B on their own material and decomposed it. Its effect is key-dominant in each of the three evaluable formats (kappa 0.618 / 0.906 / 0.864), but without a later mention the values carry a substantial share (psi_V 0.435). LETTERS-AFTER failed Gate b2. Under QNAMES and OPTIONS-AFTER the swap flips 0/150 answers, so those kappas describe a shift of the logit margin. The preregistered law (H7: kappa ≈ the natural identity key share), the shared value prediction (H8), the crossover (H9) and the rival (H10) all failed.
4. The identity-edit control on their material was met (H11). With l*_ID = 0, however, Part 1 is close to a restatement of the natural clamp, and Part 2 compares values near zero (see §e3). The out-of-sample law for identity edits is therefore untested.
5. The intervention half of the paper therefore narrows; it does not become two-sided.
   - For edits that change which value is written (our predecessor's remap, its refit, and the identity edit at block 0), the read format moves the attributed channel strongly (identity edit: kappa_ID +0.648 from NO-MENTION to OPTIONS-AFTER).
   - For Prakash et al.'s binding swap at block 28, later mentions move kappa by +0.247 [+0.211, +0.281]. The shift is real, but below the preregistered 0.4 (H9 not met).
   - The two are not compared at matched depth. At block 28 the identity edit is value-carried in every evaluable format and moves the other way (kappa_ID 0.129 → 0.006, −0.123 *(arithmetic)*), because by then no format has a key-read identity route left (s_ID(OPTIONS-AFTER) −0.014 at onset 27; ID IIA 0.00 from block 20 under QNAMES and OPTIONS-AFTER). The binding swap cannot be moved earlier: its NO-MENTION IIA is 0.00 below block 24.
   - Whether the kind of edit, rather than depth, decides how strongly the format moves the attribution is therefore untested (Gate b3 not met).

---

## (a) The new headline claims, their evidence, and what is dropped

### Headline claims (main text)

| # | Claim (paper wording to adapt) | Evidence (preregistration, verdict) | Scope to state |
|---|---|---|---|
| C1 | Later mentions of the candidates decide whether a written value's identity is looked up through the writing token's key or copied from its value. Their position is causal. | A1–A3, B1–B2, D1–D2, E1a–c (all met); unchanged from v2 | 10 models, four families, three templated tasks (belief, paint, schedule). Not IOI (see C7) |
| C2 | **The lookup has two hops: a key read at the option words, then the answer's read of the option words.** Hop 1: a small set of attention heads reads the writing token's key at the re-mentioned option words. With only the top 5 % by attention change seeing the swapped key, R(k*) = 0.967 / 0.942 (40 / 52 heads; 20 / 16 heads reach 0.8), and blinding them removes 0.977 / 0.971. Random sets of equal size give R 0.004 / 0.007 and KO 0.011 / 0.010 (means over three draws; no single draw above 0.02, recomputed). Hop 2: the answer position reads the option words directly, r_ans(KV) 0.899 / 0.859 against r_other 0.118 / 0.165. As a preregistered secondary contrast (reported, not scored), it reads them through their keys: r_ans(K) 0.797 / 0.683 against r_ans(V) 0.080 / 0.023. | H1, H2, H5 met (2/2); the K-versus-V contrast is H5's unscored secondary. The edge-level counterparts are G5 met (cutting the candidate rows' attention to the writing token gives r_K(M1) −0.105 to +0.003 in 9/9 cells) and G8b met (the answer needs no direct attention to the writing token: r_K(M3) 1.04 / 1.03 / 1.01) | Heads and hop 2: Qwen2.5-7B and Mistral-7B, OPTIONS-AFTER, BF16. Rows and edges: 7–14B. The SENTENCE-AFTER hop line is not evaluable (Gate a2) |
| C3 | **The readers' core is canonical duplicate-token heads, and the duplicate-token match is not what makes the read scale-dependent.** Over the 20 / 16 heads that carry 80 %: median duplicate score D 0.370 / 0.232, induction 0.011 / 0.022, and 6/10 and 5/10 of the top-10 shared with the top-10 by D. The re-mentions attend to the writing token, and the attention moves with the clamped key, at every scale, including 1.5B and 3B, whose key identity is about zero (within the preregistered ±1-nat margin). The format dependence (after versus before versus no mention) is what duplicate-token attention plus the causal mask predicts, so it is not evidence against that account. | H4 met (2/2). G1, G2 (H_diss), G3 met | Concede that hop 1 is duplicate-token attention (this is not novel). G2 is a dissociation across scale only; no result at 7B shows re-mention attention without the read in some format (H6, the only cross-format head test, is not evaluable). The full 5 % set is mixed: median D over k* is 0.179 / 0.065 *(recomputed)*. Hop-2 cross-scale evidence is attention only (G4) |
| C4 | **The lookup suppresses a copy route that stays available.** Ablating the reader heads at the option rows cuts ID_K to 0.238 / 0.059 of its value, while ID_V rises by +6.907 / +10.861 nats. The model still answers B in 0.87 / 1.00 of stories. Cutting the edges instead brings the copy back on the identity measure in 3/3 models but keeps the answer only at Qwen2.5-14B. | H3 met (2/2). G7a 3/3; G7b 1/3, so G7 is not met (preregistered "partial takeover") | Not "the lookup replaces the copy" (G11 and G20 not met). Head ablation and edge knockout differ: rho_K 0.238 vs r_K(M1) ≈ 0, and the populations differ. Say "raises" the value copy: at Qwen2.5-7B the rise is partial (ID_V 4.640 + 6.907 = 11.55 under ablation against 16.60 under NO-MENTION in the stage-1 summary; cross-stage, arithmetic); only at Mistral-7B does it reach about its NO-MENTION level (E-d) |
| C5 | **Where an identity edit appears to act depends on the format in which it is read.** Our predecessor's remap has nearly the same behavioural effect in every format, yet its keys or its values appear to carry it, following the format's readers. A remap refit without later mentions does the same, and so does the identity edit on Prakash et al.'s material at block 0 (kappa_ID 0.131 under NO-MENTION, 0.779 under OPTIONS-AFTER). | C3, E4 met; F1–F4 met; H11 met (Part 1 difference +0.648 [+0.630, +0.666]) | The Prakash identity-edit crossover is at block 0 only; at block 28 the identity edit is value-carried in every evaluable format (kappa_ID 0.129 / 0.007 / 0.006). H11 at l*_ID = 0 is close to the natural clamp. Present it as consistency on their stories and wrapper, not as out-of-sample support |
| C6 | **On Prakash et al.'s binding swap (one binding edit, at Qwen2.5-14B), the out-of-sample law fails, and the effect is key-dominant in each evaluable format.** We first replicate the swap (IIA 0.993 at block 28 against their released 1.00). Without any later mention, the state tokens' keys carry most of the effect and the values a substantial share: psi_K 0.703, psi_V 0.435 (kappa 0.618). With later candidates the keys carry nearly all of a non-flipping margin shift (psi_K 1.020 / 1.025). Later mentions shift kappa by +0.247 [+0.211, +0.281], less than the preregistered 0.4. The preregistered law, which predicts kappa from the natural identity key share, fails by 0.60–0.93. | H7, H8, H9, H10 not met. Gate b1 met. Replication table *(recomputed)*. Gate b3 not met | Qwen2.5-14B only (H12 not run). One edit, so no claim about binding or address edits in general. Evaluable formats: NO-MENTION, QNAMES, OPTIONS-AFTER (LETTERS-AFTER failed Gate b2). Under QNAMES and OPTIONS-AFTER the swap flips 0/150 answers, so those kappas describe a margin shift (Phi 5.8 / 6.5 nats). Prakash et al. describe this patch as swapping addresses and payloads, so it is not a pure address edit |
| C7 | Generality: plain IOI carries almost no key identity. Whether a later list opens a key read on IOI is not established. | G18 met. G19–G21 not met | State as a limitation, not as support |

**Exploratory or post hoc claims that may be stated, labelled as such:**
- **E-a.** At the same positions and depth (block 28), the binding swap is key-dominant (psi_K 0.703 / 1.020 / 1.025; psi_V 0.435 / 0.106 / 0.161), while the identity edit is value-carried (psi_K 0.129 / 0.006 / 0.005; psi_V 0.872 / 0.827 / 0.811). It is one post hoc reading consistent with the H7 failure, not an explanation of it. Caveats:
  - Under QNAMES and OPTIONS-AFTER both edits flip 0/150 answers at block 28, so four of the six cells describe margin shifts.
  - Prakash et al. describe the binding patch as moving both an address (read through QK) and a payload (moved through OV). E-a is therefore not a clean address-versus-payload split.
  - It was not preregistered: Gate b3 was built for the reverse dissociation and failed.
- **E-b.** The identity key route at the state token closes with depth under OPTIONS-AFTER: s_ID is 0.702 at l0 ≤ 3, 0.218 at 18, 0.041 at 24 and −0.014 at 27.
- **E-c.** The read needs heads in several layers jointly. No single layer recovers more than 0.05 of d_G, and the single-layer shares sum to 0.163 / 0.094 *(recomputed)*. Single-head causal rankings fail (f+ R(128) 0.06 / 0.66).
- **E-d.** At Mistral-7B, ID_V under reader ablation (13.45, arithmetic) is about its NO-MENTION level (13.65). This is a cross-stage comparison.

### Dropped or narrowed

- **Drop "the r = 0.98 relation is a law" as a general predictor of an intervention's key share.** It failed out of sample on a non-identity edit (H7: gaps 0.601 / 0.934 / 0.879; r = −0.986 over s_ID values within 0.045 of zero).
  - Keep r = 0.98 only as the post hoc in-sample relation for the released remaps, and F4 as the preregistered test for the refits.
  - Add: out of sample it fails for the binding swap (H7). The identity-edit control (H11) is met but nearly tautological at l*_ID = 0, so the out-of-sample law for identity edits is untested.
  - Report H7's secondary test beside the failure: against s_ID(f, 0) the gaps are 0.600 / 0.181 / 0.163, so there only NO-MENTION is outside 0.25.
- **Drop any claim that later mentions switch the channel of interventions in general.** H9 is not met: +0.247 [+0.211, +0.281], and kappa(QNAMES) 0.906 > kappa(OPTIONS-AFTER) 0.864. Do not replace it with the opposite claim either: the shift is positive and its CI excludes 0, so the format does move this edit's attribution, by less than predicted.
- **Drop any statement that Prakash et al.'s binding swap is value-carried, or that binding sits in values (H10 not met: kappa ≥ 0.618 in each evaluable format).** Still report the value share (psi_V 0.435 under NO-MENTION). Also drop the v2 appendix sentence "We do not test the address side": their state-token patch, which carries address and payload together in their account, is now decomposed.
- **Drop the stage-5 lead "the lookup largely replaces the copy (even when the answer is the word itself)" as a causal claim.** This follows stage-5 memo item 5. Replace it with C4.
- **Drop "we did not identify the heads"** (`results_read.tex` sec:role; the appendix lookback paragraph). Replace it with C2/C3.
- **Drop any phrase implying single heads are necessary or sufficient.** Say "a set of heads".
- **Do not claim "the same heads read the list and the sentence".** H6 is NOT EVALUABLE.
  - It may be reported, labelled exploratory: Mistral-7B 14/20 shared (met); Qwen2.5-7B 13/20, computed but not evaluable because its SENTENCE-AFTER batch-consistency gate failed (0.770 against 0.5 nats).
- **Do not claim the SENTENCE-AFTER second hop as confirmed** (r_ans 0.788 / 0.780; not evaluable overall).
- **Narrow "the re-mentioned words are the (main) readers"** to "the main readers of the key-swap effect in lists and in sentences naming at least three candidates" (G12 not met, 9/12; stage-5 memo item 6).
- **Narrow "lexical identity of a single word"** to "the word form largely up to case; synonym- and number-sensitive" (G14 TITLE/UPPER, G15, PLURAL; stage-5 memo item 7). Add that a multi-token wrapper reads only partly (G13 MODIF not met, 0/4 models).
- **Narrow "at 1.5B and 3B no identity is read"** to "the key identity is about zero (within the preregistered ±1-nat margin)" (stage-5 memo item 9).
- **Narrow generality to the three templated tasks** (IOI G19–G21 not met; stage-5 memo item 8).
- **Keep "that the readers are the same at 24B and 72B is an inference".** Heads and hop 2 were tested at 7B only.
- **Do not generalise from the binding swap to "address edits".** One binding edit was tested, at one model; a general identity-versus-address rule is a hypothesis.

---

## (b) Answers to critique items 1–6

| Item | Status after stages 5 and 6 | Answer the paper gives |
|---|---|---|
| **1. Reframe; table of what each account predicts** | **Done, with a changed lead** | Lead with C1 + C2: later mentions open a two-hop read whose first hop is a key read by generic duplicate-token attention. Then the attribution half, narrowed (C5/C6): a strong format effect for identity edits (at block 0 on Prakash et al.'s material), a weak one (+0.247) for their binding swap at block 28, with the depth confound stated. The critic's second lead ("the lookup largely replaces the copy") is not supported causally (G7 partial, G11, G20), so it becomes C4 ("suppresses a copy route that stays available"). The accounts table (§c.7) covers the duplicate-token, lookback address/payload, fit-format and later-mention accounts, with where the data refute each. |
| **2a. Attention without the read at 1.5B/3B** | **Answered: yes** | G2 H_diss met (E 0.79 / 0.88, F/E 1.03 / 0.99), at magnitudes comparable to 7B/14B (G3). The duplicate-token match is present but not sufficient across scale; this explains the scale dependence, not the format dependence. H4 now shows that, at 7B, the causally selected readers are that same kind of head. |
| **2b. Block the lookup; does the copy take over?** | **Answered: partly** | Edges: G5/G6 met; G7a 3/3, G7b 1/3 (partial takeover). Heads: H3 met, with ID_V +6.9 / +10.9 nats and the answer kept (0.87 / 1.00). Give both, without averaging them into one claim. |
| **2c. Dose-response** | **Answered, split** | G9 met (membership). G10 met for lists and not met for sentences. G11 and G12 not met. Unchanged from the stage-5 memo. |
| **2d. Token vs concept** | **Answered, not in one direction** | G15 met (form-bound on the English measure). G14 fails for case variants. G16 is graded and model-dependent. Unchanged; cite Feucht et al. here. H4 adds that the readers are duplicate-token heads, not induction heads (median I 0.011 / 0.022). |
| **3. Identify the heads; second hop** | **Answered at 7B** | H1–H4 met: about 5 % of heads suffice and are necessary, their ablation trades key for value identity, and their core is canonical duplicate-token heads. H5 met: the answer reads the option words directly, so the three-hop route is rejected; that it reads them through their keys is H5's secondary contrast (reported, not scored). Not tested at 14B+, and the cross-scale hop-2 difference remains attention-only (G4). H6 (same heads across formats) is NOT EVALUABLE. |
| **4. Change someone else's result** | **Prakash half answered (against our prediction); IOI half partly** | Prakash et al.'s state-token swap replicates at Qwen2.5-14B (IIA 0.72 / 0.75 / 0.99 / 1.00 / 0.08 at blocks 24 / 26 / 28 / 30–34 / 36, against their 0.69 / 0.72 / 1.00 / 1.00 / 0.06 *(recomputed)*). The exchange (patch at block 28) shows the effect is key-dominant in each of the three evaluable formats, with psi_V 0.435 under NO-MENTION, and a later mention shifts kappa by +0.247 [+0.211, +0.281], below the preregistered 0.4. This adds a KV-level decomposition of a patch that they describe as swapping addresses and payloads. It does not show a format flip, and it covers one edit at one model. IOI: G18 met; G19 not met (stage-5 memo). |
| **5. r = 0.98 as an out-of-sample law** | **Answered for a binding edit (fails); untested for identity edits** | H7 not met (gaps 0.60–0.93; the secondary test against s_ID(f, 0) gives 0.600 / 0.181 / 0.163). H11 is met (gaps ≤ 0.114; crossover +0.648), but at l*_ID = 0 Part 1 nearly restates the natural clamp and Part 2 compares values near zero, so it does not test the law for identity edits out of sample. The paper states the law as an in-sample relation for the released remaps (F4 for the refits) and reports H7 as a preregistered failure in the main text (sec:failed). |
| **6. Positioning** | **Writing; see §c.5** | Deltas against Ok & Lee, Wu & Shomali, Prakash et al., Zhou, Feucht et al., Ma et al. and Tan et al. are written below. Include the anonymised predecessor as supplementary material. |

---

## (c) Section-by-section changes

### c.1 Abstract (`abstract.tex`, rewrite; no longer than v2's, which is 260 words by `wc -w` on the source)

Draft (249 words). Fill every number from `numbers.tex` macros generated by `paper/make_figures.py`; the values shown are the score-file values. The key-versus-value part of hop 2 is an unscored secondary result, so the abstract says only that the answer reads the option words directly.

> A token that states a value in context, such as *shelf* in "the candle is moved to the shelf", can be read by later tokens through its attention key, matched by a later query, or through its value, which is copied. We clamp the two channels separately at this writing token in \nModels{} instruction-tuned models from four families. When the candidates are mentioned after the writing token, by an option list or, from 7B up, a neutral sentence, the value's identity is looked up through the key; otherwise it is mainly copied, even when the same list precedes the writing token. At Qwen2.5-7B and Mistral-7B, 5 % of attention heads, whose core is canonical duplicate-token heads, read the writing token's key at the option words (0.97 and 0.94 of the effect; random heads about 0.01 on average), and the answer position reads the option words directly. Such attention is also present at 1.5B and 3B, where the key identity is about zero. A learned remap from prior work has nearly the same behavioural effect in every format, yet whether its keys or values appear to carry it follows the format's readers, also for a refit without later mentions. On Prakash et al.'s binding swap, which we replicate at Qwen2.5-14B, a preregistered law predicting its key share from the natural read failed: the swap is key-dominant in every evaluable format, and later mentions shift its key share only slightly. We separate preregistered from exploratory results and report every failed prediction.

### c.2 Introduction (`intro.tex`)

- **Schematic, panel (a).** Add the second hop: an arrow from the answer position to the option word's key. Mark hop 1 as "duplicate-token heads".
- **"Later mentions decide" paragraph.**
  - Keep the first three sentences.
  - Replace "the re-mentioned words are the main readers…" with C2 in one sentence, plus G12's qualifier ("in lists and in sentences naming at least three candidates").
  - Replace "The read is an identity lookup in the style of duplicate-token attention…" with C3: its first hop is duplicate-token attention, which is present at 1.5B and 3B, whose key identity is about zero (G2), so the scale dependence lies downstream of hop-1 attention. The format dependence (after versus before versus no mention) is what duplicate-token attention plus the causal mask predicts; say so.
- **"Where an intervention appears to act" paragraph.**
  - Keep the reproduction and the crossover. Change "values where keys do not" and the refit sentence only for wording.
  - Add two sentences on Prakash et al.: the replication; the swap key-dominant in the three evaluable formats, with a value share of 0.435 without later mentions; the preregistered law failing (H7); and a later mention shifting kappa by +0.247, below the preregistered 0.4 (H9). Do not cite H11 as the law holding out of sample.
  - Change the last sentence to: "A component-level explanation of an intervention is thus a property of the intervention, of what it changes, and of the prompt in which it is read."
- **Contributions** (rewrite):
  - (i) Exact separate K/V clamps, the identity measure, row-, head- and answer-row-restricted splices (HeadSplice and HopSplice), and an exact K/V exchange on another lab's residual patch.
  - (ii) Later mentions decide look-up versus copy, and their position is causal (A–E).
  - (iii) The mechanism at 7B: sparse duplicate-token readers, then the answer's direct read of the option words, mainly through their keys (G, H1–H5; the key-versus-value part is H5's unscored secondary). The duplicate-token match is not sufficient across scale (G2).
  - (iv) Attribution: the attributed channel of identity edits flips with the read format (C, E4, F; on Prakash et al.'s material at block 0, H11). On their binding swap (Qwen2.5-14B) the preregistered law fails, and later mentions shift the key share by only +0.247 (H7–H10). The two are not compared at matched depth.
  - (v) Eight preregistrations (A–H) with every failed prediction reported.
- **Prakash et al. in the first paragraph.** Replace "Lookbacks retrieve a state token through bound ordering IDs, in our terms a copy read" with the tested statement: their state-token patch, which swaps ordering IDs (address and payload, in their terms), reaches the answer mainly through the state tokens' keys, with a value share of 0.435 without a later mention (C6), while the identity stored there is copied through the value without a later mention (H11 NO-MENTION psi_V 0.871; E4).

### c.3 Setup (`method.tex`; about 0.3 column added)

- **Heads.** One paragraph:
  - a3 (attention change under the key clamp, ranked on 60 held-out ranking stories) and HeadSplice (chosen heads see K_S in the option rows).
  - R(k) and KO(k), and mean-ablation at the option rows.
  - HopSplice: only row T−1 gets the base K/V of the option rows while K_S stays clamped at p, giving r_ans, r_other and r_all.
  - The FP32 exactness tests (Gate a1).
- **Exchange on Prakash et al.'s patch.** One paragraph:
  - Their template-2 stories, raw prompt and seed-10 pool, replicated at their release commit.
  - BIND (their reversed-sentence state-token swap at block l*) and ID (a donor state word at the same positions).
  - The seven exchange rows; psi_K, psi_V and kappa; the evaluability rule.
  - l* and l*_ID chosen by the earliest maximum of IIA on the NO-MENTION sweep.
- **Statistics paragraph.** Change "Six preregistrations (A–F)" to "Eight (A–H)", keeping v2's wording: each committed to version control before its run, with the scoring script committed before its outputs were inspected. Add that H was finalised before any stage-5 output was inspected.

### c.4 Results (`results_read.tex`, `results_intervention.tex`)

**§sec:mentions.** Unchanged except wording ("otherwise mainly copied" is no longer an exclusive alternative, per the stage-5 memo).

**§sec:general.**
- Keep Tasks (one paragraph).
- Shorten Scale to three sentences, keeping the 14B peak and E3, and move `fig_scale` to the appendix.
- Move Depth to the appendix.
- Add one IOI sentence (stage-5 memo item 8).

**§sec:readers, rewritten as "Which tokens and heads read the key, and what the answer reads"** (about +0.5 page including the merged figure*; see c.9):
1. **Rows.** D3 as in v2, shortened to one sentence with the detail in the appendix (space, c.9). G5 (necessity of the candidate rows' edges to the writing token) and G6 (matched control column). G12's two-word-sentence caveat goes in the appendix with one clause in the text.
2. **Heads (H1–H3).**
   - R(k*) 0.967 / 0.942; KO(k*) 0.977 / 0.971; random sets R 0.004 / 0.007 and KO 0.011 / 0.010 (means over three draws; no single draw above 0.02, recomputed); k80 = 20 / 16.
   - Single-head rankings fail (f+ R(128) 0.06 / 0.66), and no single layer carries more than 0.05 (exploratory). The read is a joint property of a few dozen heads over about 10–12 layers.
   - Ablation: rho_K 0.238 / 0.059; controls ≥ 0.95; ID_V +6.9 / +10.9; base answer kept in 0.87 / 1.00.
   - Contrast with G7b, where cutting edges keeps the answer only at 14B. Write both, without merging them.
3. **The second hop (H5; G8b).**
   - The answer position needs no direct attention to the writing token (G8b).
   - It reads the option words: r_ans(KV) 0.899 / 0.859; r_other 0.118 / 0.165; r_all 0.992.
   - Secondary, reported but not scored (H5): it reads them through their keys, r_ans(K) 0.797 / 0.683 against r_ans(V) 0.080 / 0.023.
   - The three-hop alternative is rejected.
   - One sentence: SENTENCE-AFTER gives 0.788 / 0.780, but the line is not evaluable because of Qwen2.5-7B's BF16 batch-consistency gate (appendix).
4. Remove "that it then reads the later mentions we did not test".

**§sec:role, rewritten as "What the duplicate-token account explains"** (about 0.1 page more than v2's sec:role, with the role control cut to one sentence; see c.9):
- G1–G3: hop-1 attention exists at every scale, including 1.5B and 3B, whose key identity is about zero (within the ±1-nat margin). Give E and F/E in one sentence.
- H4: the readers' core is canonical duplicate-token heads.
  - Median D 0.370 / 0.232 against whole-model medians 0.028 / 0.007 *(recomputed)*.
  - Induction 0.011 / 0.022; previous-token 0.017 / 0.004.
  - Top-10 overlap 6 and 5 (P = 1.4e-10, 6.7e-09).
  - State the narrowing: "the 16–20 heads that carry 80 % are dominated by duplicate-token heads; the full 5 % set is not (median D 0.18 / 0.07, recomputed), and a3 and D correlate weakly over all heads (Spearman 0.19 / 0.23)".
  - Treat T_dup as supporting, not independent, evidence.
- Part (d) of G in two sentences: form-bound on the English measure (G15), case-insensitive (G14), graded token/concept (G16).
- The contribution sentence (stage-5 memo §d): generic duplicate-token heads write the identity flag into the option words' keys, and the answer selects among the options, mainly by key (H5 secondary, not scored). The scale dependence enters downstream of hop-1 attention (G2). The format dependence is what duplicate-token attention plus the causal mask predicts. The duplicate-token match is not novel.
- Cut the role control (E2) paragraph (133 words in v2) to one sentence with an appendix pointer (space, c.9).

**§sec:intervention.** Keep its paragraphs, moving the reproduction detail and the post hoc paragraph to the appendix (space, c.9). Edit "Interpretation" to say the reading applies to edits that change which value is written; on Prakash et al.'s material it is shown at block 0 only.

**§sec:refit.** Keep. Trim the post hoc detail to one sentence; the rest goes to the appendix.

**New §sec:prakash, "A binding edit from another lab"** (about 0.6 column; `tab:prakash` goes to the appendix and its key numbers into the text):
- **Replication.** One sentence: IIA 0.993 at block 28 on 150 pairs, matching the released per-layer profile (0.69 / 0.72 / 1.00 / 0.06 at blocks 24 / 26 / 28 / 36). The full table goes in the appendix *(recomputed)*.
- **What BIND changes.** Template 2's counterfactual swaps the two sentences, so the patch moves each state token's residual to the other sentence slot. The state word's identity does not change; its ordering information does. Prakash et al. title this experiment "Binding lookback Address and Payload": the patch swaps the addresses (character and object ordering IDs) and the payload (the state ordering ID), and in their account the payload is moved through the OV circuit. Use their description, not "address edit".
- **Exchange.**
  - NO-MENTION: psi_K 0.703, psi_V 0.435, kappa 0.618.
  - QNAMES and OPTIONS-AFTER: kappa 0.906 and 0.864 (QNAMES2, exploratory: 1.014). LETTERS-AFTER failed Gate b2 (Phi 0.225) and is exploratory.
  - The swap flips 0/150 answers in the formats that name candidates, so those kappas describe a shift of the logit margin (Phi 5.8–6.5 nats), not a reproduced swap.
- **Preregistered tests, stated plainly as failures.** H8 (values carry it under NO-MENTION): not met. H9 (crossover ≥ 0.4): +0.247 [+0.211, +0.281], not met; the shift is positive with a CI excluding 0 but below the threshold, and kappa(QNAMES) is not between the other two. H10 (rival, kappa ≤ 0.25): not met; flat-high (kappa ≥ 0.75 in every evaluable format, reported beside H10) does not hold either (NO-MENTION 0.618). H7 (law): not met, gaps 0.60–0.93; the secondary test against s_ID(f, 0) gives 0.600 / 0.181 / 0.163, so there only NO-MENTION is outside 0.25.
- **Positive control.** H11 met: the identity edit at the same positions, at block 0, gives kappa_ID 0.131 / 0.773 / 0.779 against s_ID 0.017 / 0.724 / 0.702. State that at l*_ID = 0 this edit is nearly a token substitution, so H11 checks the exchange on their material rather than testing the law independently. Part 2 at block 28 compares values within 0.03 of zero under QNAMES and OPTIONS-AFTER and repeats Part 1's NO-MENTION cell (0.129 against 0.017).
- **Exploratory, labelled.** At block 28, BIND is key-dominant and the identity edit value-carried (E-a). This is one post hoc reading consistent with the H7 failure, not an explanation of it. State its caveats (E-a).
- **Depth as an alternative explanation for H9 (state it).** By block 29 no format has a key-read identity route left (s_ID(OPTIONS-AFTER) 0.702 at l0 ≤ 3, 0.041 at 24, −0.014 at 27; ID IIA 0.00 from block 20 under QNAMES and OPTIONS-AFTER). BIND flips no answers below block 24 under NO-MENTION (IIA 0.00) and flips at most 0.01 of answers at blocks 18–40 under QNAMES and OPTIONS-AFTER. No depth exists at which the binding swap and a key-read identity route coexist, which is what the Gate b3 failure says ("not evaluable as a dissociation at this depth"). The identity-versus-binding format contrast is therefore untested.

**§sec:failed.**
- Add H7–H10 (one sentence each, or a compact list) and the stage-5 failures that bear on main claims: G7 (partial), G10 sentence family, G11, G12, G13 (MODIF), G14, G16, G19–G21.
- Point to the appendix tables for the rest.

### c.5 Related work (`related.tex`; rewrite the binding paragraph and add a positioning paragraph)

- **Ok & Lee (2026).**
  - What they show: options placed before the context cannot attend to it, and accuracy drops.
  - What we add: with the behaviour preserved, the channel that carries the identity switches. Moving the same list from after to before the writing token closes the key read (E1, LIST-BEFORE control).
- **Wu & Shomali (2026).**
  - What they show: an operation's cache is read mainly as an address, not a payload. They transplant K and V together and never vary the format (verified).
  - What we add: K and V separated and the format varied. At block 28, Prakash et al.'s binding swap is key-dominant and an identity edit value-carried (E-a, exploratory; under the formats that name candidates neither flips an answer). Without a later mention, identity is payload (value). Later mentions turn identity itself into a key read (on Prakash et al.'s material, at block 0).
- **Prakash et al. (2026; ICLR).**
  - We replicate their state-token result at Qwen2.5-14B and decompose it.
  - Their state-token patch, which in their account swaps addresses (read through QK) and payloads (moved through OV), is key-dominant in each evaluable format, with a value share of 0.435 without a later mention. This is compatible with both halves of their account; it is not a clean split between them. The identity is copied through the value without a later mention.
  - Delta: their lookback is a residual-stream account; we give a KV-channel decomposition of one of its patches, and show that later mentions add a second, key-read route for identity (at block 0 on their material). The v2 appendix paragraph "Relation to lookback…" is rewritten accordingly.
- **Zhou (2026), ordinal addressing.**
  - What they show: a question points to a fact by its order of mention, via a residual-stream steering vector (no K/V split, no format variation, free-form questions).
  - What we add: one order-of-mention (binding) edit at the stored state token is read mainly through keys (Qwen2.5-14B). Identity, by contrast, is read through keys only when later mentions query it.
  - Phrase this as compatible evidence at the storage side, not as a test of their hypothesis.
- **Feucht et al. (2025), dual-route induction.**
  - What they show: token-level and concept-level induction heads.
  - What we find: the readers' core is duplicate-token heads, not induction heads (I ≈ 0.01–0.02). The read is form-bound on our English measure but concept-like for French-mixed re-mentions at the two Qwen models (G16, exploratory per model). The token-versus-concept question is open here, and our data do not settle it.
- **Ma et al. (2025), KV-cache address book.**
  - What they show: keys act as sparse routers and values as dense payloads.
  - What we find: hop 1 is a key read, and hop 2 reads keys far more than values (H5 secondary, reported, not scored); Prakash et al.'s binding swap is key-dominant; and identity is value-carried unless later mentions query it. So whether keys route identity depends on the prompt.
- **Tan et al. (2024), steering generalisation.**
  - What they show: steering effects are brittle to prompt changes.
  - What we add: the complementary case. The remap's behavioural effect is stable across formats (φ nearly constant), while its attributed component flips. The prompt changes the attribution, not only the efficacy.
- **Keep** the multiple-choice, validity and patching paragraphs, compressed by about 30 % to pay for the above.

### c.6 Discussion (`discussion.tex`)

- **Practical rule.** Add it as an observation and a hypothesis, not a finding. Formats moved the attributed channel of an identity edit strongly at block 0 (+0.648) and that of Prakash et al.'s binding swap weakly at block 28 (+0.247). Whether this reflects the kind of edit or the depth is untested (Gate b3). Suggest testing an attribution with more than one kind of edit, at matched depth.
- **Limitations, add the following:**
  - (vii) heads and hop 2 only at two 7B models in BF16, one format confirmatory;
  - (viii) Qwen2.5-7B's batch-shape BF16 offset (0.6–0.8 nats) failed a preregistered consistency gate, so the cross-format head claim is not evaluable;
  - (ix) the Prakash exchange covers one edit at one model (Qwen2.5-14B; the Llama-3-70B option H12 not run), with non-flipping margin shifts under formats that name candidates;
  - (x) l*_ID = 0 makes the identity-edit control close to the natural clamp;
  - (xi) the identity-versus-binding contrast is confounded with depth: no depth has both the binding swap and a key-read identity route (Gate b3 not met).
- **Limitations, revise the following:**
  - (vi): the revisit now covers two edits from two labs.
  - Drop "we infer, but have not tested" for nothing new; keep it for QK attribution.
- **Conclusion.** Restrict to three templated tasks. Add the two-hop read (hop 2 through keys as a secondary result) and the narrowed intervention result.

### c.7 New tables

**Claims table (`tab:claims`), rows to add or replace:**

| Section | Claim | Status |
|---|---|---|
| sec:readers | Cutting the candidate words' attention to the writing token removes the key read; the answer needs no direct attention to it | Confirmatory: G5, G6, G8b met (3/3; Qwen2.5-7B, 14B, Mistral-7B) |
| sec:readers | About 5 % of heads, ranked on held-out stories by attention change, are sufficient and necessary for the option-row key read; random sets of equal size are not | Confirmatory: H1, H2 met (Qwen2.5-7B, Mistral-7B, OPTIONS-AFTER) |
| sec:readers | Ablating the reader heads removes most of the key identity, the value identity rises, and the answer is kept | Confirmatory: H3 met (2/2). Edge-level counterpart: G7 not met (G7a 3/3, G7b 1/3; preregistered "partial takeover") |
| sec:readers | **The answer then reads the later mentions** (replaces "Hypothesis; not tested") | Confirmatory at Qwen2.5-7B and Mistral-7B under OPTIONS-AFTER (H5 met; through keys: H5 secondary, reported, not scored); SENTENCE-AFTER line not evaluable (Gate a2); cross-scale hop-2 difference attention only (G4) |
| sec:role | The re-mentions attend to the writing token through a duplicate-token pattern at every scale, including 1.5B and 3B, whose key identity is about zero | Confirmatory: G1, G2 (H_diss), G3 met; G2 replicates disclosed 0.5B pilots |
| sec:role | The reader heads' core is canonical duplicate-token heads | Confirmatory: H4 met (2/2); the full 5 % set is mixed (exploratory) |
| sec:role | The same heads read a list and a sentence | Not evaluable (H6; Qwen2.5-7B SENTENCE-AFTER Gate a2 failed); Mistral-7B 14/20 met; exploratory otherwise |
| sec:role | The read is form-bound but case-insensitive; token vs concept is model-dependent | As in the stage-5 memo (G13–G17) |
| sec:prakash | Prakash et al.'s state-token binding swap replicates at Qwen2.5-14B | Gate b1 met; per-layer comparison exploratory |
| sec:prakash | Their binding swap is key-dominant in each of the three evaluable formats (psi_V 0.435 under NO-MENTION); a later mention shifts kappa by +0.247 [+0.211, +0.281], below the preregistered 0.4 | Exploratory description; the preregistered H8, H9, H10 were not met. LETTERS-AFTER failed Gate b2; under QNAMES and OPTIONS-AFTER the swap flips 0/150 answers |
| sec:prakash | The remap-share law (kappa ≈ s_ID) predicts a new intervention's key share | Confirmatory test failed (H7). The identity-edit control H11 is met but near-tautological at l*_ID = 0, so the law for identity edits out of sample is untested |
| sec:prakash | At block 28 the binding swap is key-dominant and the identity edit value-carried | Exploratory (not preregistered; Gate b3, built for the reverse reading, not met). Both flip 0/150 answers under QNAMES and OPTIONS-AFTER. One post hoc reading, not an explanation of H7 |
| sec:general | Plain IOI carries almost no key identity; a later list's key read on IOI is not established | G18 met; G19–G21 not met (as in the stage-5 memo) |

Also update the following existing rows:
- "The later-mention words read the key at 7–14B": append the G12 qualifier.
- "The released remaps' key share tracks the natural identity key share (r = 0.98)": append "out of sample, it fails for a binding edit (H7); the identity-edit control (H11) is met but nearly tautological at l*_ID = 0, so the out-of-sample law for identity edits is untested".

**Accounts-vs-predictions table (critique item 1; main text, single column). Nine rows below; for the main text merge the first, second and fourth (hop-1) rows into one, leaving seven (space, c.9).** Columns: DT = duplicate-token attention alone; LB = lookback address/payload (Prakash et al.; Wu & Shomali); FF = fit-format (an edit's channel is set where it was fit); LM = later-mention readers (ours). Cell entries: ✓ predicted and observed; ✗ predicted otherwise (refuted); – silent; (ph) compatible, post hoc.

| Observation (evidence) | DT | LB | FF | LM |
|---|---|---|---|---|
| Re-mentions attend to the writing token, key-matched (G1, G3) | ✓ | – | – | ✓ |
| …also at 1.5B/3B, whose key identity is about zero (G2) | ✗ (attention without read) | – | – | ✓ (read decided downstream; post hoc) |
| Moving the same words before the writing token closes the key read (E1) | ✓ (causal mask) | ✗ (identity as payload predicts no key read in either position; the after position reads it) | – | ✓ |
| Readers are a sparse set whose core is duplicate-token heads (H1–H4) | ✓ | – | – | ✓ |
| Answer reads the option words, not the writing token (H5, G8b; through keys: H5 secondary, not scored) | – (hop 1 only) | ✗ for identity under lists (answer→state edge not needed) | – | ✓ |
| Without a later mention, identity is value-carried via the answer→writing-token edge (G8a, E4) | – | ✓ | – | ✓ |
| Removing the readers raises the value copy (G7a, H3d) | – | (ph) (the payload route stays available) | – | ✓ |
| A remap refit without later mentions shows the same crossover (F1–F4) | – | – | ✗ | ✓ |
| Prakash et al.'s binding swap is key-dominant in the three evaluable formats; flat-high not met (NO-MENTION kappa 0.618, psi_V 0.435); later mentions shift kappa by +0.247 (H8–H10, exploratory decomposition) | – | (ph) partly: key-dominance fits their address read via QK, the value share their payload via OV; flat-high not met | – | ✗ (the H7 law and the H9 crossover failed) |

The last row is the result the paper must show prominently: our account's preregistered extension to a binding edit failed, and the lookback account is partly compatible with it post hoc (one edit, one model).

**Prakash table (`tab:prakash`, appendix; its key numbers go into the sec:prakash text).** Rows: BIND@28, ID@0, ID@28, and s_ID(f, 29) / s_ID(f, 1). Columns: NO-MENTION, QNAMES, OPTIONS-AFTER, then LETTERS-AFTER and QNAMES2 marked exploratory. Entries: kappa (psi_K, psi_V) with the flip rate and Phi. Generate it with `make_figures.py` from `results/gpu_stage6/prakash/`.

### c.8 Figures

- **`fig_localisation` → `fig_readers`** (figure*, three panels; replaces v2's localisation figure):
  - (a) the row splice as in v2;
  - (b) R(k) and KO(k) for a3, f+, d− and the three random sets at both 7B models (OPTIONS-AFTER), with k* marked;
  - (c) the hop bars r_ans, r_other and r_all, with r_ans(K) and r_ans(V) marked as the unscored secondary contrast.
  - Data: `results/gpu_stage6/heads/*.json`.
- **`fig_frames` panel (c).** Add Prakash et al.'s cells at Qwen2.5-14B as new markers:
  - BIND@28 against s_ID(f, 29): three points, far off the diagonal, labelled "H7, preregistered, failed";
  - ID@0 against s_ID(f, 1): three points near the diagonal, labelled "H11 (block 0; near-tautological)".
  - This shows the failure in the figure that carries the law.
- **`fig_scale`** moves to the appendix. Its paragraph keeps the 14B peak, the fall and E3.
- **Schematic.** Add hop 2 (§c.2).

### c.9 Appendix additions (to stay within 8 pages)

**New preregistration tables:**
- `tab:prereg-g`: G1–G22 with every NOT MET, condensed from the outcome record.
- `tab:prereg-h`: Gates a1–a3 and b0–b3, H1–H12, with every verdict as scored.

**History of the claim, new items:**
- Item 9 (G): hop 1 conceded; the lookup-replaces-copy lead dropped.
- Item 10 (H): heads and hop 2 confirmed. The law, the crossover, H8 and H10 failed on the binding swap; the identity control was met but is near-tautological at l*_ID = 0. H was finalised (cc3a3e0, 01:16Z) before any stage-5 output was inspected; the stage-5 run had started at 21:25Z the day before, and its results were committed at 08:41Z.

**New appendix sections:**
- "Re-mention attention, knockout, membership, variants, IOI (G)", as planned in the stage-5 memo.
- "Reader heads and the second hop (H, part a)":
  - the R/KO curves for every ranking at both formats; the layer profile; split-half reliability;
  - the ablation table with every condition; the H4 sensitivity to the size of C; the hop row restrictions;
  - the SENTENCE-AFTER results labelled gate-failed or exploratory.
- "The exchange on Prakash et al.'s intervention (H, part b)":
  - population and filter; the replication table; every sweep;
  - all 13 exchange cells; the clamp onset table; kappa_w; the per-question split;
  - the FP32 re-check, which is not in the score file *(recomputed)*.

**Disclosures for App. prereg:**
- Gate a2's batched-versus-single design and the Qwen2.5-7B batch-shape offset. Its OPTIONS-AFTER pass rests on the d_full-scaled bound (0.669 against 0.719).
- The P-value conditioning.
- The dV floors.
- T_dup pooling.
- No BOS in Qwen's duplicate-score sequences (effect on D ≤ 0.033).
- Format evaluability from the NO-MENTION sweep.
- One-sided kappa versus symmetric s_ID.
- l* resting on one pair.
- l*_ID = 0 by the tie rule on a flat curve.
- The 70/80 overlap being approximate.
- The `--dtype` passed twice in the FP32 command.
- The disclosed pilots and how the outcome compares (H4 consistent; H3 (d) and H5 unlike the 0.5B pilot; l* = 28 inside the predicted 27–28, so the replication table compares against material seen before finalisation).
- The depth confound of the identity-versus-binding contrast (Gate b3).
- H12 not run: by the rule agreed before the run, the 70B model was to be run only if Qwen2.5-14B failed Gate b1, which it passed (IIA 0.993).

**Implementation:**
- HeadSplice and HopSplice hooks with their exactness tests (Gate a1).
- The exchange rows r0–r6 and their Gate b0.

**Space accounting.** v2 has no slack: in `paper/main.pdf`, page 8 ends with the Conclusion at the foot of the right column. The figures below are estimates; check them by compiling.

| Change | Main-text cost |
|---|---|
| Abstract (249 words against v2's 260) | about 0 |
| Intro: rewritten paragraphs, contributions, hop 2 in the schematic | about +0.1 page |
| Setup: heads and exchange paragraphs (about 0.3 column) | about +0.15 |
| Rewritten sec:readers, with the merged figure* | about +0.5 |
| Rewritten sec:role | about +0.1 |
| New sec:prakash text (about 0.6 column; `tab:prakash` in the appendix) | about +0.3 |
| Accounts table (single column) | about +0.3 |
| sec:failed: H7–H10 and the stage-5 failures, one line each | about +0.15 |
| Discussion: practical rule and limitations (vii)–(xi) | about +0.1 |
| **Added** | **about +1.7** |
| fig_scale and the Depth paragraph to the appendix | about −0.5 |
| Shorter sec:general | about −0.2 |
| Refit post hoc detail to the appendix | about −0.15 |
| Related work compressed by about 30 %, net of the new positioning sentences | about −0.1 |
| Discussion limitations merged into one list | about −0.1 |
| Role-control paragraph (133 words) to one sentence | about −0.1 |
| sec:intervention: reproduction detail and the post hoc paragraph to the appendix | about −0.2 |
| sec:readers rows: D3 detail to the appendix, one sentence kept | about −0.1 |
| sec:mentions tightened (E-family detail to the appendix) | about −0.1 |
| Accounts table cut to seven rows (the three hop-1 rows merged into one) | about −0.1 |
| H4 sensitivity and layer-profile sentences to the appendix | about −0.05 |
| **Cut** | **about −1.7** |
| **Net** | **about 0** |

If the compiled draft is still over, move the accounts table to the appendix with a two-sentence summary in the intro (about −0.25).

---

## (d) Title

v2: "Looked Up or Copied? Later Mentions Decide Which Attention Channel Carries an In-Context Value, and Where an Intervention Appears to Act".

**Recommendation: change it.** Keep the first clause and drop the general intervention clause:

> **Looked Up or Copied? Later Mentions Decide Which Attention Channel Carries an In-Context Value**

This is already the running title.

- **Why the second clause cannot stay:** the preregistered crossover on another lab's edit failed (H9 not met: later mentions shift the binding swap's kappa by +0.247 [+0.211, +0.281], against a predicted ≥ 0.4), and the law failed for it (H7). The shift is real but weak, and the strong format effect is shown only for identity edits (on Prakash et al.'s material, at block 0). A title asserting in general that later mentions decide where an intervention appears to act would overstate our own preregistered result.
- **Why the first clause can stay:** it is confirmatory (A, B, D, E) and is now backed by the mechanism (G, H1–H5).
- **Alternative, if the intervention result should stay visible in the title:**
  > Looked Up or Copied? Later Mentions Decide Which Attention Channel Carries an In-Context Value, and Where an Identity Edit Appears to Act

  "Identity edit" is defined in the abstract. This is accurate for the remap, the refit and the identity edit at block 0 (C3, E4, F, H11), but it is jargon in a title, and H11 rests on l*_ID = 0. Prefer the shorter title.

---

## (e) Risks a reviewer will raise, and how the text pre-empts them

1. **"This is duplicate-token attention; the heads are textbook."**
   - Concede it in sec:role, citing H4 and G1–G3.
   - Make the contribution explicit: the match is present at 1.5B and 3B, whose key identity is about zero (G2), so it does not explain the scale dependence; the format dependence is what it predicts together with the causal mask. Hop 1 is a key read, and hop 2 reads mainly keys (H5 secondary, not scored). The attributed channel of identity edits depends on the format.
   - Do not present head identity as novel.
2. **"Your law failed on the first out-of-sample intervention."**
   - Say so in the abstract and in sec:failed, with the gaps (0.60–0.93).
   - State the law as an in-sample relation for the released remaps and the refits (F4). Out of sample it failed for the binding swap and is untested for identity edits (H11 is nearly tautological).
   - Show the failed points in fig_frames(c), and give H7's secondary gaps (0.600 / 0.181 / 0.163 against s_ID(f, 0)) beside the failure.
   - Do not frame it as a revealed scope condition (identity versus address). The contrast is confounded with depth: at block 28 the identity edit is value-carried in every evaluable format, and no depth has both the binding swap and a key-read identity route (Gate b3 not met). E-a is one post hoc reading.
3. **"The positive control H11 is circular."**
   - State that at l*_ID = 0 the ID exchange is nearly the natural clamp: their one-sided kappas agree within 0.001–0.044 *(recomputed)*.
   - Under NO-MENTION the H11 gap is almost entirely the difference between the statistics (0.115 of 0.114). Under QNAMES and OPTIONS-AFTER it splits about 0.029 / 0.020 and 0.044 / 0.033 between the exchange-versus-clamp difference and the difference between the statistics *(recomputed)*.
   - Part 2 compares values near zero under QNAMES and OPTIONS-AFTER and repeats the NO-MENTION cell.
   - Do not count H11 as independent support for the law.
4. **"The binding swap does not even flip the answer with options; your kappas are noise."**
   - Report the flip rates (0/150 under QNAMES and OPTIONS-AFTER) and the Phi (5.8–6.5 nats against base margins of −20.06 and −27.69 *(recomputed)*) beside each kappa.
   - Note that the entry's evaluability rule took reproduction from the NO-MENTION sweep, as preregistered.
   - The NO-MENTION cell (IIA 0.993; psi_K 0.703 > psi_V 0.435) carries H8 by itself, and it is a full reproduction.
5. **"kappa and s_ID are different statistics, so the law was set up to fail."**
   - The one-sided clamp analogue at l0 = 29 is 0.131 / 0.012 / 0.005 *(recomputed)*, which still leaves gaps of 0.49 or more.
   - H7 was preregistered with both statistics defined as they are. Disclose the difference in the appendix.
6. **"BF16 noise: a gate failed."**
   - Show that every scored quantity is an in-batch difference, that the splice and hop are exact (Gate a1; hop 0.0000), and that the failure is a batch-shape offset at Qwen2.5-7B only (Mistral-7B 0.07–0.08).
   - Keep H6 and the sentence hop out of the confirmed claims.
   - Disclose Qwen2.5-7B's narrow OPTIONS-AFTER pass (0.669 against 0.719; CI crosses).
   - Part (b) FP32 re-check: IIA unchanged within one pair.
7. **"Mean-ablation is off-distribution / head selection is circular."**
   - Heads are ranked on R stories and evaluated on disjoint E stories; split-half Spearman is 0.97 / 0.96.
   - Zero-ablation (exploratory) gives the same direction (rho_K 0.081 / 0.057).
   - The random and high-activity control sets leave rho_K at 0.95–1.02.
   - T_dup is selection-adjacent, so cite D and I (random sequences) as the evidence.
8. **"H4 is fragile."**
   - Report the C-size sensitivity (Mistral-7B median D 0.232 at 16 heads, 0.156 at 20 *(recomputed)*).
   - Say "core" or "the 16–20 heads that carry 80 %", never "the readers are duplicate-token heads".
9. **"Only two 7B models; the heads at 24B/72B are inferred."**
   - Say it in the limitations and in sec:intervention ("that the readers are the same at 24B and 72B is an inference").
   - H12 (70B) was optional and not run, by the rule agreed before the run (run it only if Qwen2.5-14B failed Gate b1; it passed, IIA 0.993). It can still be scored later against the fixed 14B verdicts; with none of H7, H9 and H10 met, it would test Gate b1, H8 and H11 Part 1 only.
10. **"Too many tests; cherry-picking."**
    - Eight preregistrations, each committed to version control before its run, with the scoring script committed before its outputs were inspected (v2's wording, extended to eight). Do not write "scorer committed before the run": B's scorer was committed with the raw results, before inspection, and the A outcome names no scorer.
    - Every NOT MET appears in the appendix tables and the main claims are tagged.
    - The anonymised commit history lets the order be checked.
11. **"Your earlier reading of lookbacks ('a copy read') was wrong."**
    - Correct it explicitly in related work and the appendix.
    - Their state-token patch, which they describe as swapping addresses and payloads, is key-dominant with a value share of 0.435 without a later mention (C6). Identity without a later mention is value-carried (E4, H11 NO-MENTION).
12. **"Prakash et al.'s main model is Llama-3-70B; you tested 14B."**
    - Qwen2.5-14B is the model of their full-vector state-token patch, and their release gives its per-layer IIA, which we reproduce.
    - State that 70B is untested (H12 not run).
13. **"The value takeover claim contradicts stage 5."**
    - Give both: head ablation keeps the answer (H3), edge knockout does not at Qwen2.5-7B (G7b: acc_B 0.37 / 0.30).
    - The interventions differ: rho_K 0.238 against r_K(M1) ≈ 0. Do not reconcile post hoc.
14. **"Page limit: everything is in the appendix."**
    - The main text keeps one figure per mechanism claim (fig_readers, fig_frames with the H7/H11 points) and one compact table (accounts). `tab:prakash` is in the appendix, with its key numbers in the text.
    - Every number in the main text comes from `numbers.tex`, regenerated by `paper/make_figures.py` from the raw results.
