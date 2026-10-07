# Stage 5 (preregistration G): implications for the paper-v3 rewrite

The verdicts are those of `results/gpu_stage5/STAGE5_SCORE.txt` at 51e105e (12 MET, 10 NOT MET, 0 NOT EVALUABLE of 22). Five independent recomputations reproduced them, and nothing in this memo changes a verdict. Numbers come from the score file unless marked. Numbers marked *(recomputed)* were recomputed from the raw files, are not printed by the scorer, and must be labelled exploratory or post hoc in the paper. Numbers marked *(arithmetic)* are differences or ratios of printed score-file values. The predecessor is cited only as \citet{anonymous2026fitted} or "our predecessor".

Model order in triples is Qwen2.5-7B / Qwen2.5-14B / Mistral-7B (Instruct), unless stated.

---

## (a) Headline per part

**(a) Re-mention attention: G1–G4 all met.**
- The later mention of the written word attends to the writing token with a large, key-matched excess at every scale. This includes 1.5B and 3B, where the SENTENCE-AFTER key identity lies within the preregistered ±1-nat null margin (−0.53 at 1.5B, −0.02 at 3B).
  - E is 0.79 / 0.88 at 1.5B / 3B, against 0.87 / 0.92 at 7B / 14B.
  - Under the key clamp the attention moves one for one: F/E 0.98–1.03.
  - So the hop-1 duplicate-token match is present but not sufficient for the key read (G2, H_diss).
- The answer position's attention to that mention (hop 2) is weaker at the small models: G 0.086 / 0.113 at 1.5B / 3B against 0.255 / 0.329 at 7B / 14B (Q 0.34, 0.44; G4 met). This is attention evidence only, not an intervention. At 3B the hop-2 attention reaches the presence bar that G4a sets for the anchors (G 0.113 [0.095, 0.132]) without a key read, so hop-2 attention alone does not account for the scale difference.

**(b) Knockout: G5, G6 and G8 met; G7 not met.**
- Forbidding the six re-mentioned candidate words to attend to the writing token removes the key identity completely: r_K(M1) −0.105 to +0.003 in every format. A matched control column does nothing: r_K(M2) 0.99–1.09.
- The answer position does not need the writing token for the key read (r_K(M3) 1.01–1.04), so the key read is indirect. The entry's gloss of G8b is "two-hop"; how many hops follow the candidate rows is left to stage 6 H5.
- With the lookup cut, the value copy returns to its NO-MENTION level: q_V 0.39 / 0.44 / 0.45 against 0.39 / 0.34 / 0.43 (G7a 3/3).
- The answer is preserved only at Qwen2.5-14B: acc_B 1.00 / 0.99, against 0.37 / 0.30 at Qwen2.5-7B and 0.86 / 0.73 at Mistral-7B (G7b 1/3). The preregistered outcome is "partial takeover", not H_redundant and not H_replaced.

**(c) Membership and dose: G9 met; G10, G11 and G12 not met.**
- The key read needs the swapped candidates themselves among the later mentions. Naming B with two other candidates instead of S and X gives none; ID_K falls 0.13–0.35 nats below NO-MENTION (G9, 3/3 in both families, replicated at seed 1).
- In lists, two named candidates give the full read: r_2 0.96–0.98 (G10 list family met).
- In sentences, the read grows with the number of named candidates (r_2 0.28 / 0.58 / 0.46). G10's sentence family fails at Qwen2.5-7B, where r_2 = 0.28 is at or below proportional. This dose effect is in the direction of the preregistered alternative (graded list-likeness), which part (c) does not support overall (OUT lies below NO-MENTION, and lists are complete at k = 2).
- Removing S and X from a list does not bring the copy back: R_V 0.11 / 0.03 / −0.08 (G11, 0/3).
- In a two-word sentence (S2), the named words carry only 0.38–0.40 of the splice effect (G12, 9/12). G12 splits the total key-swap effect d (the factorial d_K), not ID_K. In S2 only 0.14 / 0.29 / 0.38 of d_K is identity, against 0.44 / 0.56 / 0.61 in S3 *(arithmetic, part (c) arm tables)*. So the S2 failure concerns mainly the non-specific key effect.

**(d) Non-identical re-mentions: G15 and G17 met; G13, G14 and G16 not met.**
- On the paper's English-scored measure, synonyms and mixed-language translations carry little key identity: r_K −0.13 to 0.15, at most 0.15 of the exact repeat's (G15 cut-point 1/3; 4/4 for each variant). The FRMIX values are small but their CIs exclude 0 in all four models.
- Case variants read nearly as fully as an exact repeat (r_K 0.83–1.14; the CI lies below 1 in three of the eight cells), so G14 fails for TITLE and UPPER. A multi-token wrapper reads only partly (MODIF r_K 0.32–0.64; G13 MODIF fails).
- The token/concept disambiguation (G16) matches none of the three preregistered outcomes: the profile is graded and model-dependent. The per-model profile below is exploratory; no preregistered outcome holds.
  - At the two Qwen models, the French-mixed re-mention (FRMIX) shows the concept pattern on both measures (r_K^any 0.95 / 1.06, a_v 0.86 / 0.90). Its attention leg passes only with span-summed attention (last token only, a_v 0.55 / 0.60 *(recomputed)*; mean span 2.5 tokens).
  - The German-mixed re-mention (DEMIX) is concept-like on r^any but graded on a_v there (0.66 / 0.65, n = 23). The fully French sentence does not reach the concept bar on r_K^any (0.55 / 0.47).
  - At Mistral-7B and OLMo-2-7B, the read is graded or token-level.

**(e) IOI: G18 and G22 met; G19, G20 and G21 not met.**
- Plain IOI has almost no key read in any of the three models (f_K −0.02 to +0.02, within the preregistered ±0.10; G18).
- At Qwen2.5-7B-Instruct alone, the G19 criteria hold: a later list gives a large positive (selection) key read, ID_K +7.05 [+6.26, +7.81].
- G19 is still not met. Mistral-7B's AFTER cell failed the competence gate (two-way 0.59), so the 2/2 criterion cannot be reached.
- G20 fails at Qwen2.5-7B-Instruct only on its baseline clause. f_V falls from 0.69 (QUESTION) to 0.45 (AFTER), but the predicted QUESTION level was ≥ 0.75.
- G21a fails because the list placed before the sentence gives a negative key read (−2.02).
- GPT-2 small's in-sentence re-mention gives an inhibitory key read, as in the disclosed pilot. This is a replication, not confirmatory evidence.

---

## (b) Critique items 2a–2d and the IOI half of 4

| Item | Status | How |
|---|---|---|
| 2a Do re-mentions attend where the key effect is zero? | **Answered (yes)** | G2 H_diss met at 1.5B and 3B, comparable in size to 7B/14B (G3: E(m)/min anchor E 0.91 at 1.5B, 1.01 at 3B), not embedding-level. The duplicate-token match exists without the read, so the duplicate-token account explains hop 1 but not whether the identity is used. Hop-2 attention is weaker at 1.5B/3B (Q 0.34 / 0.44, G4 met), but at 3B it reaches the anchors' presence bar without a key read, so it does not alone account for the scale difference. The cross-scale difference at hop 2 is attention evidence only, and no planned intervention tests it: stage-6 H5 tests only whether the answer reads the re-mentions at 7B. |
| 2b Block the lookup; does the copy take over? | **Answered: partly yes** | The lookup edges are necessary and specific (G5, G6), and the key read is indirect: the answer position does not need the writing token (G8b). The copy returns on the identity measure (G7a 3/3) and runs through the answer position (G8a, G8c). It keeps the answer only at Qwen2.5-14B (G7b 1/3): a preregistered "partial takeover". At Qwen2.5-7B the knocked-out model answers the stale initial location in 0.63 / 0.69 of stories *(recomputed)*. |
| 2c Dose-response from partial re-mention | **Answered, with a split result** | The read requires the swapped candidates themselves (G9). In sentences each named swapped candidate contributes about half (S3half/S3 0.61 / 0.57 / 0.51, exploratory). Lists show no dose effect (G10 list family). Sentences show a graded dose effect (G10 sentence family fails at one model). Adding non-matching candidates raises ID_K: S4 − S3 +1.11 (one candidate) and S6 − S4 +0.93 (two) at Qwen2.5-7B, CIs excluding 0. A pure duplicate-token account does not predict this. The effect is in the direction of the preregistered alternative (graded list-likeness), although that alternative fails overall. G11 and G12 failed (see (c) below). |
| 2d Non-identical re-mentions: token vs concept | **Answered, not in one direction** | On the paper's measure the read is form-bound (G15 met): synonyms and translations give little (r_K ≤ 0.15). It is largely case-insensitive (G14 TITLE and UPPER fail) and number-sensitive (PLURAL 0.11–0.49). With the answer scored in the variant's own form, the French-mixed re-mention shows the concept pattern at the two Qwen models (exploratory per model; the attention leg needs span-summed attention), the German-mixed one is graded on attention there, and Mistral-7B and OLMo-2-7B are graded or token-level (G16 not met; none of the three named outcomes). |
| 4 (IOI half): a list on a textbook value-copy task | **Partly** | No key read in plain IOI (G18 met, confirmatory at the 7B pair). The preregistered G19 was not met (1/2; Mistral-7B failed Gate e). At Qwen2.5-7B-Instruct alone the G19 criteria hold (+7.05 [+6.26, +7.81]), and at 14B-Instruct (exploratory, +10.24). The read was not established at Mistral-7B (competence gate failed). It is absent at the Qwen2.5-7B base model (−0.03, exploratory), where the total K+V effect is only 1.7 nats. The position control failed (BEFORE −2.02). "The lookup replaces the copy" on IOI is not met (G20). |

---

## (c) Concrete changes to the paper

### Claims that become stronger (move from "hypothesis" or "conceded" to tested)

1. **The key read is indirect: it needs the candidate words' attention to the writing token (G5), not the answer's (G8b).**
   - The candidate words' attention to the writing token is necessary (G5): r_K(M1) between −0.105 and +0.003 in 9/9 format × model cells.
   - The answer position needs no direct attention to the writing token (G8b: r_K(M3) 1.04 / 1.03 / 1.01, acc_B = on_B = 1.00).
   - This replaces the last sentence of the localisation paragraph in `results_read.tex` ("…that it then reads the later mentions we did not test"). Leave the number of hops to stage 6 H5; the preregistered alternative there is a three-hop route through the instruction/template rows. Keep "the answer reads the mentions" as tested only by attention (G4) until H5.
2. **The key read requires the swapped candidates themselves (membership).** G9 is met in 3/3 models in both families and replicated at seed 1. This narrows "a later mention of the candidates" to "a later mention of the candidates at stake".
3. **Without a later mention, the copy is mainly the answer position reading the writing token.** Cutting that one edge leaves 0.13 / 0.37 / 0.42 of ID_V (G8a). Add the qualifier: the cut removes most of the writing token's whole effect and the model's ability to answer (acc_B 0.00 / 0.20 / 0.15; span d_KV 42.53 → 3.93, 57.07 → 15.40, 31.53 → 12.80), and the copy's share of what remains does not fall (q_V 0.391 → 0.566, 0.335 → 0.459, 0.433 → 0.445). The causal statement is that this edge carries most of the writing token's K/V effect on the answer, the copy included. This turns the descriptive "the identity is copied instead" (`results_read.tex`, "No later mention") into a causal statement in that sense.
4. **The duplicate-token match at hop 1 is not sufficient** (G1–G3). This answers objection 1 of the critique directly. The subsection `\subsection{Identity, not belief role}` (results_read.tex), currently "…and we did not identify the heads", becomes a short measured paragraph:
   - E, F/E at 7B/14B: 0.87 [0.85, 0.89], 1.00 and 0.92 [0.91, 0.92], 0.98 (G1).
   - At 1.5B/3B, with key identity within the preregistered ±1-nat null margin (−0.53 at 1.5B, −0.02 at 3B): 0.79, 1.03 and 0.88, 0.99 (G2); R_A 1.05 / 1.06 (G3).
   - Hop 2: G 0.086 / 0.113 against 0.255 / 0.329 (G4; attention only, and at 3B G reaches the anchor bar of G4a without a read).

   Concede explicitly that hop 1 is duplicate-token attention; the contribution is that it is present whether or not the identity is used. Keep "we did not identify the heads causally" until stage 6 (H1–H4).

### Claims that must be narrowed or dropped

5. **Drop the planned v3 lead "the lookup largely replaces the copy (even when the answer is the word itself)" as a causal claim.** Three preregistered tests bear on it, and none supports it as stated:
   - G7 found only a partial takeover.
   - G11 failed 0/3: removing the alternatives from the list does not restore the copy, so the alternatives' lookup is not what suppresses it.
   - G20 failed on IOI.

   Keep the descriptive fall (ID_V 19.1 → 5.5 nats at Qwen2.5-14B under OPTIONS-AFTER; stage-5 knockout M0, NO-MENTION +19.11, OPTIONS-AFTER +5.48; the v2 stage-1 table gives 19.1 → 5.4). Replace the causal reading with what was tested: the re-mention reads suppress a copy route that stays available, and cutting them restores the copy (G7a) but not reliably the answer (G7b).

   Optionally add, clearly labelled **post hoc**: in lists that omit the written word B, the key read is at full strength while ID_V stays near or above its NO-MENTION level (above at 14B, equal at Mistral-7B, 1.7 nats below at Qwen2.5-7B).
   - ID_V(L4out) 14.90 / 30.76 / 13.68 against 16.60 / 19.12 / 13.65 (score-file arm table).
   - Naming B lowers it by 10.2 / 26.6 / 10.1 nats *(recomputed: paired L4out − L4)*.

   Together with G7a (the knockout cuts B's row as well), this suggests the copy is suppressed when the written word itself is re-mentioned. It is a hypothesis for a future preregistration (mask B's row alone), not a v3 claim.
6. **"The re-mentioned words are the (main) readers"** in the abstract, intro and discussion line 13 must be qualified. G12 is not met (9/12): in a sentence naming only the two swapped candidates (S2), the words carry 0.38–0.40 of the splice effect and the rows after the question 0.38–0.49. G12 measures which rows read the total key-swap effect d, not the key identity. In S2, 62–86 % of that effect is not identity-specific (ID_K/d_K 0.14 / 0.29 / 0.38 *(arithmetic)*), so the S2 failure concerns mainly the non-specific effect. Narrow the claim for that reason only. Suggested wording: "…are the main readers of the key-swap effect in lists and in sentences naming at least three candidates".
7. **"Identity lookup in the style of duplicate-token attention … we claim nothing more"** (`results_read.tex` sec:role; also the limitations in `appendix.tex`) needs two qualifications from part (d):
   - The read is largely case-insensitive (G14 TITLE and UPPER, r_K 0.83–1.14), so it is not exact-token matching.
   - Synonyms and, on the English measure, translations carry little key identity (r_K ≤ 0.15; ≤ 1/3 as preregistered, G15). At the two Qwen models the French-mixed re-mention does read the key when the answer is scored in its own form (r_K^form 0.99 / 1.28; G16 per-model profile, exploratory); the German-mixed one is graded on attention there.

   Add from part (c): in sentences each added non-matching candidate raises ID_K (S4 − S3 +1.11 for one, S6 − S4 +0.93 for two, at Qwen2.5-7B), which a pure duplicate-token account does not predict. The larger step S3 − S2 (+1.96) adds B, the written word itself, and does not bear on this point. "Lexical identity of a single word" should become "the word form largely up to case, as read by the models' re-mention; synonym- and number-sensitive".
8. **Generality ("across four model families and three tasks"; intro "Later mentions decide"; the conclusion).** IOI does **not** support a general claim that a later list opens a key read on value-copy tasks.
   - The preregistered G19 was not met (1/2; Mistral-7B failed Gate e). At Qwen2.5-7B-Instruct alone the G19 criteria hold (+7.05 [+6.26, +7.81]).
   - It is untested at Mistral-7B (gate failed; the gate-failed cell is negative, −1.98). It is absent at the Qwen2.5-7B base model, where the total K+V effect is only 1.7 nats.
   - The BEFORE control is negative, not null (G21a).

   Restrict the generality sentence to the three templated belief/paint/schedule tasks. Add one sentence: "On IOI, the preregistered test that a later list opens a key read (G19) was not met, because Mistral-7B failed the competence gate. At Qwen2.5-7B- and 14B-Instruct a positive (selection) key read appears (per-model observations); none is seen at the base model, where the list barely affects the answer, and the lookup-replaces-copy prediction failed (G20)." Do not cite IOI as evidence of generality across families.
9. **3B "no key read under SENTENCE-AFTER".** Phrase it as "about zero on all stories". On stories both clean runs answer correctly it is +0.71 [+0.29, +1.14] *(recomputed, not preregistered)*, against +4.84 at 7B on the same restriction *(recomputed)*.
10. **Quote ID_K from the sdpa runs, not from the eager attention run.** At Qwen2.5-7B the eager run is about 1.0 nat lower under SENTENCE-AFTER (4.50 against 5.52, part (c) factorial) and 1.8 nats lower under OPTIONS-AFTER (19.39 against 21.17, part (b) knockout M0; stage 1 gives 21.16). Disclose this in the part (a) appendix.

### New claims-table rows (one per part, headline verdict)

| Section | Claim | Status |
|---|---|---|
| sec:role (new subsection "What the duplicate-token account explains") | Later mentions of the written word attend to the writing token through a key-matched duplicate-token pattern at every scale, including 1.5B/3B, where the key identity is within the ±1-nat null margin; the match is not sufficient for the read; the answer's attention to the mention is weaker at 1.5B/3B, but at 3B it reaches the anchor-level bar without a read, so it does not alone account for the scale difference | Confirmatory: G1, G2 (H_diss), G3 met (Qwen2.5-1.5B–14B, n = 150); G4 met (attention, not causal; no planned intervention tests the cross-scale difference); G2 replicates disclosed 0.5B pilots |
| sec:readers | Cutting the re-mentioned words' attention to the writing token removes the key read; the copy returns on the identity measure but preserves the answer only at Qwen2.5-14B | Confirmatory: G5, G6, G8 met (3/3); G7 not met (G7a 3/3, G7b 1/3): preregistered "partial takeover" |
| sec:readers | The key read requires the swapped candidates among the later mentions; in lists two suffice, in sentences the read grows with the number named; the named words carry the key-swap effect except in a two-word sentence, where most of that effect is not identity-specific | Confirmatory: G9 met (3/3, both families, replicated at seed 1); G10 list met, sentence not met (2/3); G12 not met (9/12; S2 fails); G11 not met (0/3) |
| sec:role | The read is form-bound on the paper's measure (synonym and translation reads at most 0.15 of an exact repeat) but largely case-insensitive; token vs concept is model-dependent | Confirmatory: G15 met (4/4); G13 THE met, MODIF not met; G14 not met (TITLE, UPPER); G16 not met, no preregistered outcome holds (per-model profile, exploratory: FRMIX concept pattern at the two Qwen models, DEMIX concept-like on r^any but graded on a_v there); G17 met |
| sec:general | On IOI, plain sentences carry almost no key identity (f_K within ±0.02; G18) | G18 met (3/3). G19, G20, G21 not met (Mistral-7B not evaluable, gate; G20 fails on its baseline clause; BEFORE negative). Per-model observation under the NOT MET G19: at Qwen2.5-7B-Instruct a later list gives a positive key read (+7.05). GPT-2 inline inhibition replicates a disclosed pilot (G22) |

Also update the existing row "The answer then reads the later mentions | Hypothesis; not tested". New status: "Attention evidence (G4a/b met); the answer position needs no direct attention to the writing token (G8b); causal test at 7B: stage 6 H5." Update the row "The later-mention words read the key at 7–14B | Confirmatory (D3)" by appending "except in a two-word sentence (G12 not met; G12 measures the total key-swap effect, most of which is not identity-specific there)".

### Main text versus appendix (8-page limit)

**Main text (net about +0.6 page; recover space by shortening sec:general and moving the onset sweep to the appendix):**
- One compact table, "What the duplicate-token account predicts and what we find". It has one row per part (a)–(e) with the prediction, the observed headline number and the verdict. This also serves critique item 1's request for an accounts table.
- Rewrite sec:role around G1–G4 (about 6 lines).
- Two sentences in sec:readers for G5 and G8b, and one for G7 (partial takeover).
- One sentence in sec:general restricting generality (IOI).
- Edits to the abstract, the intro "Later mentions decide" paragraph and the contributions, and the conclusion. Remove "otherwise mainly copied" as an exclusive alternative, and qualify "the main readers".

**Appendix:**
- The full G1–G22 outcome table with every NOT MET.
- Part (a): per-model E, F/E, G, Q and the casing/d_K table.
- Part (b): mask definitions, a table of r_K / q_V / acc by mask, and the exploratory M3/M5/M6/M7 lines.
- Part (c): sentence and list ladders (a figure), G11, and the splice table, with ID_K/d_K beside each splice cell.
- Part (d): the variant table with r_K, r_K^any, rho_s and a_v (span-summed and last-token), plus the G16 per-model profile verbatim.
- Part (e): the IOI table by model and arm, Gate e failures, and the GPT-2 probe (CPU re-run).
- Drop or shorten the old n = 25 exploratory attention probe at 1.5B; part (a) supersedes it.

**Disclosures to carry into App. prereg:**
- the GPT-2 probe re-run on CPU;
- the scorer's combined headline lines;
- the scorer's "near-token-level" G16 label, which is not adopted;
- the scorer interpretations in parts (a) and (b);
- the fact that the 7B-pair IOI verdicts rest on one evaluable model;
- the status of each part's preregistered alternative.

---

## (d) How to frame stage 6 (preregistration H, not yet run)

H was finalised (cc3a3e0) before any stage-5 output was inspected; say so in App. prereg. Write the v3 stage-5 text so that it stands whatever stage 6 shows, and leave one sentence per stage-6 part to fill in. Report all 12 H verdicts in the appendix, with one claims-table row per part (H4 + H5 and H7/H10 + H11).

**Part (a), heads and the second hop (Qwen2.5-7B, Mistral-7B, OPTIONS-AFTER):**
- **H1/H2 (sparsity, necessity) met.** "The key read is carried by ≤ 5 % of heads." This adds to G5, which locates the read in rows; H1/H2 locate it in heads.
- **H1/H2 (sparsity, necessity) not met.** Report the read as distributed. This does not weaken part (a): G1–G3 concern the attention pattern, not head count.
- **H4 (canonical duplicate-token heads) met.** Concede fully that the readers are generic duplicate-token heads. The paper's contribution is then explicitly:
  - (i) duplicate-token attention at the re-mentions is present whether or not the identity is used (G2), so the format and scale dependence lies downstream of hop-1 attention (where exactly is not established: G4 is attention only, and H5 runs only at the 7B models);
  - (ii) channel attribution flips with format.

  Do not present the head identity as novel.
- **H4 not met (task-tuned heads).** Report it, but do not let it overturn part (a): stage 5 already shows a duplicate-token attention pattern at the re-mentions in every model. Phrase it as "the causally selected heads are not the generic duplicate heads", not as "not duplicate-token attention".
- **H5 (second hop at the answer) met.** Replace "Attention evidence" in the hop-2 row with "Confirmatory at 7B (H5)". Together with G4 and G8b, this closes critique item 3's second hop at 7B. It does not test the cross-scale difference at hop 2.
- **H5 not met, with r_all ≥ 0.8.** State the three-hop route through the instruction/template rows, and drop "the answer reads the re-mentions" from the main text. Do not cite G12's S2 cells as support: their rows after the question (0.38–0.49 of d) carry mainly the non-specific key effect, not the identity.
- **SENTENCE-AFTER line of H5 (reported, not scored).** Read it beside the exploratory part-(a) observation that at 7B and 14B the hop-2 heads' answer row attends mainly to the writing token itself (0.534, 0.468).
- **H3d (value takeover at head level).** Stage 5 found the takeover at the edge level (G7a), but answer preservation is model-dependent (G7b). If H3d holds, the edge and head levels agree. If it fails (as in the 0.5B pilot), state that the copy returns when the edges are cut but not when the reader heads are ablated. Do not average the two into one "takeover" claim. Expect the base-argmax clause to be at risk at Qwen2.5-7B, given its stage-5 behaviour under M1.
- **H6 (same readers for list and sentence).** Stage 5 is exploratory here: the top-3 H* are identical at Qwen2.5-7B and overlap little at 14B. H6 is the confirmatory test at Qwen2.5-7B and Mistral-7B; report it whichever way it comes out.

**Part (b), the exchange on Prakash et al.'s intervention (Qwen2.5-14B):**
- **H7 and H9 met (and H11).** This becomes the paper's strongest generality claim: the format decides the attributed channel on another lab's intervention. Place it in the main text. Scope it to Qwen2.5-14B (and Llama-3-70B only if H12 is run and met). After the IOI result, the family-level generality of the read itself is not established, so do not extrapolate the law across families.
- **H10 met with Gate b3 passed** ("keys carry identity, values carry binding"). This is a genuine rival result. State it in the main text as limiting the crossover claim to identity edits, such as our predecessor's remap and the ID control. Do not reconcile it post hoc.
- **H11 not met.** The in-sample law does not transfer to their stories and wrapper even for an identity edit. Restrict the r = 0.98 relation to our stimuli and say so in the limitations.
- **Not evaluable** (gates b1/b2 or the kappa rule). Report it as not evaluable, without substituting exploratory cells.
- **Item 1 reframe.** Stage 5 already removed "the lookup replaces the copy" as a causal lead. The defensible lead for v3 is the attribution half (behaviour constant, attributed channel flips), which stage 6 part (b) tests out of sample. Write the abstract so that this lead does not depend on H7/H9. If they fail, the abstract keeps the in-sample result (C, E4, F) and states the failed out-of-sample test.
