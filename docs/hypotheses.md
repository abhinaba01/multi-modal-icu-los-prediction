# Pre-registered hypotheses

**STATUS: REGISTERED on 2026-11-21, approved by the user.** The text below is the registration. Before the approval, no
result that bears on H1 to H5 was read (the image arms of Stage 3 were already training when the draft was written; their
M-versus-B3 numbers were not read until after registration). Change a hypothesis only with a dated note explaining why.

**A note on what counts as the test.** The registered tests use 5-seed ensembles. Any result looked at before the 5-seed runs exist
(for example the 1-seed first pass of Stage 3) is a preliminary look and is not the registered test.

## Shared rules

- Every comparison is on **identical test patients**, in the contract's patient-level test split (11,110 stays).
- **P1** = test stays with an eligible chest X-ray before t0 (2,360 stays). **P2** = all test stays.
- Neural arms are run with **5 seeds**. A method's score for a stay is the **mean predicted probability over its 5 seeds**
  (an ensemble of equal size for every arm). The per-seed results are reported as mean ± SD but are not the tested quantity.
- Tests are **paired patient-level bootstrap** (1,000 resamples, seed 666, `fate_icu.eval.stats`), two-sided, with a percentile
  95 % CI. Nothing is tuned on the test split.
- **Holm-Bonferroni** across the registered hypotheses (m = 4, or 5 if DPO is run), family-wise α = 0.05.
- A hypothesis is *supported* only if its adjusted p ≤ 0.05 **and** the difference has the stated sign. Otherwise it is
  reported as not supported, with the CI. Null and negative results are reported (CLAUDE.md A4).
- Explanation metrics (H2, H3, H5) are computed on a **fixed random sample of 1,000 test stays** (seed 666, drawn once, before any
  explanation is generated). Explanations are generated greedily with fixed seeds.
- The thresholds of the NLI checker (H3, H5) are fixed here: a claim is *supported* if the checker's most probable label for
  (passage, claim) is entailment. The checker is not the one used to train the explainer.

## H1: Fusion vs EHR-only

- **Statement:** fusion improves AUPRC over the EHR-only model on P1 (a null result will be reported).
- **Primary metric:** AUPRC of the long-stay probability (long stay: more than the frozen 4.1 days).
- **Population:** P1 (paired test stays).
- **Comparison:** M (attention fusion over static, time-series and image tokens) minus B3 (static + time series, concatenation + MLP).
- **Test:** paired patient-level bootstrap of AUPRC(M) − AUPRC(B3) on the seed-ensembled probabilities.
- **α / correction:** Holm-Bonferroni across the registered hypotheses, α = 0.05.
- **Decision rule:** supported if adjusted p ≤ 0.05 and AUPRC(M) − AUPRC(B3) > 0.
- **Secondary (reported, not corrected):** the same on AUROC; M vs B3 on P2; M vs B4 (image only) and B6.
- **Registered on:** 2026-11-21 (approved by the user)
- **Status / result:**

## H2: Packet-constrained explanations vs an unconstrained summary

- **Statement:** packet-constrained explanations name fewer unattributed drivers than C1.
- **Primary metric:** unattributed-driver rate of an explanation = 1 − driver precision, where driver precision is the share of
  the clinical entities named in the text (found by the deterministic lexicon of `fate_icu.eval.explanation`) that match a
  driver of the packet. Every text, C1's included, is scored against the stay's own attribution packet, although C1's model was
  never shown it.
- **Population:** the 1,000-stay explanation sample.
- **Comparison:** E5 (zero-shot MedGemma given the packet, no RAG) vs C1 (the same zero-shot MedGemma and prompt style, given a
  deterministic summary of every abnormal value instead of the packet). Same model, same instructions: only the input differs, so
  the test isolates what the packet adds. (A first draft compared E7 with C1; that would also change the model, its
  fine-tuning and the retrieval, so it could not say whether the packet was the cause.)
- **Test:** paired patient-level bootstrap of the mean per-stay difference (E5 - C1).
- **α / correction:** Holm-Bonferroni, α = 0.05.
- **Decision rule:** supported if adjusted p ≤ 0.05 and the mean unattributed-driver rate of E5 is lower than that of C1.
- **Secondary (reported, not corrected):** E6, E7 and E8 vs C1 (these also change the model); E5 vs C2 (random drivers).
- **Registered on:** 2026-11-21 (approved by the user)
- **Status / result:**

## H3: RAG and supported claims

- **Statement:** RAG raises the supported-claim rate (E7 vs E6).
- **Primary metric:** share of the *general medical claims* of an explanation (claims that go beyond the patient's own values
  and drivers) that are entailed by at least one of the passages retrieved for that stay. E6 and E7 are judged against the same
  passage pool, so E6, which was never shown the passages, is judged fairly.
- **Population:** the 1,000-stay explanation sample, stays with at least one general medical claim in both texts.
- **Comparison:** E7 (SFT + RAG) minus E6 (SFT, no RAG).
- **Test:** paired patient-level bootstrap of the mean per-stay difference.
- **α / correction:** Holm-Bonferroni, α = 0.05.
- **Decision rule:** supported if adjusted p ≤ 0.05 and the supported-claim rate of E7 is higher.
- **Secondary (reported, not corrected):** ALCE citation recall and precision of E7; the unsupported-claim rate; the
  checker's agreement with the user's 100 hand-labelled claim-passage pairs (Cohen's κ).
- **Registered on:** 2026-11-21 (approved by the user)
- **Status / result:**

## H4: Two-step driver selection

- **Statement:** two-step driver selection beats gradient-only and random selection on comprehensiveness/sufficiency.
- **Primary metric:** comprehensiveness at k = 3: the drop in the probability of the predicted class when the three selected
  units are removed together (higher is better).
- **Population:** a fixed random sample of 1,000 test stays (seed 666) of the explained model (M, or B3 while M packets do not exist).
- **Comparison:** the first three drivers of the two-step selection vs the top three units by gradient-only attribution vs three random units.
- **Test:** two paired patient-level bootstraps (two-step − gradient-only, two-step − random); the hypothesis is tested as an
  intersection: its p-value is the larger of the two.
- **α / correction:** Holm-Bonferroni, α = 0.05, with that single p-value.
- **Decision rule:** supported if the adjusted p ≤ 0.05 and two-step is higher than both comparators.
- **Secondary (reported, not corrected):** sufficiency at k = 3 (lower is better), k = 1 and k = 5, and the synthetic
  ground-truth benchmark (`docs/attribution_benchmark.md`).
- **Registered on:** 2026-11-21 (approved by the user)
- **Status / result:**

## H5: DPO (only if DPO runs)

- **Statement:** DPO improves driver and citation precision without lowering driver recall.
- **Primary metric:** driver precision and ALCE citation precision (both must improve); non-inferiority of driver recall@k.
- **Population:** the 1,000-stay explanation sample.
- **Comparison:** E8 (E7 + DPO) vs E7.
- **Test:** paired patient-level bootstraps of (E8 − E7) for each of the three metrics. Improvement of the two precisions is tested as
  an intersection (the larger p-value). Non-inferiority: the lower 95 % confidence bound of the recall difference must exceed −0.02.
- **α / correction:** Holm-Bonferroni with the intersection p-value, α = 0.05 (the family has m = 5 only if DPO is run).
- **Decision rule:** supported if both precisions improve at adjusted p ≤ 0.05 and the recall non-inferiority bound holds.
- **Registered on:** 2026-11-21 (approved by the user)
- **Status / result:**
