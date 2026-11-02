# Pre-registered hypotheses

Template written in Stage 0. Fill in the analysis fields in Stage 7 (`docs/stages.md`),
**before** any full-config result is seen. Once a hypothesis is registered, change it only
with a dated note explaining why.

Shared rules (Stage 7):
- every comparison is on identical test patients;
- patient-level bootstrap with 1,000 resamples for CIs, paired bootstrap for differences;
- Holm–Bonferroni across H1–H5;
- mean ± SD over 5 seeds for neural models;
- null results are reported.

Fields per hypothesis: statement, primary metric, population, comparison, test,
α / correction, decision rule, registration date, status / result.

---

## H1: Fusion vs EHR-only

- **Statement:** fusion improves AUPRC over the EHR-only model on P1 (a null result will be reported).
- **Primary metric:**
- **Population:** P1 (paired test stays)
- **Comparison:** M (attention fusion) vs E1 / B3 (static + time series)
- **Test:**
- **α / correction:**
- **Decision rule:**
- **Registered on:**
- **Status / result:**

## H2: Packet-constrained explanations vs an unconstrained summary

- **Statement:** packet-constrained explanations name fewer unattributed drivers than C1.
- **Primary metric:**
- **Population:**
- **Comparison:** E5–E8 vs C1
- **Test:**
- **α / correction:**
- **Decision rule:**
- **Registered on:**
- **Status / result:**

## H3: RAG and supported claims

- **Statement:** RAG raises the supported-claim rate (E7 vs E6).
- **Primary metric:**
- **Population:**
- **Comparison:** E7 vs E6
- **Test:**
- **α / correction:**
- **Decision rule:**
- **Registered on:**
- **Status / result:**

## H4: Two-step driver selection

- **Statement:** two-step driver selection beats gradient-only and random selection on comprehensiveness/sufficiency.
- **Primary metric:**
- **Population:**
- **Comparison:** two-step vs gradient-only vs random
- **Test:**
- **α / correction:**
- **Decision rule:**
- **Registered on:**
- **Status / result:**

## H5: DPO (only if DPO runs)

- **Statement:** DPO improves driver and citation precision without lowering driver recall.
- **Primary metric:**
- **Population:**
- **Comparison:** E8 vs E7
- **Test:**
- **α / correction:**
- **Decision rule:**
- **Registered on:**
- **Status / result:**
