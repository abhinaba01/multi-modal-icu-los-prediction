# Stages 1–8 — FaTE-ICU

Referenced from CLAUDE.md. Read **only the stage I name** ("Start Stage N"); don't load the others. Stage 0 is in CLAUDE.md.

**Build order (laptop, now):** 0 → 2 → 1 → 5 → 3 → 4 → 6 → 7. Stage 2 comes early because its variable list feeds the RAG queries and attribution. Stage 5 uses only public data, so it is built *and* run now.

**Run order (lab, later):** 0 → 1 → 2 → 3 → 4 → 6 → 7. Each stage needs the previous stage's outputs; Stage 5's finished index is copied across.

**Stage status** (update when a stage's build is done; the ≤10-line summary goes in `docs/STATUS.md`):

| Stage | Name | Build (laptop) | Run (lab) |
|---|---|---|---|
| 0 | Repo and contract | in CLAUDE.md | – |
| 2 | Hourly time-series tensors | – | – |
| 1 | Link MIMIC-IV and MIMIC-CXR | – | – |
| 5 | RAG layer | – | – |
| 3 | Encoders, fusion model, baselines | – | – |
| 4 | Attribution and evidence packets | – | – |
| 6 | LLM explainer (MedGemma + QLoRA) | – | – |
| 7 | Evaluation harness | – | – |
| 8 | Optional extensions | – | – |

## Stage 2 — Hourly time-series tensors

**Goal:** an hourly tensor over [intime, t0] for every stay, built from the same sources and item mappings as the internship features.

**Build now (laptop, on `mimiciv_demo`):**
1. `configs/variables.yaml`: for each variable — source table and itemids (reuse the internship's mappings), unit, plain-English label, synonyms (later used to find mentions in generated text), reference range, transform. Starting set (about 25):
   - Vitals (`mimiciv_icu.chartevents`): heart rate, systolic/diastolic/mean blood pressure, respiratory rate, SpO2, temperature, GCS.
   - Respiratory: FiO2, PaO2, PaCO2, pH.
   - Labs (`mimiciv_hosp.labevents` — no `stay_id`, so filter by `subject_id` and time): creatinine, BUN, bilirubin, platelets, WBC, lactate, glucose, sodium, potassium, haemoglobin.
   - Urine output (`outputevents`); vasopressor dose as norepinephrine equivalents (`mimiciv_derived.norepinephrine_equivalent_dose`).
   - Hourly SOFA components from `mimiciv_derived.sofa`, rows with `endtime ≤ t0`. Use the official table, not hand-rolled SQL.
2. **Availability rule:** include a measurement only if its result was available at t0 (`storetime ≤ t0` where the table has it), placed on the timeline at `charttime`. Leave the internship features unchanged and note any difference in `docs/decisions.md`.
3. **Tensors:** `values`, `mask` (1 = observed that hour) and `delta` (hours since last observation, capped), each `[N, W, C]`. Forward-fill within a stay only — never backward-fill, which moves later values into earlier hours and corrupts temporal attribution. Remaining gaps get the train median with mask = 0. log1p for skewed labs, clip to the train 0.5–99.5th percentiles, standardise with train statistics.
4. **Tests:** shapes; mask matches raw availability on the demo; no-future test; forward-fill-only test; deterministic output.

**Run later (lab):** full-cohort tensors and a per-variable missingness report (aggregates).

**Done when (build):** the smoke config builds tensors from the demo end to end; tests pass; runbook section written.

## Stage 1 — Link MIMIC-IV and MIMIC-CXR

**Background:** MIMIC-CXR uses the same `subject_id` and per-patient date shift as MIMIC-IV, so X-rays are aligned to ICU stays by `subject_id` and timestamps. It has no `hadm_id` or `stay_id`, and covers 2011–2016, so only a subset of stays will link. No open X-ray data links to the demo patients, so the build uses synthetic metadata.

**Build now (laptop):**
1. **Synthetic metadata generator:** a fake `mimic-cxr-2.0.0-metadata.csv` with the real columns (`dicom_id, subject_id, study_id, ViewPosition, StudyDate, StudyTime`), keyed to the demo's ICU stays, with planted cases and their known correct answers: images before the lookback, inside the window, after t0; AP vs PA; ties; awkward `StudyTime` values.
2. **`parse_study_datetime`:** `StudyDate` is YYYYMMDD. `StudyTime` is HHMMSS with optional fractional seconds, stored as a number, so leading zeros are lost (`1520.0` means 00:15:20). Zero-pad the integer part to six digits. Test midnight, values below 100 and missing fractions.
3. **Selection:** frontal views only (AP or PA). Eligible window `intime − 24 h ≤ study_datetime ≤ t0` (lookback configurable). Take the most recent eligible image per stay; tie-break AP over PA, then `dicom_id`. Output `paired_cohort.parquet` (`stay_id, subject_id, study_id, dicom_id, study_datetime, hours_before_t0, view`), keeping every stay with a `has_cxr` flag.
4. **Download list:** build image paths `files/p{first two digits of subject_id}/p{subject_id}/s{study_id}/{dicom_id}.jpg` and write the list; the runbook gets the `wget`/`gsutil` command. Never download the full image set.
5. **Preprocessing,** tested on a handful of public chest X-rays (not MIMIC): grayscale → torchxrayvision normalisation to [−1024, 1024] → centre crop → 224 (plus a 512 copy for Grad-CAM display) → cache.
6. **Flow-report code** (aggregates): counts after each filter step; share linked per split and per `anchor_year_group`; `hours_before_t0` distribution; LOS for paired vs unpaired stays.
7. Compare your logic with the MedFuse linkage code (github.com/nyuad-cai/MedFuse); don't copy it in.

**Rules:** use the contract split, not MIMIC-CXR's official split.

**Run later (lab):** needs my signed MIMIC-CXR-JPG data use agreement. Download the metadata, run the linkage on the full cohort, download only the selected images, build the cache, review the flow report. If the paired training set has fewer than about 2,000 stays, tell me before Stage 3 runs — the fallback is radiology report text (Stage 8).

**Done when (build):** tests pass on the synthetic metadata with every planted answer correct; preprocessing works on the public images; runbook section written.

## Stage 5 — RAG layer (public data: build and run now)

**Purpose:** patient facts come from the packet; general medical knowledge ("rising creatinine signals acute kidney injury, which is associated with longer ICU stays") comes from retrieved passages with citations. The LLM's own memory is never a cited source. BM25 and all code run on the laptop; embed the corpus on a Kaggle or Colab GPU; copy the finished index to the lab.

**Corpus:**
- PubMed abstracts: MedRAG's pre-chunked PubMed corpus on Hugging Face filtered to critical care, or abstracts pulled with NCBI E-utilities using ICU-focused queries (ICU length of stay, AKI, sepsis, ARDS, delirium, vasopressors, mechanical ventilation, …). Target 100k–500k passages.
- StatPearls (MedRAG's repository has a build script; non-commercial licence, fine for a thesis).
- A few open-access guidelines that I download (e.g. Surviving Sepsis Campaign 2021, KDIGO AKI), chunked to about 200–300 tokens with section titles.
- `corpus_manifest.csv`: source, licence, PMID or URL, retrieval date.

**Retrieval:**
- BM25 (`bm25s`); MedCPT article encoder for passages and query encoder for queries, in a FAISS index.
- Reciprocal-rank fusion of the two top-50 lists → MedCPT cross-encoder rerank → top 3 per driver → deduplicate → IDs R1…Rn.
- Queries come from the driver labels in `variables.yaml` via fixed templates, e.g. `"{label} {direction phrase} intensive care unit length of stay"`. They contain no patient values. LLM query rewriting is an ablation, not the default.
- Version the index (corpus hash + encoder revision) in the run manifest.

**Evaluation:** I hand-label relevance for about 50 driver queries × top 5. Report precision@3 and nDCG@5 for BM25, dense, hybrid, and hybrid + rerank.

**Done when:** the index is built and versioned; retrieval evaluation is reported; `retrieve(packet) → passages` is deterministic; the runbook explains how to copy the index to the lab.

## Stage 3 — Encoders, fusion model, baselines

**Architecture** (`src/fate_icu/models/`):
- **Static encoder:** MLP over the internship features plus missingness indicators → 256.
- **Time-series encoder:** 2-layer LSTM (hidden 256, as in MedFuse) over `[values, mask, delta]` → 256. Ablation: small Transformer with time embeddings.
- **Image encoder:** torchxrayvision DenseNet-121, frozen, weights `densenet121-res224-chex` (not trained on MIMIC-CXR, so it hasn't seen our test images; `-all` is an ablation only). Image token = MLP([1024-d features ‖ pathology logits for the non-empty entries of `model.pathologies`]) → 256. Cache features per `dicom_id`; a test checks cached and on-the-fly features match.
- **Fusion:** tokens [CLS, static, ts, image] + modality embeddings → 2-layer Transformer encoder (d = 256, 4 heads, dropout 0.1); a missing modality is removed with `key_padding_mask`.
- **Heads** (the internship's two stages in one model): long-stay logit (threshold from the contract) and log1p(LOS days). Loss = BCE with `pos_weight` + λ·Huber; tune λ on validation.

**Robustness training — required for valid deletion tests:** modality dropout (image p = 0.3, static and time series p = 0.1, never all at once) and evidence-unit dropout (each step, mask 10–15% of variable × 4-hour blocks and set random static groups to missing). "This evidence was removed" then becomes a normal input, so deletion tests measure reliance, not distribution shift.

**Arms** (same test patients): B0 internship XGBoost · B1 static MLP · B2 time series only · B3 static + time series · B4 image only · B5 late fusion (mean of unimodal logits) · B6 concatenation + MLP · M attention fusion (ours). Populations: P1 = paired test stays (primary), P2 = all test stays.

**Metrics** (`eval/predictive.py`): AUROC, AUPRC, Brier, ECE; MAE and RMSE in days; patient-level bootstrap 95% CIs.

**Build now (laptop):** all model, training and metric code. Smoke tests on synthetic and demo-derived tensors: shapes, missing-modality masking, overfitting a tiny batch (loss near zero), seed determinism. Image encoder tested on the public X-rays. fp32 on CPU.

**Run later (lab, full config):** cache image features; train every arm with 5 seeds — AdamW (lr 3e-4, weight decay 0.01), cosine schedule with warmup, batch 64, bf16 autocast, ≤ 50 epochs, early stopping (patience 8) on validation AUPRC. Results table with CIs and a plain statement of whether the image helps; a null gain is an acceptable finding.

**Done when (build):** the smoke config trains every arm end to end on CPU; metric tests pass; runbook section written.

## Stage 4 — Attribution and evidence packets

**Explanation target:** the long-stay logit (the LOS-days head is secondary). **Evidence units:** variable × 4-hour block; static feature groups; image (whole image, pathology concepts, anatomical regions for description only).

**Methods** (configurable, all compared):
- **Time series:** ExtremalMask (`tint`) by default; DynaMask, Integrated Gradients and occlusion for comparison; other modalities held fixed via `additional_forward_args`.
- **Static:** Integrated Gradients with the "missing" representation as baseline; group-level feature ablation.
- **Image:** Grad-CAM on DenseNet's last dense block through an `image → long-stay logit` wrapper; Integrated Gradients over the concept logits.
- **Regions:** map Grad-CAM mass to anatomy with torchxrayvision's anatomical segmentation model (ChestX-Det PSPNet; check the installed API). Use its anatomical left/right labels, never image left/right — AP films show the patient's right on the image's left. Split each lung into upper and lower zones.

**Driver selection** (two steps, because scores from different methods aren't comparable across modalities):
1. Within each modality, keep the top 5 units by that modality's method.
2. Across modalities, score each candidate by Δ = logit(x) − logit(x with the unit removed), using the in-distribution removal from Stage 3. Keep |Δ| ≥ τ (95th percentile of |Δ| for random non-candidate units, fit on validation). Merge adjacent blocks of the same variable and recompute Δ. Keep at most 5 drivers, sorted by |Δ|. Δ > 0 means the unit pushed toward a longer stay.

**Image-finding rule:** a radiographic finding may appear only if its concept is a driver *and* the image model's probability is above its operating threshold. Phrase it as an "image model estimate", never a diagnosis.

**Evidence packet** (pydantic schema, one JSON per stay; values illustrative; `observed` strings come from deterministic templates, never an LLM):

```json
{
  "prediction": {"p_long_stay": 0.78, "predicted_los_days": 6.4,
                 "long_stay_threshold_days": 4, "window_hours": 24},
  "drivers": [
    {"id": "D1", "modality": "timeseries", "label": "serum creatinine",
     "time": "hours 12-24", "observed": "rose from 1.1 to 2.3 mg/dL",
     "reference_range": "0.6-1.2 mg/dL", "direction": "longer", "effect": 0.42},
    {"id": "D2", "modality": "image", "label": "pleural effusion (image model estimate 0.81)",
     "region": "right lower lung zone", "direction": "longer", "effect": 0.21}
  ]
}
```

**Build now (laptop):**
- All of the above, running on the smoke model; packets for the demo stays.
- **Synthetic ground-truth benchmark** — real results, not just tests: synthetic stays where the label depends on a planted pattern (e.g. a creatinine rise in hours 8–12); train a small model on CPU; report each method's precision/recall@k at recovering the planted units. Add deletion/insertion curves with AOPC, comprehensiveness and sufficiency at k ∈ {1, 3, 5} (DeYoung et al., 2020), and the model-randomisation check (Adebayo et al., 2018). Metric code in `eval/attr_faithfulness.py`. Record the default method you pick, and why, in `docs/decisions.md`.

**Run later (lab):** a semi-synthetic version with patterns planted in real tensors; packets for all validation and test stays; the validation report on the real model.

**Done when (build):** I have reviewed the synthetic benchmark; demo packets validate against the schema; runbook section written.

## Stage 6 — LLM explainer: MedGemma 1.5 4B + QLoRA

**Model:** `google/medgemma-1.5-4b-it` (gated; I accept the Health AI Developer Foundations terms). Load with `AutoProcessor` / `AutoModelForImageTextToText` as the model card shows; text input only; pin the revision. Fallback: `google/medgemma-4b-it`.

**Design rule:** the explainer sees only the evidence packet and retrieved passages — never raw EHR data, notes or the image. A patient-specific statement not traceable to the packet is therefore a detectable hallucination.

**Output contract** (`llm/validate.py`): 3–5 sentences of plain clinical English, no treatment recommendations; the first sentence states the prediction; every sentence cites at least one [D#] or [R#]; only IDs present in the input; no numbers absent from the packet; each driver described with the packet's direction. On failure, resample up to 3 times, then fall back to a deterministic template. Retry and fallback rates are reported, never hidden.

**Training recipe (lab, full config):**
- **Data:** packets and passages for training-split stays (about 3–5k). Rejection-sampling SFT: 8 samples per prompt from the base model (temperature 0.8) with a strict prompt and two hand-written examples; label each with the validator plus AlignScore; keep passing samples. DPO pairs from the same samples (chosen passes every check; rejected fails one) plus corrupted negatives (swapped driver, flipped direction, mis-cited passage). The training checker (AlignScore) must differ from the evaluation checker (Stage 7 NLI model plus my labels).
- **QLoRA SFT** (TRL `SFTTrainer` + PEFT + bitsandbytes): 4-bit NF4, double quantisation, bf16 compute. LoRA r = 16, α = 32, dropout 0.05 on the language model's q, k, v, o, gate, up, down projections only — the checkpoint also has a vision tower with similar module names, so match with a regex restricted to the language model and print the selected modules. Prompt–completion format, with a test that the unmasked label tokens decode to exactly the completion. Max length 2048, batch 1, gradient accumulation 16, gradient checkpointing, paged 8-bit AdamW, lr 2e-4, cosine, 3% warmup, 2–3 epochs. Instructions go in the user turn (Gemma has no separate system role). Every N steps, log the validator pass rate on 100 validation packets. If it runs out of memory: max length 1536, or a 24 GB GPU on GCP.
- **Optional DPO** (TRL `DPOTrainer`): merge the SFT adapter into a bf16 copy of the base on CPU, then train a fresh QLoRA adapter on it so the reference policy is the SFT model; β = 0.1, lr 5e-6, 1 epoch, precomputed reference log-probabilities.
- **Inference:** greedy decoding, fixed seeds; outputs stay in the lab's artifacts directory. Use vLLM if sampling is slow.

**Build now (laptop):**
- Prompt builder, validator, template fallback, dataset builder, SFT, DPO and generation scripts, all unit-tested on CPU.
- CPU smoke run of the full generate → validate loop with a tiny instruction model (e.g. Qwen2.5-0.5B-Instruct) on demo-derived and synthetic packets.
- Kaggle or Colab smoke run of MedGemma QLoRA for a few dozen steps on demo-derived or synthetic packets (the demo is open data). Check memory use, label masking and pass-rate logging. Free-tier GPUs (T4/P100) lack bf16, so these are debugging runs only.

**Run later (lab):** sampling, SFT, optional DPO, and explanations for all test packets.

**Done when (build):** the CPU and Kaggle/Colab smoke runs pass; runbook section written.

## Stage 7 — Evaluation harness

One command runs everything: `python -m fate_icu.eval.run --config configs/eval/<smoke|full>.yaml` → `reports/<run_id>/` (metrics.json, CSV and Markdown tables, figures). Every metric has a unit test on a small hand-computed example.

1. **Predictive:** Stage 3 metrics on P1 and P2.
2. **Attribution faithfulness:** Stage 4 metrics.
3. **Explanation faithfulness (text ↔ model):** driver precision (share of clinical entities in the text that match a packet driver, found by deterministic lexicon matching on `variables.yaml` synonyms, concept names and feature-group labels); driver recall@k; direction accuracy; claim deletion validity (remove each named driver's unit and re-score — valid if the logit moves in the stated direction by ≥ τ); packet sensitivity (delete a driver from the packet → the text should drop it; add a decoy → the text should mention it); numeric consistency.
4. **Groundedness (text ↔ literature):** citation recall and precision as defined in ALCE (Gao et al., 2023), scored with an NLI model (e.g. DeBERTa-v3-large trained on MNLI/FEVER/ANLI), not AlignScore; unsupported-claim rate; and a check of the checker — I hand-label 100 claim–passage pairs, report Cohen's κ.
5. **Usability (optional):** a clinician rates about 50 blinded explanations for clarity, usefulness and perceived faithfulness.

**Circularity note:** driver selection already uses single-unit Δ, so claim deletion validity is high by design for text that sticks to the packet — it tests the explainer, not the selection method. Compare selection methods (H4) with comprehensiveness/sufficiency (joint removal of the top k) and the synthetic ground-truth benchmark instead.

**Ablation grid** (identical test stays):

| ID | Configuration |
|---|---|
| E0 | internship XGBoost + SHAP feature list |
| E1 | EHR-only neural model (static + time series) |
| E2 | image only |
| E3 | fusion (ours) |
| E4 | E3 + template explanation |
| E5 | E3 + zero-shot MedGemma, packet-constrained, no RAG |
| E6 | E3 + SFT MedGemma, no RAG |
| E7 | E3 + SFT MedGemma + RAG |
| E8 | E7 + DPO |
| C1 | control: MedGemma given a deterministic summary of all abnormal values instead of the packet |
| C2 | control: packet with random drivers |

**Statistics:** patient-level bootstrap (1,000 resamples) for CIs; paired bootstrap for differences; Holm–Bonferroni across the pre-registered hypotheses; mean ± SD over 5 seeds.

**Pre-registered hypotheses** (`docs/hypotheses.md`, written in Stage 0):
- **H1:** fusion improves AUPRC over the EHR-only model on P1 (a null result will be reported).
- **H2:** packet-constrained explanations name fewer unattributed drivers than C1.
- **H3:** RAG raises the supported-claim rate (E7 vs E6).
- **H4:** two-step driver selection beats gradient-only and random selection on comprehensiveness/sufficiency.
- **H5** (if DPO runs): DPO improves driver and citation precision without lowering driver recall.

**Build now (laptop):** all metric and statistics code; the whole grid runs end to end with the smoke config on demo and synthetic data.

**Run later (lab):** the full grid with the full config; every table reproduces from its manifest; `docs/results_log.md` updated.

## Stage 8 — Optional extensions (only after Stage 7)

- **Radiology report text** (also the fallback if the image cohort is too small): MIMIC-CXR reports (`s{study_id}.txt`) for studies before t0 only, FINDINGS and IMPRESSION sections. The report's finalisation time isn't recorded, so assume a configurable delay after the study. Encode with a biomedical BERT; attribute at sentence level with Integrated Gradients. Needs my signed MIMIC-CXR agreement.
- **Agentic variant:** specialist agents (time series, imaging, literature) read the same packet and passages, and an adjudicator writes the final explanation; the deliberation trace becomes the explanation. Evaluate with the identical harness. Local models only.
