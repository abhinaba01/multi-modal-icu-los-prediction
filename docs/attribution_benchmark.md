# Stage 4 synthetic ground-truth benchmark: results for your review

Run `uv run python -m fate_icu.attribution.benchmark --config configs/attribution/benchmark.yaml`
(about 5 minutes on the laptop CPU; deterministic, seed 666). Recorded 2026-11-16.

## What was tested

Synthetic stays (2,000; 1,400 train) whose label is "long" when at least one of four planted
causes is present (probability 0.15 each, 2 % label noise). Each cause is one evidence unit:

| Cause | Unit | Planted as |
|---|---|---|
| A | time series `c2`, hours 8-12 | a rise of +2.5 SD |
| B | time series `c4`, hours 16-20 | a fall of -2.5 SD |
| C | static group `labs` | both features +2.0 SD |
| D | image concept `k1` | its logit +3.0 |

A small attention-fusion model (the Stage 3 `M` network) was trained on CPU; it reaches test
AUROC 0.945. 60 test stays that are long, predicted long and have a planted unit were explained.
Every number is a mean over those stays with a 95 % bootstrap interval. The random columns
order the same units at random.

## Recovery and faithfulness by method

k = 3 for time series; k = 1 for static and image, where there are only three units.

| Modality | Method | precision@1 | recall@3 | comprehensiveness | (random) | sufficiency (low is good) | (random) |
|---|---|---|---|---|---|---|---|
| time series | occlusion | 0.95 (0.87-1.00) | 0.94 (0.86-1.00) | 0.60 (0.48-0.68) | 0.04 | 0.02 (-0.02-0.07) | 0.64 |
| time series | **Integrated Gradients** | 0.95 (0.87-1.00) | 0.97 (0.90-1.00) | 0.61 (0.49-0.69) | 0.04 | 0.00 (-0.02-0.03) | 0.64 |
| time series | ExtremalMask (tuned) | 0.85 (0.72-0.95) | 0.83 (0.71-0.94) | 0.59 (0.47-0.68) | 0.04 | 0.09 (0.02-0.18) | 0.64 |
| time series | DynaMask | 0.21 (0.08-0.33) | 0.37 (0.22-0.51) | 0.25 (0.14-0.35) | 0.04 | 0.41 (0.31-0.52) | 0.64 |
| static | ablation | 0.76 (0.53-0.94) | 1.00 | 0.49 (0.32-0.65) | 0.19 | 0.18 (0.04-0.34) | 0.38 |
| static | **Integrated Gradients** | 0.88 (0.71-1.00) | 1.00 | 0.49 (0.32-0.65) | 0.19 | 0.13 (0.03-0.25) | 0.38 |
| image concepts | ablation | 0.93 (0.80-1.00) | 1.00 | 0.51 (0.32-0.71) | 0.00 | 0.68 (0.64-0.70) | 0.68 |
| image concepts | **Integrated Gradients** | 0.93 (0.80-1.00) | 1.00 | 0.51 (0.32-0.71) | 0.00 | 0.68 (0.64-0.70) | 0.68 |

Recall@3 is 1.00 for static and image because k = 3 is every unit there. Image sufficiency
equals its random value because the image carries no other evidence in this data.

## The two-step driver selection

tau = 0.039 logits, fitted on validation stays.

| Methods (time series + static + image) | driver precision | driver recall | top driver is planted | effect on planted units | drivers per stay | agreement with a randomised model |
|---|---|---|---|---|---|---|
| ExtremalMask + IG + IG | 0.53 | 0.88 | 0.93 | 0.78 | 2.5 | 0.21 |
| occlusion + ablation + ablation | 0.51 | 0.89 | 0.95 | 0.77 | 2.8 | 0.48 |
| **IG + IG + IG** | 0.53 | 0.89 | 0.95 | 0.77 | 2.6 | 0.37 |

## What to take from it

1. **My recommendation was Integrated Gradients for every modality** (D-048); you chose ExtremalMask
   for the time series. With its defaults it recovered the planted unit only 39 % of the time; tuned
   on a separate seed it reaches 85 %, close to IG but not better. IG stays selectable (`ts_method: ig`).
2. **The top driver is a planted unit 93-95 % of the time, and about 89 % of the planted
   units are among the drivers.**
3. **Half of the selected drivers are not planted units.** The spec's `tau` (the 95th
   percentile of |Δ| for random units) is only 0.04 logits on this sparse model, so real but
   small effects pass it; they carry 22 % of the total effect. If you want fewer, weaker
   drivers, the cleanest change is to also require |Δ| to be a fraction of the largest one. I
   have not done that, because it is not in the spec: your decision.
4. **The model-randomisation check is not clean** (0.2 to 0.5, not 0). Cells that were never
   measured score exactly 0 in both models, which ties the rankings. ExtremalMask agrees least.
5. **Limits:** synthetic data, three planted units, 60 stays (wide intervals), one model and
   one seed. The semi-synthetic benchmark on real tensors and the validation report on the
   real model are lab tasks. The image-pixel path (Grad-CAM and regions) is tested on public
   films for orientation and mechanics, not for localisation accuracy.

## What I need from you

**Decided 2026-11-16 (you): ExtremalMask is the default for the time series (IG first), and IG for
static groups and image concepts.** The table above favours IG for the time series (precision@1 0.95
against 0.85, intervals that nearly touch); ExtremalMask agrees least with a randomised model (0.21).
Still open: whether to add the relative-size condition on drivers (point 3). Until you decide,
the spec's rule stands unchanged.
