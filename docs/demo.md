# Demo guide: the ICU stay explainer

Everything here runs on the laptop with open data only. No credentialed MIMIC data is on this laptop or on screen.

## Before the demo (once, 5 minutes)

1. The local database `mimiciv_demo` must be running (PostgreSQL service) and `.env` must have `FATE_MACHINE=laptop`.
2. Start the page and check it loads:

```bash
uv run --no-sync python -m fate_icu.demo.app
```

3. Open http://127.0.0.1:8765. Fill a few fields, press **Explain the stay**, and see an explanation appear.
4. Stop it with Ctrl+C. The model is saved under `artifacts/demo/`, so later starts need no database.
   `--retrain` trains it again; `--port N` changes the port.

If the page does not start: run `uv run --no-sync python -m pytest tests/test_demo_model.py -q`. If that passes,
the code is fine and the problem is the database or `.env`.

## What to say (about 5 minutes)

**1. The problem (30 s).** ICU beds are scarce. Predicting how long a patient will stay helps planning, but a number
alone is not trusted. Clinicians want to know why.

**2. What was built before (30 s).** The internship pipeline predicts length of stay on MIMIC-IV in two stages: first a
classifier for short versus long stays, then a regression for the short ones. FaTE-ICU extends it with hourly vital
signs and lab results, the patient's own chest X-ray, a way to find which inputs drove each prediction, and a language
model that explains the prediction in words and cites the literature.

**3. The live demo (2 minutes).** Type a patient's first 24 hours (age, a few vitals, creatinine, SOFA score) and press
the button.
- Point to the big number: expected days and the chance of a long stay.
- Read the first two sentences of the explanation aloud.
- Point to the coloured tags **D1, D2, ...**: each sentence cites the reason it comes from. Orange pushes toward a
  longer stay, blue toward a shorter one.
- Point to the bars: they show how much each reason mattered. These effects are exact for this model, not estimated.
- Change one value (for example creatinine from 0.8 to 3.4) and press the button again. The estimate and the reasons
  change.
- Point to the green line at the bottom: every sentence was checked automatically. It must cite its reasons, use only
  numbers that belong to them, name the right direction, and give no treatment advice.

**4. What is real and what is demo (1 minute).** Be direct about this.
- This page uses a small model trained on 56 stays of the open MIMIC-IV demo. It shows how the explanation works, not
  how accurate the predictions are.
- The explanation text comes from a fixed template filled with the model's own reasons. The language model
  (MedGemma, fine-tuned) is the next stage and its code is built and tested, but it needs the real data run on the
  cloud machine.
- The real model, with the full MIMIC-IV data, is being prepared on a Google Cloud machine. Patient-level results from
  it will only ever be shown as totals, because the data licence forbids showing individual patients.

**5. What is new and why it matters (1 minute).** The thesis claim is not a more accurate number. It is an explanation
that is *faithful* (it names what really moves the model, tested by removing inputs) and *grounded* (each medical claim
is backed by the passage it cites), with an evaluation that measures both. If the chest X-ray adds little, that is
reported as a result too.

## Questions you may be asked

- **"Is this accurate?"** Not claimed here. 56 training stays cannot support accuracy claims. Accuracy is measured on
  the full data later, with a held-out set of patients.
- **"Why not just use SHAP?"** SHAP gives numbers per feature. We turn the strongest reasons into a sentence and then
  check that the sentence says only what the numbers say.
- **"Can the language model make things up?"** It only sees the prediction, the listed reasons and the retrieved
  passages. Any number or reason outside those is detected by the checker, and after three failed tries the system falls
  back to the template.
- **"Is patient data safe?"** The credentialed data stays in a private Google Cloud project. Nothing from it is on this
  laptop, in git, or in any third-party service.
- **"What does it cost to run?"** A spot GPU machine, around ₹30 an hour while running, stopped when idle.

## If something goes wrong during the demo

- Page will not load: restart it with the command above and refresh.
- A field is rejected: values must be plausible (age 18-110, heart rate 20-250, and so on); the page says which one.
- No time to run it: show the saved explanation below.

### Saved example (open demo, in case the page fails)

An older patient with heart rate 118, respiratory rate 27, SpO2 92, creatinine 3.4, urea 62, total SOFA 11 gives about
10 days and an 86% chance of a long stay, mainly because of vital signs and the SOFA score. The explanation names
vital signs [D1] and SOFA score [D2] first, then demographics [D3], and ends with a smaller contribution from
laboratory results [D4], each with a "longer stay" direction.
