# Grading guide: is this passage relevant to the query?

This is the hand labelling for the Stage 5 retrieval evaluation. Your grades decide the
thesis numbers for the search engine: precision@3 and nDCG@5 for BM25, dense, hybrid and
hybrid + rerank.

## What you are grading

Each query is one driver of a prediction, written as a fixed sentence, for example:

> serum creatinine associated with longer intensive care unit length of stay

For every query you see about 8-15 passages, gathered from the top 5 results of four search
methods. The order is shuffled and the method is never shown, so the grading stays blind.
Grade each passage **on its own**, against the query.

## The three grades

| Grade | Meaning | Example for the creatinine query |
|---|---|---|
| **2** | Directly relevant. It addresses this driver's link to the ICU course, length of stay, severity or outcome in critically ill adults, or gives a guideline statement about it. | "A rise in serum creatinine in the first 48 hours was associated with longer ICU stays and more renal replacement therapy." |
| **1** | Related background. It is about the driver or its condition (definition, mechanism, normal range, management) but says nothing about ICU outcomes. | "Creatinine is produced by muscle and cleared by the kidneys; KDIGO stages acute kidney injury by its rise." |
| **0** | Off-topic. Another subject, or a population with nothing transferable (e.g. a laboratory method in rats), or text that cannot be read. | "Creatine supplements in athletes." |

Tips:
- For a "shorter" query or a "longer" one, the direction does not change the grade: a
  passage about the driver's effect on length of stay is relevant either way.
- Ignore who published it and how famous the journal is.
- If you hesitate between two grades, choose the lower one.
- A passage that is only a list of references or headings is 0.

## How to grade

The easiest way is the terminal tool. It shows one passage at a time and saves every answer
at once:

```
uv run python -m fate_icu.rag.label --config configs/rag/full.yaml
```

Type `0`, `1` or `2`, then Enter. `s` skips, `b` goes back to the previous passage, `h`
shows the grades again, and `q` saves and quits. Run the command again to carry on where you
stopped. About 400-750 passages take 3-6 hours, so spread them over several sittings.

You can also open `labelling_sheet.tsv` in Excel (File > Open, choose "All files") and type
grades in the `grade` column. Save it as "Unicode Text (*.txt)" or keep the .tsv name. Do not
change any other column.

When everything is graded, tell Claude, or run:

```
uv run python -m fate_icu.rag.evaluate import --config configs/rag/full.yaml --rater <your initials>
uv run python -m fate_icu.rag.evaluate score --config configs/rag/full.yaml
```

The import copies only the query ids, passage ids and grades into
`configs/rag/relevance_labels.tsv`, which goes into git. The passage text stays in the data
folder, because it is the publishers' text.
