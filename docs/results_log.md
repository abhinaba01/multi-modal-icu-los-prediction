# Results log

Every experiment, including null results (CLAUDE.md A4). Each entry records:
- date, stage and run id (the `run_manifest.json` in the lab's outputs);
- config, data (MIMIC version, population) and seeds;
- result with CIs, the statistical test and its outcome;
- a one-line interpretation.

Smoke runs on the demo or synthetic data never appear here.

| Date | Stage | Run id | Config | Data / population | Seeds | Result (95 % CI) | Test | Interpretation |
|---|---|---|---|---|---|---|---|---|

---

<!-- Entry template
### YYYY-MM-DD · Stage N · <run id>
- **Question:**
- **Config:** configs/<stage>/full.yaml @ <git commit>
- **Data:** MIMIC-IV v3.1, population P1/P2, n = …
- **Seeds:**
- **Result:**
- **Statistical test:**
- **Interpretation:**
-->
