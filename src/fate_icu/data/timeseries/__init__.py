"""Stage 2: hourly time-series tensors over (intime, t0] for the contract cohort.

variables.py  configs/variables.yaml as a validated model
sources.py    windowed raw rows from PostgreSQL (sql/timeseries/) or in-memory tables
extract.py    availability (storetime <= t0) and plausibility rules, provenance
tensors.py    hourly grid, train-fitted normaliser, values / mask / delta
report.py     aggregate missingness and comparison with the internship features
build.py      the end-to-end build (python -m fate_icu.data.timeseries.build)
"""
