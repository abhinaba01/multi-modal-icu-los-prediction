"""Stage 1: link MIMIC-IV ICU stays to the patient's own chest X-ray (MIMIC-CXR-JPG).

metadata.py    the metadata columns, reading, StudyDate/StudyTime parsing
synthetic.py   fake metadata keyed to a cohort, with planted cases and their answers
linkage.py     selection of one frontal image per stay, image paths, download list
report.py      aggregate flow report (counts, shares, timing, LOS, date alignment)
preprocess.py  torchxrayvision preprocessing (224 for the model, 512 for display), cache
link.py        python -m fate_icu.data.cxr.link   (linkage + list + flow report)
cache.py       python -m fate_icu.data.cxr.cache  (preprocess the selected images)
"""
