"""Synthetic MIMIC-IV tables with the real column schemas and planted test cases.

Nothing here is derived from MIMIC data: values are drawn from fixed distributions with
a seeded generator. Column names and order follow MIMIC-IV v2.2/v3.1 (a ``demo`` test
checks them against the local demo database).

Planted cases (``SyntheticMimic.roles``) have known answers under the contract's cohort
rules, e.g. a readmission within 48 h, a death in the ICU, LOS of exactly 1 and 21 days,
and one ``leakage`` stay with events exactly at intime and t0 and sentinel values after
t0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# MIMIC-IV column order per table (legacy table key -> columns).
SCHEMAS: dict[str, list[str]] = {
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
    "admissions": [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "admit_provider_id",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "edregtime",
        "edouttime",
        "hospital_expire_flag",
    ],
    "diagnoses_icd": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    "labevents": [
        "labevent_id",
        "subject_id",
        "hadm_id",
        "specimen_id",
        "itemid",
        "order_provider_id",
        "charttime",
        "storetime",
        "value",
        "valuenum",
        "valueuom",
        "ref_range_lower",
        "ref_range_upper",
        "flag",
        "priority",
        "comments",
    ],
    "icustays": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los",
    ],
    "chartevents": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "caregiver_id",
        "charttime",
        "storetime",
        "itemid",
        "value",
        "valuenum",
        "valueuom",
        "warning",
    ],
    "inputevents": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "caregiver_id",
        "starttime",
        "endtime",
        "storetime",
        "itemid",
        "amount",
        "amountuom",
        "rate",
        "rateuom",
        "orderid",
        "linkorderid",
        "ordercategoryname",
        "secondaryordercategoryname",
        "ordercomponenttypedescription",
        "ordercategorydescription",
        "patientweight",
        "totalamount",
        "totalamountuom",
        "isopenbag",
        "continueinnextdept",
        "statusdescription",
        "originalamount",
        "originalrate",
    ],
    "outputevents": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "caregiver_id",
        "charttime",
        "storetime",
        "itemid",
        "value",
        "valueuom",
    ],
    "sofa": [
        "stay_id",
        "hr",
        "starttime",
        "endtime",
        "pao2fio2ratio_novent",
        "pao2fio2ratio_vent",
        "rate_epinephrine",
        "rate_norepinephrine",
        "rate_dopamine",
        "rate_dobutamine",
        "meanbp_min",
        "gcs_min",
        "uo_24hr",
        "bilirubin_max",
        "creatinine_max",
        "platelet_min",
        "respiration",
        "coagulation",
        "liver",
        "cardiovascular",
        "cns",
        "renal",
        "respiration_24hours",
        "coagulation_24hours",
        "liver_24hours",
        "cardiovascular_24hours",
        "cns_24hours",
        "renal_24hours",
        "sofa_24hours",
    ],
}

# Folder of each table in the MIMIC-IV file layout.
TABLE_FOLDER = {
    "patients": "hosp",
    "admissions": "hosp",
    "diagnoses_icd": "hosp",
    "labevents": "hosp",
    "icustays": "icu",
    "chartevents": "icu",
    "inputevents": "icu",
    "outputevents": "icu",
    "sofa": "derived",
}

# chartevents itemid -> (mean, sd, unit); GCS components are drawn separately.
CHART_NUMERIC = {
    220045: (85.0, 10.0, "bpm"),  # heart rate
    220210: (18.0, 3.0, "insp/min"),  # respiratory rate
    220277: (96.0, 2.0, "%"),  # SpO2
    223761: (98.6, 0.8, "°F"),  # temperature
    220052: (78.0, 10.0, "mmHg"),  # arterial MAP
    223835: (40.0, 10.0, "%"),  # FiO2
}
GCS_ITEMIDS = {220739: (3, 4), 223900: (4, 5), 223901: (5, 6)}  # eye, verbal, motor ranges

# labevents itemid -> (mean, sd, unit): the 18 legacy labs + PaO2 and bilirubin (SOFA).
LAB_NUMERIC = {
    50868: (13.0, 2.0, "mEq/L"),
    50882: (24.0, 3.0, "mEq/L"),
    50902: (104.0, 4.0, "mEq/L"),
    50912: (1.1, 0.4, "mg/dL"),
    50931: (130.0, 30.0, "mg/dL"),
    50960: (2.0, 0.2, "mg/dL"),
    50971: (4.1, 0.4, "mEq/L"),
    50983: (139.0, 3.0, "mEq/L"),
    51006: (20.0, 8.0, "mg/dL"),
    51221: (32.0, 4.0, "%"),
    51222: (10.5, 1.5, "g/dL"),
    51248: (30.0, 1.5, "pg"),
    51249: (33.0, 1.0, "g/dL"),
    51250: (90.0, 4.0, "fL"),
    51265: (220.0, 60.0, "K/uL"),
    51277: (14.5, 1.0, "%"),
    51279: (3.6, 0.5, "m/uL"),
    51301: (10.0, 3.0, "K/uL"),
    50821: (95.0, 25.0, "mm Hg"),
    50885: (0.9, 0.6, "mg/dL"),
}
NOREPINEPHRINE, VASOPRESSIN, URINE_FOLEY = 221906, 222315, 226559
CREATININE, HEART_RATE, SPO2, MAP, BILIRUBIN = 50912, 220045, 220277, 220052, 50885

# Public CMS GEM rows for the ICD-9 codes used below (icd9 -> icd10 targets).
ICD9_GEM = {"5715": ["K740", "K7460", "K7469"], "4019": ["I10"], "0389": ["A419"], "486": ["J189"]}
ICD10_CODES = ["I10", "J189", "N179", "A419", "K7460", "E119", "I214", "J9601"]

CAREUNITS = [
    "Medical Intensive Care Unit (MICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
]
ADMISSION_TYPES = ["EW EMER.", "URGENT", "ELECTIVE", "SURGICAL SAME DAY ADMISSION"]
ADMISSION_LOCATIONS = ["EMERGENCY ROOM", "TRANSFER FROM HOSPITAL", "PHYSICIAN REFERRAL"]

W_HOURS = 24  # matches configs/contract.yaml window.hours

# Planted roles: stays whose cohort outcome is known.
KEPT_ROLES = ("readmit_first", "readmit_after_48h", "death_after_icu", "los_exactly_21", "leakage")
EXCLUDED_ROLES = (
    "readmit_within_48h",
    "death_in_icu",
    "los_below_1",
    "los_exactly_1",
    "los_above_21",
    "minor",
)
CASE_DROPPED_ROLES = ("missing_lab", "vital_out_of_range")  # in the base cohort, dropped later


@dataclass
class SyntheticMimic:
    """Synthetic tables keyed by legacy table name, plus planted roles."""

    tables: dict[str, pd.DataFrame]
    roles: dict[str, int] = field(default_factory=dict)
    normal_stays: list[int] = field(default_factory=list)

    @property
    def gem_text(self) -> str:
        return "".join(f"{icd9} {icd10} 10000\n" for icd9, targets in ICD9_GEM.items() for icd10 in targets)

    def expected_final_stays(self) -> set[int]:
        """Stays that survive every contract cohort rule."""
        return set(self.normal_stays) | {self.roles[r] for r in KEPT_ROLES}

    def expected_base_stays(self) -> set[int]:
        """Stays that pass the legacy ``build_cohort`` rules (before complete case)."""
        return self.expected_final_stays() | {self.roles[r] for r in CASE_DROPPED_ROLES}

    def write_csv_layout(self, root: str | Path) -> Path:
        """Write ``hosp/`` and ``icu/`` CSVs (MIMIC-IV layout) and ``2018_I9gem.txt`` under ``root``."""
        root = Path(root)
        for key, frame in self.tables.items():
            folder = root / TABLE_FOLDER[key]
            folder.mkdir(parents=True, exist_ok=True)
            frame.to_csv(folder / f"{key}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
        (root / "2018_I9gem.txt").write_text(self.gem_text, encoding="utf-8")
        return root

    def gem_path(self, root: str | Path) -> Path:
        return Path(root) / "2018_I9gem.txt"


class _Builder:
    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)
        self.rows: dict[str, list[dict]] = {key: [] for key in SCHEMAS}
        self.next_subject, self.next_hadm, self.next_stay = 10_000_000, 20_000_000, 30_000_000
        self.next_labevent, self.next_specimen, self.next_order = 1, 1, 1

    # --- entities ---------------------------------------------------------------------

    def patient(self, age: int | None = None) -> int:
        self.next_subject += 1
        self.rows["patients"].append(
            {
                "subject_id": self.next_subject,
                "gender": str(self.rng.choice(["F", "M"])),
                "anchor_age": int(age if age is not None else self.rng.randint(25, 90)),
                "anchor_year": 2150,
                "anchor_year_group": "2014 - 2016",
                "dod": None,
            }
        )
        return self.next_subject

    def admission(
        self,
        subject: int,
        admittime: pd.Timestamp,
        dischtime: pd.Timestamp,
        deathtime: pd.Timestamp | None,
        icd: list[tuple[str, int]],
    ) -> int:
        self.next_hadm += 1
        hadm = self.next_hadm
        self.rows["admissions"].append(
            {
                "subject_id": subject,
                "hadm_id": hadm,
                "admittime": admittime,
                "dischtime": dischtime,
                "deathtime": deathtime,
                "admission_type": str(self.rng.choice(ADMISSION_TYPES)),
                "admit_provider_id": "P00001",
                "admission_location": str(self.rng.choice(ADMISSION_LOCATIONS)),
                "discharge_location": None if deathtime is not None else "HOME",
                "insurance": "Medicare",
                "language": "ENGLISH",
                "marital_status": "MARRIED",
                "race": "WHITE",
                "edregtime": None,
                "edouttime": None,
                "hospital_expire_flag": int(deathtime is not None),
            }
        )
        for seq, (code, version) in enumerate(icd, start=1):
            self.rows["diagnoses_icd"].append(
                {
                    "subject_id": subject,
                    "hadm_id": hadm,
                    "seq_num": seq,
                    "icd_code": code,
                    "icd_version": version,
                }
            )
        return hadm

    def random_icd(self) -> list[tuple[str, int]]:
        codes = []
        for _ in range(3):
            if self.rng.rand() < 0.3:
                codes.append((str(self.rng.choice(list(ICD9_GEM))), 9))
            else:
                codes.append((str(self.rng.choice(ICD10_CODES)), 10))
        return codes

    def stay(
        self,
        subject: int,
        hadm: int,
        intime: pd.Timestamp,
        los_days: float,
        severity: float,
        skip_labs: frozenset[int] = frozenset(),
        hr_shift: float = 0.0,
    ) -> int:
        self.next_stay += 1
        stay = self.next_stay
        outtime = (intime + pd.Timedelta(days=los_days)).round("s")  # whole seconds, as in MIMIC
        unit = str(self.rng.choice(CAREUNITS))
        self.rows["icustays"].append(
            {
                "subject_id": subject,
                "hadm_id": hadm,
                "stay_id": stay,
                "first_careunit": unit,
                "last_careunit": unit,
                "intime": intime,
                "outtime": outtime,
                "los": (outtime - intime) / pd.Timedelta(days=1),
            }
        )
        horizon = min(los_days * 24.0, 30.0)  # events run past t0 when the stay does
        self._chart(subject, hadm, stay, intime, horizon, severity, hr_shift)
        self._labs(subject, hadm, intime, horizon, severity, skip_labs)
        self._inputs(subject, hadm, stay, intime, horizon, severity)
        self._outputs(subject, hadm, stay, intime, horizon)
        self._sofa(stay, intime, horizon)
        return stay

    # --- events -----------------------------------------------------------------------

    def chart(self, subject, hadm, stay, intime, hours, itemid, value, unit=""):
        charttime = intime + pd.Timedelta(hours=hours)
        self.rows["chartevents"].append(
            {
                "subject_id": subject,
                "hadm_id": hadm,
                "stay_id": stay,
                "caregiver_id": 1001,
                "charttime": charttime,
                "storetime": charttime + pd.Timedelta(minutes=5),
                "itemid": itemid,
                "value": f"{value:g}",
                "valuenum": float(value),
                "valueuom": unit,
                "warning": 0,
            }
        )

    def lab(self, subject, hadm, intime, hours, itemid, value, store_delay_hours=1.0):
        charttime = intime + pd.Timedelta(hours=hours)
        mean, sd, unit = LAB_NUMERIC[itemid]
        self.rows["labevents"].append(
            {
                "labevent_id": self.next_labevent,
                "subject_id": subject,
                "hadm_id": hadm,
                "specimen_id": self.next_specimen,
                "itemid": itemid,
                "order_provider_id": None,
                "charttime": charttime,
                "storetime": charttime + pd.Timedelta(hours=store_delay_hours),
                "value": f"{value:.2f}",
                "valuenum": round(float(value), 2),
                "valueuom": unit,
                "ref_range_lower": round(mean - 2 * sd, 2),
                "ref_range_upper": round(mean + 2 * sd, 2),
                "flag": None,
                "priority": "ROUTINE",
                "comments": None,
            }
        )
        self.next_labevent += 1
        self.next_specimen += 1

    def input(self, subject, hadm, stay, intime, hours, itemid, rate, unit="mcg/kg/min"):
        start = intime + pd.Timedelta(hours=hours)
        self.rows["inputevents"].append(
            {
                "subject_id": subject,
                "hadm_id": hadm,
                "stay_id": stay,
                "caregiver_id": 1002,
                "starttime": start,
                "endtime": start + pd.Timedelta(hours=6),
                "storetime": start + pd.Timedelta(minutes=10),
                "itemid": itemid,
                "amount": round(rate * 80 * 360, 3),
                "amountuom": "mcg",
                "rate": float(rate),
                "rateuom": unit,
                "orderid": self.next_order,
                "linkorderid": self.next_order,
                "ordercategoryname": "01-Drips",
                "secondaryordercategoryname": "Additive (Med)",
                "ordercomponenttypedescription": "Main order parameter",
                "ordercategorydescription": "Continuous Med",
                "patientweight": 80.0,
                "totalamount": 250.0,
                "totalamountuom": "mL",
                "isopenbag": 0,
                "continueinnextdept": 0,
                "statusdescription": "FinishedRunning",
                "originalamount": 4.0,
                "originalrate": float(rate),
            }
        )
        self.next_order += 1

    def output(self, subject, hadm, stay, intime, hours, value):
        charttime = intime + pd.Timedelta(hours=hours)
        self.rows["outputevents"].append(
            {
                "subject_id": subject,
                "hadm_id": hadm,
                "stay_id": stay,
                "caregiver_id": 1003,
                "charttime": charttime,
                "storetime": charttime + pd.Timedelta(minutes=15),
                "itemid": URINE_FOLEY,
                "value": float(value),
                "valueuom": "ml",
            }
        )

    def _chart(self, subject, hadm, stay, intime, horizon, severity, hr_shift):
        rng = self.rng
        for h in np.arange(0.5, horizon, 1.0):
            for itemid, (mean, sd, unit) in CHART_NUMERIC.items():
                if itemid == 223835 and h % 4 != 0.5:
                    continue  # FiO2 every 4 h
                if itemid == HEART_RATE:
                    mean = mean + 8.0 * severity + hr_shift
                if itemid == MAP:
                    mean = mean - 6.0 * severity
                value = rng.normal(mean, sd)
                if itemid == SPO2:
                    value = min(value, 100.0)
                self.chart(subject, hadm, stay, intime, float(h), itemid, round(value, 1), unit)
        for h in np.arange(0.5, horizon, 4.0):  # GCS components charted together
            for itemid, (low, high) in GCS_ITEMIDS.items():
                value = rng.randint(low, high + 1) - (1 if severity > 1.0 else 0)
                self.chart(subject, hadm, stay, intime, float(h), itemid, value)

    def _labs(self, subject, hadm, intime, horizon, severity, skip_labs):
        rng = self.rng
        for h in (2.0, 10.0, 18.0, 26.0):
            if h > horizon:
                continue
            for itemid, (mean, sd, _unit) in LAB_NUMERIC.items():
                if itemid in skip_labs or (itemid == 50821 and rng.rand() < 0.5):
                    continue
                if itemid == CREATININE:
                    mean = mean + 0.4 * severity
                value = max(rng.normal(mean, sd), 0.1 if itemid in (CREATININE, BILIRUBIN) else 1.0)
                self.lab(subject, hadm, intime, h, itemid, value)

    def _inputs(self, subject, hadm, stay, intime, horizon, severity):
        rng = self.rng
        if rng.rand() < 0.25 + 0.2 * max(severity, 0.0):
            for h in (1.0, 7.0, 13.0, 25.0):
                if h < horizon:
                    self.input(
                        subject, hadm, stay, intime, h, NOREPINEPHRINE, round(rng.uniform(0.02, 0.3), 3)
                    )
        if rng.rand() < 0.1:
            self.input(subject, hadm, stay, intime, 3.0, VASOPRESSIN, 2.4, "units/hour")

    def _outputs(self, subject, hadm, stay, intime, horizon):
        for h in np.arange(0.5, horizon, 1.0):
            self.output(subject, hadm, stay, intime, float(h), max(self.rng.normal(60.0, 25.0), 0.0))

    def _sofa(self, stay, intime, horizon):
        rng = self.rng
        for hr in range(int(horizon)):
            scores = {
                k: int(rng.randint(0, 3))
                for k in ("respiration", "coagulation", "liver", "cardiovascular", "cns", "renal")
            }
            row = {
                "stay_id": stay,
                "hr": hr,
                "starttime": intime + pd.Timedelta(hours=hr),
                "endtime": intime + pd.Timedelta(hours=hr + 1),
                "pao2fio2ratio_novent": None,
                "pao2fio2ratio_vent": None,
                "rate_epinephrine": None,
                "rate_norepinephrine": None,
                "rate_dopamine": None,
                "rate_dobutamine": None,
                "meanbp_min": 75.0,
                "gcs_min": 15.0,
                "uo_24hr": None,
                "bilirubin_max": None,
                "creatinine_max": None,
                "platelet_min": None,
                **scores,
                **{f"{k}_24hours": v for k, v in scores.items()},
            }
            row["sofa_24hours"] = sum(scores.values())
            self.rows["sofa"].append(row)

    def frames(self) -> dict[str, pd.DataFrame]:
        out = {}
        for key, columns in SCHEMAS.items():
            frame = pd.DataFrame(self.rows[key], columns=columns)
            for col in columns:
                if col.endswith("time") or col in ("admittime", "dischtime", "deathtime", "dod"):
                    frame[col] = pd.to_datetime(frame[col])
            out[key] = frame
        return out


def generate_synthetic_mimic(n_patients: int = 60, seed: int = 0) -> SyntheticMimic:
    """Generate synthetic MIMIC-IV tables (see module docstring).

    Args:
        n_patients: Number of ordinary patients (planted cases are added on top).
        seed: Seed for the generator; equal seeds give identical tables.

    Returns:
        The synthetic dataset with its planted roles.
    """
    b = _Builder(seed)
    rng = b.rng
    base = pd.Timestamp("2150-01-01 08:00:00")
    roles: dict[str, int] = {}
    normal: list[int] = []

    def admit(subject, intime, last_out, deathtime=None):
        return b.admission(
            subject,
            intime - pd.Timedelta(hours=6),
            last_out + pd.Timedelta(days=2),
            deathtime,
            b.random_icd(),
        )

    # Ordinary patients; ~1 in 8 has a second, later hospital admission.
    for _ in range(n_patients):
        subject = b.patient()
        n_admissions = 2 if rng.rand() < 0.125 else 1
        for a in range(n_admissions):
            severity = rng.normal()
            los = float(np.clip(np.exp(np.log(2.5) + 0.45 * severity + rng.normal(0, 0.3)), 1.05, 20.0))
            intime = base + pd.Timedelta(
                days=int(rng.randint(0, 3000)) + 400 * a, minutes=int(rng.randint(0, 1440))
            )
            hadm = admit(subject, intime, intime + pd.Timedelta(days=los))
            normal.append(b.stay(subject, hadm, intime, los, severity))

    def planted(role, los, *, age=None, death_offset=None, skip_labs=frozenset(), hr_shift=0.0, day=0):
        subject = b.patient(age)
        intime = base + pd.Timedelta(days=3500 + day)
        outtime = intime + pd.Timedelta(days=los)
        death = None if death_offset is None else outtime + pd.Timedelta(hours=death_offset)
        hadm = admit(subject, intime, outtime, death)
        roles[role] = b.stay(subject, hadm, intime, los, 0.0, skip_labs=skip_labs, hr_shift=hr_shift)

    planted("death_in_icu", 3.0, death_offset=-1.0, day=1)
    planted("death_after_icu", 3.0, death_offset=48.0, day=2)
    planted("los_below_1", 0.9, day=3)
    planted("los_exactly_1", 1.0, day=4)
    planted("los_exactly_21", 21.0, day=5)
    planted("los_above_21", 22.0, day=6)
    planted("minor", 3.0, age=17, day=7)
    planted("missing_lab", 3.0, skip_labs=frozenset({CREATININE}), day=8)
    planted("vital_out_of_range", 3.0, hr_shift=160.0, day=9)

    # Three ICU stays in one hospital admission: 2nd is a <48 h readmission, 3rd is not.
    subject = b.patient()
    t1 = base + pd.Timedelta(days=3600)
    t2 = t1 + pd.Timedelta(days=3.0) + pd.Timedelta(hours=24)
    t3 = t2 + pd.Timedelta(days=2.0) + pd.Timedelta(hours=72)
    hadm = admit(subject, t1, t3 + pd.Timedelta(days=2.5))
    roles["readmit_first"] = b.stay(subject, hadm, t1, 3.0, 0.0)
    roles["readmit_within_48h"] = b.stay(subject, hadm, t2, 2.0, 0.0)
    roles["readmit_after_48h"] = b.stay(subject, hadm, t3, 2.5, 0.0)

    # The leakage stay: boundary events and sentinel values after t0.
    subject = b.patient(60)
    intime = base + pd.Timedelta(days=3700)
    hadm = admit(subject, intime, intime + pd.Timedelta(days=4.0))
    stay = roles["leakage"] = b.stay(subject, hadm, intime, 4.0, 0.0)
    w = float(W_HOURS)
    b.chart(subject, hadm, stay, intime, 0.0, HEART_RATE, 50.0, "bpm")  # at intime: not in the mean
    b.chart(subject, hadm, stay, intime, w, HEART_RATE, 100.0, "bpm")  # at t0: in the mean
    b.chart(subject, hadm, stay, intime, 0.0, MAP, 40.0, "mmHg")  # at intime: in SOFA
    b.chart(subject, hadm, stay, intime, w, SPO2, 50.0, "%")  # at t0: mean yes, SOFA no
    b.chart(subject, hadm, stay, intime, w + 1 / 60, HEART_RATE, 999.0, "bpm")  # after t0
    b.lab(subject, hadm, intime, w + 0.5, CREATININE, 99.0)  # after t0
    b.lab(subject, hadm, intime, w + 1.0, BILIRUBIN, 50.0)  # after t0
    b.lab(subject, hadm, intime, w - 1.0, 51301, 10.0, store_delay_hours=3.0)  # result posts after t0
    b.output(subject, hadm, stay, intime, w + 0.5, 99_999.0)  # after t0
    b.input(subject, hadm, stay, intime, w + 1.0, NOREPINEPHRINE, 9.9)  # after t0

    return SyntheticMimic(tables=b.frames(), roles=roles, normal_stays=normal)


def add_post_t0_sentinels(synth: SyntheticMimic, seed: int = 1) -> dict[str, pd.DataFrame]:
    """Copy of the event tables with extreme values added strictly after every t0.

    Chart, input and output events go after their own stay's t0. Lab events join on
    ``hadm_id``, so they go after the latest t0 of any stay in the admission. Features
    built from the returned tables must equal those built from the originals.
    """
    rng = np.random.RandomState(seed)
    tables = {k: v.copy() for k, v in synth.tables.items()}
    stays = tables["icustays"].copy()
    stays["t0"] = stays["intime"] + pd.Timedelta(hours=W_HOURS)
    hadm_t0 = stays.groupby("hadm_id")["t0"].max()

    def after(t0):
        return t0 + pd.Timedelta(minutes=int(rng.randint(1, 600)))

    chart_items = list(CHART_NUMERIC) + list(GCS_ITEMIDS)
    extra_chart, extra_lab, extra_in, extra_out = [], [], [], []
    for row in stays.itertuples(index=False):
        for itemid in chart_items:
            when = after(row.t0)
            extra_chart.append(
                {
                    "subject_id": row.subject_id,
                    "hadm_id": row.hadm_id,
                    "stay_id": row.stay_id,
                    "caregiver_id": 9,
                    "charttime": when,
                    "storetime": when,
                    "itemid": itemid,
                    "value": "9999",
                    "valuenum": 9999.0,
                    "valueuom": "",
                    "warning": 0,
                }
            )
        when = after(row.t0)
        extra_in.append(
            {
                **{c: None for c in SCHEMAS["inputevents"]},
                "subject_id": row.subject_id,
                "hadm_id": row.hadm_id,
                "stay_id": row.stay_id,
                "starttime": when,
                "endtime": when + pd.Timedelta(hours=1),
                "storetime": when,
                "itemid": NOREPINEPHRINE,
                "rate": 99.0,
                "rateuom": "mcg/kg/min",
            }
        )
        extra_out.append(
            {
                "subject_id": row.subject_id,
                "hadm_id": row.hadm_id,
                "stay_id": row.stay_id,
                "caregiver_id": 9,
                "charttime": after(row.t0),
                "storetime": after(row.t0),
                "itemid": URINE_FOLEY,
                "value": 99_999.0,
                "valueuom": "ml",
            }
        )
    for hadm, t0 in hadm_t0.items():
        subject = int(stays.loc[stays["hadm_id"] == hadm, "subject_id"].iloc[0])
        for itemid in LAB_NUMERIC:
            when = after(t0)
            extra_lab.append(
                {
                    **{c: None for c in SCHEMAS["labevents"]},
                    "subject_id": subject,
                    "hadm_id": hadm,
                    "itemid": itemid,
                    "charttime": when,
                    "storetime": when,
                    "value": "9999",
                    "valuenum": 9999.0,
                }
            )

    for key, extra in (
        ("chartevents", extra_chart),
        ("labevents", extra_lab),
        ("inputevents", extra_in),
        ("outputevents", extra_out),
    ):
        frame = pd.DataFrame(extra, columns=SCHEMAS[key]).dropna(axis=1, how="all")
        tables[key] = pd.concat([tables[key], frame], ignore_index=True)[SCHEMAS[key]]
    return tables
