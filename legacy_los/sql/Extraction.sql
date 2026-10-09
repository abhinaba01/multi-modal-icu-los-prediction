-- =============================================================================
-- Extraction.sql  --  Hempel et al. (2023) faithful replication
--
-- This is an ANNOTATED copy of the reference GitLab query
--   ul-mds/data-science/length-of-stay-prediction-mimic-iv :: Data_extraction/Extraction.sql
-- kept here for the OPTIONAL PostgreSQL path and as documentation of the exact
-- cohort / feature logic the Python extractors reproduce (src/features/*).
--
-- The default project path does NOT use PostgreSQL: it streams the raw MIMIC-IV
-- CSV(.gz) directly (see src/features/build_dataset.py). Every rule below is
-- mirrored 1:1 in Python. Schema names are MIMIC-IV (v2.x/v3.x compatible for the
-- itemids used); adjust `chartevents`/`labevents`/... to your schema if running
-- this directly.
--
-- Feature itemids (verified against the reference):
--   Vitals (chartevents):
--     220045 heart_rate | 220210 respiration | 220277 saturation |
--     220739 GCS-eye     | 223761 temperature (deg F) |
--     223900 GCS-verbal  | 223901 GCS-motor
--   Labs (labevents) -- 18 items, NOTE Phosphate (50970) is intentionally absent
--   to match the reference code that produced the paper's results:
--     50868 Anion Gap | 50882 Bicarbonate | 50902 Chloride | 50912 Creatinine |
--     50931 Glucose   | 50960 Magnesium   | 50971 Potassium | 50983 Sodium |
--     51006 Urea Nitrogen | 51221 Hematocrit | 51222 Hemoglobin | 51248 MCH |
--     51249 MCHC | 51250 MCV | 51265 Platelet Count | 51277 RDW |
--     51279 Red Blood Cells | 51301 White Blood Cells
-- =============================================================================

-- 1. Readmission table: keep stays that are NOT an ICU readmission within 48h
--    (2 days) of a previous ICU discharge WITHIN THE SAME hospital admission.
drop materialized view if exists no_readmissions cascade;
create materialized view no_readmissions as
  with readmission_data as (
        select time_data.stay_id, (prev_dd > intime - interval '2 day')::int as num_readmissions
          from (select i.*,
                       lag(i.outtime) over (partition by i.hadm_id order by i.intime) as prev_dd
                from icustays i) as time_data
) select stay_id from readmission_data
where num_readmissions is null or num_readmissions <= 0;

-- 2. Chart data within the first 24h -- per-stay MEAN of each vital itemid.
drop materialized view if exists chart_data cascade;
create materialized view chart_data as
select * from crosstab(
    'with group_chart_data as (
        with chart_data as(
            select c.subject_id, c.hadm_id, c.stay_id, c.itemid, c.value, c.valuenum, c.charttime, i.intime, (c.charttime - i.intime) as timediff
            from chartevents c
                inner join icustays i on i.stay_id = c.stay_id
            )
        select subject_id, hadm_id, stay_id, itemid, value, valuenum, charttime, intime, timediff
        from chart_data
            where timediff > ''00:00:00'' and timediff <= ''24:00:00''    -- (intime, intime+24h]
            and value is not null
            and itemid in (''220739'',''223901'',''223900'', ''220045'', ''223761'', ''220277'', ''220210'')
    )
    select stay_id, itemid, avg(valuenum)
    from group_chart_data
    group by stay_id, itemid'
    ) as ct(stay_id integer, Heart_rate double precision, Respiration double precision, Saturation double precision, GLC_eye double precision,
            Temperature double precision, GLC_verbal double precision, GLC_motor double precision);

-- 3. Laboratory data within the first 24h -- per-stay MEAN. Joined on hadm_id
--    (labevents has no stay_id); window is relative to that stay's ICU intime.
drop materialized view if exists lab_data cascade;
create materialized view lab_data as
select * from crosstab(
    'with group_lab_data as(
        with lab_data as(
            select l.subject_id, l.hadm_id, i.stay_id, dli.label as itemlabel, l.itemid, l.value, l.valuenum, l.charttime, i.intime, (l.charttime - i.intime) as timediff
            from labevents l
                inner join icustays i on i.hadm_id = l.hadm_id
                inner join d_labitems dli on dli.itemid = l.itemid
            )
        select subject_id, stay_id, hadm_id, itemlabel, itemid, value, valuenum, charttime, intime, timediff
        from lab_data
            where timediff > ''00:00:00'' and timediff <= ''24:00:00''
            and value is not null
            and itemid in (''50868'',''50882'',''50902'', ''50912'', ''50931'', ''50960'', ''50971'', ''50983'', ''51006'', ''51221'', ''51222'',
                           ''51248'',''51249'',''51250'',''51265'',''51277'',''51279'',''51301'')
    )
    select stay_id, itemlabel, avg(valuenum)
    from group_lab_data
    group by stay_id, itemlabel
    order by 1',
    $$Values ('Anion Gap'), ('Bicarbonate'), ('Chloride'), ('Creatinine'), ('Glucose'),
    ('Magnesium'), ('Potassium'), ('Sodium'), ('Urea Nitrogen'), ('Hematocrit'),
    ('Hemoglobin'), ('MCH'), ('MCHC'), ('MCV'), ('Platelet Count'), ('RDW'),
    ('Red Blood Cells'), ('White Blood Cells')$$)
    as ct(stay_id integer, "Anion Gap" double precision, "Bicarbonate" double precision, "Chloride" double precision, "Creatinine" double precision,
            "Glucose" double precision, "Magnesium" double precision, "Potassium" double precision, "Sodium" double precision,
            "Urea Nitrogen" double precision, "Hematocrit" double precision, "Hemoglobin" double precision, "MCH" double precision,
            "MCHC" double precision, "MCV" double precision, "Platelet Count" double precision, "RDW" double precision,
            "Red Blood Cells" double precision, "White Blood Cells" double precision);

-- 4. Diagnosis data: top-2 diagnoses per stay (seq_num 1,2), keep icd_version.
drop materialized view if exists diagnosis_data cascade;
create materialized view diagnosis_data as
select * from crosstab(
    'select i.stay_id, icd_version, concat(''seq_num'', cast(di.seq_num as text)) as label, icd_code from diagnoses_icd di
        inner join icustays i on i.hadm_id = di.hadm_id
        where seq_num in (1,2)
        order by 1,2',
        $$Values ('seq_num1'::text), ('seq_num2')$$)
as ct(stay_id integer, icd_version integer, "Diagnosis_1" text, "Diagnosis_2" text);

-- 5. Final extraction: exclude death-in-ICU (deathtime after ICU outtime, or no
--    death), and keep LOS strictly > 1 day and <= 21 days. Requires complete
--    vitals + labs + diagnoses (INNER JOINs) -- the complete-case rule.
select * from (
  select i.los, i.first_careunit, a.admission_type, a.admission_location, d."Diagnosis_1", d."Diagnosis_2", d.icd_version, p.gender, p.anchor_age,
         cd.heart_rate, cd.respiration, cd.saturation, cd.temperature, cd.glc_eye, cd.glc_verbal, cd.glc_motor,
         ld."Anion Gap", ld."Bicarbonate", ld."Chloride", ld."Creatinine", ld."Glucose", ld."Magnesium", ld."Potassium",
         ld."Sodium", ld."Urea Nitrogen", ld."Hematocrit", ld."Hemoglobin", ld."MCH", ld."MCHC", ld."MCV", ld."Platelet Count", ld."RDW", ld."Red Blood Cells", ld."White Blood Cells"
  from icustays i
    inner join admissions a on a.hadm_id = i.hadm_id
    inner join no_readmissions nr on nr.stay_id = i.stay_id
    inner join chart_data cd on cd.stay_id = i.stay_id
    inner join lab_data ld on ld.stay_id = i.stay_id
    inner join patients p on p.subject_id = i.subject_id
    inner join diagnosis_data d on d.stay_id = i.stay_id
      where (a.deathtime - i.outtime) > '00:00:000' or a.deathtime is null
    ) as data where los > 1 and los <= 21;
