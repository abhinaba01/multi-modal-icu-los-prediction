-- mimic-code's norepinephrine-equivalent dose intervals that overlap (intime, intime + :hours].
-- The derived table has no storetime; intervals are clipped at t0 in Python.
SELECT e.stay_id, e.starttime, e.endtime, e.norepinephrine_equivalent_dose AS value
FROM {derived}.norepinephrine_equivalent_dose AS e
JOIN {icu}.icustays AS i ON i.stay_id = e.stay_id
WHERE e.stay_id = ANY(CAST(:stay_ids AS integer[]))
  AND e.starttime < i.intime + CAST(:hours AS integer) * INTERVAL '1 hour'
  AND e.endtime > i.intime
ORDER BY e.stay_id, e.starttime, e.endtime, e.norepinephrine_equivalent_dose
