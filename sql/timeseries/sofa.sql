-- mimic-code's hourly SOFA rows with intime < endtime <= intime + :hours (Stage 2: endtime <= t0).
-- The per-hour component scores are used, not the rolling *_24hours columns.
SELECT e.stay_id, e.endtime, e.respiration, e.coagulation, e.liver, e.cardiovascular, e.cns, e.renal
FROM {derived}.sofa AS e
JOIN {icu}.icustays AS i ON i.stay_id = e.stay_id
WHERE e.stay_id = ANY(CAST(:stay_ids AS integer[]))
  AND e.endtime > i.intime
  AND e.endtime <= i.intime + CAST(:hours AS integer) * INTERVAL '1 hour'
ORDER BY e.stay_id, e.endtime
