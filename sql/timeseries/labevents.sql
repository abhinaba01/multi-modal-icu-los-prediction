-- Lab results of the requested ICU stays with intime < charttime <= intime + :hours.
-- labevents has no stay_id, so rows are matched on subject_id and time (Stage 2).
-- The storetime and plausibility rules are applied in Python.
SELECT i.stay_id, e.itemid, e.charttime, e.storetime, e.valuenum AS value
FROM {hosp}.labevents AS e
JOIN {icu}.icustays AS i ON i.subject_id = e.subject_id
WHERE i.stay_id = ANY(CAST(:stay_ids AS integer[]))
  AND e.itemid = ANY(CAST(:itemids AS integer[]))
  AND e.charttime > i.intime
  AND e.charttime <= i.intime + CAST(:hours AS integer) * INTERVAL '1 hour'
ORDER BY i.stay_id, e.charttime, e.itemid, e.storetime, e.valuenum
