-- Output events (urine) of the requested ICU stays with intime < charttime <= intime + :hours.
-- The storetime and plausibility rules are applied in Python.
SELECT e.stay_id, e.itemid, e.charttime, e.storetime, e.value AS value
FROM {icu}.outputevents AS e
JOIN {icu}.icustays AS i ON i.stay_id = e.stay_id
WHERE e.stay_id = ANY(CAST(:stay_ids AS integer[]))
  AND e.itemid = ANY(CAST(:itemids AS integer[]))
  AND e.charttime > i.intime
  AND e.charttime <= i.intime + CAST(:hours AS integer) * INTERVAL '1 hour'
ORDER BY e.stay_id, e.charttime, e.itemid, e.storetime, e.value
