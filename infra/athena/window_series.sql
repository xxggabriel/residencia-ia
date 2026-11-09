-- Window series by step (= end_seq), prequential vs holdout kept apart.
SELECT policy, partition, step, start_time_ms, end_time_ms, value AS accuracy
FROM windows
WHERE campaign_id = '${campaign}' AND metric = 'accuracy'
ORDER BY policy, partition, step;
