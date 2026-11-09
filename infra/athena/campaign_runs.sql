-- The consolidated campaign row and the final manifest of each run.
SELECT c.campaign_id, c.release, c.snapshot_id, c.succeeded, c.total, c.complete, r.policy, r.run_id,
       r.final_uri, r.report_uri, r.predictions_uri, r.model_versions_count
FROM campaigns c JOIN runs r ON r.campaign_id = c.campaign_id
WHERE c.campaign_id = '${campaign}'
ORDER BY r.policy;
