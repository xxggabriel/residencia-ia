-- Per-attempt timing and estimated USD (reserved Fargate resources × billed seconds).
SELECT a.run_id, a.job_id, a.status, a.exit_code, a.task_arn, c.queue_seconds, c.seconds, c.billed_seconds,
       c.vcpu_usd, c.memory_usd, c.ephemeral_storage_usd, c.total_usd, c.price_snapshot
FROM attempts a LEFT JOIN cost_estimates c ON c.job_id = a.job_id
WHERE a.campaign_id = '${campaign}'
ORDER BY a.run_id, a.created_at;
