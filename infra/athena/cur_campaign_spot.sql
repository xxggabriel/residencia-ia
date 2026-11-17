-- Fargate Spot unblended CUR 2.0 costs for a benchmark campaign (vCPU, memory and ephemeral storage).
-- Database: continual_bench_billing. Batch propagates the job tags to each Fargate task; activate
-- cb-campaign, cb-run-id and cb-component as cost allocation tags.
SELECT resource_tags['user_cb_run_id'] AS run_id,
       line_item_resource_id AS task_arn,
       line_item_usage_type AS usage_type,
       sum(line_item_usage_amount) AS usage_amount,
       sum(line_item_unblended_cost) AS unblended_usd,
       min(line_item_usage_start_date) AS first_usage,
       max(line_item_usage_end_date) AS last_usage
FROM cur2
WHERE billing_period = '${period}'
  AND resource_tags['user_cb_campaign'] = '${campaign}'
  AND resource_tags['user_cb_component'] = 'benchmark'
  AND line_item_product_code = 'AmazonECS'
  AND line_item_usage_type LIKE '%Fargate%'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;
