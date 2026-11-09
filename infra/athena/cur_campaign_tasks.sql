-- Observed CUR 2.0 cost of the campaign's Fargate tasks (split cost allocation data).
-- Database: continual_bench_billing. Requires the cb-campaign/cb-run-id cost allocation tags
-- activated in the payer account and split cost allocation data enabled for ECS.
SELECT resource_tags['user_cb_run_id'] AS run_id,
       line_item_resource_id AS task,
       sum(split_line_item_split_cost) AS split_cost_usd,
       sum(split_line_item_unused_cost) AS unused_cost_usd,
       sum(line_item_unblended_cost) AS unblended_usd,
       min(line_item_usage_start_date) AS first_usage, max(line_item_usage_end_date) AS last_usage
FROM cur2
WHERE billing_period = '${period}'
  AND resource_tags['user_cb_campaign'] = '${campaign}'
  AND line_item_product_code = 'AmazonECS'
GROUP BY 1, 2
ORDER BY 1;
