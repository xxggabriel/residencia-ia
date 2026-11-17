-- EC2 and EBS unblended CUR 2.0 costs for a Spot benchmark campaign.
-- Database: continual_bench_billing. Requires activated cb-campaign and cb-run-id tags.
SELECT resource_tags['user_cb_run_id'] AS run_id,
       line_item_resource_id AS resource_id,
       line_item_usage_type AS usage_type,
       sum(line_item_unblended_cost) AS unblended_usd,
       min(line_item_usage_start_date) AS first_usage,
       max(line_item_usage_end_date) AS last_usage
FROM cur2
WHERE billing_period = '${period}'
  AND resource_tags['user_cb_campaign'] = '${campaign}'
  AND resource_tags['user_cb_component'] = 'benchmark-spot'
  AND line_item_product_code = 'AmazonEC2'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;
