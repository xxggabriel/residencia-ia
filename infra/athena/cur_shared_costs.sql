-- Shared costs of the period by service (CI/CD, network endpoints, S3, CloudWatch, streaming, Feature Store).
SELECT line_item_product_code AS service, line_item_usage_type AS usage_type,
       sum(line_item_unblended_cost) AS usd, sum(line_item_usage_amount) AS usage
FROM cur2
WHERE billing_period = '${period}'
  AND resource_tags['user_cb_stack'] = 'continual-bench'
  AND line_item_product_code IN ('AWSCodePipeline', 'CodeBuild', 'AmazonECR', 'AmazonVPC', 'AmazonS3',
      'AmazonCloudWatch', 'AWSLambda', 'AWSStepFunctions', 'AmazonKinesis', 'AmazonKinesisFirehose',
      'AmazonKinesisAnalytics', 'AmazonSageMaker', 'AWSGlue', 'AmazonAthena', 'AmazonSNS')
GROUP BY 1, 2
ORDER BY usd DESC;
