-- Final prequential and holdout metrics of every policy of one campaign.
-- roc_auc is macro one-vs-rest; pr_auc is the positive class of binary tasks (NULL for multiclass).
-- Usage: continual-bench aws query infra/athena/compare_policies.sql --param campaign=social-full-v1
SELECT policy,
       max(CASE WHEN partition = 'prequential' AND metric = 'accuracy' THEN value END) AS prequential_accuracy,
       max(CASE WHEN partition = 'prequential' AND metric = 'balanced_accuracy' THEN value END) AS prequential_balanced_accuracy,
       max(CASE WHEN partition = 'prequential' AND metric = 'macro_roc_auc' THEN value END) AS prequential_roc_auc,
       max(CASE WHEN partition = 'prequential' AND metric = 'pr_auc' THEN value END) AS prequential_pr_auc,
       max(CASE WHEN partition = 'prequential' AND metric = 'log_loss' THEN value END) AS prequential_log_loss,
       max(CASE WHEN partition = 'holdout' AND metric = 'accuracy' THEN value END) AS holdout_accuracy,
       max(CASE WHEN partition = 'holdout' AND metric = 'macro_roc_auc' THEN value END) AS holdout_roc_auc,
       max(CASE WHEN partition = 'holdout' AND metric = 'pr_auc' THEN value END) AS holdout_pr_auc,
       max(CASE WHEN partition = 'holdout' AND metric = 'log_loss' THEN value END) AS holdout_log_loss,
       max(CASE WHEN partition = 'prequential' AND metric = 'update_count' THEN value END) AS updates,
       max(CASE WHEN partition = 'prequential' AND metric = 'scenario_total_cost' THEN value END) AS scenario_cost_units,
       max(CASE WHEN partition = 'prequential' AND metric = 'aws_cost_total_usd' THEN value END) AS evaluator_aws_estimate_usd
FROM metrics
WHERE campaign_id = '${campaign}'
GROUP BY policy
ORDER BY policy;
