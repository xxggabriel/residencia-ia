-- Total for the period to compare with Cost Explorer (same account, unblended, all line item types).
SELECT line_item_line_item_type, sum(line_item_unblended_cost) AS usd
FROM cur2 WHERE billing_period = '${period}' GROUP BY 1 ORDER BY usd DESC;
