#!/usr/bin/env bash
# One-time (idempotent) bootstrap in us-east-1. Re-running converges to the same state.
#   1. persistent stack: KMS, data/artifacts/billing/access-log buckets, ECR repositories;
#   2. account defaults: Container Insights (enhanced) for new ECS clusters, service-linked roles;
#   3. CI/CD stack: CodeConnections (existing ARN or a new PENDING connection), CodeBuild, CodePipeline V2;
#   4. optional runtime overrides (activation flags, budget, alert email) uploaded for the deploy stage.
# The runtime stack is created by the first pipeline execution (push/merge to main or the
# manual "Release change"); bootstrap never downloads datasets or starts campaigns.
#
# Usage: REPOSITORY=owner/repo [CONNECTION_ARN=arn:...] [OVERRIDES=infra/params/runtime-overrides.json] \
#        scripts/aws/bootstrap.sh
set -euo pipefail
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
PROJECT="${PROJECT:-continual-bench}"
REPOSITORY="${REPOSITORY:?owner/repo of the GitHub source}"
BRANCH="${BRANCH:-main}"
cd "$(dirname "$0")/../.."
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
echo "account $ACCOUNT region $AWS_REGION project $PROJECT branch $BRANCH"

aws cloudformation deploy --stack-name "$PROJECT-persistent" --template-file infra/cloudformation/persistent.yaml \
  --parameter-overrides Project="$PROJECT" --tags cb-stack="$PROJECT" cb-component=persistent \
  --no-fail-on-empty-changeset
aws cloudformation update-termination-protection --enable-termination-protection --stack-name "$PROJECT-persistent" >/dev/null

# Account-wide default; affects ECS clusters created afterwards (Batch creates its own cluster).
aws ecs put-account-setting-default --name containerInsights --value enhanced >/dev/null
for service in batch.amazonaws.com ecs.amazonaws.com; do
  aws iam create-service-linked-role --aws-service-name "$service" >/dev/null 2>&1 || true
done

aws cloudformation deploy --stack-name "$PROJECT-cicd" --template-file infra/cloudformation/cicd.yaml \
  --capabilities CAPABILITY_NAMED_IAM --tags cb-stack="$PROJECT" cb-component=cicd --no-fail-on-empty-changeset \
  --parameter-overrides Project="$PROJECT" FullRepositoryId="$REPOSITORY" BranchName="$BRANCH" ConnectionArn="${CONNECTION_ARN:-}" \
  RuntimeStackName="$PROJECT-runtime"
CONNECTION="$(aws cloudformation describe-stacks --stack-name "$PROJECT-cicd" \
  --query "Stacks[0].Outputs[?OutputKey=='ConnectionArn'].OutputValue" --output text)"
STATUS="$(aws codeconnections get-connection --connection-arn "$CONNECTION" --query Connection.ConnectionStatus --output text)"
echo "connection $CONNECTION status $STATUS"
if [[ "$STATUS" != AVAILABLE ]]; then
  echo "Authorize it: Console > Developer Tools > Settings > Connections > $PROJECT-github > Update pending connection,"
  echo "install/choose the AWS Connector for GitHub app for $REPOSITORY, then re-run: scripts/aws/wait_connection.sh $CONNECTION"
fi

if [[ -n "${OVERRIDES:-}" ]]; then
  BUCKET="$(aws cloudformation list-exports --query "Exports[?Name=='$PROJECT-ArtifactsBucket'].Value" --output text)"
  MONITOR_ARN="${EXISTING_SERVICE_ANOMALY_MONITOR_ARN:-$(aws ce get-anomaly-monitors --query "AnomalyMonitors[?MonitorType=='DIMENSIONAL'&&MonitorDimension=='SERVICE'].MonitorArn | [0]" --output text 2>/dev/null || true)}"
  if [[ -n "$MONITOR_ARN" && "$MONITOR_ARN" != "None" ]]; then
    python3 -c "import json, sys; d = json.load(open('$OVERRIDES')); d.setdefault('ExistingServiceAnomalyMonitorArn', '$MONITOR_ARN'); print(json.dumps(d, indent=2))" > /tmp/runtime-parameters-bootstrap.json
    aws s3 cp /tmp/runtime-parameters-bootstrap.json "s3://$BUCKET/config/runtime-parameters.json"
    rm -f /tmp/runtime-parameters-bootstrap.json
  else
    aws s3 cp "$OVERRIDES" "s3://$BUCKET/config/runtime-parameters.json"
  fi
fi
echo "bootstrap complete; merge to main (or: aws codepipeline start-pipeline-execution --name $PROJECT-release)"
