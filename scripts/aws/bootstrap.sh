#!/usr/bin/env bash
# One-time (idempotent) bootstrap in us-east-1. Re-running converges to the same state.
#   1. persistent stack: KMS, data/artifacts/billing/access-log buckets, ECR repositories;
#   2. account defaults: Container Insights (enhanced) for new ECS clusters, service-linked roles;
#   3. CI/CD stack: GitHub OIDC provider + deploy role for GitHub Actions (sets the AWS_ROLE_ARN repo variable via gh);
#   4. optional runtime overrides (activation flags, budget, alert email) uploaded for the deploy stage.
# The runtime stack is created by the first deploy workflow run (push/merge to main or the
# manual "Release change"); bootstrap never downloads datasets or starts campaigns.
#
# Usage: REPOSITORY=owner/repo [OIDC_PROVIDER_ARN=arn:...] [OVERRIDES=infra/params/runtime-overrides.json] \
#        scripts/aws/bootstrap.sh
set -euo pipefail
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
PROJECT="${PROJECT:-continual-bench}"
REPOSITORY="${REPOSITORY:?owner/repo of the GitHub source}"
cd "$(dirname "$0")/../.."
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
echo "account $ACCOUNT region $AWS_REGION project $PROJECT"

aws cloudformation deploy --stack-name "$PROJECT-persistent" --template-file infra/cloudformation/persistent.yaml \
  --parameter-overrides Project="$PROJECT" --tags cb-stack="$PROJECT" cb-component=persistent \
  --no-fail-on-empty-changeset
aws cloudformation update-termination-protection --enable-termination-protection --stack-name "$PROJECT-persistent" >/dev/null

# Account-wide default; affects ECS clusters created afterwards (Batch creates its own cluster).
aws ecs put-account-setting-default --name containerInsights --value enhanced >/dev/null
for service in batch.amazonaws.com ecs.amazonaws.com; do
  aws iam create-service-linked-role --aws-service-name "$service" >/dev/null 2>&1 || true
done

# Repositories with immutable OIDC subjects (repo:owner@id/name@id) need the exact prefix in the trust policy.
SUBJECT_PREFIX=""
if command -v gh >/dev/null 2>&1; then
  SUBJECT_PREFIX="$(gh api "repos/$REPOSITORY/actions/oidc/customization/sub" --jq 'if .use_immutable_subject then .sub_claim_prefix else "" end' 2>/dev/null || true)"
fi
aws cloudformation deploy --stack-name "$PROJECT-cicd" --template-file infra/cloudformation/cicd.yaml \
  --capabilities CAPABILITY_NAMED_IAM --tags cb-stack="$PROJECT" cb-component=cicd --no-fail-on-empty-changeset \
  --parameter-overrides Project="$PROJECT" FullRepositoryId="$REPOSITORY" \
  OidcSubjectPrefix="$SUBJECT_PREFIX" ExistingOidcProviderArn="${OIDC_PROVIDER_ARN:-}" RuntimeStackName="$PROJECT-runtime"
ROLE_ARN="$(aws cloudformation describe-stacks --stack-name "$PROJECT-cicd" \
  --query "Stacks[0].Outputs[?OutputKey=='GitHubActionsRoleArn'].OutputValue" --output text)"
echo "GitHub Actions role $ROLE_ARN"
if command -v gh >/dev/null 2>&1; then
  gh variable set AWS_ROLE_ARN --repo "$REPOSITORY" --body "$ROLE_ARN"
fi

if [[ -n "${OVERRIDES:-}" ]]; then
  BUCKET="$(aws cloudformation list-exports --query "Exports[?Name=='$PROJECT-ArtifactsBucket'].Value" --output text)"
  aws s3 cp "$OVERRIDES" "s3://$BUCKET/config/runtime-parameters.json"
fi
echo "bootstrap complete; merge to main (or: gh workflow run deploy.yml)"
