#!/usr/bin/env bash
# Print one runtime stack output (e.g. BenchmarkStateMachineArn).
aws cloudformation describe-stacks --stack-name "${RUNTIME_STACK:-continual-bench-runtime}" --region us-east-1 \
  --query "Stacks[0].Outputs[?OutputKey=='${1:?output key}'].OutputValue" --output text
