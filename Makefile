.PHONY: test lint smoke compose-check aws-validate aws-smoke aws-images aws-lambda
test:
	uv run pytest -q
lint:
	uv run ruff check src tests dags examples scripts
smoke:
	uv run continual-bench smoke --state state/smoke --samples 600 --baselines
compose-check:
	docker compose --env-file .env.example config --quiet
aws-validate:
	CFN_LINT="uvx --from cfn-lint==1.38.0 cfn-lint" uv run python scripts/aws/validate_templates.py
	uv run python scripts/aws/validate_asl.py infra/statemachines/benchmark.asl.json infra/statemachines/dataset.asl.json
aws-smoke:
	uv run continual-bench aws local-smoke --root state/aws-local-smoke
aws-images:
	docker buildx build --platform linux/amd64 -f docker/aws/Dockerfile --target runtime -t continual-bench-runtime:local --load .
	docker buildx build --platform linux/amd64 -f docker/aws/Dockerfile --target ml -t continual-bench-ml:local --load .
	docker run --rm --platform linux/amd64 continual-bench-runtime:local
	bash scripts/aws/check_inference_image.sh continual-bench-ml:local
aws-lambda:
	uv run python scripts/aws/build_control_lambda.py local --output build/control.zip --build-output build/build-output.json
