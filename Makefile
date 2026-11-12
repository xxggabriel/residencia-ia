.PHONY: test lint smoke compose-check aws-validate aws-smoke aws-images aws-lambda \
        aws-bootstrap aws-dataset-smoke aws-dataset-full \
        aws-campaign-smoke aws-campaign-full aws-campaign-matrix \
        aws-cancel aws-teardown aws-teardown-all help

REPOSITORY ?= xxggabriel/residencia-ia
OVERRIDES ?= infra/params/runtime-overrides.example.json

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

aws-bootstrap:
	REPOSITORY=$(REPOSITORY) OVERRIDES=$(OVERRIDES) scripts/aws/bootstrap.sh

aws-dataset-smoke:
	scripts/aws/start_dataset.sh infra/examples/dataset-social-smoke.json

aws-dataset-full:
	scripts/aws/start_dataset.sh infra/examples/dataset-social-full.json

aws-campaign-smoke:
	scripts/aws/start_campaign.sh infra/examples/campaign-social-smoke.json

aws-campaign-full:
	scripts/aws/start_campaign.sh infra/examples/campaign-social-full.json

aws-campaign-matrix:
	scripts/aws/start_campaign.sh infra/examples/campaign-binary-matrix.json

aws-cancel:
	@if [ -z "$(CAMPAIGN)" ]; then echo "Uso: make aws-cancel CAMPAIGN=<id>"; exit 1; fi
	scripts/aws/cancel_campaign.sh $(CAMPAIGN)

aws-teardown:
	scripts/aws/teardown.sh

aws-teardown-all:
	scripts/aws/teardown.sh --with-cicd

