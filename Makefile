.PHONY: test lint format install-hooks smoke compose-check up down health logs \
        bench run-experiment bench-paper tf-init tf-plan tf-apply tf-destroy \
        aws-validate aws-smoke aws-images aws-lambda \
        aws-bootstrap aws-dataset-smoke aws-dataset-full \
        aws-dataset-fraud-full aws-dataset-ecommerce-full \
        aws-campaign-smoke aws-campaign-full aws-campaign-fraud-full aws-campaign-ecommerce-full \
        aws-campaign-full-cost-aware aws-campaign-matrix \
        aws-mlflow-ui aws-studio-ui aws-cancel aws-teardown aws-teardown-all

REPOSITORY ?= xxggabriel/residencia-ia
BRANCH ?= $(shell git rev-parse --abbrev-ref HEAD)
OVERRIDES ?= infra/params/runtime-overrides.example.json

ENV ?= docker
CONFIG ?=
PROTOCOL ?= cara
DATASET ?= electricity
CAMPAIGN ?=
MODEL ?= logistic
POLICIES ?=
SEEDS ?= 42,43,44,45,46
SAVE_CHECKPOINTS ?= false
DAG ?= benchmark_campaign
WAIT ?= 1
TF_ENV ?= local
TF_DIR := infra/terraform/environments/$(TF_ENV)

test:
	MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE=1 uv run --frozen pytest -q -n 6 --dist=loadfile

lint:
	uv run ruff check src tests dags examples scripts

format:
	uv run ruff format src tests dags examples scripts
	uv run ruff check --fix src tests dags examples scripts

install-hooks:
	git config core.hooksPath .githooks
	chmod +x .githooks/*

smoke:
	uv run continual-bench smoke --state state/smoke --samples 600 --baselines

compose-check:
	docker compose --env-file .env.example config --quiet

up:
	docker compose up -d --build

down:
	docker compose down

health:
	docker compose ps

logs:
	docker compose logs -f --tail=100

bench run-experiment:
	uv run python scripts/run_experiment.py \
		--env "$(ENV)" \
		--config "$(CONFIG)" \
		--protocol "$(PROTOCOL)" \
		--dataset "$(DATASET)" \
		--campaign "$(CAMPAIGN)" \
		--model "$(MODEL)" \
		--policies "$(POLICIES)" \
		--seeds "$(SEEDS)" \
		--save-checkpoints "$(SAVE_CHECKPOINTS)" \
		--dag "$(DAG)" \
		--wait "$(WAIT)"

bench-paper:
	@if [ -z "$(CAMPAIGN)" ] && [ -z "$(CONFIG)" ]; then \
		echo "Uso: make bench-paper CAMPAIGN=cara-electricity [ENV=docker|local|aws]"; \
		exit 1; \
	fi
	$(MAKE) bench \
		ENV="$(ENV)" \
		CONFIG="$(if $(CONFIG),$(CONFIG),configs/campaigns/$(CAMPAIGN).yaml)" \
		CAMPAIGN="$(CAMPAIGN)" \
		SAVE_CHECKPOINTS="$(SAVE_CHECKPOINTS)" \
		DAG="$(DAG)" \
		WAIT="$(WAIT)"

tf-init:
	terraform -chdir=$(TF_DIR) init -backend=false

tf-plan:
	terraform -chdir=$(TF_DIR) plan

tf-apply:
	terraform -chdir=$(TF_DIR) apply -auto-approve

tf-destroy:
	terraform -chdir=$(TF_DIR) destroy -auto-approve

aws-validate:
	CFN_LINT="uvx --from cfn-lint==1.60.0 cfn-lint" uv run python scripts/aws/validate_templates.py
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
	scripts/aws/start_dataset.sh infra/examples/dataset-social-full.json $(SUFFIX)

aws-dataset-fraud-full:
	scripts/aws/start_dataset.sh infra/examples/dataset-fraud-full.json $(SUFFIX)

aws-dataset-ecommerce-full:
	scripts/aws/start_dataset.sh infra/examples/dataset-ecommerce-full.json $(SUFFIX)

aws-campaign-smoke:
	scripts/aws/start_campaign.sh infra/examples/campaign-social-smoke.json $(SUFFIX)

aws-campaign-full:
	scripts/aws/start_campaign.sh infra/examples/campaign-social-full.json $(SUFFIX)

aws-campaign-fraud-full:
	scripts/aws/start_campaign.sh infra/examples/campaign-fraud-full.json $(SUFFIX)

aws-campaign-ecommerce-full:
	scripts/aws/start_campaign.sh infra/examples/campaign-ecommerce-full.json $(SUFFIX)

aws-campaign-full-cost-aware:
	scripts/aws/start_campaign.sh infra/examples/campaign-social-full-cost-aware.json $(SUFFIX)

aws-campaign-matrix:
	scripts/aws/start_campaign.sh infra/examples/campaign-binary-matrix.json

aws-mlflow-ui:
	uv run continual-bench aws mlflow-url

aws-studio-ui:
	uv run continual-bench aws studio-url

aws-cancel:
	@if [ -z "$(CAMPAIGN)" ]; then echo "Uso: make aws-cancel CAMPAIGN=<id>"; exit 1; fi
	scripts/aws/cancel_campaign.sh $(CAMPAIGN)

aws-teardown:
	scripts/aws/teardown.sh

aws-teardown-all:
	scripts/aws/teardown.sh --with-cicd

