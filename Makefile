.PHONY: test lint smoke compose-check
test:
	uv run pytest -q
lint:
	uv run ruff check src tests dags examples scripts
smoke:
	uv run continual-bench smoke --state state/smoke --samples 600 --baselines
compose-check:
	docker compose --env-file .env.example config --quiet
