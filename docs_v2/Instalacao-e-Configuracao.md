# Instalação e Configuração

Esta página explica como instalar o pacote `continual-bench`, quais dependências e grupos opcionais (*extras*) existem, como os arquivos de trava de versões (*lockfiles*) são usados, quais variáveis de ambiente o código lê e como os arquivos de `configs/` se organizam e herdam valores.

## Requisitos

| Requisito | Valor | Fonte |
|---|---|---|
| Python | `>=3.12,<3.13` (somente 3.12) | `pyproject.toml`, `requires-python` |
| Gerenciador de ambiente | [uv](https://docs.astral.sh/uv/) | `Makefile` e `README.md` usam `uv run` e `uv sync` |
| Sistema operacional | POSIX (Linux ou macOS) | `infrastructure/storage.py` usa `fcntl.flock` para os *locks*, disponível apenas em sistemas POSIX |
| Docker e Docker Compose | Apenas para o perfil com serviços | `compose.yaml`, `Dockerfile` |

A imagem do worker usa `python:3.12.13-slim-bookworm` (`Dockerfile`). O Airflow roda em um ambiente Python separado, na imagem `apache/airflow:3.3.2-python3.12` (`docker/airflow/Dockerfile`), sem instalar o pacote completo: apenas `dags/`, `application/project_suite.py` e `configs/base/tracking.yaml` são copiados.

## Instalação

Na raiz do repositório:

```bash
uv sync --frozen --extra dev --extra services --extra datasets --extra large-datasets
```

`--frozen` instala exatamente as versões de `uv.lock`, sem resolver dependências novamente. Para apenas executar o núcleo e os testes rápidos, `--extra dev` basta; os demais extras são necessários para serviços, datasets de imagem/Parquet e XGBoost/DuckDB. Depois da instalação, os comandos rodam com `uv run --frozen continual-bench ...`.

### Dependências principais

O pacote base declara quatro dependências fixadas: `river==0.26.1`, `scikit-learn==1.9.1`, `pydantic==2.12.5` e `PyYAML==6.0.3`. O *build backend* é `hatchling==1.29.0`.

### Extras

| Extra | Pacotes (versões em `pyproject.toml`) | Uso |
|---|---|---|
| `dev` | `pytest==9.1.1`, `pytest-xdist>=3.8.0`, `ruff==0.15.7` | Testes (inclusive paralelos com `-n`) e lint |
| `services` | `zenml==0.98.0`, `mlflow-skinny==3.17.2`, `boto3`, `confluent-kafka==2.16.0`, `s3fs`, `pandas`, `matplotlib`, `psutil` | Integração com ZenML, MLflow, Kafka e MinIO/S3 |
| `datasets` | `Pillow==13.0.0`, `pyarrow==26.0.1` | Loaders nativos de imagens e Parquet |
| `large-datasets` | `duckdb==1.5.4`, `xgboost-cpu==3.2.0` (Linux x86_64) ou `xgboost==3.2.0` (demais plataformas) | Fluxos Zia e avaliação UPF de referência |
| `plots` | `matplotlib==3.11.2` | Figuras detalhadas registradas com `mlflow.log_figure()` |
| `aws` | `boto3`, `pandas`, `psutil` | Perfil AWS gerenciado, sem ZenML, Kafka, MinIO nem Airflow |
| `aws-mlflow` | `boto3`, `pandas`, `mlflow-skinny`, `sagemaker-mlflow==0.5.0`, `matplotlib`, `psutil` | MLflow gerenciado do SageMaker; sem ele os workers usam o cliente REST/SigV4 |
| `replication` | nenhum | Mantido vazio por compatibilidade com comandos antigos |

O wheel inclui quatro contratos de `configs/contracts/` como dados do pacote (`metric-definitions.json`, `image-datasets.lock.json`, `real-streams.lock.json`, `temporal-vision.lock.json`), por meio de `[tool.hatch.build.targets.wheel.force-include]`.

## Lockfiles

| Arquivo | Conteúdo | Quem usa |
|---|---|---|
| `uv.lock` | Resolução completa de todas as dependências e extras | `uv sync --frozen`, `uv run --frozen`, `make test` |
| `requirements.lock` | Exportação com *hashes* dos extras `dev`, `services`, `datasets` e `large-datasets` (comando no cabeçalho do arquivo) | `Dockerfile` (`pip install --require-hashes`) e job `test` do CI |
| `requirements-aws.lock` | Exportação com *hashes* dos extras `aws`, `aws-mlflow`, `datasets`, `large-datasets` e `plots` | `docker/aws/Dockerfile` |
| `infra/ci-requirements.lock` | `cfn-lint==1.60.0` e dependências | Validação de templates CloudFormation no CI |

`requirements.lock` está defasado em relação a `uv.lock`: reexportar com o comando registrado no cabeçalho acrescenta `pytest-xdist==3.8.0` e `execnet==2.1.2`. Consequência prática: ambientes instalados a partir de `requirements.lock` (imagem do worker e CI) não têm o plugin de paralelismo do pytest, e `make test` (que usa `-n 6`) só funciona no ambiente criado por `uv sync`. O CI não usa `-n` e não é afetado.

Para regenerar o lock de requisitos após mudar dependências, use o comando exato do cabeçalho:

```bash
uv export --frozen --extra dev --extra services --extra datasets --extra large-datasets \
  --no-emit-project --format requirements-txt --output-file requirements.lock
```

## Variáveis de ambiente

### Arquivo de ambiente do Docker Compose

`.env.example` é apenas um modelo, com valores `CHANGE_ME`. O arquivo real deve ficar fora do Git (`.gitignore` ignora `.env` e `.env.*`, exceto `.env.example`) e é passado com `--env-file`. O Compose interrompe com erro se qualquer uma delas faltar (`${VAR:?required}`).

| Variável | Propósito em `compose.yaml` |
|---|---|
| `WEB_BIND_IP` | Endereço privado onde as interfaces web são publicadas; também entra em `--allowed-hosts` e `--cors-allowed-origins` do MLflow |
| `WEB_USER` | Usuário administrativo do Airflow, ZenML e MLflow; também é `MLFLOW_TRACKING_USERNAME` dos workers |
| `WEB_PASSWORD` | Senha do Airflow e do ZenML |
| `MLFLOW_PASSWORD` | Senha exclusiva do administrador do MLflow; o código exige pelo menos 12 caracteres |
| `MINIO_USER`, `MINIO_PASSWORD` | Credenciais raiz do MinIO, repassadas como `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` aos workers e ao MLflow |
| `POSTGRES_PASSWORD` | Senha do banco do Airflow; deve ser segura para URL (sem `@:/?#%`) |
| `AIRFLOW_FERNET_KEY` | Chave Fernet do Airflow; deve decodificar para exatamente 32 bytes |
| `AIRFLOW_JWT_SECRET` | Segredo JWT da API do Airflow |
| `MLFLOW_FLASK_SECRET` | Chave secreta do servidor Flask do MLflow |

`scripts/preflight.py` valida essas regras a partir das variáveis do processo, sem imprimir valores. O script também aceita apenas um endereço privado específico, fixado no código, para `WEB_BIND_IP`; em outra implantação ele precisa ser ajustado.

### Variáveis lidas pelo código (perfil local e Compose)

| Variável | Padrão | Onde é lida | Efeito |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | nenhum | `application/readiness.py`, `infrastructure/services.py`, `cli.py` | Servidor MLflow; no `iterative-worker`, ativa a publicação |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | nenhum | `application/readiness.py` | Credenciais exigidas por `--services` (senha com 12 ou mais caracteres) |
| `KAFKA_BOOTSTRAP_SERVERS` | nenhum | `infrastructure/services.py` | Endereço do Kafka no modo `streaming` com `--services` |
| `S3_ENDPOINT_URL` | nenhum | `infrastructure/services.py`, `scripts/bootstrap_zenml.py` | Endpoint S3 (MinIO) |
| `BENCH_BUCKET` | `benchmarks` | `infrastructure/services.py` | Bucket de publicação dos relatórios |
| `ZENML_STORE_URL` | nenhum | `orchestration/zenml_pipeline.py` | Servidor ZenML; obrigatório (HTTP) com `--zenml --services` |
| `BENCH_ZENML_PROJECT`, `BENCH_ZENML_STACK`, `BENCH_ZENML_ARTIFACT_STORE` | `continual-benchmark`, `benchmark-local-v1`, nenhum | `orchestration/zenml_pipeline.py` | Projeto, stack e armazenamento de artefatos do ZenML |
| `ZENML_CONFIG_PATH` | nenhum | `orchestration/server.py` | Copiado para um diretório isolado por job |
| `BENCH_STATE` | `/state/runs` | `orchestration/server.py` e outros | Raiz de estado do servidor de jobs |
| `BENCH_CONCURRENCY` | `1` | `orchestration/server.py` | Jobs simultâneos por servidor |
| `BENCH_WORKER_URLS`, `BENCH_SHARED_TRACKING` | nenhum, `false` | `orchestration/worker_pool.py` | Backends do distribuidor e verificação de prontidão compartilhada |
| `BENCH_DATASETS` | `/state/datasets` | `datasets/openml.py`, `datasets/real_streams.py`, `datasets/pinned_streams.py` | Diretório de arquivos de datasets reais |
| `BENCH_EPOCH_ROOT`, `BENCH_DATASET_SMOKES`, `BENCH_DATASET_CACHE`, `BENCH_DATASET_DOWNLOAD` | ver código | `orchestration/epoch_server.py` | Raízes e permissão de download do servidor por épocas |
| `BENCH_GIT_REVISION` | `uncommitted` | `evaluation/environment.py`, `cli.py` | Revisão registrada na proveniência |
| `BENCH_EXECUTION_ENV` | detectado | `evaluation/environment.py` | Rótulo do ambiente de execução no relatório |
| `PYTHONHASHSEED` | `random` | `evaluation/environment.py` | Registrado e incluído no contrato de retomada; o Compose fixa `0` |
| `BENCH_SYSTEM_METRICS_INTERVAL` | `5.0` segundos | `evaluation/system_metrics.py` | Intervalo do amostrador de CPU e memória |
| `BENCH_MLFLOW_PUBLISH_CONCURRENCY` | `4` (`DEFAULT_CONCURRENCY`) | `infrastructure/tracking/backends.py` | Paralelismo das publicações no MLflow |
| `MLFLOW_ENABLE_ASYNC_LOGGING` | desligado | `infrastructure/tracking/mlflow_async.py` | Ativa o registro assíncrono do MLflow |
| `BENCH_VCPUS`, `BENCH_BENCHMARK_VCPUS`, `OMP_NUM_THREADS` | `OMP_NUM_THREADS=2` se não houver outro valor | `models/training/loop.py` | Define o número de *threads* numéricas e sobrescreve `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` e `MKL_NUM_THREADS`; o Compose e o `Dockerfile` fixam 1 |
| `BENCH_POLICY_DATASET_PARALLELISM` | `1` (entre 1 e 8) | `dags/benchmark_pipelines.py` | Quantos datasets a DAG `benchmark_policies` processa em paralelo |

As variáveis específicas do perfil AWS (`BENCH_KMS_KEY_ID`, `BENCH_JOB_QUEUE`, `BENCH_CAMPAIGN_ID` e outras) estão em [Execução na AWS](Execucao-na-AWS).

## Configurações

Os arquivos de `configs/` são YAML (ou JSON) validados por esquemas Pydantic. Os esquemas recusam campos desconhecidos.

| Pasta | Arquivos atuais | Esquema e comando |
|---|---|---|
| `base/` | `run.yaml` (padrões do motor binário), `smoke.yaml` (caso curto), `tracking.yaml` (perfil de métricas), `tracking-fargate.yaml`, `tracking-corrective.yaml`, `tracking-dataset-suite.yaml` | Bases herdadas pelos demais |
| `runs/` | `streaming.yaml`, `incremental.yaml`, `offline.yaml`, `metrics.yaml` | `Config`, `continual-bench run` |
| `typed/` | `friedman-smoke.yaml` | `TypedConfig`, `continual-bench typed` |
| `campaigns/` | `task-smoke.yaml` (seis casos, um por família de política) | `Campaign`, `continual-bench campaign` |
| `matrices/` | `base.yaml` (sementes 42, 43, 44 e `on_incompatible: skip`), `binary-comparison.yaml`, `any-model.yaml`, `insects-72.yaml`, `insects-pilot.yaml`, `insects-corrected.yaml`, `corrective-smoke.yaml`, `studio-smoke.yaml` | `ExperimentMatrix`, `continual-bench matrix` |
| `contracts/` | Locks de datasets (`real-sources.lock.json`, `real-streams.lock.json`, `image-datasets.lock.json`, `temporal-vision.lock.json`), `metric-definitions.json`, `dataset-catalog.json`, `aws-prices.us-east-1.json` | Contratos versionados; mudanças alteram *fingerprints* e o wheel |
| `history/` | `campaign-controlled-v3.json`, `campaign-versioned-v4.json`, `campaign-nonlinear-v5.json` | Manifestos congelados de campanhas executadas; não devem ser editados |

`configs/base/run.yaml` define, entre outros, `dataset.samples: 1200`, `seed: 42`, `label_delay: 3`, política `none` com `action: reset_retrain`, `warmup: 100` e `checkpoint_every: 100`. Modelo e detector, quando omitidos, usam os padrões do esquema `Config`: `logistic` e `adwin`.

### Herança com `extends`

Implementada em `src/continual_bench/config/loader.py`. Qualquer mapeamento pode declarar `extends` com um caminho, ou uma lista de caminhos, relativo ao arquivo que o declara:

- as bases são aplicadas em ordem e as chaves do próprio mapeamento sobrescrevem as herdadas;
- mapeamentos são mesclados recursivamente;
- listas e escalares substituem o valor herdado por inteiro;
- `null` explícito sobrescreve com nulo;
- `extends` funciona em qualquer nível, inclusive dentro de seções (`tracking: {extends: ../base/tracking.yaml}`) e de itens de lista (casos de uma campanha);
- ciclos, arquivos ausentes e `extends` mal formado falham antes da validação.

```yaml
# configs/runs/streaming.yaml
extends: ../base/run.yaml
name: sea-drift
mode: streaming
dataset: {name: sea}
model: {name: logistic, params: {learning_rate: 0.05}}
detector: {name: adwin, params: {delta: 0.002}}
policy: {name: drift}
```

A herança só é resolvida ao ler arquivos locais (`load_document` e `load_model`). Os esquemas recusam a chave `extends`, então os corpos HTTP enviados aos workers precisam ser autocontidos.

Para ver a configuração resolvida:

```bash
uv run --frozen python -c "import json, sys; from continual_bench.config import load_document; print(json.dumps(load_document(sys.argv[1]), indent=2))" configs/runs/streaming.yaml
```

### Identidade

`Config.identity()` é o SHA-256 (24 primeiros caracteres hexadecimais) do JSON ordenado da configuração validada. Por isso, alterar um valor numa base muda a identidade de todos os arquivos que o herdam sem sobrescrevê-lo. `tests/test_config_files.py` fixa as identidades dos exemplos versionados e falha se uma mudança em `configs/base/` alterá-las sem intenção. Campos opcionais novos usam `exclude_if` para manter as identidades históricas inalteradas quando estão no valor padrão (`config/benchmark.py`).

### Limites validados pelo esquema `Config`

| Campo | Limite |
|---|---|
| `dataset.samples` | 100 a 100.000.000 |
| `dataset.label_delay` | 0 a 10.000 |
| `replay_window`, `metric_window`, `checkpoint_every` | 1 a 10.000 |
| `batch_size` | 1 a 1000 |
| `max_seconds` | 10 a 86.400 |
| `name` | 1 a 64 caracteres `[A-Za-z0-9_-]` |

Além do esquema, `application/runner.py` recusa materializações de eventos acima de 512 MiB, e o servidor HTTP aceita corpos de 1 a 65.536 bytes.

## Estado das capacidades

| Capacidade | Classificação | Evidência |
|---|---|---|
| Instalação reprodutível por `uv.lock` | Implementado | `uv run --frozen` usado no `Makefile` |
| Herança `extends` | Validado | `tests/test_config_files.py` |
| Consistência entre `uv.lock` e `requirements.lock` | Pendente | `requirements.lock` sem `pytest-xdist` e `execnet` |
| Validação do arquivo de ambiente | Implementado | `scripts/preflight.py`, restrito a um endereço fixo |
