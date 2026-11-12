# continual-bench

Laboratório reprodutível e limitado para **aprendizado contínuo com mudança de conceito (drift)**.
Compara modelos, detectores de drift e políticas de retreinamento sobre fluxos ordenados no tempo,
com protocolos de avaliação explícitos, rastreamento de experimentos e execução local, em homelab
(Docker Compose) ou na AWS.

> **Escopo.** Os testes rápidos e os *smokes* validam algoritmo, protocolo e integração, não desempenho.
> Nenhum resultado aqui é alegação de estado da arte. Veja os limites em [docs/reliability.md](docs/reliability.md).

## Sumário

- [Visão geral](#visão-geral)
- [Instalação](#instalação)
- [Uso rápido](#uso-rápido)
- [Protocolos de avaliação](#protocolos-de-avaliação)
- [Componentes e extensão](#componentes-e-extensão)
- [Avaliação iterativa (Semana 09)](#avaliação-iterativa-semana-09)
- [Conjuntos de dados](#conjuntos-de-dados)
- [Execução em serviços reais (homelab)](#execução-em-serviços-reais-homelab)
- [Perfil AWS](#perfil-aws)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Testes e qualidade](#testes-e-qualidade)
- [Documentação](#documentação)

## Visão geral

| Camada | Tecnologia | Papel |
|---|---|---|
| Núcleo | Python 3.12, River | Ciclo por amostra/lote, políticas, métricas |
| Orquestração | Apache Airflow | Tarefas finitas, manuais e parametrizadas (`schedule=None`) |
| Etapas e artefatos | ZenML (projeto independente) | Quatro etapas com artefatos versionados |
| Eventos | Kafka | Transporte do fluxo (somente com `--services`) |
| Armazenamento | MinIO / S3 | Dados, pontos de recuperação e artefatos |
| Rastreamento | MLflow | Parâmetros, métricas e artefatos por execução |

Somente o núcleo Python controla o ciclo contínuo; Airflow e ZenML não controlam o fluxo de dados.

## Instalação

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/). As dependências estão fixadas em `uv.lock`
(exportadas com hashes em `requirements.lock`).

```bash
uv sync --frozen --extra dev --extra services --extra datasets
```

| Extra | Conteúdo |
|---|---|
| `dev` | pytest, ruff |
| `services` | ZenML, MLflow, boto3, Kafka, s3fs, pandas |
| `datasets` | Pillow, pyarrow (loaders nativos de imagens e Parquet) |
| `large-datasets` | DuckDB, XGBoost |
| `aws` | boto3, pandas (perfil gerenciado, lock próprio `requirements-aws.lock`) |

O Airflow usa um ambiente Python separado.

## Uso rápido

```bash
PYTHONPATH=. uv run pytest -q
uv run continual-bench run configs/runs/streaming.yaml
uv run continual-bench run configs/runs/incremental.yaml
uv run continual-bench run configs/runs/offline.yaml
uv run continual-bench smoke --state state/smoke --samples 600 --baselines
```

O `smoke` executa **21 experimentos limitados** (3 fontes × 2 protocolos prequential × 3 modelos de
referência, mais 3 avaliações com separação temporal). O modo local lê JSONL, sem Kafka
(`transport: local_jsonl`).

Comandos da CLI (`continual-bench <comando>`):

| Comando | Função |
|---|---|
| `run` | Uma configuração (streaming, incremental ou offline) |
| `smoke` | Conjunto rápido de verificação |
| `matrix` | Expande/executa datasets × modelos × estratégias × modos × sementes (`--dry-run` lista o plano) |
| `campaign` | Campanha versionada |
| `typed` | Avaliação tipada (multiclasse/regressão) |
| `datasets`, `epoch-job` | Aquisição de datasets e treino por épocas |
| `iterative-worker` | Avaliação iterativa com early stopping |
| `components` | Lista o catálogo de componentes |
| `serve`, `serve-datasets` | API de tarefas e servidor de datasets |
| `aws` | Subcomandos do perfil AWS (ex.: `aws local-smoke`) |

Comparação ampliada (execução explícita, nunca pelo pytest):

```bash
uv run python scripts/benchmark_matrix.py --samples 30000 --seeds 42 43 44
```

Executa 63 configurações em subprocessos separados e grava `matrix-summary.json`. Acrescente
`--services --zenml` somente com os serviços ativos.

## Protocolos de avaliação

- **Streaming (prequential por evento):** prediz antes de treinar, a cada evento. Kafka real só com `--services`.
- **Incremental (por lote):** todas as predições do lote ocorrem antes de qualquer treino dele; rótulos são liberados no fim do lote conforme `available_at`. Pode divergir do streaming.
- **Offline (holdout temporal):** treina no prefixo, avalia congelado no sufixo, removendo do treino rótulos ainda indisponíveis no início do teste. Política deve ser `none`; não é comparável ao prequential.

Regras comuns: ordem original nunca embaralhada; timestamps **simulados** (`epoch + sequência`);
`warmup` treina uma única vez e é excluído das métricas; ao final há drenagem dos rótulos pendentes.
Fontes novas exigem `timestamp` estritamente crescente e `available_at` não decrescente.
Conjuntos de diagnóstico (*probes*) usam outra semente e nunca entram no treino, no detector ou na política.

## Componentes e extensão

Arquitetura em camadas ([docs/architecture.md](docs/architecture.md)): `core/` (contratos, políticas, motor,
métricas, sem dependências externas) → `config/` → `datasets/`, `models/`, `detectors/` (catálogos) →
`evaluation/` → `application/` → `infrastructure/` e `orchestration/`.

| Componente | Implementações |
|---|---|
| Modelo | Regressão logística e linear, Hoeffding Tree, kNN (River) |
| Detector | ADWIN, DDM, EDDM, HDDM-A/W, Page-Hinkley, KSWIN, nenhum |
| Política | nenhuma, online, periódica, por drift, por desempenho, heurística de custo |
| Persistência | Ponto de recuperação local atômico, S3/MinIO, MLflow |

O detector **não** decide retreinamento: as políticas impõem `cooldown`, `min_samples`, `min_window` e
frequência. `reset_retrain` recria modelo e preprocessador e treina na janela limitada de rótulos já
disponíveis; `learn_one` atualiza só com o exemplo liberado.

**Plugins:** implemente o contrato, registre uma função importável no nível do módulo e declare
`plugins: [modulo:register]` (veja `examples/custom_plugin.py` e `examples/temporal_csv.py`). Plugins executam
código confiável instalado pelo operador; a API de tarefas rejeita plugins enviados por HTTP.

## Avaliação iterativa (Semana 09)

Cada fonte é cortada em atualizações sequenciais (uma por tarefa, se houver até 20 tarefas contíguas; senão
10 blocos iguais). Cada atualização é pontuada primeiro no modo test-then-train e depois treinada por até
100 épocas com *early stopping* (paciência 10, 10% cronológico final como validação, restauração dos
melhores pesos). Um guarda de memória ciente de cgroup converte iminência de OOM em falha registrada.

A Semana 09 também trouxe políticas por desempenho e custo, detectores River, promoção independente do
treino, fontes com drift explícito, admissão de datasets locais, avaliação tipada de regressão/multiclasse e
métricas futuras com modelos congelados. Detalhes em [docs/task-coverage.md](docs/task-coverage.md) e
[docs/task-report.md](docs/task-report.md).

No homelab, `scripts/run_iterative_homelab.sh` treina os 17 datasets do catálogo em 8 containers
(2 vCPU / 2 GiB cada) que consomem uma fila compartilhada, registrando no MLflow:

```bash
ITER_ROOT=/caminho/absoluto bash scripts/run_iterative_homelab.sh
```

Variáveis: `ITER_IMAGE`, `ITER_WORKERS`, `ITER_CAMPAIGN`, `ITER_SERVICE_ENV`, `ITER_MLFLOW_URI`.
As credenciais vêm do arquivo de ambiente do serviço e nunca são impressas.

## Conjuntos de dados

O catálogo (`continual-bench components`, [docs/dataset-catalog.md](docs/dataset-catalog.md)) lista 17 candidatos
e se declara incompleto; não há download automático, e fontes reais exigem manifesto local revisado.

| Tipo | Fontes |
|---|---|
| Sintéticos (River, BSD-3-Clause) | SEA, Agrawal, STAGGER, Hyperplane, RBF drift, Friedman drift (regressão) |
| Reais com versão e SHA-256 fixados | Electricity (OpenML 151), Airlines (OpenML 1169) |
| Manifestos locais (licença, hashes e anotações exigidos) | Insects, Gas Sensor Array Drift |
| Imagens | Split MNIST, Permuted MNIST, Split CIFAR-10/100, CORe50, CLEAR-10/100 |

Os [benchmarks Zia Data Labs](docs/zia-benchmarks.md) (fraude, rede social, devoluções) usam fluxos Parquet
temporais próprios e rodam no homelab e na AWS.

Cada execução registra fonte, versão, semente, SHA-256 do JSONL, limites dos conceitos e tipo de timestamp.
Nos datasets reais não há pontos de mudança nem probes inventados; licenças seguem o rótulo do publicador
([fontes fixadas](configs/contracts/real-sources.lock.json)). Nada é baixado ou ordenado silenciosamente.
Veja [docs/image-datasets.md](docs/image-datasets.md), [docs/real-streams.md](docs/real-streams.md),
[docs/temporal-vision.md](docs/temporal-vision.md) e
[aquisição e execução](docs/task-dataset-execution.md).

## Execução em serviços reais (homelab)

Serviços: Airflow, ZenML, MLflow, MinIO, Kafka e Postgres via Docker Compose, com limites de CPU/RAM por
container e sem montar Docker socket. Apenas as interfaces web são publicadas, vinculadas ao IP privado
aprovado; Kafka, Postgres, API S3 e API de tarefas ficam só na rede Docker.

1. Mantenha o arquivo de ambiente privado fora do Git (`.env.example` traz só exemplos). MLflow exige senha
   de pelo menos 12 caracteres em `MLFLOW_PASSWORD`.
2. Valide a estrutura sem imprimir valores: `python3 scripts/preflight.py`.
3. Suba o conjunto:

```bash
docker compose -f compose.yaml -f compose.homelab.yaml --env-file /caminho/privado/benchmark.env config --quiet
docker compose -f compose.yaml -f compose.homelab.yaml --env-file /caminho/privado/benchmark.env build
docker compose -f compose.yaml -f compose.homelab.yaml --env-file /caminho/privado/benchmark.env up -d --wait --wait-timeout 300
```

Não use `down -v` nem remova volumes. Valide os serviços reais (exige Kafka, MinIO, MLflow e ZenML ativos):

```bash
docker compose -f compose.yaml -f compose.homelab.yaml --env-file /caminho/privado/benchmark.env \
  exec worker-1 pytest -q -m services tests/test_services.py
```

DAGs do Airflow (todas manuais): `benchmark_smoke`, `benchmark_streaming`, `benchmark_incremental`,
`benchmark_offline`, `benchmark_strategies`, `benchmark_full`, `benchmark_controlled`,
`benchmark_versioned`, `benchmark_task_suite` e `benchmark_zia_<dataset>`. A `benchmark_task_suite` exige
que todas as fontes tenham sido aprovadas no mesmo contrato de código, dependências e locks.

Operação completa, ZenML/MinIO e evidências: [docs/benchmark-operations.md](docs/benchmark-operations.md),
[docs/zenml-observability.md](docs/zenml-observability.md), [docs/mlflow-tracking.md](docs/mlflow-tracking.md)
e [reports/validation.md](reports/validation.md).

> O MinIO comunitário está arquivado. O laboratório fixa a versão de segurança de outubro de 2025, compila
> do código-fonte e mantém a licença AGPL. Não equivale a um serviço mantido para produção.

## Perfil AWS

Perfil gerenciado com CodePipeline, Batch sobre Fargate, Step Functions, S3, Athena, SageMaker e CUR 2.0,
sem MLflow, ZenML ou Kafka. Smoke local com fixtures:

```bash
uv run continual-bench aws local-smoke
```

Os alvos do `Makefile` cobrem validação, imagens, bootstrap, campanhas (`aws-campaign-*`), datasets
(`aws-dataset-*`), UIs (`aws-mlflow-ui`, `aws-studio-ui`), cancelamento e `aws-teardown`. Veja
[docs/aws/README.md](docs/aws/README.md), [custos](docs/aws/costs.md) e
[contabilidade de custos](docs/cost-accounting.md).

## Estrutura do repositório

```
src/continual_bench/   core, config, datasets, models, detectors, evaluation, application, infrastructure, orchestration
configs/               runs, matrices, campaigns, typed, contratos e locks
dags/                  DAGs do Airflow
scripts/               campanhas, validações, análise e execução no homelab
infra/ cfn-project/    CloudFormation, buildspecs, Athena, Flink
docker/ compose*.yaml  imagens e conjuntos de serviços
docs/ reports/         documentação e evidências de validação
examples/              plugins de referência
tests/                 testes unitários e de integração
```

## Testes e qualidade

```bash
make test    # pytest rápido (sem serviços nem benchmarks)
make lint    # ruff
```

- `pytest` padrão não usa serviços nem campanhas completas (`-m 'not services and not benchmark'`).
- `-m services` é integração com serviços reais; `-m benchmark` são experimentos completos, executados à parte.
- Não considere a configuração implantada com base só em testes locais; confira a evidência em `reports/`.

## Documentação

[Arquitetura](docs/architecture.md) ·
[Operação](docs/benchmark-operations.md) ·
[Confiabilidade](docs/reliability.md) ·
[Pesquisa](docs/research.md) ·
[Métricas](docs/task-metrics.md) ·
[Políticas](docs/task-policies.md) ·
[AWS](docs/aws/README.md) ·
[Diagrama do ciclo de vida](docs/diagrams/ciclo-de-vida-sistema.drawio.pdf)
