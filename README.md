# continual-bench

Laboratório reprodutível e limitado para **aprendizado contínuo com mudança de conceito (drift)**.
Compara modelos, detectores de drift e políticas de retreinamento sobre fluxos ordenados no tempo,
com protocolos de avaliação explícitos e rastreamento de experimentos. A execução pode ser local,
com Docker Compose ou na AWS.

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
- [Execução em serviços reais](#execução-em-serviços-reais)
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
uv sync --frozen --extra dev --extra services --extra datasets --extra large-datasets
```

| Extra | Conteúdo |
|---|---|
| `dev` | pytest, ruff |
| `services` | ZenML, MLflow, boto3, Kafka, s3fs, pandas |
| `datasets` | Pillow, pyarrow (loaders nativos de imagens e Parquet) |
| `large-datasets` | DuckDB, XGBoost (Zia e avaliação UPF de referência) |
| `plots` | Matplotlib, para gerar figuras localmente |
| `replication` | Alias mantido por compatibilidade; não instala dependências adicionais |
| `aws` | boto3, pandas (perfil gerenciado, lock próprio `requirements-aws.lock`) |
| `aws-mlflow` | SDK opcional para o MLflow gerenciado, com `sagemaker-mlflow` e matplotlib |

O Airflow usa um ambiente Python separado.

## Uso rápido

```bash
make test
uv run continual-bench run configs/runs/streaming.yaml
uv run continual-bench run configs/runs/incremental.yaml
uv run continual-bench run configs/runs/offline.yaml
uv run continual-bench smoke --state state/smoke --samples 600 --baselines
```

Com `--baselines`, o `smoke` executa **21 casos limitados**: três fontes, três políticas
(`none`, `periodic`, `drift`) em cada modo prequential e um holdout por fonte. Todos
usam o modelo padrão. O modo local lê JSONL, sem Kafka
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

O motor prequential preserva a ordem da fonte. Nos exemplos com relógio simulado,
`timestamp = epoch + sequência`. No motor por evento ou lote com políticas comuns,
o aquecimento inicial fica fora do placar e os rótulos pendentes são drenados ao final.
CARA, UPF e Oracle usam uma [fase offline da matriz](docs/cost-aware-retraining.md#binário-incremental).
O fluxo binário exige `timestamp` estritamente crescente e `available_at` não decrescente.
No protocolo prequential binário, os diagnósticos de retenção usam probes independentes em fontes
sintéticas ou exemplos reservados em fontes reais que suportam esse diagnóstico. No caminho com
reserva no próprio fluxo, esses exemplos ficam fora do treino, do detector e do placar prequential.
Veja [observações e retenção](docs/benchmark-observations.md#retenção-por-episódio).

## Componentes e extensão

Arquitetura em camadas ([docs/architecture.md](docs/architecture.md)): `core/` (contratos, políticas, motor,
métricas, sem dependências externas) → `config/` → `datasets/`, `models/`, `detectors/` (catálogos) →
`evaluation/` → `application/` → `infrastructure/` e `orchestration/`.

| Componente | Implementações |
|---|---|
| Modelo | Regressão logística e linear, Hoeffding Tree, kNN (River) |
| Detector | ADWIN, DDM, EDDM, HDDM-A/W, Page-Hinkley, KSWIN, nenhum |
| Política | nenhuma, online, periódica, por drift, por desempenho; em lotes (binário incremental e Zia): CARA-T/CT/P, UPF e Oracle ([docs](docs/cost-aware-retraining.md)) |
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

O worker iterativo treina os 17 datasets do catálogo por padrão. Workers que compartilham o diretório
de saída usam uma fila de datasets. Configure `MLFLOW_TRACKING_URI` para publicar as métricas no MLflow.
Para instâncias autenticadas, forneça `MLFLOW_TRACKING_USERNAME` e `MLFLOW_TRACKING_PASSWORD` pelo ambiente.

```bash
uv run continual-bench iterative-worker --cache /caminho/cache \
  --real-directory /caminho/datasets --out /caminho/estado --campaign minha-campanha
```

## Conjuntos de dados

O [catálogo de aquisição e experimentos](docs/dataset-catalog.md) distingue as 17 entradas da
suíte por épocas das fontes adicionais disponíveis para matrizes.
`continual-bench components` lista as fontes registradas no catálogo executável. A preparação e a admissão
são explícitas e seguem os locks ou manifestos de cada fonte.

| Tipo | Fontes |
|---|---|
| Sintéticos (River, BSD-3-Clause) | SEA, Agrawal, STAGGER, Hyperplane, RBF drift, Friedman drift (regressão) |
| Reais com versão e SHA-256 fixados | Electricity (OpenML 151), Airlines (OpenML 1169) |
| Fluxos reais com loaders próprios | INSECTS balanceado, Gas Sensor Array Drift e duas variantes binárias desbalanceadas de INSECTS |
| Imagens | Split MNIST, Permuted MNIST, Split CIFAR-10/100, CORe50, CLEAR-10/100 |

Os [benchmarks Zia Data Labs](docs/zia-benchmarks.md) (rede social, fraude e e-commerce) usam fluxos
Parquet temporais próprios e rodam com Docker Compose e na AWS. Na AWS, o catálogo do SageMaker Unified Studio
publica os snapshots e reúne os links para os runs Zia correspondentes no MLflow.

Cada execução registra fonte, versão, semente, SHA-256 do JSONL, limites dos conceitos e tipo de timestamp.
Os datasets OpenML não têm fronteiras de drift anotadas no contrato. As duas variantes binárias de
INSECTS têm fronteiras publicadas, reservadas à avaliação e à auditoria. Consulte os locks
[OpenML](configs/contracts/real-sources.lock.json) e [USP/UCI](configs/contracts/real-streams.lock.json).
Veja [docs/image-datasets.md](docs/image-datasets.md), [docs/real-streams.md](docs/real-streams.md),
[docs/temporal-vision.md](docs/temporal-vision.md) e
[aquisição e execução](docs/task-dataset-execution.md).

## Execução em serviços reais

Serviços: Airflow, ZenML, MLflow, MinIO, Kafka e Postgres via Docker Compose, com limites de CPU/RAM por
container e sem montar Docker socket. Apenas as interfaces web são publicadas, vinculadas ao IP privado
aprovado; Kafka, Postgres, API S3 e API de tarefas ficam só na rede Docker.

1. Mantenha o arquivo de ambiente privado fora do Git (`.env.example` traz só exemplos). MLflow exige senha
   de pelo menos 12 caracteres em `MLFLOW_PASSWORD`.
2. Em uma implantação com endereço previamente aprovado, valide o ambiente com `python3 scripts/preflight.py`.
3. Valide a configuração do Compose, gere as imagens e suba os serviços:

```bash
docker compose -f compose.yaml --env-file /caminho/privado/benchmark.env config --quiet
docker compose -f compose.yaml --env-file /caminho/privado/benchmark.env build
docker compose -f compose.yaml --env-file /caminho/privado/benchmark.env up -d --wait --wait-timeout 300
```

Não use `down -v` nem remova volumes. Valide os serviços reais (exige Kafka, MinIO, MLflow e ZenML ativos):

```bash
docker compose -f compose.yaml --env-file /caminho/privado/benchmark.env \
  exec worker-1 pytest -q -m services tests/test_services.py
```

As DAGs do Airflow são manuais. Elas incluem `benchmark_smoke`, a [família de campanhas binárias](docs/benchmark-operations.md#executar-no-airflow),
`benchmark_task_suite` e `benchmark_zia_<dataset>`. A `benchmark_task_suite` exige
que todas as fontes tenham sido aprovadas no mesmo contrato de código, dependências e locks.

Operação completa, ZenML/MinIO e evidências: [docs/benchmark-operations.md](docs/benchmark-operations.md),
[docs/zenml-observability.md](docs/zenml-observability.md) e [docs/mlflow-tracking.md](docs/mlflow-tracking.md).

> O MinIO comunitário está arquivado. O laboratório fixa a versão de segurança de outubro de 2025, compila
> do código-fonte e mantém a licença AGPL. Não equivale a um serviço mantido para produção.

## Perfil AWS

O deploy usa GitHub Actions com OIDC. Na AWS, o Step Functions coordena aquisição, preparação
e campanhas Batch. A configuração base usa Fargate Spot para o benchmark e Fargate On-Demand
para o preparo. S3, Glue e Athena preservam e consultam os resultados; o CUR 2.0 fornece dados de
faturamento. O runtime aceita uma ou duas sub-redes privadas. O gateway S3 atende as sub-redes
configuradas; os endpoints de interface são provisionados sob demanda e compartilhados pelos
workflows. Consulte o [ciclo dos endpoints](docs/aws/operations.md#7-ciclo-dos-endpoints-de-interface).

No SageMaker Studio, `continual-bench-campaign` acompanha a preparação, o workflow Batch e a
verificação do resultado. `continual-bench-native` acompanha jobs de Processing, Training e Tuning.
O app serverless `continual-bench-mlflow` exibe as métricas das campanhas Zia, das matrizes e dos
jobs nativos. O catálogo tabular fica em **SageMaker Unified Studio**. Veja o [guia de operação na AWS](docs/aws/operations.md#campanhas-no-sagemaker-studio).

Smoke local com fixtures:

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
scripts/               campanhas, validações, análise e execução de benchmarks
infra/                 CloudFormation, Step Functions, Athena, Glue, Flink e SageMaker
docker/ compose*.yaml  imagens e conjuntos de serviços
docs/                  documentação e evidências de validação versionadas
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
- Para o escopo e as evidências de implantação, veja [validação na AWS](docs/aws/validation.md).

## Documentação

O [protocolo corretivo e a campanha de avaliação](docs/benchmark-corrections.md)
descrevem disponibilidade de rótulos, separação treino/teste, retenção exclusiva de
métricas e execução com o mesmo código em ambiente local ou AWS. A campanha concluiu
60 de 60 runs INSECTS e 24 ensaios UPF em oito jobs, nas seeds 42, 43 e 44, com
cobertura integral das fontes admitidas. Online ficou adiado para a próxima rodada,
com quatro configurações e 12 runs. Os resultados históricos da Semana 10 permanecem
preservados.

O [dossiê da campanha corretiva](docs/evidencias/benchmark-corrigido/README.md) reúne
o registro da execução e da conferência dos artefatos. Conclusão de execução não
substitui essa conferência. O protocolo UPF integral difere das janelas e da baseline
do artigo; sua AUC não representa uma reprodução equivalente dos números publicados.
Custos de cenário permanecem separados de faturamento real. A portabilidade do código
não comprova, por si só, que esses resultados tenham sido executados na AWS.

| Para | Consulte |
|---|---|
| Entender e estender o código | [Arquitetura](docs/architecture.md), [configurações](configs/README.md), [diagrama do ciclo de vida](docs/diagrams/ciclo-de-vida-sistema.drawio.pdf) |
| Selecionar dados e comparar métodos | [Catálogo](docs/dataset-catalog.md), [fluxos INSECTS e Gas](docs/real-streams.md), [Zia](docs/zia-benchmarks.md), [políticas por custo](docs/cost-aware-retraining.md), [referências de pesquisa](docs/research.md) |
| Operar o laboratório | [Docker Compose e Airflow](docs/benchmark-operations.md), [AWS](docs/aws/README.md), [recuperação](docs/reliability.md) |
| Interpretar métricas e artefatos | [Contrato de métricas](docs/tracking-metrics.md), [observações por ajuste](docs/benchmark-observations.md), [MLflow](docs/mlflow-tracking.md), [custos](docs/cost-accounting.md) |
| Consultar resultados arquivados | [Semana 09](docs/task-report.md), [Semana 10: CARA/UPF](docs/evidencias/semana-10/cara-upf/README.md), [Semana 10: INSECTS](docs/evidencias/semana-10/insects/README.md), [campanha corretiva](docs/evidencias/benchmark-corrigido/README.md), [validações AWS](docs/aws/validation.md) |

A campanha INSECTS da Semana 10 arquivou **66 de 72 unidades concluídas**; o comparativo de
Electricity/Airlines arquivou o protocolo pareado local e experimentos de referência dos artigos.
Em ambos os casos, os resultados se aplicam aos recortes, modelos e protocolos indicados nos relatórios.
