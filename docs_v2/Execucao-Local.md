# Execução Local

Esta página mostra como executar o benchmark na máquina local pela linha de comando `continual-bench`, o que cada subcomando faz e grava, e como funciona o perfil Docker Compose com Airflow, ZenML, MinIO, MLflow e Kafka. A instalação está em [Instalação e Configuração](Instalacao-e-Configuracao); o perfil gerenciado na nuvem, em [Execução na AWS](Execucao-na-AWS).

## Linha de comando

O ponto de entrada é `continual-bench` (`src/continual_bench/cli.py`). Cada subcomando é registrado por um decorador e aceita `--help`. Os subcomandos atuais são:

| Subcomando | Argumentos e opções (padrões) | Função |
|---|---|---|
| `run` | `config`; `--state` (`state`), `--services`, `--zenml` | Uma execução binária (`streaming`, `incremental` ou `offline`) a partir de um arquivo `Config` |
| `smoke` | `--state` (`state`), `--services`, `--zenml`, `--baselines`, `--samples` (600) | Conjunto curto de execuções binárias sintéticas |
| `matrix` | `config`; `--state` (`state/matrices`), `--cache` (`state/datasets`), `--real-directory`, `--dry-run`, `--services` | Expande e executa uma `ExperimentMatrix` |
| `campaign` | `config`; `--state` (`state/task_evaluation`), `--services`, `--zenml` | Campanha declarativa |
| `typed` | `config`; `--state` (`state/task_evaluation-typed`), `--services` | Avaliação prequential multiclasse ou de regressão (`TypedConfig`) |
| `datasets` | `--dataset` (`all`), `--cache` (`state/datasets`), `--state` (`state/dataset-epochs`), `--real-directory`, `--download` | Treino por épocas de um dataset do catálogo ou verificação de todos |
| `epoch-job` | `config`; `--cache`, `--state`, `--real-directory` (obrigatórios), `--download`, `--services` | Treino por épocas a partir de um arquivo `EpochConfig` |
| `iterative-worker` | `--cache`, `--real-directory`, `--out` (obrigatórios), `--dataset` (repetível; padrão: os 17 datasets de `datasets.catalog.DATASETS`), `--max-epochs` (100), `--patience` (10), `--worker`, `--campaign` (`local`) | Avaliação iterativa com *early stopping* |
| `components` | `--plugin` (repetível) | Lista datasets, modelos, detectores e políticas registrados |
| `serve` | nenhum | Servidor HTTP de jobs binários e Zia na porta 8088 |
| `serve-datasets` | nenhum | Servidor HTTP de jobs por épocas |
| `aws` | subcomandos `acquire`, `prepare`, `publish-snapshot`, `publish-matrix-source`, `run-unit`, `selftest`, `local-smoke`, `plan`, `refresh-prices`, `query`, `reconcile`, `cancel`, `mlflow-url`, `studio-url` | Perfil AWS; ver [Execução na AWS](Execucao-na-AWS) |

Todos os subcomandos que executam avaliações imprimem um resumo em JSON no terminal. As opções comuns significam:

- `--state`: diretório raiz onde cada execução cria `<run_key>/` (ver [Arquitetura](Arquitetura#conteúdo-de-staterun_key)). `state/` está no `.gitignore`.
- `--services`: exige MLflow autenticado e publica em MLflow e MinIO; no modo `streaming`, também usa Kafka. Sem essa opção, nada sai da máquina.
- `--zenml`: executa a mesma avaliação como pipeline ZenML local de quatro etapas. Requer o extra `services`.

## Primeiros comandos

```bash
make test
uv run --frozen continual-bench components
uv run --frozen continual-bench run configs/runs/streaming.yaml
uv run --frozen continual-bench run configs/runs/incremental.yaml
uv run --frozen continual-bench run configs/runs/offline.yaml
make smoke
```

| Configuração | Dataset | Modo | Detector | Política |
|---|---|---|---|---|
| `configs/runs/streaming.yaml` | SEA | `streaming` | ADWIN (`delta: 0.002`) | `drift` |
| `configs/runs/incremental.yaml` | Agrawal | `incremental` (lotes de 32) | DDM | `periodic` |
| `configs/runs/offline.yaml` | STAGGER | `offline` (60% para treino) | ADWIN (padrão) | `none` |
| `configs/runs/metrics.yaml` | Agrawal, 3000 amostras | `streaming` | nenhum | `online` com Hoeffding Tree |

Os três primeiros herdam de `configs/base/run.yaml` 1200 amostras, semente 42 e atraso de rótulo 3. O resumo impresso por `run` contém `run_key`, `protocol`, `metrics` e `complete`.

### Smoke

`make smoke` executa `uv run continual-bench smoke --state state/smoke --samples 600 --baselines`. O código percorre SEA, Agrawal e STAGGER nos três modos. Com `--baselines`, os modos `streaming` e `incremental` usam as políticas `none`, `periodic` e `drift`, e o modo `offline` usa `none`, totalizando 21 execuções (3 fontes × (3 + 3 + 1)). Sem `--baselines`, são 9 execuções, com `drift` nos modos prequential. Todas usam o modelo e o detector padrão (`logistic`, `adwin`), `warmup: 50` e atraso de rótulo 3, e leem o fluxo de um JSONL local, sem Kafka. O resumo vai para `state/smoke/smoke-summary.json`.

O smoke valida integração e protocolo, não desempenho.

### Matriz de experimentos

```bash
uv run --frozen continual-bench matrix configs/matrices/binary-comparison.yaml --dry-run
uv run --frozen continual-bench matrix configs/matrices/binary-comparison.yaml --state state/matrices
```

`--dry-run` só imprime o plano expandido e as combinações descartadas por incompatibilidade, sem executar. Para `binary-comparison.yaml` (5 datasets × 3 modelos × 4 estratégias × 2 modos × 3 sementes), o plano tem 360 casos e nenhum descartado. A execução grava em `state/matrices/<identidade>/` um diário retomável (`matrix.json`) e um resumo (`summary.json`); casos concluídos são reaproveitados numa nova execução. `configs/matrices/any-model.yaml` usa o avaliador typed com regressão.

### Outras avaliações

```bash
uv run --frozen continual-bench typed configs/typed/friedman-smoke.yaml
uv run --frozen continual-bench campaign configs/campaigns/task-smoke.yaml
uv run --frozen continual-bench iterative-worker --cache /caminho/cache \
  --real-directory /caminho/datasets --out /caminho/estado --campaign minha-campanha
```

O `iterative-worker` reivindica cada dataset criando `--out/claims/<dataset>` de forma atômica, então vários workers podem compartilhar o mesmo diretório de saída sem repetir trabalho. Falhas são registradas em `results/<dataset>/failed.json` e o worker segue para o próximo dataset. Com `MLFLOW_TRACKING_URI` definido, ele publica no MLflow; credenciais autenticadas vêm de `MLFLOW_TRACKING_USERNAME` e `MLFLOW_TRACKING_PASSWORD`. Datasets reais precisam estar previamente disponíveis em `--real-directory`; a preparação e a admissão estão em [Datasets](Datasets).

### Comparação ampliada por script

```bash
uv run --frozen python scripts/benchmark_matrix.py --samples 30000 --seeds 42 43 44
```

O script repete a grade do smoke com baselines para cada semente (21 configurações por semente, 63 para três sementes), cada uma em um subprocesso separado para isolar a medição de memória, e grava `state/full-benchmark/matrix-summary.json`. É um experimento explícito, nunca disparado pelo pytest. `--services --zenml` só devem ser usados com os serviços ativos.

## Retomada e reexecução

Reexecutar a mesma configuração com o mesmo `--state` retoma do último ponto de recuperação (`checkpoint.bin`), desde que o contrato de retomada seja idêntico: configuração, SHA-256 da fonte, Python, pacotes, hash do código, `PYTHONHASHSEED`, definições de métricas, plugins e transporte. Qualquer diferença faz a execução falhar com erro explícito, sem carregar estado antigo. Se a execução já terminou, o relatório existente é devolvido. Duas execuções da mesma identidade no mesmo diretório são impedidas por um *lock* de arquivo.

Como `PYTHONHASHSEED` entra no contrato e o padrão do Python é aleatório, uma retomada local fora do Compose só é aceita quando a variável tem o mesmo valor nas duas execuções. O Compose fixa `PYTHONHASHSEED=0`.

### Garantias e limites da recuperação

- **Checkpoints locais.** Ficam apenas no volume de estado. Ao concluir com `--services`, os arquivos da execução (relatório, eventos e checkpoint final) são copiados para o MinIO, mas não há restauração automática a partir dessa cópia: se o volume local se perder no meio da execução, a execução recomeça do início a partir da fonte e da semente.
- **Kafka.** O produtor usa idempotência e `acks=all`, num tópico `bench-<run_key>` de uma partição com retenção de 24 h ou 256 MiB (`infrastructure/services.py`). Reprocessamentos podem duplicar mensagens no tópico; o consumidor descarta sequências já vistas e rejeita lacunas, inversões e conteúdo diferente do `events.jsonl`. Se a retenção apagou parte do tópico, a leitura falha em vez de aceitar um fluxo incompleto. Não há garantia de processamento distribuído exatamente uma vez.
- **Publicação.** O MinIO usa chaves determinísticas `<run_key>/...` e o MLflow localiza o run pela etiqueta `run_key`; mais de um run com a mesma etiqueta faz a publicação falhar para reconciliação manual. Uma queda entre os dois destinos pode deixar um deles completo; repetir a publicação converge ao mesmo estado. `receipt.json` só é gravado depois dos dois destinos. Não é uma transação distribuída.
- **Prontidão.** Os servidores de jobs distinguem processo vivo (`/health`) de pilha pronta (`/ready`) e respondem 503 a submissões quando o MLflow não está pronto. A CLI e a pipeline ZenML falham antes de criar artefatos nessa situação.

Para uma réplica nova, mude `name` ou a semente; para comparar transportes (JSONL local e Kafka), use diretórios de estado distintos. Os detalhes estão em [Reprodutibilidade e Artefatos](Reprodutibilidade-e-Artefatos).

## Perfil Docker Compose

O perfil com serviços reais é definido em `compose.yaml` e descrito aqui sem procedimento de implantação completo. Ele publica apenas as interfaces web no endereço `WEB_BIND_IP`; Kafka, PostgreSQL, a API S3 do MinIO e a API de jobs ficam restritos à rede Docker `bench`. Os containers usam `no-new-privileges`, limites de PIDs, rotação de logs e não montam o socket do Docker.

### Serviços

| Serviço | Imagem | Porta publicada | Limites | Papel |
|---|---|---|---|---|
| `postgres` | `postgres:17.10-bookworm` | nenhuma | 1 CPU, 512 MiB | Banco do Airflow |
| `minio` | `continual-bench-minio:2025-10-15-9e49d5e7` (compilada de `docker/minio/Dockerfile`) | 19001 (console) | 2 CPUs, 1 GiB | Armazenamento compatível com S3 |
| `kafka` | `apache/kafka:4.3.1` | nenhuma | 2 CPUs, 1536 MiB | Nó único KRaft; retenção de 24 h ou 256 MiB; criação automática de tópicos desligada |
| `zenml` | `zenmldocker/zenml-server:0.98.0` | 18082 | 2 CPUs, 1536 MiB | Servidor ZenML com SQLite |
| `mlflow` | `continual-bench-mlflow:3.17.2` | 15000 | 4 CPUs, 3 GiB | MLflow com `basic-auth`, SQLite e artefatos em `s3://mlflow` |
| `storage-init` | `continual-bench-worker:0.1.0` | nenhuma | 1 CPU, 256 MiB | Cria os buckets `benchmarks` e `mlflow` e termina |
| `worker` | `continual-bench-worker:0.1.0` | nenhuma | 0,25 CPU, 256 MiB | Distribuidor `worker_pool` para oito backends |
| `worker-1` a `worker-8` | `continual-bench-worker:0.1.0` | nenhuma | 1 CPU, 2 GiB, sem swap | `continual-bench serve`, um job por vez (`BENCH_CONCURRENCY=1`) |
| `airflow-init`, `airflow-api`, `airflow-scheduler`, `airflow-dag-processor` | `continual-bench-airflow:3.3.2` | 18081 (API e interface) | 2 CPUs e 2 GiB; scheduler com 4 CPUs e 4 GiB | Airflow 3.3.2 com `LocalExecutor`, paralelismo 8, DAGs pausadas na criação |

O distribuidor `worker` consulta a ocupação dos backends, encaminha cada `POST` a um backend livre e devolve HTTP 429 quando todos estão ocupados. Cada backend recusa um segundo job local, valida o corpo (até 65.536 bytes), rejeita `plugins` e responde 503 se o MLflow não estiver pronto.

### Arquivos complementares

| Arquivo | Uso |
|---|---|
| `compose.policies-parallel.yaml` | Aplicado depois de `compose.yaml`: MLflow com 8 CPUs, prontidão compartilhada e `BENCH_POLICY_DATASET_PARALLELISM=8` para a DAG `benchmark_policies` |
| `compose.dataset-benchmark.yaml` | Pool alternativo `dataset-worker-*` com `serve-datasets`, montando código e estado externos |
| `compose.task-validation.yaml` | Cliente de teste transitório (`run --rm --no-deps validation ...`), nunca `up` |
| `compose.zia.yaml` | Monta os dados Zia somente leitura nos workers |
| `compose.homelab.yaml` | Define a sub-rede da rede `bench` |

### DAGs do Airflow

Todas as DAGs usam `schedule=None`, isto é, só rodam por disparo manual.

| DAG | Arquivo | Casos por semente com parâmetros padrão |
|---|---|---|
| `benchmark_smoke` | `dags/benchmark_smoke.py` | 9 (3 datasets × 3 modos) |
| `benchmark_full` | `dags/benchmark_pipelines.py` | 33 |
| `benchmark_streaming`, `benchmark_incremental`, `benchmark_offline` | idem | 15, 15 e 3 |
| `benchmark_strategies` | idem | 30 |
| `benchmark_controlled`, `benchmark_versioned`, `benchmark_nonlinear`, `benchmark_metrics`, `benchmark_metrics_smoke` | idem | Conforme as campanhas v3, v4, v5 e o perfil de métricas (o smoke de métricas tem 8 casos) |
| `benchmark_policies` | idem | Conforme dataset e modelos escolhidos |
| `benchmark_task_suite` | `dags/benchmark_task_suite.py` | Suíte de datasets admitidos |
| `benchmark_zia_fraud`, `benchmark_zia_social`, `benchmark_zia_ecommerce` | `dags/benchmark_zia.py` | Campanhas Zia |

Na DAG `benchmark_full`, os parâmetros de disparo são `samples` (300 a 100.000, padrão 3000), `seeds` (1 a 10 sementes distintas, padrão `[42]`) e `label_delay` (0 a 100, padrão 3). As campanhas `controlled`, `versioned`, `nonlinear` e `metrics` usam por padrão as sementes 42 a 46. `benchmark_policies` aceita ainda `dataset` (um entre `electricity`, `airlines`, `sea`, `agrawal`, `stagger`, `hyperplane`, `rbf_drift`), `models`, `campaign` e `experiment` (`dags/benchmark_pipelines.py`, `dags/bench_jobs.py`).

As DAGs de `benchmark_pipelines.py` seguem o fluxo `services_ready → plan → evaluate_and_publish [mapeada] → consolidate_results`. A primeira tarefa falha explicitamente se algum serviço não responder, e nenhuma execução é submetida. Os números de casos de `full`, `streaming`, `incremental`, `offline` e `strategies` correspondem ao que `dags/bench_jobs.configurations` gera para uma semente.

### Comandos de operação

Os comandos abaixo constam do `README.md` e do `Makefile`. Eles constroem imagens e sobem serviços, por isso devem ser usados apenas numa implantação aprovada:

```bash
make compose-check   # docker compose --env-file .env.example config --quiet
python3 scripts/preflight.py
docker compose -f compose.yaml --env-file /caminho/privado/benchmark.env config --quiet
docker compose -f compose.yaml --env-file /caminho/privado/benchmark.env build
docker compose -f compose.yaml --env-file /caminho/privado/benchmark.env up -d --wait --wait-timeout 300
docker compose -f compose.yaml --env-file /caminho/privado/benchmark.env \
  exec worker-1 pytest -q -m services tests/test_services.py
```

Não use `docker compose down -v` nem remova volumes: eles guardam execuções, checkpoints, bancos SQLite e artefatos.

Interfaces publicadas em `WEB_BIND_IP`: Airflow na porta 18081, ZenML na 18082, MLflow na 15000 e console do MinIO na 19001. O MLflow usa `WEB_USER` com a senha exclusiva `MLFLOW_PASSWORD`. Se `--cors-allowed-origins` não corresponder ao endereço usado no navegador, a interface do MLflow mostra `INTERNAL_ERROR` mesmo com a API funcionando.

### ZenML com MinIO

Nos workers, o Compose define o projeto ZenML `default`, a stack e o armazenamento de artefatos `benchmark-minio-v2` (`BENCH_ZENML_STACK`, `BENCH_ZENML_ARTIFACT_STORE`). A stack combina o orquestrador local do worker com o bucket privado `s3://zenml-artifacts` no MinIO, de onde o servidor ZenML lê os logs das etapas e as visualizações dos artefatos. Um armazenamento de artefatos local (o padrão `benchmark-local-v1`, usado fora do Compose) não pode ser lido pelo servidor: o painel mostra o grafo, mas não os logs nem as visualizações.

Antes do bootstrap devem existir:

- o bucket privado `zenml-artifacts`;
- uma identidade MinIO restrita a esse bucket;
- o segredo ZenML `benchmark-minio-artifacts`, com os campos `key` e `secret` dessa identidade.

Política mínima de exemplo para a identidade, sem acesso aos buckets `benchmarks` e `mlflow`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::zenml-artifacts"]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": ["arn:aws:s3:::zenml-artifacts/*"]
    }
  ]
}
```

Depois de provisionar esses recursos, o bootstrap idempotente registra o armazenamento S3 (endpoint `S3_ENDPOINT_URL`, credenciais referenciadas ao segredo, sem valores literais) e a stack:

```bash
docker compose --env-file /caminho/privado/benchmark.env \
  -f compose.yaml exec -T worker-1 python scripts/bootstrap_zenml.py
```

No painel do ZenML, cada execução aparece como a pipeline `finite_benchmark`, com grafo e logs por etapa; os artefatos têm a aba **Visualization**.

`scripts/preflight.py` lê as variáveis do processo, então o arquivo de ambiente precisa estar exportado no shell antes de executá-lo. A operação detalhada de campanhas e diagnósticos está em [Operação e Diagnóstico](Operacao-e-Diagnostico).

Limites conhecidos do perfil: Kafka de nó único, SQLite no ZenML e no MLflow, volumes Docker locais sem cópia de segurança automática e MinIO comunitário compilado de uma versão de outubro de 2025 (projeto arquivado, licença AGPL). Não é um ambiente de produção.

## Estado das capacidades

| Capacidade | Classificação | Evidência |
|---|---|---|
| Subcomandos da CLI listados | Implementado | `continual-bench --help` e `<subcomando> --help` |
| `run`, `smoke`, `matrix`, `typed`, `campaign` locais | Validado | `tests/test_application.py`, `tests/test_config_files.py` e testes de matriz como `tests/test_matrix_checkpoint_resume.py` |
| Contagem de casos das DAGs | Validado | `tests/test_benchmark_dags.py` |
| Perfil Compose com serviços reais | Experimental | Testes `services` (`tests/test_services.py`) exigem a pilha ativa e não rodam no CI |
| Restauração automática de checkpoint remoto no perfil local | Pendente | `application/runner.py` lê apenas `LocalCheckpoints`; nenhum código baixa o checkpoint publicado no MinIO |
