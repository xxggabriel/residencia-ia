# Auditoria e plano de otimização do tracking MLflow

Este documento registra a auditoria, o plano e as medições daquela revisão. Para os
nomes de experimentos e os artefatos da revisão atual, consulte o [guia MLflow](mlflow-tracking.md).

Sintoma: a execução do benchmark termina rápido, mas a run no MLflow continua
`RUNNING` por muito tempo depois. A auditoria abaixo mede o caminho entre o fim
da avaliação e o `runs/update status=FINISHED`.

## 1. Gargalos encontrados

| # | Onde | Problema | Impacto |
|---|------|----------|---------|
| G1 | `aws/mlflow_tracking.py` `_request` | `urllib.request.urlopen` abre **uma conexão TCP+TLS nova por requisição** (sem keep-alive). | ~100–300 ms extras por chamada ao SageMaker MLflow; uma publicação faz centenas de chamadas. |
| G2 | `aws/mlflow_tracking.py` `_session_credentials` | Cria um `boto3.Session` novo **a cada requisição** quando nenhuma sessão é injetada (resolve a cadeia de credenciais, IMDS no EC2). | Dezenas de ms por chamada. |
| G3 | `SageMakerMlflowTracker.log_batch` | Params, tags e métricas são enviados em **requisições separadas**, e as métricas em lotes de 500, embora o `runs/log-batch` aceite 1000 métricas + 100 params + 100 tags numa chamada só. | 2–3× mais round-trips do que o necessário. |
| G4 | `metric_tracking.missing_metric_points` | Reconciliação idempotente faz **um `get-history` por chave, em série** (N+1), inclusive quando a chave tem um único ponto e o valor mais recente já confirma o ponto. | Em retomadas/Spot e nas runs filhas: N chamadas em série. |
| G5 | `progress()` (AWS) | O flush do progresso ao vivo roda **na thread de treino**; uma requisição lenta (timeout de 10 s × endpoints) bloqueia a avaliação. | A avaliação fica mais lenta e o tempo de tracking entra no relógio do benchmark. |
| G6 | `upload_directory_to_mlflow_artifacts` (REST) | `artifacts/plots` e `artifacts/data` são espelhados em `plots/`, `data/` **e depois reenviados** pelo walk completo; `tracking-observations/fits/*` é enviado para as runs filhas **e de novo** para a run pai. | Mais que o dobro de PUTs no S3. |
| G7 | `sagemaker_mlflow._log_artifacts` (SDK) | `publish_sdk_observations` envia `tracking-observations/`, e `_log_artifacts` envia o diretório inteiro **outra vez**; um `log_artifact` por arquivo, em série. | Uploads duplicados e sequenciais. |
| G8 | Uploads de artefatos | Um arquivo por vez (I/O de rede ocioso). | Tempo linear no nº de arquivos (≈120 numa run de replicação local). |
| G9 | `publish_training_rounds` / `publish_sdk_training_rounds` | Cada fit gera create + log-batch(métricas) + log-batch(tags) + update, **fit por fit em série**. | 4 chamadas × nº de fits, em série. |
| G10 | `publish_tracking`, `publish_local_benchmark`, `publish_run_artifacts` (homelab) | Várias tags enviadas com `set_tag` uma a uma. | 6–8 chamadas evitáveis por publicação. |

Problemas de arquitetura encontrados na mesma auditoria:

* **A1 – Duplicação AWS × homelab.** `SageMakerMlflowTracker.publish_training_rounds`
  (REST) e `training_tracking.publish_sdk_training_rounds` (MlflowClient)
  implementam o mesmo algoritmo duas vezes. O mesmo vale para o chunking do
  `log-batch` (`log_batch`, `_log_verified_points`, `run_artifacts`,
  `sagemaker_mlflow`) e para o layout de artefatos (REST × SDK).
* **A2 – Acoplamento benchmark × Zia.** `evaluation/zia.py` lê `DATASETS`,
  `load_manifest` e `sha256_file` de `datasets/zia.py` diretamente, então o
  benchmark batch-prequencial só roda os três datasets Zia (fraud, social,
  ecommerce), embora o projeto tenha outros 17 no catálogo.

## 2. Plano de otimização

### Transporte (G1, G2)
* `infrastructure/tracking/http.py`: `PooledHttpOpener`, um opener compatível
  com `urlopen` sobre um `urllib3.PoolManager` (keep-alive e pool por host).
  Vira o padrão do tracker; testes ainda injetam `http_opener`.
* O tracker cria o `boto3.Session` uma vez só e reutiliza as credenciais
  (renováveis) do botocore.

### Escrita em lote (G3, G10)
* `infrastructure/tracking/batching.py`: `batch_payloads()` é a única
  implementação dos limites do MLflow (1000 métricas, 100 params, 100 tags,
  1000 entidades por requisição). Params e tags vão no primeiro lote junto com
  as métricas.
* As publicações do homelab trocam a sequência de `set_tag` por um único `log_batch`.

### Reconciliação (G4)
* `infrastructure/tracking/reconcile.py`: `missing_metric_points()` busca os
  históricos das chaves já existentes **em paralelo** (pool limitado por
  `concurrency` do backend; stores locais `file:`/SQLite ficam seriais). A
  detecção de conflito continua estrita: nenhum atalho pelo "último valor".

### Progresso assíncrono (G5)
* `infrastructure/tracking/async_writer.py`: `AsyncMetricWriter`, uma thread
  daemon com coalescência por chave e flush periódico. `progress()` só enfileira;
  `publish_run_report`, `finish` e `mark_killed` drenam a fila antes de escrever.
  As tentativas Spot ficam síncronas, porque a reconciliação delas depende da
  ordem.

### Artefatos (G6, G7, G8)
* `infrastructure/tracking/artifacts.py`: `artifact_plan(output_dir)` define o
  layout **uma vez** (`artifacts/plots → plots/`, `artifacts/data → data/`, sem
  duplicar, ignorando arquivos temporários e caminhos já publicados em runs
  filhas) e `upload_concurrently()` envia em paralelo. Os publicadores REST e
  SDK usam o mesmo plano.

### Runs filhas (G9)
* Algoritmo único `publish_training_rounds(backend, …)`: tags e params vão no
  mesmo `log-batch` das métricas, e os fits são publicados em paralelo, porque
  são runs independentes.

## 3. Refatoração de arquitetura (A1)

Padrões: **Ports & Adapters** (porta `TrackingBackend`), **Strategy**
(transporte REST/SigV4 ou MlflowClient) e **Template Method** (os algoritmos
de publicação compartilhados recebem a porta).

```
infrastructure/tracking/
  ports.py        TrackingBackend (Protocol) + MetricPoint
  backends.py     MlflowClientBackend (homelab/SDK)  |  RestBackend (SigV4 AWS)
  batching.py     batch_payloads()                   (limites do MLflow)
  reconcile.py    missing_metric_points()            (paralelo + atalho)
  rounds.py       publish_training_rounds(backend)   (runs filhas, AWS e homelab)
  artifacts.py    artifact_plan() + upload_concurrently()
  async_writer.py AsyncMetricWriter
  http.py         PooledHttpOpener
```

Os módulos e nomes antigos (`publish_sdk_training_rounds`,
`SageMakerMlflowTracker.publish_training_rounds`,
`metric_tracking.missing_metric_points`, …) continuam existindo como fachadas
finas, então campanhas e testes atuais não quebram.

## 4. Desacoplamento do benchmark (A2)

* `datasets/streams.py`: registro de **provedores de stream** (Strategy +
  Registry). Cada provedor expõe `spec(name)`, `prepare(root, name, scope)` e
  `load_manifest(root, name, scope)` no mesmo formato ordenado em Parquet
  (`event_id, event_time, label, features…`).
  * `ZiaStreamProvider`: os três datasets Zia (comportamento atual).
  * `CatalogStreamProvider`: **qualquer dataset do catálogo** (`sea`,
    `agrawal`, `electricity`, `insects`, `split_mnist`, …), materializado de
    forma determinística no mesmo formato (features nominais codificadas por
    hash estável, rótulos mapeados para índices).
* `evaluation/stream_benchmark.py`: o runner batch-prequencial genérico
  (antes `evaluation/zia.py`) resolve o dataset pelo registro. O
  `evaluation/zia.py` continua como fachada de compatibilidade
  (`python -m continual_bench.evaluation.zia` segue funcionando).
* O fator de modelo deixa de depender do nome do dataset: os hiperparâmetros
  dos boosters (ex.: `scale_pos_weight` de fraud/ecommerce) vêm do `spec` do
  provedor (`model_params`).

Uso com qualquer dataset do catálogo (sem arquivos brutos):

```bash
python -m continual_bench.evaluation.stream_benchmark state/streams stagger smoke drift_ddm state/runs/stagger --prepare
```

Datasets do catálogo que exigem aquisição explícita (`insects`, `gas`,
imagens) continuam exigindo o cache local, exatamente como nos outros
avaliadores.

Na revisão desta auditoria, os contratos de snapshot cobriam Zia. O fluxo atual
admite matrizes do catálogo com [publicação de fontes separada](aws/operations.md#suíte-de-todos-os-datasets):
o pedido fixa o SHA-256 dos inputs externos, que o worker valida antes de materializar.

## 5. Status da implementação e medições

Implementado conforme as seções 2–4. Configuração:

* `BENCH_MLFLOW_PUBLISH_CONCURRENCY` (padrão `4`): requisições MLflow
  concorrentes na publicação (históricos, runs filhas).
* Uploads de artefatos: 8 em paralelo (`tracking.artifacts.DEFAULT_UPLOAD_WORKERS`).
* Progresso assíncrono: ligado na fábrica do worker AWS (`aws_cli`,
  `async_progress=True`) e nas runs de matriz fora do Spot; tentativas Spot
  continuam síncronas.

Benchmark com latência simulada (40 ms por requisição MLflow, 25 ms por PUT no
S3; 20 fits × 20 épocas, 200 métricas de resumo, 120 arquivos), ambos já com
lotes combinados e sem uploads duplicados:

| Configuração | Tempo de publicação | Requisições MLflow | PUTs S3 |
|---|---|---|---|
| Sequencial (1 requisição/upload por vez) | 7,04 s | 82 | 120 |
| Novo padrão (4 req. concorrentes, 8 uploads) | 2,08 s | 82 | 120 |

No layout antigo, os mesmos 120 arquivos geravam **200 PUTs** (`artifacts/plots`
e `artifacts/data` reenviados) e cada `log_batch` com params+tags custava 2–3
requisições. Na AWS real ainda soma o ganho do keep-alive (G1/G2), que a
simulação não mede: hoje cada requisição pagava um handshake TLS novo.
