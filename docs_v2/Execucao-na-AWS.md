# Execução na AWS

Esta página descreve o perfil AWS do `continual-bench`: serviços e stacks provisionados, módulos ativáveis, publicação por GitHub Actions, preparação de datasets, campanhas de benchmark, cancelamento e remoção. A operação do dia a dia, a recuperação de falhas e o diagnóstico estão em [Operação e Diagnóstico](Operacao-e-Diagnostico); custos e recursos medidos estão em [Custos e Recursos](Custos-e-Recursos).

O perfil AWS é independente do laboratório local (Docker Compose, Airflow, ZenML, MinIO e Kafka), descrito em [Execução Local](Execucao-Local). O protocolo científico é o mesmo nos dois caminhos: o transporte AWS só move entradas e saídas, e o teste `test_worker_matches_the_local_protocol` (`tests/test_aws_planning_worker.py`) compara janelas, digests de predições e de modelos e a matriz de confusão do holdout entre uma run do caminho AWS e uma run local.

## Estado das capacidades

| Capacidade | Estado | Evidência |
|---|---|---|
| Templates CloudFormation e invariantes de segurança | Validado | `make aws-validate` aprova 16 templates com cfn-lint e as invariantes de `scripts/aws/validate_templates.py`; testes em `tests/test_aws_infra.py` |
| Definições Step Functions (ASL) | Validado | `scripts/aws/validate_asl.py` em modo offline aprova as duas definições de `infra/statemachines/` |
| Deploy por GitHub Actions com OIDC | Validado | Execução do workflow `Build and deploy` de 12/11/2026 (ver Validações registradas) |
| Workflow de datasets (aquisição, preparação, manifesto) | Validado | Dataset `social-smoke-prepare` com `SUCCEEDED` em 09/11/2026 (ver Validações registradas); testes em `tests/test_aws_snapshots_sources.py` |
| Campanha de benchmark no Batch com Fargate | Validado | Campanha `social-smoke-v4` com 4/4 unidades em 09/11/2026 (ver Validações registradas); suíte de quatro streams com 132 unidades e 133 tentativas em `evidencias/aws-benchmarks/provenance.json` |
| Checkpoint e retomada no Fargate Spot | Validado por testes automatizados | `tests/test_aws_spot_worker.py`, `tests/test_fargate_spot.py`. Não há registro de uma interrupção Spot real exercitada na conta (Pendente) |
| Cancelamento de campanha | Validado por testes automatizados | `tests/test_aws_lifecycle_control.py`. O cancelamento manual ficou fora dos smokes registrados (Pendente na conta) |
| SageMaker Pipelines (`continual-bench-campaign` e `continual-bench-native`) | Validado | Execuções `studio-matrix-151104` e `native-smoke-155750` verificadas em 16/11/2026 (ver Validações registradas) |
| MLflow App serverless e catálogo do Unified Studio | Validado | App `continual-bench-mlflow` e listagens `ACTIVE` verificados em 16/11/2026 (ver Validações registradas) |
| SageMaker Model Registry | Validado | Versões `PendingManualApproval` nos grupos `continual-bench-zia-social-*-smoke` (campanha `social-smoke-v4`, ver Validações registradas) |
| Módulos Streaming, Feature Store, ML avançado e BI | Experimental | Código e testes unitários (`tests/test_aws_streaming_ml_costs.py`), sem registro de ativação na conta |
| Conciliação com o faturamento (CUR 2.0) | Experimental | Ver [Custos e Recursos](Custos-e-Recursos#faturamento-observado-e-rateio) |

## Visão geral

O perfil opera na região `us-east-1` (fixada em `scripts/aws/bootstrap.sh` e `scripts/aws/outputs.sh`). Nenhuma instância EC2 é criada pelo perfil: o processamento usa AWS Batch sobre Fargate (contêineres sem servidor gerenciado pelo usuário), CodeBuild para aquisição e Lambda para controle.

Fluxo resumido:

1. O operador executa o bootstrap uma vez, criando as stacks persistente e de CI/CD.
2. Um push ou merge na `main` dispara o workflow `Build and deploy`, que testa, constrói as imagens, aplica o change set do runtime e grava o manifesto da release no S3.
3. O operador inicia manualmente o workflow de datasets, que adquire, verifica e prepara um snapshot.
4. O operador inicia manualmente o workflow de benchmark, que planeja a campanha, provisiona os endpoints de rede privados, submete um job Batch por unidade, consolida os resultados e confere que nenhuma task ficou ativa.

Nenhum evento inicia campanhas automaticamente. O EventBridge apenas observa jobs e execuções e aciona a Lambda de controle para registro, notificação e reconciliação (`src/continual_bench/orchestration/aws_control.py`, função `observe`). `scripts/aws/validate_templates.py` verifica essa regra nos templates.

## Serviços utilizados

| Serviço | Papel |
|---|---|
| Amazon S3 | Buckets de dados, artefatos, billing e logs de acesso; snapshots, planos, manifestos, evidências por tentativa e tabelas analíticas |
| AWS KMS | Criptografia de buckets, logs, SNS, Kinesis e ECR |
| Amazon ECR | Repositórios `continual-bench-runtime` e `continual-bench-ml` (imagens `linux/amd64` por digest) |
| AWS Step Functions | Workflows `continual-bench-benchmark` e de datasets (`infra/statemachines/*.asl.json`) |
| AWS Batch com Fargate | Fila `continual-bench-jobs` para o benchmark (Fargate Spot por padrão) e fila separada em Fargate On-Demand para a preparação |
| AWS CodeBuild | Aquisição de datasets fora da VPC privada |
| AWS Lambda | Lambda de controle (planejamento, reconciliação, registro, consolidação) e Lambda de ciclo de vida dos endpoints |
| Amazon EventBridge e SNS | Observação de jobs e execuções; alertas |
| CloudWatch Logs, métricas EMF e dashboard | Logs `/continual-bench/batch` e `/continual-bench/acquisition`; dashboard `continual-bench-operations` |
| AWS Glue Data Catalog e Amazon Athena | Tabelas `runs`, `metrics`, `windows`, `model_versions`, `attempts`, `campaigns` e `cost_estimates`; workgroup `continual-bench` |
| Amazon SageMaker | MLflow App serverless, Pipelines, Model Registry, jobs de Processing, Training e Tuning |
| SageMaker Unified Studio (DataZone) | Catálogo dos snapshots Zia com links para avaliações no MLflow |
| AWS Data Exports (CUR 2.0), Budgets e Cost Anomaly Detection | Faturamento observado e alertas de gasto (módulo `billing`) |

## Stacks CloudFormation

| Stack | Template | Como é criada | Conteúdo |
|---|---|---|---|
| `continual-bench-persistent` | `infra/cloudformation/persistent.yaml` | `scripts/aws/bootstrap.sh` | KMS, buckets de dados, artefatos, billing e logs de acesso, ECR. Recursos com retenção e proteção contra término |
| `continual-bench-cicd` | `infra/cloudformation/cicd.yaml` | `scripts/aws/bootstrap.sh` | Provedor OIDC do GitHub e roles de deploy |
| `continual-bench-runtime` | `infra/cloudformation/root.yaml` | Workflow `Build and deploy` | Stacks aninhadas dos módulos (próxima seção) |
| `continual-bench-interface-endpoints` | `infra/cloudformation/modules/interface-endpoints.yaml` | Sob demanda, pelos workflows de benchmark e de datasets | Endpoints de interface da VPC; removida quando filas e execuções ficam ociosas |
| `continual-bench-studio` | `infra/cloudformation/studio.yaml` | Implantação separada | Pipeline `continual-bench-campaign` com callbacks para Step Functions |
| `continual-bench-studio-native` | `infra/cloudformation/studio-native.yaml` | `scripts/aws/deploy_native_pipeline.sh` | Pipeline `continual-bench-native` (Processing, Training e Tuning do Social) |
| `continual-bench-studio-catalog` | `infra/cloudformation/studio-catalog.yaml` | `scripts/aws/deploy_studio_catalog.sh` | Publicação automática de snapshots e avaliações no catálogo |

O ECR fica na stack persistente para evitar dependência circular entre pipeline, imagens e job definitions. As job definitions do Batch usam `DeletionPolicy: Retain`.

## Módulos do runtime

O template `infra/cloudformation/root.yaml` aninha um módulo por stack. Os quatro primeiros são sempre criados; os demais dependem de flags.

| Módulo | Template | Flag e padrão | Função |
|---|---|---|---|
| Rede | `modules/network.yaml` | sempre | VPC privada, sub-redes sem rota para a internet, gateway endpoint do S3 |
| Execução | `modules/execution.yaml` | sempre | Batch, job definitions, CodeBuild, Lambdas, state machines, MLflow App (`EnableMlflowTracking=true`) |
| Observabilidade | `modules/observability.yaml` | sempre | SNS, regras EventBridge, alarmes e dashboard |
| Analytics | `modules/analytics.yaml` | sempre | Banco Glue `continual_bench`, tabelas e workgroup Athena |
| Billing | `modules/billing.yaml` | `EnableBilling=true` | Export CUR 2.0, tabela `continual_bench_billing.cur2`, orçamento e monitor de anomalias |
| Streaming | `modules/streaming.yaml` | `EnableStreaming=false` | Kinesis, Firehose e Managed Flink |
| Feature Store | `modules/features.yaml` | `EnableFeatureStore=false` (`EnableOnlineStore=false`) | Feature group offline, Glue ETL e Data Quality |
| ML avançado | `modules/ml.yaml` | `EnableMl=false` (`EnableEndpoint=false`) | Role e recursos para jobs SageMaker avulsos e endpoint de inferência |
| BI | `modules/bi.yaml` | `EnableBi=false` | Fonte, dataset e análise no Amazon Quick Sight; exige `QuickSightPrincipalArn` |

Regras do template impedem `EnableEndpoint=true` sem `EnableMl=true` e `EnableBi=true` sem principal do Quick Sight. Billing vem ativado por padrão, mas é uma flag como as demais.

Para ativar um módulo, altere a flag no objeto `config/runtime-parameters.json` do bucket de artefatos, preservando os demais valores. O deploy lê esse objeto (variável `RUNTIME_CONFIG_KEY` do GitHub) e aplica os parâmetros no change set. O modelo do arquivo é `infra/params/runtime-overrides.example.json`.

## Parâmetros de dimensionamento

| Parâmetro | Padrão em `root.yaml` | Exemplo de override | Efeito |
|---|---:|---:|---|
| `FargateCapacity` | `FARGATE_SPOT` | `FARGATE_SPOT` | Capacidade do ambiente de benchmark; `FARGATE` é On-Demand |
| `BenchmarkVcpus` / `BenchmarkMemoryMiB` | 2 / 4096 | 2 / 4096 | Reserva por task de benchmark |
| `BenchmarkEphemeralGiB` | 50 | 50 | Disco efêmero por task |
| `MaxVcpus` | 64 | 64 | Teto do ambiente Batch (até 32 tasks de 2 vCPU) |
| `BenchmarkTimeoutSeconds` | 14400 | (não definido) | Prazo por tentativa na job definition |
| `BenchmarkCheckpointSeconds` | 30 | (não definido) | Intervalo do checkpoint periódico no Spot |
| `JobAttempts` | 5 | (não definido) | Tentativas da `RetryStrategy` do Batch |
| `QueueWaitTimeoutSeconds` | 3600 | 3600 | Tempo máximo em `RUNNABLE` para as razões de cancelamento configuradas |
| `PrepareVcpus` / `PrepareMemoryMiB` / `PrepareEphemeralGiB` | 2 / 8192 / 150 | 2 / 8192 / 150 | Reserva do job de preparação |
| `AvailabilityZoneCount` | 1 | 1 | Sub-redes privadas do Batch (1 ou 2) |
| `InterfaceEndpointAzMode` | `single` | `single` | Endpoints de interface em uma AZ ou em todas |
| `LogRetentionDays` | 14 | 14 | Retenção dos logs |
| `MonthlyBudgetUsd` / `AnomalyThresholdUsd` | 100 / 10 | 100 / 10 | Orçamento mensal e limiar de anomalia |

A suíte de quatro streams exige valores diferentes dos padrões em dois pontos: `scripts/aws/run_dataset_campaigns.py` recusa a submissão se a job definition não tiver exatamente 2 vCPU e 4096 MiB em Fargate Spot com teto de 64 vCPU, e se as regras de cancelamento da fila em `RUNNABLE` permitirem menos de 86.400 segundos (`QUEUE_WAIT_SECONDS`). Portanto, para essa suíte, `QueueWaitTimeoutSeconds` precisa ser elevado no arquivo de parâmetros. Campanhas `online` (`learn_one`) de maior porte, como as execuções completas de INSECTS, já foram dimensionadas com `BenchmarkVcpus=4`, `BenchmarkMemoryMiB=16384` e `MaxVcpus=64` (até 16 tasks simultâneas); como a job definition é única, essa configuração e a da suíte (2 vCPU, 4096 MiB) não coexistem.

## Rede e segurança

- Tasks de benchmark e de preparação rodam em sub-redes privadas sem IP público (`AssignPublicIp: DISABLED`). Não há Internet Gateway, NAT, Elastic IP nem load balancer; o validador de templates falha se algum aparecer.
- O S3 é acessado pelo gateway endpoint. ECR API, ECR Docker e CloudWatch Logs usam endpoints de interface; SageMaker API e MLflow entram com `EnableModelRegistry` e `EnableMlflowTracking`; Kinesis e Feature Store runtime entram com os módulos correspondentes (`infra/cloudformation/modules/interface-endpoints.yaml`).
- O acesso dos workers ao S3 exige a condição `aws:SourceVpce` do gateway endpoint. `TerminateJob` exige a tag `cb-stack`. Cada serviço tem role própria e `iam:PassRole` é restrito.
- Buckets com KMS, versionamento, bloqueio de acesso público e TLS obrigatório. O bucket de dados nega `DeleteObjectVersion` em `datasets/`, `campaigns/` e `releases/` para qualquer principal exceto a role `continual-bench-data-steward`, criada manualmente.
- As fontes de dados Zia são públicas; o GitHub Actions obtém credenciais temporárias por OIDC (OpenID Connect, federação de identidade sem chaves de longa duração).

## Bootstrap

```bash
REPOSITORY=owner/residencia-ia scripts/aws/bootstrap.sh
# ou: make aws-bootstrap REPOSITORY=owner/residencia-ia OVERRIDES=infra/params/runtime-overrides.json
```

O script (`scripts/aws/bootstrap.sh`) é idempotente e executa, em ordem:

1. `aws cloudformation deploy` da stack persistente e ativação da proteção contra término;
2. padrão de conta `containerInsights=enhanced` para clusters ECS e criação das service-linked roles de Batch e ECS;
3. deploy da stack de CI/CD com o repositório informado; `OIDC_PROVIDER_ARN` reutiliza um provedor OIDC existente;
4. publicação de `AWS_ROLE_ARN` como variável do repositório via `gh`, quando disponível;
5. com `OVERRIDES`, cópia do arquivo de parâmetros para `s3://<bucket-de-artefatos>/config/runtime-parameters.json`.

O bootstrap não cria o runtime, não baixa datasets e não inicia campanhas. Etapas fora do CloudFormation, como ativação de tags de alocação de custo e Split Cost Allocation Data, estão em [Custos e Recursos](Custos-e-Recursos#faturamento-observado-e-rateio).

## Deploy por GitHub Actions

O workflow `.github/workflows/deploy.yml` roda em push na `main` e por `workflow_dispatch`. O grupo de concorrência `deploy-main` com `cancel-in-progress: false` serializa os deploys. Os jobs que acessam a AWS usam o ambiente `production` do GitHub e assumem `vars.AWS_ROLE_ARN` por OIDC. Variáveis esperadas no repositório: `AWS_REGION`, `PROJECT`, `RUNTIME_STACK`, `RUNTIME_CONFIG_KEY` e `AWS_ROLE_ARN`.

| Job | O que faz |
|---|---|
| `verify` | Reutiliza `.github/workflows/ci.yml` (ver [Testes e CI/CD](Testes-e-CI-CD)) |
| `build` | Constrói e publica as imagens `linux/amd64` com `scripts/aws/build_images.sh` (tag imutável `git-<sha>`, digest resolvido no ECR), valida a imagem `runtime` com `selftest` e a `ml` com `scripts/aws/check_inference_image.sh`, empacota templates, Lambda, Glue e Flink |
| `deploy` | Aplica o change set do runtime com `scripts/aws/deploy_change_set.sh`, usando os parâmetros do objeto indicado por `RUNTIME_CONFIG_KEY` |
| `release` | `scripts/aws/write_release.py` confere saídas da stack e digests e grava `releases/<sha>/release.json` uma única vez, movendo `releases/latest.json` |

O manifesto da release contém o Git SHA, as imagens por digest, o backend de computação do benchmark (`batch-fargate-spot` ou `batch-fargate`), as revisões das job definitions e a versão da Lambda de controle. O plano de cada campanha fixa a release; um deploy durante a campanha não altera as tentativas em andamento.

Rollback sem rebuild: `scripts/aws/rollback.sh <sha>` reaplica templates e artefatos de uma release anterior, desde que as revisões Batch, a versão da Lambda e os digests ainda existam.

## Preparação de datasets

O workflow de datasets (`infra/statemachines/dataset.asl.json`) executa: `ResolveSource` (decide entre reaproveitar, publicar, preparar ou adquirir), `Acquire` (CodeBuild), `VerifyRaw`, `EnsureEndpoints`, `Prepare` (job Batch privado), `PublishManifest` e `SnapshotReady`. Snapshots válidos são reaproveitados.

```bash
scripts/aws/start_dataset.sh infra/examples/dataset-social-smoke.json        # make aws-dataset-smoke
scripts/aws/start_dataset.sh infra/examples/dataset-social-full.json [sufixo] # make aws-dataset-full SUFFIX=...
```

| Modo (`request.mode`) | Comportamento |
|---|---|
| `download` | O CodeBuild baixa a fonte do catálogo (repositório, revisão e arquivo fixados), com limite de bytes, retentativas, retomada por `Range` e verificação SHA-256 antes do upload condicional |
| `import` | Copia de `s3://<bucket-de-dados>/imports/...` e verifica o SHA-256 contra o catálogo (`dataset-social-import.json`) |
| `prepare` | Usa dados brutos já publicados e falha se não existirem (`dataset-social-prepare-only.json`) |

O nome da execução é `<dataset>-<scope>-<mode>`, com `--<sufixo>` opcional. Os exemplos em `infra/examples/` cobrem `social`, `fraud` e `ecommerce` nos escopos `smoke` e `full`. O snapshot fica pronto quando existe `datasets/<dataset>/<revisão>/manifests/<versão>/<escopo>/_SUCCESS.json`, escrito por último.

A suíte de streams ordenados (Electricity, Airlines e duas variantes INSECTS) não usa esse workflow: as fontes são publicadas a partir de uma máquina com acesso a elas por `continual-bench aws publish-matrix-source`, e os SHA-256 retornados entram em `request.source_digests`. Ver [Datasets](Datasets).

## Campanhas de benchmark

### Pedido de campanha

Um pedido é um JSON com o objeto `request`. Campos usados nos exemplos de `infra/examples/`:

| Campo | Significado |
|---|---|
| `campaign_id` | 1 a 40 caracteres minúsculos, dígitos ou hífens (`scripts/aws/campaign_request.py`) |
| `release` | SHA de 40 caracteres de uma release publicada; `latest` é recusado em campanhas Zia |
| `snapshot` | Dataset, revisão, versão de preparação e escopo (campanhas Zia) |
| `matrix.kind` | `zia` (padrão), `experiment_matrix`, `project_suite`, `stream` ou `benchmark` (`src/continual_bench/application/cloud/contracts.py`) |
| `matrix.policies` / `matrix.path` | Políticas Zia ou arquivo de matriz em `configs/matrices/` |
| `max_parallel` | Unidades simultâneas no Map; sem valor, 32 no Spot e 4 no On-Demand (`planning.py`) |
| `max_attempts` | Jobs reais por unidade antes de desistir (1 a 5, padrão 2) |
| `queue_timeout_seconds` / `run_timeout_seconds` | Espera na fila (600 a 86.400 s, padrão 3600) e prazo por tentativa (`AttemptDurationSeconds` do job, padrão 14.400 s) |
| `source_digests` | SHA-256 das fontes publicadas, por dataset (matrizes e streams) |

| Exemplo | Conteúdo |
|---|---|
| `campaign-social-smoke.json`, `campaign-fraud-smoke.json`, `campaign-ecommerce-smoke.json` | 12 políticas Zia no snapshot `smoke`, `max_parallel=12`, prazo de 3600 s |
| `campaign-social-full.json`, `campaign-fraud-full.json`, `campaign-ecommerce-full.json` | 12 políticas Zia no snapshot `full`, prazo de 172.800 s (48 h) |
| `campaign-social-full-cost-aware.json` | Cinco políticas por custo (`cara`, `cara_cumulative`, `cara_periodic`, `upf`, `oracle`) |
| `campaign-social-spot-acceptance.json` | `drift_adwin` e `cara` no `smoke`, `max_attempts=4`, para ensaios de interrupção |
| `campaign-binary-matrix.json` | Matriz `configs/matrices/binary-comparison.yaml` (`experiment_matrix`) |

As políticas estão descritas em [Modelos e Políticas](Modelos-e-Politicas).

### Iniciar

```bash
export RELEASE_SHA=<sha-publicado>
scripts/aws/start_campaign.sh infra/examples/campaign-social-smoke.json [sufixo]   # make aws-campaign-smoke
CAMPAIGN_ID=novo-id scripts/aws/start_campaign.sh infra/examples/campaign-social-full.json
```

`start_campaign.sh` materializa o pedido com `RELEASE_SHA` e `CAMPAIGN_ID` (variáveis opcionais que substituem os campos do JSON) e chama `StartExecution` com o nome `<campaign_id>` ou `<campaign_id>--<sufixo>`.

### O que o workflow faz

A definição `infra/statemachines/benchmark.asl.json` segue esta sequência:

1. `Plan` (Lambda `plan_campaign`): resolve a release, recusa campanhas explicitamente canceladas, verifica o snapshot ou as fontes, gera uma unidade por combinação com `run_id` determinístico (hash da campanha e do contrato de execução) e nome de job `cb-<campaign_id>-<run_id>`, verifica capacidade e quota e publica `plan.json` uma única vez. Falhas levam a `PreflightFailed`.
2. `EnsureEndpoints`: cria ou reutiliza a stack de endpoints de interface e espera até ficarem disponíveis.
3. `InlineRuns` (até 40 unidades) ou `DistributedRuns` (Distributed Map acima disso, lendo os itens do S3): para cada unidade, reconcilia o estado antes de submeter, submete o job Batch com `submitJob.sync`, registra a tentativa e decide entre sucesso, nova submissão ou desistência. A concorrência do Map é `plan.concurrency`.
4. `Consolidate`: lê os manifestos finais, grava `summary.json` e a linha da tabela `campaigns`.
5. `VerifyNoActiveTasks`: confere até 10 vezes, com 60 s de intervalo, que não restam jobs ativos nem tasks em execução.
6. `CampaignSucceeded` se todas as unidades foram concluídas e verificadas; caso contrário `CampaignIncomplete` ou `TasksStillActive`.

No Fargate Spot, o planejador submete todas as unidades até o limite de concorrência; o Batch inicia quantas couberem no teto `MaxVcpus` e na quota regional e mantém as demais em `RUNNABLE`. No On-Demand, o planejador recusa a campanha se o paralelismo pedido não couber (`admit_capacity` em `src/continual_bench/application/cloud/planning.py`).

### Suíte de quatro streams

A seleção AWS atual usa Electricity, Airlines e duas variantes INSECTS com 11 políticas e três seeds: 132 unidades em oito campanhas (84 base e 48 que reutilizam as matrizes CARA). O coordenador é `scripts/aws/run_dataset_campaigns.py`, com os subcomandos `start-base`, `status` e `advance`; ele não faz deploy e verifica release, imagem, job definition, fila e quota antes de submeter. As condições de admissão e as duas fases estão em [Operação e Diagnóstico](Operacao-e-Diagnostico#suíte-de-quatro-streams); a metodologia está em [Metodologia Experimental](Metodologia-Experimental) e as evidências exportadas em [Tracking e Evidências](Tracking-e-Evidencias).

## SageMaker Studio, MLflow e catálogo

- `scripts/aws/start_studio_campaign.sh <campanha.json> [dataset.json]` envia o pedido ao pipeline `continual-bench-campaign`, cujas etapas `PrepareDataset`, `RunBatchCampaign` e `VerifyResults` acionam e acompanham os mesmos workflows Step Functions. Cancelar pelo Studio cancela o workflow associado.
- `scripts/aws/deploy_native_pipeline.sh` implanta o pipeline `continual-bench-native` (`ProfileSnapshot`, `TrainBaseline`, `TuneHyperparameters`) para o snapshot Social.
- As métricas ficam no MLflow App serverless `continual-bench-mlflow`. `make aws-mlflow-ui` (`continual-bench aws mlflow-url`) gera uma URL pré-assinada; `make aws-studio-ui` (`continual-bench aws studio-url`) gera links para o Studio, o portal do Unified Studio e o MLflow. A aba Experiments do menu principal do Studio consulta servidores MLflow clássicos e não lista o app serverless.
- A stack `continual-bench-studio-catalog` publica no Unified Studio cada novo snapshot (`_SUCCESS.json`) e vincula as avaliações (`final.json`) por S3, EventBridge, SQS FIFO e Lambda.

A navegação e as tags do MLflow estão em [Tracking e Evidências](Tracking-e-Evidencias) e em [Operação e Diagnóstico](Operacao-e-Diagnostico#observação).

## Módulos opcionais

| Módulo | Uso | Parada |
|---|---|---|
| Streaming | `KinesisTransport` implementa a mesma porta do transporte Kafka local; Flink (`infra/flink/event_time_windows.py`) calcula janelas por tempo do evento | `scripts/aws/flink.sh stop`; a aplicação é criada parada |
| Feature Store | Feature group offline, materialização point-in-time (`infra/glue/materialize_features.py`) e Data Quality; smoke em `scripts/aws/feature_store_smoke.py` | `EnableOnlineStore=false` ao fim do teste |
| ML avançado | `scripts/aws/sagemaker_jobs.py processing|training|tuning|inference` com perfis em `infra/sagemaker/profiles/` | `inference --delete-after` remove o endpoint |
| BI | Dataset e análise no Quick Sight sobre o Athena | Desativar a flag |

Todos são Experimental: há código e testes unitários, mas não há registro de ativação deles na conta.

Detalhes por módulo:

- **Streaming** (`EnableStreaming=true`): Kinesis Data Streams `continual-bench-events` (1 shard, KMS). `KinesisTransport` implementa a mesma porta `contract()`/`events()` do transporte Kafka; cada evento leva `event_id` e `seq`, o consumidor restaura a ordem com buffer limitado, descarta redeliveries verificadas, recusa lacunas e payloads alterados e grava o cursor no S3 (retomada com `AFTER_SEQUENCE_NUMBER`, só para o cursor da mesma run). O Firehose arquiva o stream em `archive/streams/` (GZIP, KMS). A aplicação Flink 1.20 (`infra/flink/event_time_windows.py`) calcula janelas por tempo do evento com watermark, tabela separada de eventos atrasados e checkpoints a cada 60 s; saídas em `archive/windows/` do bucket de artefatos. A aplicação nasce em `READY`; `scripts/aws/flink.sh start [--restore-latest]`, `status` e `stop` controlam o custo. A campanha Zia de referência continua lendo o Parquet temporal preparado.
- **Feature Store** (`EnableFeatureStore=true`, online com `EnableOnlineStore=true`): feature group offline `continual-bench-zia-social-edges` (Iceberg, KMS) com as features que o `prepare()` calcula só com datas anteriores; `event_time` é enviado como texto ISO 8601 em UTC. O Glue ETL `continual-bench-materialize-features` gera Parquet imutável em `features/materialized/<versão>/<job run>/` e falha se houver feature posterior ao instante da predição. O Glue Data Quality `continual-bench-prepared-social` verifica contagem, completude, domínio dos rótulos e faixas. Smoke: `scripts/aws/feature_store_smoke.py --group continual-bench-zia-social-edges --parquet <smoke.parquet> --online`.
- **ML avançado** (`EnableMl=true`, endpoint com `EnableEndpoint=true`): a imagem `ml` reaproveita as dependências da `runtime` e muda só o ponto de entrada (`process`, `train`, `serve`). O perfil avulso de tuning minimiza `validation:log_loss` com no máximo 8 jobs (2 em paralelo) num split de validação cronológico anterior ao holdout. Com `EnableModelRegistry=true`, os jobs Zia publicam um candidato `PendingManualApproval` referenciando snapshot, run, relatório, `model.tar.gz` e a imagem `ml` por digest; antes, o worker confirma que o `ZiaPredictor` reproduz as predições arquivadas do holdout.
- **BI** (`EnableBi=true`, `QuickSightPrincipalArn=...`): exige assinatura Amazon Quick Sight Enterprise em `us-east-1` e um usuário ou grupo Quick Sight. A role de serviço do Quick Sight precisa de acesso ao Athena, ao bucket de dados (incluindo `athena-results/`) e à chave KMS (Quick Sight, Manage, Security & permissions). O módulo cria a fonte Athena, o dataset que junta métricas por política com as estimativas de custo por job e a análise "policy comparison".

## Cancelamento e remoção

```bash
scripts/aws/cancel_campaign.sh <campaign_id> [nome-da-execução]   # make aws-cancel CAMPAIGN=<id>
scripts/aws/flink.sh stop                                         # se o streaming estiver ativo
scripts/aws/teardown.sh                                           # make aws-teardown
scripts/aws/teardown.sh --with-cicd                               # make aws-teardown-all
```

O cancelamento grava `campaigns/<id>/cancelled.json` (escrita única), para a execução Step Functions com o erro `CampaignCancelled` e reconcilia até 20 vezes com 30 s de intervalo, encerrando os jobs da campanha. Os workers param no próximo ponto seguro (código de saída 76). Uma campanha cancelada não pode ser replanejada com o mesmo ID. Detalhes em [Operação e Diagnóstico](Operacao-e-Diagnostico#cancelamento).

`teardown.sh` para o Flink, cancela as execuções de benchmark em andamento, espera a remoção da stack de endpoints e remove o runtime. Preserva a stack persistente (buckets, KMS, ECR), os log groups de jobs e aquisição e os grupos do Model Registry. `--with-cicd` também remove a stack de OIDC. As stacks do Studio são removidas separadamente.

## Alvos do Makefile

| Alvo | Comando | Contata a AWS |
|---|---|---|
| `aws-validate` | cfn-lint e invariantes; `validate_asl.py` offline | Não |
| `aws-smoke` | `continual-bench aws local-smoke` com fixtures e armazenamento local | Não |
| `aws-images` | Build local das imagens `runtime` e `ml` e checagem de inferência | Não |
| `aws-lambda` | Zip reproduzível da Lambda de controle | Não |
| `aws-bootstrap` | `scripts/aws/bootstrap.sh` | Sim |
| `aws-dataset-smoke`, `aws-dataset-full`, `aws-dataset-fraud-full`, `aws-dataset-ecommerce-full` | `start_dataset.sh` com o exemplo correspondente | Sim |
| `aws-campaign-smoke`, `aws-campaign-full`, `aws-campaign-fraud-full`, `aws-campaign-ecommerce-full`, `aws-campaign-full-cost-aware`, `aws-campaign-matrix` | `start_campaign.sh` com o exemplo correspondente | Sim |
| `aws-mlflow-ui`, `aws-studio-ui` | URLs pré-assinadas | Sim |
| `aws-cancel` | `cancel_campaign.sh $(CAMPAIGN)` | Sim |
| `aws-teardown`, `aws-teardown-all` | `teardown.sh` | Sim |

Os subcomandos `continual-bench aws` disponíveis são `acquire`, `prepare`, `publish-snapshot`, `publish-matrix-source`, `run-unit`, `selftest`, `local-smoke`, `plan`, `refresh-prices`, `query`, `reconcile`, `cancel`, `mlflow-url` e `studio-url` (`src/continual_bench/orchestration/aws_cli.py`). Os comandos que acessam a AWS exigem o extra `aws` (ver [Instalação e Configuração](Instalacao-e-Configuracao)).

## Validações registradas

Registros datados do que foi exercitado na conta AWS (região `us-east-1`). Horários em BRT.

**09/11/2026, validação local inicial** (macOS arm64, Docker Desktop com emulação `linux/amd64`). A suíte padrão passou com 669 testes aprovados e 50 desmarcados (`services` e `benchmark`); `make aws-validate` aprovou 12 templates; o validador ASL offline aprovou as 2 definições; `continual-bench aws local-smoke` concluiu 4 de 4 runs com execuções sobrepostas; o zip da Lambda de controle foi reproduzível (mesmo SHA-256) e planejou a matriz binária de 360 unidades em modo distribuído. As imagens `runtime` e `ml` foram construídas para `linux/amd64`, mas a execução local sob QEMU não foi demonstrada: o selftest travou e `import pyarrow` falhou na emulação. A execução da imagem passou a ser validada no job `build` em host x86_64 nativo.

**09/11/2026, primeira validação na AWS.** O pipeline de CI/CD de então publicou a release `f1cb05e1bd22005f3994869fd664e4159a122cb2`. O dataset `social-smoke-prepare` terminou com `SUCCEEDED` e publicou o Parquet de 10.000 linhas (job Batch `3c5b80fc-0e5e-4a6b-bad9-25c30bdefcbe`), com verificação independente do SHA-256 do manifesto. As campanhas seguintes usaram o snapshot `smoke` de 10.000 linhas e 4 políticas (`frozen`, `drift`, `incremental`, `periodic`):

| Campanha | Release | Resultado | O que se observou |
|---|---|---|---|
| `social-smoke-v1` (11:22 a 11:24) | `f1cb05e1...` | `FAILED`, 0/4, `CampaignIncomplete` | 8 jobs com código 1 (1 retry por unidade). `AccessDeniedException` em `sagemaker:ListModelPackages`, porque a ação era concedida em `model-package-group` e avaliada sobre `model-package` |
| `social-smoke-v3` (12:38 a 12:40) | `92c7c539502dc6426a02065225ab4126cb24052a` | `FAILED`, 0/4, `CampaignIncomplete` | `CreateModelPackage` negado: o SageMaker não podia ler a imagem no ECR `continual-bench-ml` (`ecr:BatchGetImage`) |
| `social-smoke-v4` (13:12 a 13:14) | `a43d95ffe3deb864f96b17d706aac4dd5d6d2f86` | `SUCCEEDED`, 4/4 na primeira tentativa | Jobs `SUCCEEDED`, tasks ECS com código 0, janela comum das 4 tasks de 7,4 s |

Em `v1` e `v3`, o Step Functions fez um retry por unidade e a reconciliação terminou com `clean: true` (zero jobs ativos, zero tasks em execução). As ENIs das tasks foram observadas em sub-rede privada sem `Association.PublicIp`, com rota ao gateway endpoint do S3. A correção de IAM (`ecr:BatchGetImage` e `ecr:DescribeImages` só em `continual-bench-ml`, confirmada por simulação da role) foi aplicada na conta e depois incorporada ao template, e a verificação de drift da role `BenchmarkJobRole` retornou `IN_SYNC`. Em `v4` foram lidos os 4 `final.json`, verificados no S3 os 4 arquivos de modelo com KMS, as 4 linhas de tentativa e 16 objetos analíticos, e o carregador nativo reproduziu 256 predições por modelo com diferença máxima 0. O SageMaker retornou `Completed` e `PendingManualApproval` para a versão 1 de cada grupo `continual-bench-zia-social-{frozen,drift,incremental,periodic}-smoke`. Nesse build, o Docker Hub devolveu HTTP 429 duas vezes e a terceira tentativa usou temporariamente o espelho oficial do ECR Public com o mesmo digest da imagem base. Não foram exercitados: cancelamento manual, snapshot sem marcador e correlação das gravações com linhas dos logs de acesso do bucket.

**16/11/2026, verificações da revisão.** A suíte padrão passou com 785 testes e 50 fora da seleção, o Ruff passou e `scripts/aws/validate_templates.py` aprovou 15 templates. O workflow `Build and deploy` concluiu os quatro jobs na execução de 12/11/2026. O runtime estava em `UPDATE_COMPLETE` com `AvailabilityZoneCount=2` e `InterfaceEndpointAzMode=single`: duas sub-redes privadas (`us-east-1a` e `us-east-1b`) na mesma tabela de rotas com o gateway endpoint do S3, e cinco endpoints de interface em `us-east-1a`, `available`. Quatro jobs `cb-network-smoke-20261116132423-*` terminaram `SUCCEEDED`, com `S3_GET_OK` nas duas zonas e o digest da imagem em execução igual ao da URI no ECR. O arquivo de overrides no S3 ainda indicava uma AZ para o Batch e omitia o modo dos endpoints.

**Campanhas Zia integrais, resumos de 16/11/2026.** As campanhas `zia-{social,fraud,ecommerce}-full-v1` tinham `complete: true` e 7/7 runs concluídas por dataset. Os resumos `full-v2` tinham `complete: false` (4/7 no Social, 6/7 no Fraud e 6/7 no E-commerce). Os pedidos versionados atuais listam 12 políticas por dataset, portanto não reproduzem essas campanhas de 7 runs.

**16/11/2026, SageMaker Pipelines, MLflow e catálogo.** `studio-matrix-151104` terminou com `Succeeded` em `PrepareDataset`, `RunBatchCampaign` e `VerifyResults` (unidade da matriz 1/1, run MLflow `f26aa09c40134af68eff22de3a708e72` `FINISHED` com 36 métricas `progress/*` e `final/*`). `native-smoke-155750` terminou com `Succeeded` em `ProfileSnapshot`, `TrainBaseline` e `TuneHyperparameters`; os dois trials de tuning ficaram `Completed`, com `validation:log_loss` de 0,91775048 (melhor) e 0,91872990. O app `continual-bench-mlflow` estava no estado `Created`, e a DataZone retornou listagens `ACTIVE` para Social, Fraud e E-commerce `full` e para o Social `smoke`. Um objeto de teste acionou S3, EventBridge, SQS FIFO e Lambda; a repetição de eventos confirmou o reaproveitamento de listagem e vínculo, e a função restaurou as tags de catálogo do run E-commerce `f891e314b890465289e7e51a86ba2c3f` (253 métricas preservadas).

**Suíte de quatro streams.** O conjunto exportado em `evidencias/aws-benchmarks/` registra 132 unidades concluídas em 133 tentativas operacionais (uma publicação de métricas foi retomada sem novo treinamento), na revisão `f1092ee0232a6d16504b3a2211a037c41a02068a`. A campanha INSECTS da Semana 10 concluiu 66 de 72 unidades e não substitui um teste de interrupção e retomada Spot.

Ordem recomendada numa conta nova: bootstrap e OIDC; workflow de deploy (conferir os quatro jobs, o digest no ECR e `releases/<sha>/release.json`); dataset smoke e `_SUCCESS.json`; campanha smoke com `RELEASE_SHA` conferido (IDs dos jobs, `describe-jobs` `SUCCEEDED`, `describe-tasks` `STOPPED`, ENIs sem IP público); falha induzida, cancelamento e retry com sufixo; e, antes de uma campanha `full`, pico de RSS e métricas `system/` do smoke, disco contra `BenchmarkEphemeralGiB`, quota Fargate e orçamento.

## Limitações

- Não há registro de uma interrupção Spot real nem de um cancelamento manual exercitados na conta; ambos são cobertos por testes automatizados.
- Em 16/11/2026, o arquivo de parâmetros no S3 ainda indicava uma AZ para o Batch enquanto o runtime implantado usava duas. Confira o objeto `config/runtime-parameters.json` antes de cada deploy.
- A correlação das gravações dos workers com linhas individuais dos logs de acesso do bucket ficou pendente nas validações de 09/11/2026.
- Os módulos opcionais e a conciliação com o CUR não têm evidência de execução registrada.

Ver também [Limitações e Trabalhos Pendentes](Limitacoes-e-Trabalhos-Pendentes).
