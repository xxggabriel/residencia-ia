# Plataforma AWS gerenciada do continual-bench

Perfil de execução em us-east-1 que adquire e prepara datasets, executa campanhas de
benchmark em paralelo no AWS Batch sobre Fargate, publica resultados consultáveis no S3,
Glue e Athena, registra candidatos no SageMaker Model Registry e concilia custos com o CUR 2.0.
O perfil roda sem Airflow, ZenML, MinIO, Kafka, MLflow ou servidor HTTP de workers. Os fluxos
locais (Compose, DAGs, MLflow) continuam inalterados.

| Documento | Conteúdo |
|---|---|
| [plan.md](plan.md) | Plano registrado antes da implementação |
| [operations.md](operations.md) | Bootstrap, CI/CD, rollback, datasets, campanhas, cancelamento, consultas e remoção |
| [costs.md](costs.md) | CUR 2.0, Split Cost Allocation Data, tags, Budgets, anomalias, Price List e conciliação |
| [modules.md](modules.md) | Streaming, Feature Store, ML avançado e BI: ativação, custo e parada |
| [validation.md](validation.md) | O que foi validado localmente e o que ainda depende da conta AWS |

## Três fluxos independentes

```
GitHub main ──push/merge──▶ CodePipeline V2 ─▶ testes, lint, cfn-lint, ASL, smoke local
                                              ─▶ imagens x86_64 (tag git-<sha>, digest) ─▶ change set ─▶ releases/<sha>
operador ──start-execution──▶ state machine de datasets ─▶ resolver ─▶ adquirir (CodeBuild, fora da VPC)
                                                        ─▶ validar ─▶ preparar (Batch privado) ─▶ manifesto + _SUCCESS
operador ──start-execution──▶ state machine de benchmark ─▶ plano no S3 ─▶ Map (1 job Batch/Fargate por run)
                                                        ─▶ consolidar ─▶ verificar tasks paradas
EventBridge (Batch e Step Functions) ─▶ Lambda de controle ─▶ registro, SNS e reconciliação (nunca inicia campanhas)
```

O merge publica imagens e atualiza a infraestrutura. Ele não baixa datasets nem inicia
campanhas: os buildspecs não contêm `start-execution`, `acquire` nem `submit-job`, e um teste
garante isso (`tests/test_aws_infra.py`).

## Fronteiras dos módulos Python

| Camada | Módulo | Responsabilidade |
|---|---|---|
| `infrastructure` | `object_store.py` | Porta de armazenamento e implementação local com criação condicional |
| `infrastructure` | `stream_order.py` | Ordem, duplicatas e lacunas de eventos, independente do transporte |
| `infrastructure.aws` | `s3`, `batch`, `model_registry`, `kinesis`, `feature_store`, `athena`, `quotas`, `pricing`, `emf`, `session` | Adaptadores boto3 |
| `application.cloud` | `contracts`, `layout` | Entradas validadas, manifesto de release e layout das chaves S3 |
| `application.cloud` | `sources`, `snapshots` | Fontes do catálogo, download limitado e verificado, preparação e manifesto do snapshot |
| `application.cloud` | `planning`, `worker` | Plano da campanha, `run_id` determinístico e execução de uma run |
| `application.cloud` | `lifecycle`, `records`, `costs` | Reconciliação, consolidação, registros do Athena e contabilidade em USD |
| `application.cloud` | `streaming`, `features`, `fixtures` | Semântica das janelas, checagens point-in-time e fixtures do smoke |
| `orchestration` | `aws_control`, `aws_cli`, `sagemaker` | Lambda de controle, `continual-bench aws ...` e contrato SageMaker |

`core`, `models` e `detectors` não importam SDKs da AWS. O teste de arquitetura existente
continua valendo. O worker reaproveita `evaluation.zia.run`, `prediction_manifest`,
`tracking_report`, `write_zia_artifacts`, o catálogo `datasets.zia`, `prepare()`,
`load_manifest()` e `ZiaPredictor`. A publicação MLflow e a AWS compartilham
`zia_publication.run_documents`, extraído sem mudar o conteúdo publicado. O carregador nativo
substitui `mlflow.pyfunc` no caminho AWS. A matriz genérica entra pelo mesmo planejador:
cada caso de `configs/matrices/*.yaml` vira uma unidade independente executada por
`application.matrix._execute_case`.

## Protocolo científico preservado

O transporte AWS só move entradas e saídas. A ordem temporal, o desempate por ID na preparação,
as features disponíveis no instante da predição, a avaliação antes da atualização, o ADWIN,
o holdout congelado e as métricas vêm do mesmo código. O teste
`test_worker_matches_the_local_protocol` compara uma run do caminho AWS com uma run local:
janelas, digests das partes de predição, digests das versões de modelo e matriz de confusão do
holdout são idênticos.

## Layout no S3 (bucket de dados)

```
datasets/<dataset>/<revisão>/raw/<arquivo>                         fonte pública, verificada por SHA-256
datasets/<dataset>/<revisão>/raw/_manifest.json                    evidência da aquisição (write-once)
datasets/<dataset>/<revisão>/prepared/<versão>/<escopo>/data.parquet
datasets/<dataset>/<revisão>/prepared/<versão>/<escopo>.json       manifesto do prepare() (bytes originais)
datasets/<dataset>/<revisão>/prepared/<versão>/<escopo>.provenance.json  digest da imagem, schema, código
datasets/<dataset>/<revisão>/manifests/<versão>/<escopo>/snapshot.json    URIs, versões, hashes, contagens, licença
datasets/<dataset>/<revisão>/manifests/<versão>/<escopo>/_SUCCESS.json    marcador, escrito por último
releases/<sha>/release.json, releases/latest.json
campaigns/<id>/request.json, plan.json, plan-items.json, units/<run_id>.json, preflight/<data>.json
campaigns/<id>/runs/<run_id>/attempts/<job>-<tentativa>/...        evidência de cada tentativa
campaigns/<id>/runs/<run_id>/final.json                            manifesto final (If-None-Match: *)
campaigns/<id>/runs/<run_id>/events/...                            ciclo de vida vindo do EventBridge e do workflow
campaigns/<id>/summary.json, reconciliation/<data>.json
analytics/<tabela>/<campanha>__<run_id>[__<job>].json              JSON Lines para o Glue/Athena
```

## Infraestrutura

| Stack | Template | Ciclo de vida |
|---|---|---|
| `continual-bench-persistent` | `infra/cloudformation/persistent.yaml` | Bootstrap. KMS, buckets de dados, artefatos, billing e logs de acesso, ECR. Tudo com `Retain` e proteção contra término. |
| `continual-bench-cicd` | `infra/cloudformation/cicd.yaml` | Bootstrap. CodeConnections, CodeBuild, CodePipeline V2 e roles por estágio. |
| `continual-bench-runtime` | `infra/cloudformation/root.yaml` | Pipeline. Aninha `network`, `execution`, `observability`, `analytics` (Core) e `billing`, `streaming`, `features`, `ml`, `bi` (ativáveis). |

Não há ciclo entre pipeline, ECR e job definition. O ECR fica na stack persistente, o
pipeline publica imagens por digest e passa o digest como parâmetro da stack de execução.
Cada mudança de digest cria uma revisão nova da job definition, gravada no manifesto de
release depois do deploy.

## Segurança

- **Rede**: as tasks rodam em sub-redes privadas sem rota para a internet, com
  `AssignPublicIp=DISABLED`. Não há Internet Gateway, NAT, Elastic IP nem load balancer.
  O S3 é acessado pelo gateway endpoint, cuja política inclui o bucket de camadas do ECR. ECR
  API, ECR Docker e CloudWatch Logs passam por endpoints de interface com DNS privado e política
  restrita à conta. SageMaker API, Kinesis, Feature Store runtime, KMS e Secrets Manager ganham
  endpoints só quando o módulo correspondente está ativo. O security group dos workers só sai
  por HTTPS para os endpoints e para a prefix list do S3.
- **IAM**: há roles separadas para CodePipeline, cada estágio do CodeBuild (teste, build,
  deploy, release), execução do CloudFormation, Step Functions, execução da task, job de
  benchmark, job de preparação, aquisição, Lambda de controle, Firehose, Flink, Feature Store,
  Glue e SageMaker. As roles dos workers exigem `aws:SourceVpce` do gateway endpoint em todo
  acesso ao S3. `TerminateJob` exige a tag `cb-stack`. `iam:PassRole` só aparece no deploy,
  restrito à role do CloudFormation, e na role do CloudFormation, restrito às roles do projeto
  e ao serviço de destino.
- **Dados**: KMS com rotação em buckets, logs, SNS, Kinesis e ECR. Versionamento, bloqueio de
  acesso público, TLS obrigatório e lifecycle em todos os buckets. O bucket de dados nega
  `DeleteObjectVersion` em `datasets/`, `campaigns/` e `releases/` para qualquer principal
  exceto uma role `continual-bench-data-steward` criada manualmente. O bucket policy não exige
  VPC endpoint, para permitir as gravações autorizadas do CodeBuild de aquisição, do Firehose,
  do Lambda e do Data Exports, que ocorrem fora da VPC.
- **Segredos**: nenhum módulo precisa de segredo. As fontes Zia são públicas e a autenticação
  GitHub é feita pela CodeConnections. Um token futuro de fonte privada deve ir para o Secrets
  Manager, ser lido pelo projeto de aquisição como variável `secrets-manager` e ter o endpoint
  ativado com `EnableKmsAndSecretsEndpoints`.

## APIs sem suporte nativo no CloudFormation

| Configuração | Como é tratada | Motivo |
|---|---|---|
| Container Insights para o cluster ECS criado pelo Batch | `bootstrap.sh`: `aws ecs put-account-setting-default containerInsights=enhanced`, idempotente | O Batch cria o cluster; não há propriedade de cluster nem recurso de account setting no CloudFormation |
| Autorização do GitHub App na CodeConnections | Passo manual documentado e `wait_connection.sh` | O handshake OAuth exige o console |
| Ativação de tags de alocação | CLI `ce update-cost-allocation-tags-status` na conta pagadora | Só a pagadora pode ativar; as tags só existem depois de aparecer no billing |
| Split Cost Allocation Data para ECS | Console da conta pagadora | Não há API pública nem recurso CloudFormation |
| Início e parada da aplicação Flink | `scripts/aws/flink.sh` | O CloudFormation cria a aplicação em READY; iniciar é uma decisão de custo |
| Jobs SageMaker (Processing, Training, Tuning) | `scripts/aws/sagemaker_jobs.py` a partir de perfis versionados | São execuções, não recursos de infraestrutura |
| Model package groups do Registry | Criados pelo worker sob demanda (`ensure_group`) | Ficam fora das stacks e sobrevivem à remoção delas |
