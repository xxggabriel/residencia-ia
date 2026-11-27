# Operação e Diagnóstico

Esta página reúne o que é preciso para acompanhar e recuperar campanhas no perfil AWS: onde ficam os registros, o significado de cada estado, os mecanismos de retentativa, checkpoint e idempotência, e um roteiro de diagnóstico por sintoma. A arquitetura e os comandos de início estão em [Execução na AWS](Execucao-na-AWS).

## Onde olhar

| Pergunta | Onde |
|---|---|
| A campanha terminou? Quantas unidades concluíram? | `campaigns/<id>/summary.json` (`succeeded`, `total`, `complete`, `analytics_errors`) e o estado da execução no Step Functions |
| Qual `run_id` corresponde a cada política? | `campaigns/<id>/plan.json` (rótulos de cada unidade) |
| O que aconteceu com cada job Batch? | `campaigns/<id>/runs/<run_id>/events/` e a tabela Athena `attempts` |
| Quais tentativas houve e em que estado terminaram? | `campaigns/<id>/runs/<run_id>/attempts/<job>-<n>/` (`status.json` e evidências) e, no Spot, `runs/<run_id>/spot/attempt-NNN/claim.json` |
| Há checkpoint confirmado? | `campaigns/<id>/runs/<run_id>/spot/checkpoints/` (manifestos em `spot/checkpoints/manifests/`) |
| A unidade foi publicada? | `campaigns/<id>/runs/<run_id>/final.json` (escrita única) e `analytics.json` (marcador das linhas do Athena) |
| Restou algum job ou task ativo? | `campaigns/<id>/reconciliation/<data>.json` (`clean`, `active_jobs`, `running_tasks`) |
| Por que o worker falhou? | CloudWatch Logs `/continual-bench/batch`, stream do job; aquisição em `/continual-bench/acquisition` |
| Qual capacidade o preflight admitiu? | `campaigns/<id>/preflight/<data>.json` |
| Métricas da run | MLflow App `continual-bench-mlflow`; o `final.json` aponta para o run |
| Snapshot pronto? | `datasets/<dataset>/<revisão>/manifests/<versão>/<escopo>/_SUCCESS.json` e `snapshot.json` |
| Release usada | `releases/<sha>/release.json` e `releases/latest.json` |

O layout completo das chaves está em `src/continual_bench/application/cloud/layout.py`. As tabelas analíticas ficam em `analytics/<tabela>/<campanha>__<run_id>[__<job>].json` (JSON Lines lidos pelo Glue e pelo Athena).

## Estados e códigos

### Códigos de saída do worker

Definidos em `src/continual_bench/application/cloud/worker.py`:

| Código | Significado | Tratamento |
|---:|---|---|
| 0 | Unidade concluída e publicada | Sucesso, se houver `final.json` |
| 1 | Falha genérica | A `RetryStrategy` do Batch encerra o job; o workflow pode submeter outro job até `max_attempts` |
| 2 | Contrato de execução inválido | Falha sem retentativa útil; corrija o pedido |
| 3 | Snapshot indisponível ou divergente | Prepare ou corrija o snapshot |
| 75 | Interrompido (SIGTERM do Fargate Spot) após gravar o estado `INTERRUPTED` | `RETRY` pela `RetryStrategy` do job |
| 76 | Campanha cancelada pelo operador | Encerramento sem nova tentativa |

### Estados registrados por unidade

`record_attempt` (`src/continual_bench/application/cloud/lifecycle.py`) classifica o resultado de cada unidade após o job:

| Estado | Condição |
|---|---|
| `SUCCEEDED` | Existe `final.json` |
| `FAILED_PUBLICATION` | O job terminou com sucesso, mas não há `final.json`. Código zero sem manifesto final não é sucesso |
| `TERMINATING` | Ainda há job ativo após o fim do `.sync` |
| `FAILED` | Demais casos |

Antes de cada submissão, `reconcile_unit` decide entre `skip` (já existe `final.json`, a unidade é reaproveitada), `wait` (há job ativo com o mesmo nome), `submit` ou `exhausted` (jobs terminados já atingiram `max_attempts`). Em `consolidate`, campanhas Spot também verificam o checksum do relatório, o índice de artefatos e as linhas analíticas; uma falha nessa verificação marca a unidade como `PUBLICATION_FAILED` e a campanha como incompleta.

### Estados finais do workflow

| Estado | Significado |
|---|---|
| `CampaignSucceeded` | Todas as unidades concluídas, publicação verificada e nenhuma task ativa |
| `CampaignIncomplete` | Alguma unidade sem manifesto final ou com erro de consolidação |
| `TasksStillActive` | Após 10 verificações com 60 s de intervalo ainda havia jobs ou tasks da campanha |
| `PreflightFailed` | O planejamento recusou o pedido (snapshot, release, capacidade, conflito) |
| `EndpointSetupFailed` | Os endpoints de interface não ficaram disponíveis |

## Recuperação de falhas

### Camadas de retentativa

| Camada | Regra | Onde |
|---|---|---|
| Chamadas à Lambda | Até 6 retentativas para erros transitórios da Lambda, intervalo de 2 s, fator 2 | `infra/statemachines/*.asl.json` |
| Endpoints pendentes | Até 60 retentativas de 15 s enquanto a stack de endpoints não está pronta | Estado `EnsureEndpoints` |
| Aquisição | Até 3 retentativas com 30 s e fator 2 para limites do CodeBuild | Estado `Acquire` |
| Task Batch de benchmark | `RetryStrategy` com `JobAttempts` (padrão 5): `RETRY` para código 75, interrupção Spot, `ResourceInitializationError` e `CannotPullContainerError`; `EXIT` para o resto | `infra/cloudformation/modules/execution.yaml` |
| Task Batch de preparação | `RETRY` apenas para `ResourceInitializationError` e `CannotPullContainerError` | Mesmo arquivo |
| Unidade no workflow | Novo job enquanto os jobs terminados forem menos que `max_attempts` do pedido | `reconcile_unit` |
| Submissão sem job identificado | Orçamento de 5 falhas por unidade, com esperas de 30, 60, 120 e 240 s e reconciliação após cada espera; a quinta falha encerra só aquela unidade e o contador não é zerado quando um job real aparece | Estados `CountSubmissionFailure` a `AdvanceSubmissionBackoff`; testes em `tests/test_aws_submission_retry_workflow.py` |
| Fila | Jobs em `RUNNABLE` por mais de `QueueWaitTimeoutSeconds` são cancelados pelo Batch quando a razão é de configuração ou falta de capacidade | `JobStateTimeLimitActions` em `execution.yaml` |

Um estado `Fail` dentro do Map é proibido pelo validador ASL, para que a falha de uma unidade não aborte as demais.

### Idempotência

- **Identidade determinística.** O `run_id` é derivado do `campaign_id` e do hash do contrato de execução (`run_identity` em `contracts.py`). O nome do job é `cb-<campaign_id>-<run_id>`, o que permite reencontrar jobs cuja resposta de submissão se perdeu.
- **Duplicatas.** Se houver mais de um job ativo com o mesmo nome, `reconcile_unit` mantém o mais antigo e encerra os demais.
- **Escrita condicional.** Plano, manifesto final, claims de tentativa, linhas de `attempts` e `cost_estimates` e o marcador de cancelamento são gravados com criação condicional (`If-None-Match: *`). Uma repetição com conteúdo idêntico é aceita; conteúdo divergente falha (`put_row_once` em `records.py`).
- **Reaproveitamento.** Uma unidade com `final.json` não é executada de novo; repetir a campanha com sufixo só processa as unidades que faltam.
- **Publicação em duas fases.** As linhas do Athena da tentativa vencedora ficam arquivadas na evidência da tentativa antes do `final.json`; o marcador `analytics.json` é escrito depois de publicá-las. Se a execução cair entre as duas etapas, `ensure_rows` repara a publicação na próxima leitura (job reaproveitado ou consolidação).
- **Snapshots.** Objetos de dados válidos são reaproveitados; evidência divergente nunca é sobrescrita; o marcador `_SUCCESS.json` é escrito por último.
- **Escopo do nome do job.** A reconciliação filtra pelo padrão exato `^cb-<campaign_id>-<20 hex>$`, para que a campanha `a` não encerre jobs da campanha `a-b`.

### Checkpoint e retomada no Fargate Spot

Quando o backend da release é `batch-fargate-spot`, o worker recebe `BENCH_FARGATE_SPOT=true`. Nos avaliadores com checkpoint durante a avaliação:

1. Ao receber SIGTERM, o worker para no próximo ponto seguro e confirma um checkpoint em `runs/<run_id>/spot/checkpoints/`. Há também checkpoint periódico a cada `BenchmarkCheckpointSeconds` (padrão 30 s).
2. Drena a fila de logging assíncrono do MLflow, grava `INTERRUPTED` na tentativa e sai com código 75.
3. A tentativa seguinte é numerada por um claim de escrita única (`spot/attempt-NNN/claim.json`), valida as entradas contra o contrato, restaura o checkpoint confirmado mais recente e continua o mesmo run lógico no MLflow, registrando `recovery_reason`.

O trabalho posterior ao último checkpoint é refeito. Limitações conhecidas: no caminho binário de CARA, UPF e Oracle o estado de avaliação só é confirmado ao terminar, então uma interrupção durante a construção da matriz pode exigir recomeço; no modo `artifact_mode=metrics_only` (usado pela suíte de quatro streams) a avaliação incompleta reinicia sem restaurar treino, e só a publicação de métricas já concluídas é recuperável.

Estado: Validado por testes automatizados (`tests/test_aws_spot_worker.py`, `tests/test_fargate_spot.py`, `tests/test_aws_spot_tracking.py`). A suíte de quatro streams registrou uma publicação de métricas retomada sem novo treinamento (`historical_failed_publication_attempts_recovered: 1`, `total_operational_job_attempts: 133` em `evidencias/aws-benchmarks/provenance.json`). Uma interrupção Spot real com retomada de checkpoint não está registrada no repositório.

Para ensaiar a interrupção, use só campanhas reduzidas. Inicie `CAMPAIGN_ID=<id-novo> scripts/aws/start_campaign.sh infra/examples/campaign-social-spot-acceptance.json` (políticas `drift_adwin` e `cara`, `max_attempts=4`), leia `plan.json` para mapear política e `run_id` e espere surgir um manifesto em `runs/<run_id>/spot/checkpoints/manifests/`. Então pare a task do job de teste: `aws ecs stop-task --cluster <cluster-do-batch> --task <task-arn> --reason "spot drill"` (o ECS envia SIGTERM e, 30 s depois, SIGKILL; a ação `aws:ecs:stop-task` do AWS FIS tem o mesmo efeito). A tentativa deve terminar `INTERRUPTED` com código 75 e a seguinte deve registrar `recovery_reason=INTERRUPTED` e o manifesto retomado. Para exercitar `PUBLICATION_FAILED`, simule a perda da confirmação após a escrita de um lote MLflow e confirme a conclusão na tentativa seguinte, com o mesmo identificador de run. Compare os artefatos com uma referência da mesma política (mesmo digest e snapshot): corte temporal, matrizes de confusão, predições, linhagem de ajustes e métricas de qualidade; tempos e custos dependentes deles variam.

### Repetir uma campanha ou um dataset

| Situação | Ação |
|---|---|
| Campanha falhou e o pedido não mudou | `scripts/aws/start_campaign.sh <pedido> <sufixo>`: a execução `<campaign_id>--<sufixo>` reaproveita as unidades concluídas |
| Plano, dados ou release mudaram | Novo `campaign_id` (o plano publicado é imutável) |
| Campanha foi cancelada | Novo `campaign_id`; `plan_campaign` recusa IDs com `cancelled.json` |
| Workflow de dataset falhou | `scripts/aws/start_dataset.sh <pedido> <sufixo>`; objetos válidos são reaproveitados |
| Stack de endpoints foi removida durante a espera | Nova execução com sufixo, que passa pelo provisionamento outra vez |

O nome de uma execução Step Functions já usado não pode ser reaproveitado; por isso as repetições usam sufixo.

### Submissões parciais da suíte de quatro streams

`scripts/aws/run_dataset_campaigns.py` guarda em um diretório de estado local os pedidos e as identidades devolvidas pela AWS, protegido por um bloqueio de arquivo contra dois coordenadores simultâneos:

- `status` consulta campanhas e jobs já registrados sem submeter nada, mantendo os denominadores de 132 unidades e oito campanhas e `complete=false` enquanto faltar algo.
- Repetir `start-base` com o mesmo contrato retoma as campanhas base ausentes. Uma submissão com resposta incerta é conciliada pelo nome determinístico e pelo conteúdo do pedido; o comando não cria outro nome para contornar uma identidade existente.
- `advance` inicia apenas as fases que reutilizam matrizes CARA cujos três caches estejam verificados.

As condições de admissão e as fases da suíte estão na seção [Suíte de quatro streams](#suíte-de-quatro-streams) abaixo.

## Cancelamento

`scripts/aws/cancel_campaign.sh <campaign_id> [execução]` chama `continual-bench aws cancel`, que:

1. confere que a execução pertence à campanha;
2. grava `campaigns/<id>/cancelled.json` com escrita única, o que bloqueia novos planos e faz os workers pararem no próximo ponto seguro (código 76);
3. para a execução com o erro `CampaignCancelled`;
4. reconcilia até 20 vezes, com 30 s de intervalo, encerrando os jobs ativos da campanha e gravando o relatório em `reconciliation/`. O comando sai com código 1 se ainda houver job ou task ativo.

Quando uma execução de benchmark termina por qualquer motivo, o EventBridge aciona a Lambda de controle, que também reconcilia a campanha (e grava o marcador de cancelamento se a execução foi abortada). Pelo SageMaker Studio, o callback cancela o workflow associado e a etapa termina como `Failed` com o motivo `Cancelled from SageMaker Studio`.

`continual-bench aws reconcile --campaign <id>` executa só a reconciliação, sem cancelar.

## Observação

| Ferramenta | Conteúdo |
|---|---|
| MLflow App | Runs por unidade com séries `progress/*` e métricas `final/*`; o run lógico de uma unidade Spot é preservado entre tentativas (tags `cb-spot-attempt-*`), e no On-Demand cada tentativa tem seu run. Filtros úteis: `cb-campaign-id`, `cb-unit-id`. Ver [Tracking e Evidências](Tracking-e-Evidencias) |
| `make aws-mlflow-ui` | URL pré-assinada do MLflow (`continual-bench aws mlflow-url`, validade padrão de 300 s para abrir e sessão de 43.200 s) |
| `make aws-studio-ui` | Links para o SageMaker Studio, o portal do Unified Studio e o MLflow |
| Dashboard `continual-bench-operations` | Métricas EMF `RunSucceeded`, `RunFailed` e `EvaluationSeconds` no namespace `ContinualBench`, com dimensões de baixa cardinalidade (dataset, escopo, política); `run_id` nunca é dimensão (`src/continual_bench/infrastructure/aws/emf.py`) |
| Alarmes CloudWatch e SNS | Falhas e timeouts das state machines e erros da Lambda; e-mail em `AlertEmail` |
| Eventos de job | A Lambda de controle grava cada mudança de estado de job em `runs/<run_id>/events/` e notifica falhas pelo SNS |
| Métricas de sistema | O worker amostra contêiner e processo durante a avaliação e publica `system/<nome>` no MLflow (ver [Custos e Recursos](Custos-e-Recursos#recursos-computacionais-medidos)) |

## Consultas Athena

```bash
continual-bench aws query infra/athena/compare_policies.sql --param campaign=<id>
continual-bench aws query infra/athena/window_series.sql --param campaign=<id>
continual-bench aws query infra/athena/campaign_runs.sql --param campaign=<id>
continual-bench aws query infra/athena/attempt_costs.sql --param campaign=<id>
```

Os parâmetros são substituídos no SQL e só aceitam letras, dígitos, `-` e `_`. A tabela `predictions` usa projeção de partições: filtre por `campaign_id`, `run_id` e `attempt_key` lidos do `final.json`. Consultas de faturamento usam `--database continual_bench_billing` (ver [Custos e Recursos](Custos-e-Recursos)).

## Suíte de quatro streams

A seleção AWS atual (Electricity, Airlines e duas variantes INSECTS) usa o protocolo `ordered-stream-batch-prequential-metrics-v2` com 11 políticas e as seeds 42, 43 e 44: 132 unidades lógicas em oito campanhas, com até 100 épocas e `artifact_mode=metrics_only`. As fontes integrais somam 1.392.014 exemplos (45.312, 539.383, 355.275 e 452.044). A ordem de um stream real é fixa entre seeds, então as três seeds não são três populações independentes.

**Fases.** A fase base tem `frozen`, `drift`, `drift_adwin`, `drift_ddm`, `incremental`, `periodic` e `cara`: 84 unidades. Cada execução CARA gera uma matriz de métricas verificada. Depois que os três resultados CARA de um stream são publicados, a segunda fase executa `cara_cumulative`, `cara_periodic`, `upf` e `oracle`, reutilizando as matrizes (48 unidades). `online` não integra esses pedidos. Cada tarefa reserva 2 vCPU e 4096 MiB em Fargate Spot, em um único ambiente Batch com teto de 64 vCPU (até 32 tarefas simultâneas; as demais esperam na fila e a capacidade Spot real pode reduzir o paralelismo).

**Coordenador.** `scripts/aws/run_dataset_campaigns.py` recebe pedidos novos e um SHA de release explícito e não faz deploy. Antes da primeira submissão, ele verifica:

- QA completo da matriz declarada (três seeds, métricas reconstruídas, integridade das predições, ausência de arquivos de modelo, catálogo de métricas e marcadores de publicação no rastreamento nativo); o QA local pode cobrir os sete streams e 231 casos, desde que os quatro selecionados estejam completos;
- vínculo do hash de código do smoke ao recibo da release e sucesso atual do workflow `Build and deploy`, consultado pela CLI do GitHub; o recibo de deploy contém `complete`, `git_sha`, `smoke_code_sha256` e `workflow` (`databaseId`, `headSha`, `status`, `conclusion`);
- stack concluída, release publicada e imagem por digest compatíveis;
- job definition Fargate com exatamente 2 vCPU e 4096 MiB, ambiente Spot válido com teto de 64 vCPU e quota regional suficiente;
- regras reais de cancelamento da fila em `RUNNABLE` com espera mínima de 86.400 s para as razões de capacidade e configuração (os recibos guardam as regras retornadas e `advance` as confere de novo);
- uso regional de Fargate: se houver tarefas externas ao ambiente da suíte, a submissão é interrompida.

O índice de pedidos (`scripts/run_dataset_suite.py --write-requests ... --request-stage base`) precisa ter `stage=base`, os quatro streams, três seeds, `artifact_mode=metrics_only`, `dataset` explícito e `matrix.kind=stream`; pedidos Zia são rejeitados.

```bash
python scripts/run_dataset_suite.py --root /inputs/prepared --output state/four-stream-full-plan \
  --datasets electricity airlines insects_abrupt_imbalanced \
    insects_incremental_abrupt_reoccurring_imbalanced \
  --scope full --max-epochs 100 --tracking --plan-only \
  --write-requests state/dataset-requests-four-streams --request-stage base --release "$RELEASE_SHA"

python scripts/aws/run_dataset_campaigns.py start-base --state-dir state/aws-four-stream-suite \
  --requests-dir state/dataset-requests-four-streams --smoke-receipt state/dataset-smoke/suite-qa.json \
  --deployment-receipt state/deployment-receipt.json --release "$RELEASE_SHA" \
  --bucket "$DATA_BUCKET" --region us-east-1
python scripts/aws/run_dataset_campaigns.py status  --state-dir state/aws-four-stream-suite --region us-east-1
python scripts/aws/run_dataset_campaigns.py advance --state-dir state/aws-four-stream-suite --region us-east-1
```

Publicar as fontes: `continual-bench aws publish-matrix-source --dataset <nome> --cache state/datasets --bucket "$BUCKET"` (com `BENCH_KMS_KEY_ID` definido), usando `--real-directory` para os arquivos ARFF de Electricity e Airlines preparados por `scripts/prepare_real_sources.py` e `--download` para autorizar a aquisição; o `dataset` e o `sha256` devolvidos entram em `request.source_digests`. As variantes INSECTS desbalanceadas usam os CSVs integrais com SHA-256 fixado. O planejador valida os objetos antes de publicar o plano e cada worker confere o SHA-256 ao baixar. O Distributed Map exporta o resultado da orquestração para `map-results/<campanha>/` no bucket de artefatos.

**Orçamento de submissão.** Uma resposta de submissão sem novo job consome o orçamento `NoJob` da unidade (esperas de 30, 60, 120 e 240 s; a quinta falha encerra a unidade). `max_attempts` conta só jobs reais.

**Aceitação.** Cada unidade exige manifesto final compatível com o plano, relatório com SHA-256 conferido, cobertura integral, seed correspondente, rastreamento obrigatório concluído e inventário sem pesos nem checkpoints, com listagens paginadas por completo. A suíte só está concluída com 132 unidades verificadas, oito campanhas com resumos completos e nenhuma tarefa ativa; código de saída zero ou `SUCCEEDED` isolados não bastam. Interrupções Spot preservam evidências por tentativa, mas a avaliação incompleta reinicia sem restaurar treino, sob o mesmo contrato, release e imagem.

**Exportação.** `scripts/aws/export_dataset_evidence.py --state-dir ... --output ... --scratch-root ... --scratch-gib 20 --max-file-mib 64` exige a conclusão integral e só faz listagem, metadados e leitura no S3. Baixa e confere por SHA-256 os relatórios, reconstrói as métricas por janela, partição e coorte comum a partir das predições (um caso por vez, com limite de espaço temporário), e grava o pacote com manifesto e `SHA256SUMS`. A retomada reaproveita casos já auditados por recibo local. O pacote não incorpora dados brutos nem pesos, e não substitui uma auditoria independente das APIs do MLflow. O resultado da suíte exportada está em `evidencias/aws-benchmarks/`.

## Campanhas no SageMaker Studio

A stack `continual-bench-studio` (`infra/cloudformation/studio.yaml`) recebe `DataBucket`, `KeyArn`, `CampaignWorkflowArn` e `DatasetWorkflowArn` do runtime. Para iniciar:

```bash
RELEASE_SHA=<sha> scripts/aws/start_studio_campaign.sh infra/examples/campaign-social-smoke.json
RELEASE_SHA=<sha> scripts/aws/start_studio_campaign.sh infra/examples/campaign-social-smoke.json infra/examples/dataset-social-smoke.json
```

O script grava os pedidos em `studio/requests/<id>/` do bucket de dados (escrita única, com `VersionId`) e inicia o pipeline `continual-bench-campaign` com o nome de exibição igual ao `campaign_id`. No Studio, em Pipelines, abra a execução pelo ID da campanha: `PrepareDataset` reutiliza o workflow de datasets quando recebe a solicitação, `RunBatchCampaign` acompanha o workflow de benchmark e publica seu ARN, e `VerifyResults` expõe o URI do `summary.json`. Os estados de cada job ficam no Step Functions e no Batch.

**MLflow.** Studio, Applications, MLflow, `continual-bench-mlflow`, Open MLflow, workspace `default`, Experiments. O app serverless contém os runs das matrizes, dos jobs nativos e das campanhas Zia. Para matrizes, procure `<sha-curto>-continual-bench-matrix-<política>` e filtre por `cb-campaign-id` ou `cb-unit-id`; para Zia, o experimento `<sha-curto>-continual-bench-zia-experiments` (padrão `<projeto>-experiments` em `infrastructure/aws/experiments.py`) filtrado por `cb-campaign-id`. A run pai abre como `RUNNING` ao iniciar a avaliação. `progress/rows_processed` e `progress/rows_total` mostram a cobertura. O catálogo do Unified Studio liga cada snapshot às avaliações pelas tags `cb-catalog-asset-id`, `cb-catalog-asset-url` e `cb-catalog-snapshot-sha256`.

**Catálogo.** As mensagens com falha da automação ficam nas filas indicadas pelos outputs `DeliveryFailureQueueUrl` e `ProcessingFailureQueueUrl`; após corrigir a causa, republique a mensagem na `PublicationQueueUrl` com o grupo FIFO `catalog` ou faça redrive. A automação confere de novo identidade e vínculos ao receber o mesmo evento. A notificação EventBridge do bucket de dados vem de `infra/cloudformation/persistent.yaml`; a autorização da função no projeto do Unified Studio é feita por `scripts/aws/deploy_studio_catalog.sh <domain-id> <project-id>`, num ambiente em que o tipo `ContinualBenchTabularSnapshot` já existe.

**Pipeline nativo.** `scripts/aws/deploy_native_pipeline.sh "$RUNTIME_IMAGE" "$ML_IMAGE" "$GIT_SHA"` implanta `continual-bench-native` (`ProfileSnapshot`, `TrainBaseline`, `TuneHyperparameters`). Exemplo de execução: `aws sagemaker start-pipeline-execution --pipeline-name continual-bench-native --pipeline-parameters Name=MaxEpochs,Value=24 Name=TuningMaxEpochs,Value=24 Name=MaxTuningJobs,Value=2 Name=MaxParallelTuningJobs,Value=1`. Os padrões apontam para o snapshot `smoke`; para outro, informe juntos `PreparedParquetUri`, `PreparedManifestUri` e `Scope`. No MLflow, filtre por `cb-pipeline-execution-id` no experimento `continual-bench-native`. Em `TuneHyperparameters`, os trials aparecem ordenados por `validation:log_loss` do menor para o maior; `aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job --sort-by FinalObjectiveMetricValue --sort-order Ascending` cruza os dados com a API. Esses ativos de snapshot não aparecem em Assets, Datasets do Studio (reservado a customização de modelos generativos); localize-os no prefixo `datasets/` do bucket de dados.

## Ciclo dos endpoints de interface

Os workflows de benchmark e de datasets provisionam a stack `continual-bench-interface-endpoints` depois do preflight (ou antes da etapa Batch de preparação) e esperam todos os endpoints ficarem `available` antes de submeter jobs. As execuções compartilham a stack, com leases registrados em `runtime/interface-endpoints.json` no S3 (escrita condicional por ETag). Uma rotina agendada a cada cinco minutos remove a stack quando todas as execuções admitidas terminaram, o intervalo de propagação passou (120 s no fluxo normal; 180 s para benchmarks encerrados sem sucesso, cobrindo tarefas Spot que ainda gravam checkpoint) e as filas de benchmark e de preparação não têm jobs ativos; em falha ou cancelamento, também verifica Maps distribuídos remanescentes. Com `AUTO_STOP_IDLE_SERVICES=true` (configurado na Lambda), a rotina pode ainda parar a aplicação Flink do projeto e apagar endpoints SageMaker do projeto marcados com `cb-ephemeral=true`. Mudar os parâmetros de rede exige esperar o escoamento dos jobs. O deploy da `main` mantém o MLflow App, os workflows do Studio, a VPC e o gateway do S3. Ao migrar uma stack que tinha endpoints permanentes, faça o primeiro deploy com a fila do Batch vazia e sem execuções, para liberar o DNS privado usado pela stack temporária. Para recriar o runtime preservando os logs, confirme que `/continual-bench/batch` e `/continual-bench/acquisition` existem com a chave KMS e a retenção esperadas e acrescente `"ReuseRuntimeLogGroups": "true"` ao `config/runtime-parameters.json`.

## Diagnóstico por sintoma

| Sintoma | Causa provável | Onde confirmar | Ação |
|---|---|---|---|
| `PreflightFailed` com `SnapshotNotReady` | Snapshot `missing` (ausente), `incomplete` (sem marcador) ou `incompatible` (divergente) | Saída do estado `Plan`; prefixo `manifests/` do snapshot | Executar o workflow de datasets |
| `PreflightFailed` com conflito de campanha | ID cancelado, plano já publicado com conteúdo diferente ou release sem Spot | Saída do estado `Plan`; `campaigns/<id>/cancelled.json` e `plan.json` | Novo `campaign_id` ou release correta |
| `PreflightFailed` com erro de capacidade | Fila não `ENABLED/VALID` ou paralelismo On-Demand acima do teto ou da quota | `preflight/<data>.json` | Ajustar `max_parallel`, `MaxVcpus` ou quota |
| Jobs parados em `RUNNABLE` | Quota de vCPU ocupada ou falta de capacidade Spot | Console do Batch; `preflight/` | Esperar; jobs são cancelados após `QueueWaitTimeoutSeconds` nas razões configuradas |
| `EndpointSetupFailed` | Stack de endpoints em estado inesperado | Eventos da stack `continual-bench-interface-endpoints` | Corrigir a stack e repetir com sufixo |
| Job `FAILED` com código 1 | Erro no worker, permissão negada ou dado inválido | Stream do job em `/continual-bench/batch`; `attempts/<job>-<n>/status.json` | Corrigir e repetir com sufixo |
| `FAILED_PUBLICATION` ou `PUBLICATION_FAILED` | Saída zero sem `final.json`, ou checksum, índice ou linhas analíticas ausentes | `summary.json` (`analytics_errors`), `attempts/` | Repetir com sufixo; a próxima tentativa pode concluir a publicação |
| `CampaignIncomplete` | Alguma unidade sem manifesto final | `summary.json` | Inspecionar as unidades com `final_uri` nulo |
| `TasksStillActive` | Jobs ou tasks não pararam após 10 verificações | `reconciliation/<data>.json` (`running_tasks`) | `continual-bench aws reconcile --campaign <id>` |
| Execução recusada por nome | Nome de execução já usado | Erro do `StartExecution` | Repetir com sufixo |
| `qemu: uncaught target signal 11` ao rodar a imagem localmente | Emulação x86_64 em Mac arm64 trava no `pyarrow` | Selftest da imagem sob QEMU | Validar a imagem em host x86_64 (o job `build` faz isso) |

Exemplos reais: as campanhas `social-smoke-v1` e `social-smoke-v3` (09/11/2026) terminaram em `CampaignIncomplete` por `AccessDeniedException` em `sagemaker:ListModelPackages` e por acesso negado do SageMaker ao ECR `continual-bench-ml`; a causa apareceu nos streams de `/continual-bench/batch` e no CloudTrail, e a correção de IAM levou à campanha `social-smoke-v4` com 4/4 unidades.

## Verificação antes de campanhas maiores

Antes de uma campanha `full`, convém conferir o pico de memória do smoke e as métricas `system/` do relatório, o disco (`BenchmarkEphemeralGiB` contra o tamanho do Parquet e das predições), a quota Fargate e o orçamento; e repetir o smoke com uma falha induzida (snapshot sem marcador), um cancelamento e um retry com sufixo, conferindo que nenhuma task fica ativa. Esses três ensaios não estão registrados como executados.

Ver também [Reprodutibilidade e Artefatos](Reprodutibilidade-e-Artefatos) e [Limitações e Trabalhos Pendentes](Limitacoes-e-Trabalhos-Pendentes).
