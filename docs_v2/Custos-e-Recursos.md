# Custos e Recursos

Esta página explica as grandezas de custo produzidas pelo projeto (custo de cenário, estimativas de infraestrutura em USD, faturamento observado e rateio), como cada uma é calculada, de onde vêm os preços e o que nenhuma delas representa. Também descreve os recursos computacionais medidos (tempo, CPU e memória) e as suas limitações.

## As grandezas não se somam

O projeto mantém grandezas de custo separadas, com unidades e origens diferentes. Elas nunca são somadas entre si: o código grava cada uma em um campo próprio, com um rótulo `kind` e um texto `scope` que delimita o que entra no cálculo.

| Grandeza | Campos | Unidade | `kind` / origem | Estado |
|---|---|---|---|---|
| Custo de cenário | `cost_prediction_errors`, `cost_training_presentations`, `cost_update_events`, `cost_training_time`, `cost_prediction_time`, `cost_total` (`scenario_total_cost` nos relatórios de stream) | unidade de cenário (`scenario_unit` por padrão) | `scenario_estimate`, `src/continual_bench/evaluation/costs.py` (`account`) | Validado (`tests/test_costs.py`) |
| Estimativa de infraestrutura do aprendiz | `aws_cost_cpu_usd`, `aws_cost_ram_usd`, `aws_cost_model_storage_usd`, `aws_cost_total_usd` (no relatório: `costs.aws.metrics.cost_*_usd`) | USD de referência | `infrastructure_estimate`, `aws_account` em `evaluation/costs.py` | Validado (`tests/test_costs.py`; valores em `evidencias/aws-benchmarks/resources-summary.csv`) |
| Estimativa sobre o tempo de avaliação | `aws_evaluation_cost_*_usd` | USD de referência | Mesmo `aws_account`, aplicado ao tempo de computação da avaliação (`src/continual_bench/evaluation/stream_benchmark.py`) | Validado (valores no mesmo CSV) |
| Estimativa por job Batch | Tabela Athena `cost_estimates` (`vcpu_usd`, `memory_usd`, `ephemeral_storage_usd`, `total_usd`) | USD | `estimated`, `attempt_estimate` em `src/continual_bench/application/cloud/costs.py` | Validado por testes (`tests/test_aws_streaming_ml_costs.py`) |
| Faturamento observado | Linhas do CUR 2.0 (`line_item_unblended_cost`, `split_line_item_split_cost`, `split_line_item_unused_cost`) | USD faturado | AWS Data Exports, tabela `continual_bench_billing.cur2` | Experimental (sem evidência no repositório) |
| Rateio de custos compartilhados | Saída de `reconcile` (`allocated_shared_usd`, `unattributed_period_usd`) | USD faturado, alocado | `reconcile` em `application/cloud/costs.py` | Experimental (testado, sem comando que o exponha) |
| Preço por retreino | `custo_retreino_usd`, usado em `α = λ × custo_retreino_usd` | USD de referência | `src/continual_bench/retraining/pricing.py` | Entrada de decisão de CARA, UPF e Oracle, não uma medida da execução (ver [Modelos e Políticas](Modelos-e-Politicas)) |

Regras práticas:

- `cost_total` está em unidade de cenário e `aws_cost_total_usd` está em USD; somá-los não tem significado.
- As três estimativas em USD aplicam preços de lista a durações e reservas. Nenhuma delas é fatura.
- Só o CUR 2.0 contém valores faturados. O repositório não contém nenhuma consulta ao CUR executada, portanto nenhum número desta documentação é faturamento observado.
- Custos normalizados das políticas (`upf_total_cost`, `retraining_c_hat`, `cost_scpe`, `upf_paper_auc` e afins) são outra família, sem unidade monetária; estão em [Métricas](Metricas).

## Custo de cenário

O custo de cenário é uma contabilidade parametrizada, independente da decisão da política, que traduz erros e esforço de treino em uma unidade arbitrária. A configuração fica em `tracking.costs` (fluxo principal) ou `costs` (fluxos tipado e por épocas), validada por `CostConfig` em `src/continual_bench/config/benchmark.py`.

| Coeficiente | Padrão em `CostConfig` | Perfil de `TrackingConfig` | Uso |
|---|---:|---:|---|
| `unit` | `scenario_unit` | `scenario_unit` | Nome da unidade |
| `false_positive`, `false_negative` | 1 | 1 | Binário do fluxo principal: FP × `false_positive` + FN × `false_negative` |
| `classification_error` | 1 | 1 | Fluxos tipado e por épocas: erros × coeficiente |
| `regression_error`, `regression_scale` | 1, 1 | 1, 1 | Soma de \|alvo - predição\| / escala × coeficiente |
| `train_sample` | 0 | 0,001 | Por apresentação de treino, incluindo aquecimento e replay |
| `update` | 0 | 0,05 | Por atualização após o aquecimento |
| `train_second`, `predict_second` | 0 | 0 | Por segundo de treino e de predição |

`cost_total` é a soma dos cinco componentes. Se um tempo necessário (com preço não nulo) estiver ausente, o total fica indisponível (`null`), não zero. O escopo declarado é "treino e predições do aprendiz", excluindo aquisição, publicação e referências de avaliação. Direção desejável: menor. Limitação: o valor depende dos coeficientes escolhidos, e comparações entre tarefas exigem escalas e protocolos comparáveis. Por fluxo: no principal, o binário usa FP × `false_positive` + FN × `false_negative`; nos fluxos tipado e por épocas, erros × `classification_error`, e a regressão soma \|alvo - predição\| / `regression_scale` × `regression_error` (escala fornecida pelo experimento, sem ajuste no teste). No fluxo por épocas, cada passagem é uma atualização e cada repetição conta como apresentação; o fluxo tipado precifica erros apenas nos exemplos avaliados, incluindo validação e teste, mas seu tempo de predição inclui o aquecimento. Esses escopos diferentes devem acompanhar qualquer comparação. Um perfil explícito mantém os padrões de preço zero para treino e atualização quando os campos são omitidos; declare todos os coeficientes em comparações controladas. O relatório guarda coeficientes, versão, unidade e consumo (em `costs`, ou `tracking.costs` no fluxo principal), e execuções concluídas são idempotentes.

## Estimativa de infraestrutura do aprendiz

Configurada em `costs.aws` (ou `tracking.costs.aws`) por `AwsCostConfig`. Habilitada por padrão; `aws: null` a desliga. Projeta uma task Fargate por run usando CPU e memória **reservadas**, não medidas:

```
billed_seconds         = max(minimum_compute_seconds, ceil(train_seconds + prediction_seconds))
cost_cpu_usd           = vcpus × billed_seconds / 3600 × vcpu_hour_usd
cost_ram_usd           = memory_gb × billed_seconds / 3600 × memory_gb_hour_usd
cost_model_storage_usd = model_storage_bytes / 1024³ × retention_days / billing_month_days × storage_gb_month_usd
cost_total_usd         = soma dos três (null se algum componente for null)
```

| Parâmetro | Padrão do código | `configs/base/tracking.yaml` | `configs/base/tracking-fargate.yaml` e `tracking-dataset-suite.yaml` |
|---|---:|---:|---:|
| `vcpus` / `memory_gb` | 2 / 4 | 1 / 2 | 2 / 4 |
| `vcpu_hour_usd` | 0,04048 | 0,04048 | 0,04048 |
| `memory_gb_hour_usd` | 0,004445 | 0,004445 | 0,004445 |
| `storage_gb_month_usd` | 0,023 | 0,023 | 0,023 |
| `minimum_compute_seconds` | 60 | 60 | 60 |
| `retention_days` / `billing_month_days` | 30 / 30 | 30 / 30 | 30 / 30 |

O validador só aceita combinações de vCPU e memória suportadas pelo Fargate, e outra região exige informar explicitamente preços, fontes e data.

Variante sobre o tempo de avaliação: nos relatórios de stream, `aws_evaluation_cost_*_usd` aplica a mesma fórmula com `train_seconds = evaluation_compute_seconds` e `prediction_seconds = 0`, isto é, a duração total da avaliação em vez da soma dos ajustes lógicos promovidos e das predições. O dicionário `evidencias/aws-benchmarks/metric-dictionary.csv` classifica `aws_cost_*` com direção "minimizar" e `aws_evaluation_cost_*` como "descritivo".

Exclusões declaradas no próprio relatório: inicialização do contêiner, ociosidade, aquisição de dados, referências de avaliação, publicação, serviços de tracking, checkpoints, metadados de modelo, requisições S3, transferência de rede, impostos, free tier e descontos.

Efeito do mínimo de 60 s: com 2 vCPU e 4 GB, o valor mínimo é 60 × (2 × 0,04048 + 4 × 0,004445) / 3600 = 0,0016457 USD. Em `evidencias/aws-benchmarks/resources-summary.csv`, `aws_cost_total_usd` vale exatamente 0,0016456666 para `electricity/frozen`, `electricity/cara` e `airlines/cara`, cujos tempos de aprendiz ficam abaixo de 60 s. Nessas unidades a estimativa não distingue políticas: o piso de cobrança domina. A variante sobre a avaliação diferencia mais (`airlines/cara`: 0,0077346 USD, com `evaluation_compute_seconds` médio de 281,7 s).

## Estimativa por job Batch

Quando um job Batch termina, a Lambda de controle executa `record_attempt` (`src/continual_bench/application/cloud/lifecycle.py`), que grava uma linha em `analytics/attempts/` (job, task ECS, código de saída, `createdAt`, `startedAt`, `stoppedAt`, job definition, recursos reservados e lista de tentativas internas) e, se houver preços e horários, uma linha em `analytics/cost_estimates/`:

```
seconds        = (stoppedAt - startedAt) / 1000
billed_seconds = max(60, ceil(seconds))
vcpu_usd       = vCPU reservadas × billed_seconds / 3600 × fargate_vcpu_hour
memory_usd     = GiB reservados × billed_seconds / 3600 × fargate_memory_gb_hour
ephemeral_storage_usd = max(0, GiB efêmeros - 20) × billed_seconds / 3600 × fargate_ephemeral_storage_gb_hour
vcpu_seconds   = vCPU reservadas × seconds
queue_seconds  = (startedAt - createdAt) / 1000
```

Os preços vêm do snapshot `configs/contracts/aws-prices.us-east-1.json`, empacotado na imagem e na Lambda. A consulta `infra/athena/attempt_costs.sql` junta `attempts` e `cost_estimates` por job. Escopo declarado no código: apenas a reserva da task Fargate, sem S3, logs, endpoints e serviços compartilhados.

Limitações verificadas no código:

- As tarifas do snapshot são as de Fargate On-Demand. A mesma fórmula é aplicada aos jobs executados em Fargate Spot; o preço Spot efetivo não está no snapshot e só aparece no CUR.
- O cálculo usa o par `startedAt`/`stoppedAt` do job Batch, não a soma dos intervalos de cada tentativa interna da `RetryStrategy`, e aplica o mínimo de 60 s uma vez por job. Com retentativas internas, o resultado pode diferir do tempo efetivo das tasks.
- Só jobs em estado terminal (`SUCCEEDED` ou `FAILED`) geram linhas, e as linhas são imutáveis.

Exemplo de ordem de grandeza (aritmética com o snapshot, não faturamento): uma task de 2 vCPU, 4 GiB e 50 GiB de disco custa 2 × 0,04048 + 4 × 0,004445 + 30 × 0,000111 = 0,10207 USD por hora reservada; 32 tasks simultâneas, o teto de 64 vCPU, somam 3,27 USD por hora.

## Faturamento observado e rateio

O módulo `billing` (`infra/cloudformation/modules/billing.yaml`, ativado por `EnableBilling=true`) cria:

- um export AWS Data Exports CUR 2.0 horário, com `INCLUDE_RESOURCES` e `INCLUDE_SPLIT_COST_ALLOCATION_DATA`, em Parquet no bucket de billing;
- a tabela `continual_bench_billing.cur2`, particionada por `billing_period`;
- um orçamento mensal (`MonthlyBudgetUsd`, padrão 100) com alertas acima de 80% do gasto real e de 100% do previsto;
- um monitor de anomalias por serviço com limiar `AnomalyThresholdUsd` (padrão 10), ou uma assinatura num monitor existente (`ExistingServiceAnomalyMonitorArn`).

Etapas manuais na conta pagadora (management account), fora do CloudFormation. Se a conta já tiver um monitor de anomalias `DIMENSIONAL` por `SERVICE`, informe o ARN em `ExistingServiceAnomalyMonitorArn` no `config/runtime-parameters.json`; a stack cria apenas a assinatura do projeto, e o monitor e as assinaturas anteriores continuam sob responsabilidade da conta. Etapas:

1. ativar as tags de alocação `cb-stack`, `cb-campaign`, `cb-run-id`, `cb-component` e `cb-snapshot` (disponíveis para ativação até 24 h após o primeiro recurso marcado);
2. habilitar Split Cost Allocation Data para Amazon ECS, que faz o CUR trazer `split_line_item_*` por task (os jobs propagam as tags para as tasks);
3. em contas membro, o export depende de a pagadora liberar o acesso aos dados de faturamento.

Consultas (`infra/athena/`):

| Arquivo | Resultado |
|---|---|
| `cur_campaign_spot.sql` | Custo não mesclado do Fargate por run e task de uma campanha (`cb-component=benchmark`) |
| `cur_campaign_tasks.sql` | Custo por task de uma campanha |
| `cur_shared_costs.sql` | Custos compartilhados do período por serviço, para a tag `cb-stack=continual-bench` |
| `cur_period_total.sql` | Total do período por tipo de linha, para comparar com o Cost Explorer |

```bash
continual-bench aws query infra/athena/cur_campaign_spot.sql --database continual_bench_billing \
  --param campaign=<id> --param period=<AAAA-MM>
```

A função `reconcile(campaign_id, attempts, cur_rows, period_vcpu_seconds)` em `application/cloud/costs.py` junta as estimativas com as linhas do CUR e devolve, por campanha:

- `estimated_usd`: soma das estimativas por job;
- `observed_task_usd`: linhas do CUR com a tag da campanha e de um `run_id` conhecido (`split_cost`, ou `cost` na falta dele);
- `shared_period_usd` e `allocated_shared_usd`: linhas de serviços compartilhados (categorias de `SHARED_CATEGORIES`: CI/CD, rede, armazenamento S3, observabilidade, streaming, SageMaker e Glue, Athena) multiplicadas pela fração `vCPU-segundos da campanha / vCPU-segundos do período`;
- `unattributed_period_usd`: linhas que não se encaixam em nenhuma categoria, mantidas explícitas.

Estado: Experimental. A função é coberta por `tests/test_aws_streaming_ml_costs.py`, mas não há subcomando da CLI nem etapa de workflow que a execute; o operador precisa extrair as linhas do CUR e das tentativas e chamá-la. O CUR fecha o mês com atraso, então a conciliação final de uma campanha deve esperar o fechamento. Não há resultado de conciliação registrado no repositório (Pendente).

## Preços usados

| Entrada do snapshot | USD | Unidade | Status no arquivo |
|---|---:|---|---|
| `fargate_vcpu_hour` | 0,04048 | vCPU-hora | `verified_public_offer_file`, vigência 19/10/2026 |
| `fargate_memory_gb_hour` | 0,004445 | GB-hora | `verified_public_offer_file`, vigência 19/10/2026 |
| `s3_standard_gb_month` | 0,023 | GB-mês | `verified_public_offer_file`, vigência 04/11/2026 |
| `fargate_ephemeral_storage_gb_hour` | 0,000111 | GB-hora | `seed_pending_price_list_refresh` |
| `vpc_interface_endpoint_hour` | 0,01 | hora por AZ | `seed_pending_price_list_refresh` |
| `vpc_interface_endpoint_gb` | 0,01 | GB | `seed_pending_price_list_refresh` |
| `cloudwatch_logs_ingest_gb` | 0,5 | GB | `seed_pending_price_list_refresh` |
| `kinesis_shard_hour` | 0,015 | shard-hora | `seed_pending_price_list_refresh` |
| `codebuild_general1_medium_minute` | 0,01 | minuto de build | `seed_pending_price_list_refresh` |
| `lambda_gb_second` | 0,0000166667 | GB-segundo | `seed_pending_price_list_refresh` |

Fonte: `configs/contracts/aws-prices.us-east-1.json` (região `us-east-1`, obtido em 08/11/2026). Os preços de Fargate Linux x86 On-Demand e de S3 Standard foram conferidos em 08/11/2026 nos arquivos públicos de ofertas da AWS (ECS, publicado em 19/10/2026; S3, publicado em 04/11/2026; armazenamento na primeira faixa, 50 TB por mês). As três entradas verificadas vêm dos arquivos públicos de ofertas da AWS para ECS (Fargate Linux x86 On-Demand) e S3 Standard, citados em `AwsCostConfig` (`compute_source`, `storage_source`). As demais são valores semente que aguardam `continual-bench aws refresh-prices`, que consulta a AWS Price List Query API e reescreve o arquivo com SKU, unidade e data de vigência (`src/continual_bench/infrastructure/aws/pricing.py`). Esse comando precisa de credenciais e não está registrado como executado (Pendente). O treino nunca consulta a Price List API.

Os preços da estimativa do aprendiz ficam na configuração da execução e fazem parte da sua identidade: alterar tarifas ou alocação gera outra execução, e relatórios históricos não são recalculados.

## Requisitos dos módulos de faturamento e BI

O módulo `billing` depende de configuração na conta pagadora (ativação de tags e Split Cost Allocation Data, descritas acima); o export de uma conta membro depende de a pagadora permitir o acesso aos dados de faturamento. O módulo `bi` (`EnableBi=true`) exige `QuickSightPrincipalArn`, uma assinatura Amazon Quick Sight Enterprise em `us-east-1` e acesso da role de serviço do Quick Sight ao Athena, ao bucket de dados (incluindo `athena-results/`) e à chave KMS (Quick Sight, Manage, Security & permissions). O dataset do BI junta as métricas por política com `cost_estimates` (linhas com `status = 'SUCCEEDED'`), portanto exibe estimativas, não faturamento.

## Custos fixos da infraestrutura

Além das tasks, alguns recursos cobram por hora enquanto existem. Valores calculados com o snapshot (preço de lista, não fatura):

| Recurso | Fórmula | Observação |
|---|---|---|
| Endpoints de interface | endpoints × AZs × horas × 0,01 + GB × 0,01 | Core: 3 (ECR API, ECR Docker, Logs); `EnableModelRegistry` e `EnableMlflowTracking` somam 2; Streaming e Feature Store somam 1 cada. Com 5 endpoints em uma AZ: 0,05 USD por hora enquanto a stack existe |
| Gateway endpoint do S3 | sem cobrança pelo endpoint | Continuam valendo as tarifas do S3 |
| Preparação | 2 × 0,04048 + 8 × 0,004445 + 130 × 0,000111 | Cerca de 0,13 USD por hora (2 vCPU, 8 GiB, 150 GiB) |
| Kinesis | shards × horas × 0,015 | Só com `EnableStreaming=true` |
| Managed Flink, Feature Store online, endpoint de inferência | Por KPU, por unidade de leitura e escrita, por requisição | Desligados por padrão; cobrados só quando iniciados |

Os endpoints de interface não ficam provisionados continuamente: a stack `continual-bench-interface-endpoints` é criada pelos workflows e removida por uma rotina que roda a cada cinco minutos (`rate(5 minutes)` em `execution.yaml`) quando execuções e filas ficam ociosas. Mantê-los o mês inteiro (730 h) custaria, com os preços semente, 21,90 USD por mês para os 3 endpoints do Core em uma AZ e 36,50 USD com os 5 (registro de modelos e MLflow), ou o dobro com `InterfaceEndpointAzMode=all` em duas AZs. Com `AvailabilityZoneCount=2` e `single`, os endpoints ficam em uma AZ e os workers da outra cruzam a zona (latência e tráfego entre AZs), e uma falha nessa AZ afeta o acesso nas duas sub-redes; para disponibilidade entre AZs use `all`. Ambientes sem registro de modelos e tracking podem usar `EnableModelRegistry=false` e `EnableMlflowTracking=false` para remover dois endpoints.

## Recursos computacionais medidos

Os recursos aparecem em duas formas que não devem ser confundidas: **reservados** (o que a job definition pede ao Fargate: vCPU, memória e disco) e **medidos** (o que o processo realmente usou). As estimativas em USD usam os reservados; as métricas abaixo são medidas.

| Métrica | Unidade | Direção | Como é medida | Limitações |
|---|---|---|---|---|
| `train_seconds` | s | menor | Soma dos ajustes lógicos promovidos (inicial e atualizações) | Não inclui candidatos descartados nem construção de matrizes reutilizadas |
| `initial_train_seconds`, `update_train_seconds` | s | menor | Subconjuntos de `train_seconds` | Não somar de novo ao total |
| `actual_total_fit_seconds` | s | equilíbrio descritivo | Todo o trabalho físico de ajuste da unidade, incluindo materialização | Unidades que reutilizam matrizes CARA não pagam a construção original |
| `retraining_matrix_fit_seconds` | s | menor | Tempo de construção da matriz de desempenho | Pode ser herdado da execução CARA de origem |
| `prediction_seconds` | s | menor | Soma dos tempos de predição | Mede o processo de avaliação, não um serviço |
| `prediction_latency_mean_seconds`, `prediction_throughput_per_second` | s, exemplos/s | menor, maior | Derivados de `prediction_seconds` e da contagem de predições | Idem |
| `evaluation_compute_seconds`, `evaluation_attempt_wall_seconds` | s | menor | Relógio da avaliação na tentativa | Excluem publicação e provisionamento; escopo difere entre protocolos |
| `process_peak_rss_bytes` | bytes | menor | Pico de RSS (memória residente) do processo por `getrusage(RUSAGE_SELF).ru_maxrss` (`src/continual_bench/core/observability.py`); em retomadas, o máximo entre tentativas | É do processo, não a RAM reservada do contêiner |
| `sampled_rss_byte_seconds` | byte·s | menor | Integral trapezoidal do RSS amostrado, por segmento de observação | Sem série temporal o valor é nulo; pico × duração não substitui a integral |
| `logical_model_serialized_bytes` | bytes | descritivo | Tamanho serializado dos modelos lógicos | Não são pesos persistidos |
| `persisted_model_storage_bytes` / `model_storage_bytes` | bytes | descritivo | Estado de modelo efetivamente salvo | Zero no modo `metrics_only` |
| `training_presentations`, `update_count`, `retrain_count` | contagem | menor | Contadores do protocolo | Ver [Métricas](Metricas) |
| `system/<nome>` | MB, %, vCPU | descritivo | `SystemMetricsSampler` (`src/continual_bench/evaluation/system_metrics.py`): a cada 5 s (0,5 s no escopo `smoke` da suíte), lê cgroup v2 (`memory.current`, `memory.max`, `cpu.max`, `cpu.stat`) para memória e CPU do contêiner, e psutil para processo, disco e rede | Sem cgroup cai para métricas do host; acima de 3600 amostras metade é descartada e o intervalo dobra, exceto no modo `metrics_only`, que mantém todas; exclui gráficos e publicação |
| `cpu_model`, `logical_cpus`, limites cgroup | texto, contagem | descritivo | `runtime_resources` em `src/continual_bench/evaluation/environment.py`, lendo `/proc/cpuinfo` e cgroup | A mesma reserva não implica o mesmo processador |
| `EvaluationSeconds` (EMF) | s | menor | Publicada pelo worker no CloudWatch por dataset, escopo e política | Agregada, sem `run_id` |

As séries de sistema incluem `container_cpu_usage_vcpus`, `container_cpu_utilization_percentage`, `container_memory_usage_megabytes`, `container_memory_limit_megabytes`, `process_memory_rss_megabytes`, `process_cpu_utilization_percentage`, `disk_usage_megabytes` e `network_receive_megabytes`, entre outras. GPU só aparece se a biblioteca NVIDIA estiver presente.

### Evidência registrada

`evidencias/aws-benchmarks/` contém os recursos das 132 unidades da suíte de quatro streams: `resources-by-seed.csv` (6.336 células, com 168 ausências preservadas como NULL) e `resources-summary.csv` (média e desvio padrão amostral entre as três seeds). `hardware.csv` registra oito modelos de CPU observados nas avaliações, de AMD EPYC 7R13 a Intel Xeon Platinum 8488C; a reserva de 2 vCPU e 4 GiB foi a mesma, mas o desempenho físico dos processadores não é idêntico, o que afeta comparações de tempo entre unidades. As unidades, direções e definições estão em `metric-dictionary.csv`. A leitura desses arquivos está em [Tracking e Evidências](Tracking-e-Evidencias).

## O que não é faturamento

- `cost_total`, `scenario_total_cost` e seus componentes: unidade de cenário parametrizada.
- `aws_cost_*` e `aws_evaluation_cost_*`: preço de lista On-Demand aplicado a durações do aprendiz ou da avaliação e a recursos reservados.
- `cost_estimates`: preço de lista On-Demand aplicado à janela do job Batch, mesmo em Fargate Spot.
- `custo_retreino_usd`: cenário de preço que alimenta a decisão das políticas por custo.
- Custos normalizados (`upf_total_cost`, `retraining_c_hat`, `cost_scpe`, `upf_paper_auc`): objetivos de decisão sem unidade monetária.
- Valores mensais de endpoints: ilustração com 730 h e preços semente.

Somente as linhas do CUR 2.0, conferidas com o Cost Explorer após o fechamento do mês, representam o valor faturado. Diferenças entre o total do CUR e o Cost Explorer vêm de impostos, créditos e defasagem de atualização.

Ver também [Limitações e Trabalhos Pendentes](Limitacoes-e-Trabalhos-Pendentes).
