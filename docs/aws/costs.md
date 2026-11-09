# Custos na AWS

O perfil AWS separa quatro grandezas e nunca as soma entre si:

| Grandeza | Unidade | Origem |
|---|---|---|
| Custo de cenário (`cost_*`, `scenario_total_cost`) | unidade de cenário | Calculado pelo avaliador como antes (`docs/cost-accounting.md`) |
| Estimativa do avaliador (`aws_cost_*`) | USD | Tempos de treino e predição × tarifa fixada em `configs/base/tracking.yaml`, preservada no relatório |
| Estimado por tentativa (`cost_estimates`) | USD | Recursos reservados do Batch × segundos faturáveis entre `startedAt` e `stoppedAt` × snapshot de preços versionado |
| Observado (CUR 2.0) e rateio | USD | Linhas faturadas da task (Split Cost Allocation Data) e custos compartilhados do período rateados por vCPU-segundo reservado |

## Por tentativa

`record_attempt` grava em `analytics/attempts/` o ID do job, o ARN da task ECS, o código de
saída, `createdAt`, `startedAt` e `stoppedAt`, a job definition com revisão, vCPU, memória e
disco reservados. Grava também, em `analytics/cost_estimates/`, a espera na fila, a duração, os
segundos faturáveis (mínimo de 60 s, arredondamento por segundo) e os componentes em USD. O
worker registra no `final.json` e no `status.json` os bytes publicados, a duração da avaliação
e o digest da imagem. A estimativa cobre só a reserva Fargate. S3, logs, endpoints e serviços
compartilhados entram pelo CUR.

`configs/contracts/aws-prices.us-east-1.json` é o snapshot de preços versionado. Fargate e S3
coincidem com os catálogos públicos verificados em 08/11/2026. As demais entradas ficam
marcadas `seed_pending_price_list_refresh` até rodar, com credenciais:

```bash
continual-bench aws refresh-prices        # AWS Price List Query API → SKU, unidade e data efetiva
git diff configs/contracts/aws-prices.us-east-1.json
```

O treino nunca consulta a Price List API. O arquivo entra na imagem e no Lambda de controle.

## CUR 2.0, tags e Split Cost Allocation Data

O módulo `billing` cria um export AWS Data Exports CUR 2.0 horário, com
`INCLUDE_RESOURCES` e `INCLUDE_SPLIT_COST_ALLOCATION_DATA`, em Parquet no bucket de billing
retido. Também cria a tabela `continual_bench_billing.cur2` com projeção por `billing_period`,
o orçamento mensal (alertas a 80% do real e 100% do previsto) e um monitor de anomalias por
serviço com assinatura no SNS.

Se a conta já tiver um monitor de anomalias do tipo `DIMENSIONAL` para `SERVICE`, informe
o ARN em `ExistingServiceAnomalyMonitorArn` no objeto `config/runtime-parameters.json`
do bucket de artefatos. A stack cria a assinatura do projeto no monitor existente.
A conta continua responsável pelo monitor e pelas assinaturas que já estavam associadas a ele.

Etapas que exigem a conta pagadora (management account), fora do CloudFormation:

1. **Tags de alocação**: Billing → Cost allocation tags → ativar `cb-stack`, `cb-campaign`,
   `cb-run-id`, `cb-component` e `cb-snapshot`. As tags aparecem para ativação até 24 h depois
   do primeiro recurso marcado. Pela CLI, na pagadora:
   `aws ce update-cost-allocation-tags-status --cost-allocation-tags-status TagKey=cb-campaign,Status=Active TagKey=cb-run-id,Status=Active TagKey=cb-stack,Status=Active`.
2. **Split Cost Allocation Data para ECS**: Billing → Cost Management Preferences → Split cost
   allocation data → Amazon ECS. Os jobs propagam as tags para as tasks (`propagateTags`),
   e o CUR passa a trazer `split_line_item_*` por task.
3. Em contas membro de uma organização, Budgets e anomalias funcionam na própria conta. O
   export CUR de uma conta membro depende de a pagadora permitir acesso aos dados de
   faturamento da conta.

As tags de usuário aparecem no CUR 2.0 em `resource_tags['user_cb_campaign']` e afins. O CUR é
atualizado várias vezes ao dia e fecha o mês com atraso. A conciliação final de uma campanha
deve esperar o fechamento.

## Consultas

```bash
continual-bench aws query infra/athena/cur_campaign_tasks.sql --database continual_bench_billing \
  --param campaign=social-full-v1 --param period=2026-11
continual-bench aws query infra/athena/cur_shared_costs.sql --database continual_bench_billing --param period=2026-11
continual-bench aws query infra/athena/cur_period_total.sql --database continual_bench_billing --param period=2026-11
continual-bench aws query infra/athena/attempt_costs.sql --param campaign=social-full-v1
```

`application.cloud.costs.reconcile` recebe as tentativas (`cost_estimates`) e as linhas do CUR
e devolve, por campanha, o estimado das quatro tasks, o observado por task e o rateio dos
custos compartilhados. O rateio cobre CI/CD, rede, S3, observabilidade, streaming, Feature
Store e Athena, dividido pela fração de vCPU-segundos reservados da campanha no período. O
total não atribuído fica explícito.

## Conferir no Cost Explorer

1. Cost Explorer → Report parameters: granularidade mensal, mês da campanha, *Unblended costs*,
   agrupado por *Service*.
2. Compare o total com `cur_period_total.sql` (soma de `line_item_unblended_cost` de todos os
   tipos de linha). Diferenças vêm de impostos, créditos e da defasagem de atualização.
3. Depois de ativar as tags, filtre por *Tag: cb-campaign = <id>* e compare com
   `cur_campaign_tasks.sql`. Use `split_line_item_split_cost` mais
   `split_line_item_unused_cost` para o custo da task.

## Custos fixos e cobrados por hora

| Recurso | Fórmula | Estimativa (preço de lista do snapshot) |
|---|---|---|
| Endpoints de interface | endpoints × AZs × horas × `vpc_interface_endpoint_hour` + GB × `vpc_interface_endpoint_gb` | Core com registro de modelos: 4 × 2 AZ × 730 h × USD 0,01 ≈ USD 58/mês; com `AvailabilityZoneCount=1`, ≈ USD 29/mês. Streaming, Feature Store e KMS/Secrets somam 1, 1 e 2 endpoints. |
| Gateway endpoint do S3 | sem custo por hora | USD 0 |
| Batch/Fargate | vCPU-h × 0,04048 + GB-h × 0,004445 + GB de disco acima de 20 × h × 0,000111 | 4 tasks de 1 vCPU/2 GiB ≈ USD 0,20/h; 48 h ≈ USD 9,50 (sem disco extra) |
| Preparação | 2 vCPU/8 GiB e 150 GiB de disco | ≈ USD 0,13/h |
| CodeBuild | minutos × tarifa do tipo | Cobrado por build (CI e aquisição) |
| Kinesis (streaming) | shards × h × 0,015 | 1 shard ≈ USD 11/mês, além do volume |
| Managed Flink (streaming) | KPUs × h (+1 KPU de orquestração) | Cobrado só enquanto `RUNNING`. Pare com `scripts/aws/flink.sh stop`. |
| Feature Store online | unidades de leitura/escrita + GB-mês | Desligado por padrão (`EnableOnlineStore=false`) |
| Endpoint de inferência | serverless: por requisição e duração | Desligado por padrão. `sagemaker_jobs.py inference --delete-after` remove após o teste. |

As entradas sem verificação na Price List API estão marcadas no snapshot. Rode
`refresh-prices` antes de usar estes valores em decisões.
