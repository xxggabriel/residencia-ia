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
  --param campaign=zia-social-full-v1 --param period=2026-11
continual-bench aws query infra/athena/cur_shared_costs.sql --database continual_bench_billing --param period=2026-11
continual-bench aws query infra/athena/cur_period_total.sql --database continual_bench_billing --param period=2026-11
continual-bench aws query infra/athena/attempt_costs.sql --param campaign=zia-social-full-v1
```

`application.cloud.costs.reconcile` recebe as tentativas (`cost_estimates`) e as linhas do CUR
e devolve, por campanha, o estimado das tentativas, o observado por task e o rateio dos
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
| Endpoints de interface | endpoints × AZs dos endpoints × horas × `vpc_interface_endpoint_hour` + GB × `vpc_interface_endpoint_gb` | Core: 3 endpoints (ECR API, ECR Docker e CloudWatch Logs). Registro de modelos e MLflow acrescentam 2; Streaming e Feature Store acrescentam 1 cada. KMS e Secrets Manager somam 2 se habilitados diretamente no módulo de rede. |
| Gateway endpoint do S3 | sem cobrança adicional pelo endpoint | USD 0 pelo gateway; continuam valendo as tarifas do S3. |
| Batch/Fargate | vCPU-h × 0,04048 + GB-h × 0,004445 + GB de disco acima de 20 × h × 0,000111 | 4 tasks de 1 vCPU/2 GiB ≈ USD 0,20/h; 48 h ≈ USD 9,50 (sem disco extra) |
| Preparação | 2 vCPU/8 GiB e 150 GiB de disco | ≈ USD 0,13/h |
| GitHub Actions e CodeBuild | uso dos runners e minutos de build do CodeBuild | O GitHub Actions publica releases. O CodeBuild é usado na aquisição dos datasets. |
| Kinesis (streaming) | shards × h × 0,015 | 1 shard ≈ USD 11/mês, além do volume |
| Managed Flink (streaming) | KPUs × h (+1 KPU de orquestração) | Cobrado só enquanto `RUNNING`. Pare com `scripts/aws/flink.sh stop`. |
| Feature Store online | unidades de leitura/escrita + GB-mês | Desligado por padrão (`EnableOnlineStore=false`) |
| Endpoint de inferência | serverless: por requisição e duração | Desligado por padrão. `sagemaker_jobs.py inference --delete-after` remove após o teste. |

As entradas sem verificação na Price List API estão marcadas no snapshot. Rode
`refresh-prices` antes de usar estes valores em decisões.

### Reduzir o custo de rede

O S3 regional já passa pelo gateway endpoint da tabela de rotas compartilhada pelas sub-redes privadas,
inclusive para baixar as camadas de imagens do ECR. Mantenha os buckets usados pelos workers na
mesma região da VPC. Consulte a [documentação do gateway endpoint do S3](https://docs.aws.amazon.com/vpc/latest/privatelink/vpc-endpoints-s3.html).

O deploy lê o objeto `config/runtime-parameters.json` do bucket de artefatos quando
`RUNTIME_CONFIG_KEY` aponta para essa chave. Preserve os outros overrides ao editar o perfil:

| Perfil | Parâmetros | Core com registro e MLflow | Core sem registro e MLflow |
|---|---|---:|---:|
| Redundância de rede (padrão do template) | `AvailabilityZoneCount=2`, `InterfaceEndpointAzMode=all` | USD 73,00/mês | USD 43,80/mês |
| Duas sub-redes e endpoints em uma AZ | `AvailabilityZoneCount=2`, `InterfaceEndpointAzMode=single` | USD 36,50/mês | USD 21,90/mês |
| Ambiente com uma AZ | `AvailabilityZoneCount=1` | USD 36,50/mês | USD 21,90/mês |

Em 16/11/2026, o runtime de referência estava com `AvailabilityZoneCount=2` e
`InterfaceEndpointAzMode=single`. O arquivo de overrides no S3 ainda indicava uma AZ
para o Batch e omitia o modo dos endpoints. Para preservar o perfil implantado, atualize
os dois parâmetros no objeto antes de um novo deploy por GitHub Actions. Consulte
[a evidência de rede](validation.md#rede-do-batch).

Valores ilustrativos da parcela fixa, com 730 h/mês e USD 0,01/h por endpoint por AZ.
`single` mantém as duas opções de sub-rede para o Batch, mas os workers da outra AZ cruzam a zona
para chegar aos endpoints de interface. Considere a latência e os custos de tráfego aplicáveis:
uma falha na AZ dos endpoints afeta o acesso a esses serviços nas duas sub-redes.
Para cargas que exigem disponibilidade entre AZs, mantenha `all`. Veja a
[documentação de resiliência do PrivateLink](https://docs.aws.amazon.com/vpc/latest/privatelink/privatelink-access-aws-services.html)
e os [preços do PrivateLink](https://aws.amazon.com/privatelink/pricing/).

Em ambientes que dispensam registro de modelos e tracking, `EnableModelRegistry=false` remove
os endpoints do SageMaker API e MLflow e desativa a integração correspondente. Mantenha
Streaming e Feature Store desabilitados quando estiverem fora do escopo do ambiente.
Use as saídas `InterfaceEndpoints` (soma dos endpoints) e `InterfaceEndpointAvailabilityZoneCount`
para estimar a parcela fixa; `S3GatewayEndpointId` identifica o gateway no inventário da stack.
