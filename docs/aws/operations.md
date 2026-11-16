# Operação

Os exemplos usam `us-east-1`. Configure a região e as credenciais da conta alvo no shell.

## 1. Bootstrap

```bash
REPOSITORY=owner/residencia-ia scripts/aws/bootstrap.sh
```

O script cria ou atualiza a stack persistente e liga a proteção contra término. Depois define
o padrão de conta `containerInsights=enhanced` para o cluster ECS criado pelo Batch, cria as
service-linked roles e implanta a stack de CI/CD. Para reutilizar o provedor OIDC da conta,
exporte `OIDC_PROVIDER_ARN`. Para enviar um arquivo de parâmetros operacionais ao bucket de
artefatos durante o bootstrap, defina `OVERRIDES` com o caminho do arquivo.

### Configuração do GitHub Actions

Crie o ambiente `production` no GitHub e restrinja as branches de deploy conforme as regras do
repositório. A role AWS confia no subject OIDC desse ambiente. O bootstrap configura
`AWS_ROLE_ARN` com `gh` quando disponível. Caso contrário, use a saída `GitHubActionsRoleArn`
da stack `continual-bench-cicd`. Configure também as variáveis do workflow:

```bash
gh variable set AWS_REGION --body us-east-1
gh variable set PROJECT --body continual-bench
gh variable set RUNTIME_STACK --body continual-bench-runtime
gh variable set RUNTIME_CONFIG_KEY --body config/runtime-parameters.json
```

O workflow [Build and deploy](../../.github/workflows/deploy.yml) inicia com push ou merge em
`main` e também aceita execução manual. O grupo `deploy-main` serializa as execuções com
`cancel-in-progress: false`. O workflow [CI](../../.github/workflows/ci.yml) verifica pull
requests e pushes fora de `main`.

## 2. CI/CD

| Job | O que faz |
|---|---|
| `verify` | Executa o workflow CI reutilizável com dependências fixadas por hash, Ruff, pytest, CloudFormation, ASL offline e smoke local de Social, Fraud e E-commerce. |
| `build` | Valida o ASL na AWS e publica imagens `linux/amd64` com `build_images.sh`. O script reaproveita a tag imutável `git-<sha>` existente e resolve o digest no ECR. Valida imagens novas com selftest e contrato SageMaker antes do push. Também empacota os templates, o zip do Lambda e o código de Glue e Flink. |
| `deploy` | Aplica o change set do runtime com `deploy_change_set.sh` e lê os parâmetros operacionais do objeto S3 indicado por `RUNTIME_CONFIG_KEY`. Um change set vazio é removido sem falhar. |
| `release` | `write_release.py` confere as saídas da stack contra o build e os digests no ECR. Grava `releases/<sha>/release.json` uma vez e move `releases/latest.json`. |

O manifesto de release contém o Git SHA, as imagens por digest, as revisões das job
definitions e a versão do Lambda de controle. A campanha resolve `latest` uma única vez, no
plano. Daí em diante, jobs e chamadas ao Lambda usam os ARNs com revisão e versão daquela
release, mesmo que outro merge aconteça durante a campanha.

### Verificar no ECR a imagem publicada

```bash
SHA=$(git rev-parse origin/main)
aws ecr describe-images --repository-name continual-bench-runtime --image-ids imageTag=git-$SHA \
  --query 'imageDetails[0].[imageDigest,imagePushedAt]' --output text
aws s3 cp s3://continual-bench-data-$ACCOUNT-us-east-1/releases/$SHA/release.json - | python -m json.tool
aws batch describe-job-definitions --job-definitions "$(scripts/aws/outputs.sh BenchmarkJobDefinitionArn)" \
  --query 'jobDefinitions[0].containerProperties.image' --output text
```

O digest do ECR, o `images.runtime.digest` do release e o sufixo `@sha256:` da imagem da job
definition devem coincidir.

### Recriar o runtime com os logs preservados

Para recriar a stack de runtime após rollback ou remoção, confirme se os grupos
`/continual-bench/batch` e `/continual-bench/acquisition` foram preservados. Se ambos
existirem com a chave KMS e a retenção esperadas, acrescente
`"ReuseRuntimeLogGroups": "true"` ao objeto `config/runtime-parameters.json` no bucket de artefatos.
O runtime passa a usar esses grupos existentes, e a retenção deles é gerenciada separadamente.

### Rollback por digest

```bash
scripts/aws/rollback.sh <sha-de-uma-release-anterior>
```

O rollback reaplica os templates e os artefatos da release sem rebuild, preservando os
parâmetros ativos do runtime. O script exige as revisões Batch ativas, a versão da Lambda
disponível e os digests no ECR antes de aplicar o change set. Ele repete a verificação
antes de mover `releases/latest.json` e recusa sobrescrever um ponteiro alterado
durante a operação. Releases que já perderam essas referências exigem uma nova publicação.
Faça rollback em uma janela sem deploy concorrente.

## 3. Datasets (manual)

```bash
scripts/aws/start_dataset.sh infra/examples/dataset-social-smoke.json      # baixar + preparar 10 mil linhas
scripts/aws/start_dataset.sh infra/examples/dataset-social-full.json       # 100 milhões de arestas
scripts/aws/start_dataset.sh infra/examples/dataset-social-import.json     # importar cópia já no S3 (imports/)
scripts/aws/start_dataset.sh infra/examples/dataset-social-prepare-only.json
```

- **Baixar** (`mode=download`): o CodeBuild de aquisição roda fora da VPC privada. Ele deriva a
  URL do catálogo (repositório, revisão e arquivo fixados) e aceita redirecionamentos somente
  por HTTPS para `huggingface.co` e `hf.co`. Há limite de bytes, timeout, retry com backoff e
  retomada por `Range`. O SHA-256 é verificado antes do upload multipart condicional.
- **Importar** (`mode=import`): copia de `s3://<bucket-de-dados>/imports/...` e verifica o
  SHA-256 contra o catálogo. A role de aquisição só lê esse prefixo.
- **Preparar** (`mode=prepare`): usa raw já publicado e falha se ele não existir.

Execuções com o mesmo nome são recusadas pelo Step Functions enquanto a anterior roda. Para
retomar depois de uma falha, repita a entrada com sufixo (`start_dataset.sh ... retry1`).
Objetos válidos são reaproveitados e evidência divergente nunca é sobrescrita.

Para identificar o snapshot pronto:

```bash
aws s3 cp s3://$BUCKET/datasets/social/8518bf04c105df75ab72c13bbec1e12b270eb92f/manifests/zia-temporal-v1/smoke/_SUCCESS.json -
```

O benchmark falha no preflight (`PreflightFailed`, erro `SnapshotNotReady`) se o snapshot
estiver ausente (`missing`), sem marcador (`incomplete`) ou divergente (`incompatible`).

## 4. Campanhas (manual)

```bash
# RELEASE_SHA contém o SHA publicado da release escolhida.
scripts/aws/start_campaign.sh infra/examples/campaign-social-smoke.json
scripts/aws/start_campaign.sh infra/examples/campaign-binary-matrix.json    # matriz existente, Distributed Map
```

O nome da execução é o `campaign_id`. Uma nova execução da mesma campanha usa
`<campaign_id>--<sufixo>` e reaproveita as runs com manifesto final. O Lambda `plan_campaign`
valida a entrada, resolve a release e verifica o snapshot. Também confere a quota regional do
Fargate (`L-3032A538`), o `maxvCpus` dos compute environments e o estado da fila, depois
publica o plano. Antes de cada submissão, `reconcile_unit` procura o manifesto final e jobs
com o nome determinístico `cb-<campanha>-<run_id>`. Isso evita reenvio quando a resposta
de uma submissão se perde. Jobs duplicados ativos são terminados, mantendo o mais antigo.
Cada tentativa grava a própria evidência em `attempts/<job>-<n>/`, incluindo as linhas do
Athena (`analytics-rows.json`). O manifesto final usa `If-None-Match: *`. Depois dele, as
linhas da tentativa vencedora são publicadas e o marcador `analytics.json` é gravado. Se essa
etapa for interrompida, um job reaproveitado ou a consolidação da campanha refazem a
publicação. Uma saída 0 sem manifesto final conta como `FAILED_PUBLICATION`. A reconciliação
só termina jobs cujo nome é exatamente `cb-<campanha>-<run_id de 20 hex>`, para não alcançar
campanhas cujo ID começa com o mesmo prefixo.

As regras de `jobStateTimeLimitActions` da fila cancelam jobs presos em `RUNNABLE`.
O pedido `run_timeout_seconds` chega ao Batch em `Timeout.AttemptDurationSeconds`
nos dois modos do Map. O `TimeoutSecondsPath` do `.sync` limita espera mais
execução por item. Códigos de saída: 0 sucesso ou reaproveitamento, 1 falha de execução,
2 contrato (imagem ou perfil divergente do plano), 3 snapshot.

### Campanha integral dos três datasets Zia

Os exemplos `dataset-{fraud,social,ecommerce}-full.json` preparam 50, 100 e 50 milhões
de eventos, respectivamente. Aguarde os três marcadores `_SUCCESS.json` no escopo `full`
e confira as contagens no manifesto antes de iniciar as campanhas.

Os exemplos `campaign-{fraud,social,ecommerce}-full.json` definem 12 jobs independentes
por dataset, 36 no total. Os três exemplos `*-smoke.json` usam a mesma matriz e snapshots
de 10 mil linhas. Os scripts exigem um SHA de release publicado para Zia. Defina o mesmo
`RELEASE_SHA` nas três submissões ou grave o SHA em `request.release` em cada arquivo.
Use `CAMPAIGN_ID` ou edite o campo `request.campaign_id` quando mudar um plano já publicado.
Cada plano reserva até duas execuções simultâneas. Com 1 vCPU e 2 GiB por job, as três
campanhas em conjunto podem ocupar até 6 das 8 vCPUs configuradas por padrão. Se a
reserva da release ou a capacidade livre da fila mudar, ajuste a concorrência antes
da submissão. O prazo por tentativa é de 1 hora no smoke e 48 horas no full.

As variantes são `frozen`, `online`, `drift`, `drift_adwin`, `drift_ddm`, `incremental`,
`periodic`, `cara`, `cara_cumulative`, `cara_periodic`, `upf` e `oracle`.
Todas usam o mesmo modelo por dataset, o mesmo warm-up e
as mesmas fronteiras temporais. O modo `online` atualiza em minibatches após a predição
do bloco. Em Social, River SGD atualiza os coeficientes da regressão logística multinomial.
Em Fraud e E-commerce, a perda logística do River fornece os gradientes e o XGBoost
atualiza os pesos das folhas das árvores existentes. O holdout usa o último checkpoint.

Os dois detectores recebem o mesmo sinal binário de erro de cada predição e podem acionar
um retreino por bloco. O relatório registra os parâmetros de decisão e o método de atualização.

As [políticas com custo](../cost-aware-retraining.md) fazem um ajuste por bloco e medem pares de
modelos e blocos da matriz de desempenho (O(B²)). Para dimensionar o full, use medições de tempo,
memória e disco em um volume intermediário representativo. Para planejar somente as cinco
políticas por custo de Social, use `campaign-social-full-cost-aware.json`.

### Suíte de todos os datasets

`matrix.kind=project_suite` usa o mesmo gerador de casos da DAG `benchmark_task_suite`.
O padrão tem 518 unidades e usa 3.000 eventos nos casos binários, 100 exemplos e duas
épocas nos casos por épocas, atraso de três e seed 42. `matrix` aceita os campos
`samples`, `dataset_examples`, `epochs`, `label_delay` e `seed`.

Publique os inputs numa máquina com acesso às fontes. Prepare primeiro os arquivos ARFF
de Electricity e Airlines com `scripts/prepare_real_sources.py` e informe o diretório
via `--real-directory`. Para as outras nove fontes, use o cache existente ou acrescente
`--download` para aquisição explícita. Exemplo para INSECTS:

```bash
continual-bench aws publish-matrix-source --dataset insects --cache state/datasets \
  --examples 100 --seed 42 --bucket "$BUCKET"
```

Defina `BENCH_KMS_KEY_ID` com o ARN da chave do bucket de dados. Guarde o `dataset` e o
`sha256` retornados por cada publicação no objeto `request.source_digests`, mapeado por
nome de dataset. A mesma publicação OpenML atende aos modos binário, tipado e por épocas.
Os bundles das outras fontes atendem ao avaliador por épocas e precisam corresponder à
contagem e à seed da campanha. Fixe o SHA em `request.release` e use um novo `campaign_id`
ao alterar dados ou release.

O planejador valida os objetos antes de publicar o plano. Cada worker confere o SHA-256
do conteúdo ao baixar pelo gateway do S3. O Distributed Map lê o plano do bucket de dados
e exporta o resultado da orquestração para `map-results/<campanha>/` no bucket de
artefatos com SSE-KMS por padrão. Manifestos e métricas das unidades continuam no bucket
de dados.

## 5. Cancelamento e reconciliação

```bash
scripts/aws/cancel_campaign.sh social-smoke-v1                  # stop-execution + reconcile até tudo parar
continual-bench aws reconcile --campaign social-smoke-v1 --bucket $BUCKET --queue "$(scripts/aws/outputs.sh JobQueueArn)"
```

A interrupção do `.sync` é de melhor esforço. Por isso a regra do EventBridge para execuções
`ABORTED`, `TIMED_OUT` ou `FAILED` chama `reconcile_campaign`. Ele termina somente jobs com tag
`cb-stack` do projeto e nome da campanha, a única permissão de `TerminateJob` concedida, e
grava `campaigns/<id>/reconciliation/`. Ao fim de toda campanha, o workflow repete a
verificação por até 10 minutos e falha com `TasksStillActive` se algum job não estiver
terminal ou alguma task ECS não estiver `STOPPED`.

## 6. Consultar resultados

```bash
continual-bench aws query infra/athena/compare_policies.sql --param campaign=social-smoke-v1
continual-bench aws query infra/athena/window_series.sql --param campaign=social-smoke-v1
continual-bench aws query infra/athena/campaign_runs.sql --param campaign=social-smoke-v1
continual-bench aws query infra/athena/attempt_costs.sql --param campaign=social-smoke-v1
```

A tabela `predictions` usa projeção injetada: filtre `campaign_id`, `run_id` e `attempt_key`
(lidos em `runs`/`final.json`). Logs: CloudWatch Logs `/continual-bench/batch`, em JSON com
`campaign_id`, `run_id` e `job_id`. O dashboard `continual-bench-operations` mostra as
métricas EMF por dataset, escopo e política, sem `run_id` como dimensão.

## 7. Remover recursos cobrados

```bash
scripts/aws/flink.sh stop                 # se o streaming estiver ativo
scripts/aws/teardown.sh                   # runtime: endpoints da VPC, Batch, Lambda, state machines, módulos
scripts/aws/teardown.sh --with-cicd       # também a stack de OIDC e roles de deploy
```

O script preserva a stack persistente (buckets, KMS, ECR), os log groups de jobs e aquisição
e o Model Registry. Planeje separadamente a parada de execuções e remoção das stacks
complementares do Studio. Elas usam os workflows, buckets e app MLflow do ambiente.

## Campanhas no SageMaker Studio

A stack complementar `infra/cloudformation/studio.yaml` usa os ARNs dos workflows de
dataset e benchmark da stack de runtime. Empacote-a com `aws cloudformation package` no
bucket de artefatos e passe `DataBucket`, `KeyArn`, `CampaignWorkflowArn` e
`DatasetWorkflowArn` no deploy. O parâmetro `EnableMlflowTracking` do runtime habilita o
MLflow App e o acesso da rede privada. O parâmetro `EnableModelRegistry` controla o
registro de modelos no SageMaker.

```bash
# RELEASE_SHA contém o SHA publicado da release escolhida.
scripts/aws/start_studio_campaign.sh infra/examples/campaign-social-smoke.json
# Ou solicite a preparação do snapshot antes da campanha
scripts/aws/start_studio_campaign.sh \
  infra/examples/campaign-social-smoke.json infra/examples/dataset-social-smoke.json
```

No Studio, em **Pipelines -> continual-bench-campaign**, abra a execução pelo ID da
campanha. A etapa `PrepareDataset` reutiliza o workflow de dataset quando recebe uma
solicitação. `RunBatchCampaign` acompanha o workflow de benchmark e publica seu ARN.
`VerifyResults` verifica a conclusão e expõe o URI do resumo no S3. O mapa da campanha
e os estados de cada job ficam no Step Functions e no Batch. Se houver cancelamento pelo
Studio, o callback cancela o workflow associado e o reconciliador existente encerra os
jobs da campanha. A etapa fica como `Failed`, com motivo `Cancelled from SageMaker Studio`,
para impedir que o SageMaker inicie etapas dependentes após o cancelamento.

Para abrir os experimentos, siga **Studio → Applications → MLflow →
continual-bench-mlflow → Open MLflow → workspace default → Experiments**.
A aba **Experiments** do menu principal do Studio consulta os recursos **MLflow
Tracking Server**. Neste ambiente ela exibe `No tracking server`, pois os jobs
publicam no **MLflow App** serverless. O app contém os experimentos e o histórico
dos runs da matriz, dos jobs nativos e das campanhas Zia.

No app MLflow, filtre o experimento `continual-bench-matrix` por `cb-campaign-id`.
Cada tentativa do Batch registra um run com o ID do job, a revisão e o contrato da
unidade. As métricas `progress/*` usam a janela ou época do avaliador, e `final/*`
registra as métricas finais. O manifesto final da unidade aponta para o run MLflow.

## Jobs nativos no SageMaker Studio

### Onde localizar datasets e avaliações

O menu **Assets -> Datasets / Evaluators** do SageMaker Studio lista ativos
registrados para customização de modelos generativos. O formulário de datasets
espera JSONL no esquema da técnica escolhida. O de evaluators registra funções
ou prompts de recompensa. Veja os [tipos de ativos na documentação da AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-open-weight-create-assets-ui.html).

Para localizar os snapshots tabulares do projeto, abra o prefixo `datasets/` do
bucket de dados. Cada manifesto fica em `datasets/<dataset>/<revisão>/manifests/<versão>/<escopo>/`.
O marcador `_SUCCESS.json` indica que o snapshot foi publicado, e `snapshot.json`
referencia o Parquet preparado. Para as avaliações, abra **Applications → MLflow →
continual-bench-mlflow → Open MLflow → default → Experiments**. Use
`continual-bench-native` para os jobs nativos ou `continual-bench-matrix` para as
matrizes. As tabelas `runs` e `metrics` do Athena também permitem consultar os
resultados do benchmark.

No [catálogo do SageMaker Unified Studio](https://dzd-6vrgrvxbeefk2v.sagemaker.us-east-1.on.aws/projects/dtpvnbwwrsq6xz/search?searchText=Continual%20Bench),
abra os ativos **Continual Bench - Social (full)**, **Continual Bench - Fraud (full)**
ou **Continual Bench - E-commerce (full)**. O formulário `SnapshotMetadata` registra
a revisão da fonte, o URI e a versão do Parquet, o manifesto e os hashes SHA-256.
A seção **Avaliações no MLflow** agrupa por campanha os links para os runs que
usaram esses snapshots completos. Nos runs correspondentes, `cb-catalog-asset-url`
aponta de volta para o ativo, e `cb-catalog-snapshot-sha256` identifica o manifesto.

#### Publicação automática no catálogo

O EventBridge envia os eventos `_SUCCESS.json` de Social, Fraud e E-commerce e os
`final.json` das campanhas Batch para a fila FIFO da stack `continual-bench-studio-catalog`.
A função valida o manifesto e a versão do Parquet no S3. Uma nova identidade de
snapshot (dataset, revisão da fonte, versão da preparação e escopo) gera um ativo
distinto, mantendo acessíveis os snapshots anteriores. A função reaproveita os
ativos já cadastrados. Ao encontrar uma avaliação concluída e ativa no MLflow
que corresponde ao snapshot, publica uma revisão dos metadados e grava o vínculo
inverso no run.
Até 20 avaliações recentes ficam no README do ativo. Para localizar todas as
avaliações de um snapshot no MLflow, filtre pela tag `cb-catalog-asset-id`.

Para implantar, habilite a notificação EventBridge no bucket de dados com
`infra/cloudformation/persistent.yaml` e execute o comando abaixo no ambiente
em que o tipo `ContinualBenchTabularSnapshot` e seu formulário já existem.
O script inclui a função como colaboradora no projeto do catálogo.

```bash
scripts/aws/deploy_studio_catalog.sh <domain-id> <project-id>
```

Os outputs `DeliveryFailureQueueUrl` e `ProcessingFailureQueueUrl` identificam as
filas de falhas. Depois de corrigir a causa, republique a mensagem original na
`PublicationQueueUrl` com o grupo FIFO `catalog`, ou faça redrive a partir da fila
de processamento. A automação confere novamente a identidade e os vínculos ao
receber o mesmo evento. As avaliações nativas e de matriz continuam nos
experimentos existentes do MLflow. O vínculo automático usa o `final.json` das
campanhas Zia, que registra a identidade e os hashes do snapshot e o ID do run.

### Executar o pipeline nativo

O pipeline `continual-bench-native` tem três tipos nativos de etapa. `ProfileSnapshot`
valida o hash e a contagem de linhas do snapshot Social, grava o perfil e publica o
canal de treinamento. `TrainBaseline` treina o modelo com esse canal.
`TuneHyperparameters` usa o mesmo canal e procura valores de regularização e de
`max_epochs`, minimizando `validation:log_loss` na validação temporal.

Para publicar outra revisão, construa as duas imagens AWS a partir da mesma revisão
e passe os seus digests ao script. A implantação usa a stack complementar
`continual-bench-studio-native` e a aplicação MLflow da stack de runtime.

```bash
scripts/aws/deploy_native_pipeline.sh "$RUNTIME_IMAGE" "$ML_IMAGE" "$GIT_SHA"

aws sagemaker start-pipeline-execution --region us-east-1 --pipeline-name continual-bench-native \
  --pipeline-parameters Name=MaxEpochs,Value=24 Name=TuningMaxEpochs,Value=24 \
    Name=MaxTuningJobs,Value=2 Name=MaxParallelTuningJobs,Value=1
```

Os padrões de entrada apontam para o snapshot `smoke`. Para outra revisão ou escopo,
informe juntos `PreparedParquetUri`, `PreparedManifestUri` e `Scope`. No Studio, o grafo
mostra o status e os detalhes de cada job nativo. No MLflow, use o experimento
`continual-bench-native` e filtre pela tag `cb-pipeline-execution-id`. O processamento
registra o progresso por lote de linhas, o treinamento publica métricas por época e
cada trial de tuning tem seu próprio run com os hiperparâmetros e o objetivo medido.


Na etapa `TuneHyperparameters`, abra o job de tuning e consulte **Top Training Jobs**.
O SageMaker ordena os trials pelo valor final de `validation:log_loss`, do menor para o
maior. O treinamento emite a métrica no formato esperado pela definição do job. Para cruzar
os dados da interface com a API:

```bash
aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job \
  --hyper-parameter-tuning-job-name "$TUNING_JOB" \
  --sort-by FinalObjectiveMetricValue --sort-order Ascending
```
