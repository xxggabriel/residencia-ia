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

O manifesto de release contém o Git SHA, as imagens por digest, o tipo de execução do
benchmark, as revisões das job definitions e a versão do Lambda de controle. A campanha
resolve `latest` uma única vez, no plano. Cada tentativa Spot usa a imagem e a versão do
Lambda fixadas no plano, mesmo que haja outro deploy durante a campanha.

### Verificar no ECR a imagem publicada

```bash
SHA=$(git rev-parse origin/main)
aws ecr describe-images --repository-name continual-bench-runtime --image-ids imageTag=git-$SHA \
  --query 'imageDetails[0].[imageDigest,imagePushedAt]' --output text
aws s3 cp s3://continual-bench-data-$ACCOUNT-us-east-1/releases/$SHA/release.json - | python -m json.tool
scripts/aws/outputs.sh RuntimeImage
scripts/aws/outputs.sh BenchmarkComputeBackend
```

O digest do ECR, `images.runtime.digest` da release e o sufixo `@sha256:` da saída
`RuntimeImage` devem coincidir. O backend esperado é `batch-fargate-spot`.

### Recriar o runtime com os logs preservados

Para recriar a stack de runtime após rollback ou remoção, confirme se os grupos
`/continual-bench/batch` e `/continual-bench/acquisition` foram preservados. Se ambos
existirem com a chave KMS e a retenção esperadas, acrescente
`"ReuseRuntimeLogGroups": "true"` ao objeto `config/runtime-parameters.json` no bucket de artefatos.
A retenção dos grupos reutilizados é gerenciada separadamente.

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

O planejador valida os dados e fixa a release e o digest da imagem. A execução é 100%
serverless: cada unidade vira um job do AWS Batch no compute environment `FARGATE_SPOT`
(2 vCPU e 4 GiB por tarefa; o Fargate não aceita 2 GiB com 2 vCPU). Nenhuma instância EC2
é criada. O Map submete **todas as unidades ao mesmo tempo**: o Batch inicia quantas tarefas
couberem no limite de vCPU (`MaxVcpus`, padrão 64, igual à quota de Fargate Spot
`L-36FBB829`; com 2 vCPU são 32 tarefas simultâneas) e mantém as demais em `RUNNABLE` até
liberar capacidade. O preflight registra `running_in_parallel` e `queued` em
`campaigns/<id>/preflight/`. A preparação de datasets usa uma fila separada em Fargate
on-demand, porque não é retomável.

Quando o Fargate recupera a capacidade Spot, a tarefa recebe SIGTERM e, 30 segundos depois,
SIGKILL (o Batch não expõe `stopTimeout`). O worker então:

1. para no próximo ponto seguro do avaliador e confirma um checkpoint no S3
   (`runs/<run_id>/spot/checkpoints/`), além do checkpoint periódico a cada
   `BenchmarkCheckpointSeconds` (padrão 30 s);
2. drena a fila de logging assíncrono do MLflow e grava o estado `INTERRUPTED` da tentativa na
   run lógica;
3. sai com o código 75, que a `RetryStrategy` do job trata como `RETRY` (até `JobAttempts`,
   padrão 5). Se a tarefa for encerrada antes, o motivo de parada do Spot também gera `RETRY`.

A tentativa seguinte é numerada por um claim write-once (`runs/<run_id>/spot/attempt-NNN/claim.json`),
valida as entradas, restaura o checkpoint confirmado mais recente e continua a mesma run no
MLflow. O trabalho feito depois do último checkpoint é refeito. Ao concluir, o worker confirma
a publicação no S3 e no MLflow.

Cada tentativa escreve em `attempts/<job_id>-<n>/`. O relatório continua usando
`final.json`, `analytics.json`, `report.json`, `training-history.json` e o manifesto de
predições. As janelas de progresso e os ajustes no MLflow mantêm a identidade da run lógica.
Se a publicação falhar após os resultados intermediários chegarem ao S3, a próxima tentativa
pode concluir a publicação desse conjunto de resultados.

As métricas ao vivo usam o logging assíncrono nativo do MLflow (`MLFLOW_ENABLE_ASYNC_LOGGING=true`,
`mlflow.config.enable_async_logging`). O treino só enfileira os pontos; o pool do MLflow
(`MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE=4`, lotes a cada `MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS=5`)
os envia ao MLflow App. A fila é drenada no aviso Spot, antes da publicação final e ao encerrar
a run. Sem o pacote MLflow, o worker usa o transporte REST/SigV4.

Use `<campaign_id>--<sufixo>` para repetir uma execução falha com o mesmo pedido. Use um novo
ID se mudar o plano ou se a campanha anterior foi explicitamente cancelada.

### Campanha integral dos três datasets Zia

Os exemplos `dataset-{fraud,social,ecommerce}-full.json` preparam 50, 100 e 50 milhões
de eventos, respectivamente. Aguarde os três marcadores `_SUCCESS.json` no escopo `full`
e confira as contagens no manifesto antes de iniciar as campanhas.

Os exemplos `campaign-{fraud,social,ecommerce}-full.json` definem 12 políticas por dataset
com `max_parallel=12`. As três campanhas juntas submetem 36 jobs (72 vCPUs). Com a quota de
64 vCPUs de Fargate Spot, 32 começam juntos e os outros 4 esperam em `RUNNABLE`. Sem capacidade
Spot disponível, os jobs aguardam até `queue_timeout_seconds`. O prazo para rodar cada tentativa é de uma hora no smoke e 48 horas
no full. Use o mesmo `RELEASE_SHA` publicado nas três submissões e um novo `CAMPAIGN_ID`
para qualquer plano alterado.

```bash
export RELEASE_SHA=<sha-publicado>
scripts/aws/start_campaign.sh infra/examples/campaign-social-full.json
# Para os outros datasets, use os pedidos campaign-fraud-full.json e campaign-ecommerce-full.json.
```

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

### Aceitação com duas políticas

Use o snapshot Social `smoke` e uma release `batch-fargate-spot` publicada. Execute a referência
com um ID novo e guarde o ARN que o comando retorna.

```bash
CAMPAIGN_ID=zia-spot-accept-ref scripts/aws/start_campaign.sh \
  infra/examples/campaign-social-spot-acceptance.json
aws batch list-jobs --job-queue continual-bench-jobs --filters name=JOB_NAME,values=cb-zia-spot-accept-ref-*
```

Repita com um novo ID para cada cenário. Leia `plan.json` para mapear política e `run_id`.
Espere aparecer um manifesto em `runs/<run_id>/spot/checkpoints/manifests/` antes de injetar
a interrupção. Faça os testes apenas nas tarefas das campanhas reduzidas.

- Para exercitar o aviso Spot, pare a tarefa Fargate do job de teste. O ECS envia SIGTERM e,
  30 segundos depois, SIGKILL, como numa interrupção Spot:
  `aws ecs stop-task --cluster <cluster-do-batch> --task <task-arn> --reason "spot drill"`.
  A tentativa deve terminar `INTERRUPTED` com o código 75, e a seguinte deve registrar
  `recovery_reason=INTERRUPTED` e o manifesto retomado. A ação `aws:ecs:stop-task` do AWS FIS
  produz o mesmo efeito.
- Para exercitar uma falha de publicação, simule a perda da confirmação após a escrita de um
  lote MLflow. Confirme `PUBLICATION_FAILED` no estado da tentativa e a conclusão na tentativa
  seguinte com o mesmo identificador de run e de ajustes no MLflow.

Compare os artefatos publicados com a referência por política. Use o mesmo digest e o mesmo
snapshot. Compare o corte temporal, as matrizes de confusão, as predições, a linhagem de ajustes
e as métricas de qualidade. Tempos de execução e custos dependentes desses tempos variam.
Confira os estados das tentativas em `runs/<run_id>/attempts/` e os streams do job no
CloudWatch (`/continual-bench/batch`).

Use `make aws-mlflow-ui` e os IDs em `final.json` para abrir as runs. Confira o histórico de
`progress/rows_processed`, as tags `cb-spot-attempt-*`, os ajustes filhos e os artefatos
`report.json` e `predictions-manifest.json` na run lógica.

## 5. Cancelamento e reconciliação

```bash
scripts/aws/cancel_campaign.sh zia-social-smoke-spot-v1
```

O cancelamento registra a intenção no S3 antes de parar o workflow. Essa intenção bloqueia
novas tentativas. A reconciliação do EventBridge encerra os jobs da execução interrompida e
preserva os checkpoints confirmados e os logs. O histórico da reconciliação fica em
`campaigns/<id>/reconciliation/`. Consulte o estado no Step Functions e confira que não restam
tasks em execução antes de considerar a limpeza concluída.

## 6. Consultar resultados

```bash
continual-bench aws query infra/athena/compare_policies.sql --param campaign=social-smoke-v1
continual-bench aws query infra/athena/window_series.sql --param campaign=social-smoke-v1
continual-bench aws query infra/athena/campaign_runs.sql --param campaign=social-smoke-v1
continual-bench aws query infra/athena/attempt_costs.sql --param campaign=social-smoke-v1
```

A tabela `predictions` usa projeção injetada: filtre `campaign_id`, `run_id` e `attempt_key`
(lidos em `runs`/`final.json`). Os logs dos workers Fargate Spot e do preparo de datasets ficam
no CloudWatch Logs `/continual-bench/batch`. O dashboard `continual-bench-operations` mostra as
métricas EMF por dataset, escopo e política, sem `run_id` como dimensão.

## 7. Ciclo dos endpoints de interface

O benchmark provisiona a stack `continual-bench-interface-endpoints` depois do preflight
e aguarda todos os endpoints ficarem `available` antes de submeter os jobs Fargate Spot. O workflow
de datasets faz o mesmo quando precisa executar a etapa de preparação no Batch. Use os workflows para executar os workers nas sub-redes privadas. Para repetir uma execução encerrada, inicie uma
nova execução com o mesmo pedido e o sufixo de tentativa do projeto para passar pelo
provisionamento novamente.

As execuções compartilham a stack. A limpeza roda a cada cinco minutos e aguarda todas
as execuções admitidas encerrarem, um intervalo de dois minutos para propagar as
transições e as filas de benchmark e de preparação ficarem sem jobs ativos. Isso abrange falhas e cancelamentos.
O deploy da `main` atualiza o runtime e mantém o MLflow App, os workflows do Studio,
a VPC e o gateway do S3. Os workflows de benchmark e de preparação de datasets provisionam a stack de endpoints.

Na migração de uma stack com endpoints permanentes, aplique primeiro o template atualizado
de CI/CD. Faça o primeiro deploy da versão nova com a fila do Batch vazia e sem execuções
dos workflows de benchmark e dataset. Aguarde o deploy concluir antes de enviar novas
campanhas. Essa atualização remove os endpoints que estavam na stack de rede para liberar
o DNS privado usado pela stack temporária.

## 8. Remover todo o runtime

```bash
scripts/aws/flink.sh stop                 # se o streaming estiver ativo
scripts/aws/teardown.sh                   # runtime: endpoints da VPC, Batch, Lambda, state machines, módulos
scripts/aws/teardown.sh --with-cicd       # também a stack de OIDC e roles de deploy
```

O script aguarda a remoção da stack temporária antes de excluir o runtime.
Ele preserva a stack persistente (buckets, KMS, ECR), os log groups de jobs e aquisição
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

Para Zia, use o experimento `<sha-curto>-continual-bench-zia-experiments` e filtre por
`cb-campaign-id`. Ao iniciar uma tentativa de avaliação, o worker abre a run pai
como `RUNNING`. A série `progress/*` registra exemplos processados e avaliados,
atualizações, retreinamentos e métricas acumuladas com prefixos `prequential/` e
`holdout/`. As políticas por custo também publicam `progress/matrix/*` durante a
construção da matriz. Nelas, a coordenada de step inclui as linhas lidas nessa fase.
Use `progress/rows_processed` e `progress/rows_total` para acompanhar a cobertura da
avaliação. O worker agrupa os pontos de cada série em intervalos de 30 segundos
quando há blocos concluídos e publica os pontos pendentes antes do relatório final.
A publicação final usa a mesma run pai e registra as runs filhas, os checkpoints
e os artefatos. Uma retomada só de publicação reutiliza a tentativa arquivada.

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
