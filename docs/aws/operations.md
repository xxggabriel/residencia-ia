# Operação

Todos os comandos usam us-east-1 e credenciais com permissão de administração da conta alvo.
Antes de qualquer comando que crie recursos, confirme a autorização para provisionar recursos
cobrados na conta.

## 1. Bootstrap (uma vez, idempotente)

```bash
REPOSITORY=owner/residencia-ia OVERRIDES=infra/params/runtime-overrides.example.json scripts/aws/bootstrap.sh
```

O script cria ou atualiza a stack persistente e liga a proteção contra término. Depois define
o padrão de conta `containerInsights=enhanced`, para que o cluster ECS criado pelo Batch tenha
Container Insights, e cria os service-linked roles. Por fim, implanta a stack de CI/CD.
Para usar uma conexão já existente, exporte `CONNECTION_ARN=arn:aws:codeconnections:...`.

### Conexão GitHub

Sem `CONNECTION_ARN`, a stack cria a conexão no estado `PENDING`. O CloudFormation não conclui
a autorização OAuth.

1. Console → Developer Tools → Settings → Connections → `continual-bench-github` → *Update pending connection*.
2. Instale ou escolha o GitHub App "AWS Connector for GitHub" com acesso ao repositório.
3. Aguarde `AVAILABLE`:

```bash
scripts/aws/wait_connection.sh "$(aws cloudformation describe-stacks --stack-name continual-bench-cicd --query "Stacks[0].Outputs[?OutputKey=='ConnectionArn'].OutputValue" --output text)"
```

### Gatilho e regras do GitHub

O pipeline V2 tem um único gatilho, `push` filtrado para `main`. O push gerado pelo merge de
um pull request também dispara. Não há regra do EventBridge, webhook ou polling para o mesmo
repositório, então uma revisão não dispara duas execuções. A detecção nativa da ação de origem
é substituída pelo filtro do gatilho. `ExecutionMode: QUEUED` aplica os merges em ordem. Exigir
merge em vez de push direto é responsabilidade das regras do GitHub: ative branch protection ou
ruleset em `main` com pull request obrigatório e checks exigidos.

## 2. CI/CD

| Estágio | Projeto | O que faz |
|---|---|---|
| Verify | `continual-bench-ci-test` | `pip install --require-hashes`, `ruff`, `pytest`, `validate_templates.py` (cfn-lint e invariantes), `validate_asl.py` offline e `--service` (`ValidateStateMachineDefinition`) e `continual-bench aws local-smoke`. |
| Build | `continual-bench-ci-build` | `build_images.sh`: tag imutável `git-<sha>`, reutiliza a tag se já existir, build `linux/amd64` a partir de `requirements-aws.lock`, selftest e contrato SageMaker antes do push, digest resolvido no ECR. Gera o zip determinístico do Lambda (`lambda/<sha>/control-<hash>.zip`), o código de Glue e Flink e os templates empacotados. |
| Deploy | `continual-bench-ci-deploy` | `deploy_change_set.sh`: cria, descreve e executa o change set. Um change set vazio, como na reexecução do mesmo commit, é removido sem falhar. |
| Deploy | `continual-bench-ci-release` | `write_release.py`: confere as saídas da stack contra o build e os digests no ECR. Grava `releases/<sha>/release.json` uma vez e move `releases/latest.json`. |

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

### Rollback por digest

```bash
scripts/aws/rollback.sh <sha-de-uma-release-anterior>
```

O rollback reaplica templates, imagens e Lambda daquela release, sem rebuild, e move
`releases/latest.json`. Campanhas em andamento continuam com a própria release.

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
scripts/aws/start_campaign.sh infra/examples/campaign-social-smoke.json
scripts/aws/start_campaign.sh infra/examples/campaign-social-full.json      # após verificar memória, disco, quotas e orçamento
scripts/aws/start_campaign.sh infra/examples/campaign-binary-matrix.json    # matriz existente, Distributed Map
```

O nome da execução é o `campaign_id`. Uma nova execução da mesma campanha usa
`<campaign_id>--<sufixo>` e reaproveita as runs com manifesto final. O Lambda `plan_campaign`
valida a entrada, resolve a release e verifica o snapshot. Também confere a quota regional do
Fargate (`L-3032A538`), o `maxvCpus` dos compute environments e o estado da fila, depois
publica o plano. Antes de cada submissão, `reconcile_unit` procura o manifesto final e jobs
com o nome determinístico `cb-<campanha>-<run_id>`. Isso evita reenvio quando a resposta
de uma submissão se perde. Jobs duplicados ativos são terminados, mantendo o mais antigo.
Cada tentativa grava a própria evidência em `attempts/<job>-<n>/`. O manifesto final usa
`If-None-Match: *`. Uma saída 0 sem manifesto final conta como `FAILED_PUBLICATION`.

Timeouts: `jobStateTimeLimitActions` da fila cancela jobs presos em `RUNNABLE` (espera na fila).
`AttemptDurationSeconds` limita a execução. `TimeoutSecondsPath` do `.sync` limita espera mais
execução por item. Códigos de saída: 0 sucesso ou reaproveitamento, 1 falha de execução,
2 contrato (imagem ou perfil divergente do plano), 3 snapshot.

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
scripts/aws/teardown.sh --with-cicd       # também o pipeline
```

A stack persistente (buckets, KMS, ECR), os log groups de jobs e aquisição e o Model Registry
são preservados. Removê-los é uma decisão manual e irreversível, fora destes scripts.
