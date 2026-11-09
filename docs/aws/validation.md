# Evidências de validação

Data: 09/11/2026. Máquina: macOS arm64, Docker Desktop 29.9.2 com emulação `linux/amd64`.
Não havia AWS CLI nem credenciais nesta máquina. Nenhum recurso foi criado na conta AWS.

## Validado localmente

| Verificação | Comando | Resultado |
|---|---|---|
| Suíte completa (585 testes existentes + 80 novos) | `uv run pytest -q` | 665 aprovados, 50 desmarcados (`services`/`benchmark`, como antes) |
| Lint | `uv run ruff check src tests dags examples scripts` | sem erros |
| CloudFormation | `make aws-validate` (cfn-lint 1.38.0 + invariantes) | 12 templates, 0 erros de invariantes, cfn-lint sem erros (W3002 ignorado: caminhos resolvidos por `package`) |
| ASL | `scripts/aws/validate_asl.py` offline | 2 definições, 0 erros |
| Smoke AWS local | `continual-bench aws local-smoke` | 4/4 runs concluídas, códigos 0, execuções sobrepostas no tempo |
| Protocolo | `test_worker_matches_the_local_protocol` | janelas, digests de predições e de modelos e confusão do holdout iguais aos do caminho local |
| Carregador nativo | `final.json → model.loader_check` | 256 linhas do holdout reproduzidas, diferença máxima 0,0 |
| Lambda de controle | `scripts/aws/build_control_lambda.py` duas vezes + container `linux/amd64` Python 3.12 | zip reproduzível (mesmo SHA-256), importa e planeja a matriz binária (360 unidades, modo distribuído) só com pydantic/PyYAML, sem numpy, pyarrow, river ou boto3 carregados |
| Imagens | `make aws-images` | ver a seção abaixo |

Os testes novos cobrem escrita condicional no S3 (PutObject e multipart com
`If-None-Match`, via botocore Stubber) e contratos e admissão pelo catálogo. Cobrem também
download com retomada, limite, allowlist de redirecionamentos e SHA-256, além de snapshots
write-once com marcador por último e preflight `missing`/`incomplete`/`incompatible`. No
planejamento, testam `run_id` determinístico, plano publicado uma vez, conflito de campanha e
admissão de capacidade. No worker, testam reaproveitamento de run concluída, preservação da
tentativa falha e códigos de saída. Na reconciliação, testam espera, duplicatas, tentativas
esgotadas, `FAILED_PUBLICATION`, timeout, cancelamento e tasks ECS ativas. O EventBridge só
observa e reconcilia. As políticas IAM são verificadas (VPC endpoint, `TerminateJob` por tag,
`PassRole`, `ListBucket` para 404 corretos), assim como o gatilho do pipeline restrito a push
na `main`, os buildspecs sem aquisição ou campanha e os módulos desligados por padrão. Também
há testes para a ordem, as duplicatas, as lacunas e a retomada por cursor do Kinesis, para as
janelas por tempo do evento com restauração, para as checagens point-in-time, para os perfis
SageMaker, para a idempotência do Model Registry, para as dimensões EMF e para a contabilidade
em USD.

## Imagens (linux/amd64)

- **Build**: `runtime` e `ml` construídas para `linux/amd64` a partir de
  `requirements-aws.lock` com hashes (740 MB cada, camada de dependências compartilhada).
- **Execução local não comprovada**: no Mac arm64, o x86_64 roda via QEMU. O selftest da
  `runtime` recebeu `qemu: uncaught target signal 11` e ficou preso. Isolando os imports,
  numpy e scipy carregaram e `import pyarrow` travou sob a emulação. É uma limitação conhecida
  do QEMU com o pyarrow, mas, sem host x86_64, a execução da imagem não foi demonstrada aqui.
- **Onde fica comprovada**: o estágio Build roda em CodeBuild x86_64 nativo e, antes do push,
  executa `selftest --require-x86-64` na `runtime` e `check_inference_image.sh` na `ml`
  (treino da fixture, `/ping`, `/invocations`). Uma falha impede a publicação.
- **Alternativa local**: `IMAGE_PLATFORM=linux/arm64` (argumento do Dockerfile) e `PLATFORM=linux/arm64`
  (`check_inference_image.sh`) permitem uma checagem nativa no Apple Silicon. Ela foi iniciada
  e interrompida durante o primeiro build, sem resultado. Ativar "Use Rosetta for x86_64/amd64
  emulation" no Docker Desktop costuma permitir rodar a imagem amd64 localmente.

## Não executado em AWS

Nenhum destes itens foi executado. Eles dependem de autorização para provisionar recursos e de
credenciais da conta alvo:

- conexão GitHub (CodeConnections `AVAILABLE`), execução do pipeline e publicação no ECR;
- `ValidateStateMachineDefinition` pelo serviço (`validate_asl.py --service`, roda no estágio Verify);
- aquisição smoke e full, preparação no Batch privado, manifesto e marcador no S3;
- as quatro tasks Fargate privadas em paralelo, acesso ao S3 sem IP público, estado terminal no
  Batch e `STOPPED` no ECS, falha, cancelamento e retry reais;
- Glue/Athena sobre dados reais, Model Registry, Kinesis/Firehose/Flink, Feature Store, perfis
  SageMaker, Quick Sight, CUR 2.0, Budgets e Cost Anomaly Detection;
- `refresh-prices` na Price List API.

## Ordem de validação recomendada numa conta autorizada

1. `scripts/aws/bootstrap.sh`, autorização do GitHub App e `wait_connection.sh`.
2. Merge em `main`. Conferir as quatro etapas do pipeline, o digest no ECR (`operations.md`)
   e `releases/<sha>/release.json`.
3. `start_dataset.sh infra/examples/dataset-social-smoke.json`. Conferir `_SUCCESS.json`.
4. `start_campaign.sh infra/examples/campaign-social-smoke.json` e registrar:
   - os IDs dos jobs Batch e os ARNs das tasks em `analytics/attempts/`;
   - `describe-jobs` com `SUCCEEDED` e `describe-tasks` com `STOPPED`;
   - a sobreposição de `startedAt`/`stoppedAt` das quatro tasks;
   - `describe-network-interfaces` das tasks sem `Association.PublicIp`;
   - os acessos ao S3 pelo endpoint (`aws:SourceVpce` nas roles e nos logs de acesso do bucket).
5. Repetir com uma falha induzida (snapshot sem marcador), um cancelamento
   (`cancel_campaign.sh`) e um retry com sufixo. Conferir que nenhuma task fica ativa.
6. Antes do full: memória (pico de RSS do smoke e `system/` do relatório), disco
   (`BenchmarkEphemeralGiB` contra o tamanho do Parquet e das predições), quota Fargate e
   orçamento. Só então `campaign-social-full.json`.
