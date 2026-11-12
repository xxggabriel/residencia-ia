# Evidências de validação

Validação local em 09/11/2026 em macOS arm64, com Docker Desktop 29.9.2 e emulação
`linux/amd64`. As verificações na conta AWS foram feitas separadamente pelo AWS Core MCP.

## Validado localmente

| Verificação | Comando | Resultado |
|---|---|---|
| Suíte completa (585 testes existentes + 84 novos) | `uv run pytest -q` | 669 aprovados, 50 desmarcados (`services`/`benchmark`, como antes) |
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

## Validado na AWS em 09/11/2026

- O pipeline `f2bb99b5-e81d-44b5-aad1-89968f6d31ba` publicou a release
  `f1cb05e1bd22005f3994869fd664e4159a122cb2` após Verify, Build e Deploy bem-sucedidos.
  [Release no S3](https://s3.console.aws.amazon.com/s3/object/continual-bench-data-<ACCOUNT_ID>-us-east-1?region=us-east-1&prefix=releases/f1cb05e1bd22005f3994869fd664e4159a122cb2/release.json)
- O dataset `social-smoke-prepare` terminou com `SUCCEEDED` após reutilizar os dois
  arquivos brutos da aquisição. O job Batch `3c5b80fc-0e5e-4a6b-bad9-25c30bdefcbe`
  publicou o Parquet de 10.000 linhas. A verificação independente no S3 confirmou a
  versão e o SHA-256 do manifesto referenciado pelo marcador
  (`762204a460a8974dd614f66f251f33715ceaa0107513c1ef71ab10cce414c6d1`).
  [Marcador no S3](https://s3.console.aws.amazon.com/s3/object/continual-bench-data-<ACCOUNT_ID>-us-east-1?region=us-east-1&prefix=datasets/social/8518bf04c105df75ab72c13bbec1e12b270eb92f/manifests/zia-temporal-v1/smoke/_SUCCESS.json)

### Campanha `social-smoke-v1`

Em `us-east-1`, a execução começou às 11:22:20 e terminou às 11:24:54 (BRT) com
`FAILED` (`CampaignIncomplete`). O preflight confirmou o snapshot, a release, a
fila `ENABLED/VALID` e capacidade para 4 jobs de 1 vCPU. O compute environment
tinha limite de 8 vCPUs e a quota regional de Fargate on-demand era de 64 vCPUs.
[Execução](https://us-east-1.console.aws.amazon.com/states/home?region=us-east-1#/v2/executions/details/arn:aws:states:us-east-1:<ACCOUNT_ID>:execution:continual-bench-benchmark:social-smoke-v1)

As quatro unidades iniciaram a avaliação sobre 10.000 linhas. O Step Functions
submeteu um retry por unidade. Os oito jobs terminaram `FAILED` com código 1, e
as oito tasks terminaram `STOPPED`. O resumo registrou 0/4 unidades concluídas e
nenhum manifesto final.
[Resumo no S3](https://s3.console.aws.amazon.com/s3/object/continual-bench-data-<ACCOUNT_ID>-us-east-1?region=us-east-1&prefix=campaigns/social-smoke-v1/summary.json)

| Política | Tentativa | Job AWS Batch | ARN da task ECS | Início–fim (BRT) | ENI verificada em execução |
|---|---:|---|---|---|---|
| `frozen` | 1 | `e73178fa-a1c2-4012-a94f-ca5e15f63bd3` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/c371662a0385416da27f7f32fea7bb8e` | 11:22:56.137–11:23:33.221 | `eni-0d94ad3c282a098ff` |
| `frozen` | 2 | `6a234261-211b-4f54-9e33-9d5078d47dad` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/86ed42140eec4771b5344528710b5638` | 11:24:12.136–11:24:47.149 | — |
| `drift` | 1 | `95451715-18de-45ed-8cfd-8f7608252ff9` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/5dbed8935b134127a9d77d4cbf81258c` | 11:23:02.453–11:23:35.678 | `eni-00bf4f8e2e1cd39e7` |
| `drift` | 2 | `1102e2ae-34e7-49d6-a661-60b11d01ec03` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/b6a7fb5e99314cfe8f7d9f6b51afad21` | 11:24:11.087–11:24:44.471 | — |
| `incremental` | 1 | `4b1f1ab4-354a-4c43-933d-8ed442c832f3` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/f79f6995c6d140c9b2089ae76630b2ed` | 11:23:03.567–11:23:40.314 | `eni-0f84632accef26800` |
| `incremental` | 2 | `7fa869f3-bad0-4bfa-b19a-b019cc82ff7f` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/b91299377a464a7e91dd35374e775566` | 11:24:17.923–11:24:50.753 | — |
| `periodic` | 1 | `8259d5d4-aede-47a4-b496-0f37e20f8caa` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/9a2060adf2df497a91a2ffc6808c4344` | 11:23:00.264–11:23:34.776 | `eni-091ed047a71d4b438` |
| `periodic` | 2 | `5be61e8d-360f-48f4-9228-d1974688a75d` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/324f591a94a84ce985a4aa7c938315ac` | 11:24:17.627–11:24:51.571 | — |

- **Paralelismo.** A janela comum das quatro tasks iniciais foi de 11:23:03.567 a
  11:23:33.221 (29,654 s). As quatro tasks de retry também se sobrepuseram por
  26,548 s. Os horários vêm de `DescribeTasks`.
- **Rede.** As quatro ENIs iniciais estavam na sub-rede
  `subnet-0a3eef86260fb57e8` em `us-east-1a`, sem `Association.PublicIp`
  em `DescribeNetworkInterfaces`. O acesso ao S3 foi verificado pelas leituras
  do snapshot e gravações de artefatos das tasks, pela rota da sub-rede ao
  gateway `vpce-0935fe0a9f6cf1d92` e pela condição `aws:SourceVpce` nas
  permissões da role. O bucket tinha logs de acesso habilitados. A correlação
  com as linhas individuais desses logs ficou pendente.
- **Falha dos workers.** Os oito streams de `/continual-bench/batch` registraram
  `AccessDeniedException` em `sagemaker:ListModelPackages`. A role
  `continual-bench-benchmark-job` concedia a ação em `model-package-group/...`;
  a chamada foi avaliada sobre `model-package/<grupo>/*`.
- **Registro de tentativas.** O histórico da execução registrou quatro negativas
  de `s3:GetObject` para a role `continual-bench-control` em `analytics/attempts/`.
  Os oito `status.json` dos workers ficaram em
  `campaigns/social-smoke-v1/runs/<run_id>/attempts/<job_id>-1/`.
- **Reconciliação.** O workflow e o EventBridge gravaram, respectivamente,
  `reconciliation/20261109T142454Z.json` e `reconciliation/20261109T142455Z.json`
  sob o prefixo da campanha, ambos com `clean: true`, zero jobs ativos e zero
  tasks em execução. Nenhum job precisou ser terminado pela reconciliação.
  O retry automático foi observado. O cancelamento manual e o teste com marcador
  ausente continuam pendentes.

### Release do PR #12 e campanha `social-smoke-v3`

O pipeline `2830406f-e8ee-43e9-b1e6-f30f27ec1638` foi iniciado manualmente
com o SHA do merge do PR #12. Source, Verify, Build, ApplyChangeSet e WriteRelease
terminaram com `Succeeded`. A stack de runtime e suas stacks aninhadas ficaram
em `UPDATE_COMPLETE`. O manifesto `92c7c539502dc6426a02065225ab4126cb24052a`
foi confirmado no S3. A release usou os digests de runtime
`sha256:e3be0d1217317e31adf10cd62ed3544749b5c22a59844132e18c6f0bb0c28d95`
e de inferência
`sha256:f4b7a36790314068c4a2c7339dada1933334b78e51009d63aacb518362630c92`.
[Release no S3](https://s3.console.aws.amazon.com/s3/object/continual-bench-data-<ACCOUNT_ID>-us-east-1?region=us-east-1&prefix=releases/92c7c539502dc6426a02065225ab4126cb24052a/release.json)

A campanha reutilizou o snapshot smoke de 10.000 linhas. Começou às 12:38:16.038
e terminou às 12:40:49.873 (BRT) com `FAILED` (`CampaignIncomplete`).
O resumo registrou **0/4**, sem manifestos finais e sem erros de consolidação
analítica. O Step Functions fez um retry por unidade. O Batch reportou
`FAILED` para os oito jobs, e o ECS reportou `STOPPED` com código 1 para as
oito tasks.
[Execução](https://us-east-1.console.aws.amazon.com/states/home?region=us-east-1#/v2/executions/details/arn:aws:states:us-east-1:<ACCOUNT_ID>:execution:continual-bench-benchmark:social-smoke-v3)
[Resumo no S3](https://s3.console.aws.amazon.com/s3/object/continual-bench-data-<ACCOUNT_ID>-us-east-1?region=us-east-1&prefix=campaigns/social-smoke-v3/summary.json)

| Política | Tentativa | Job AWS Batch | ARN da task ECS | Início–fim (BRT) | ENI verificada em execução |
|---|---:|---|---|---|---|
| `frozen` | 1 | `a657590c-75e2-449d-9cba-c31964693cff` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/3c2812ef635b4a3ab190d27037e02f15` | 12:38:56.702–12:39:34.196 | `eni-0f324c2ec81834928` |
| `frozen` | 2 | `95ee1c1b-ae5f-4ab7-9a82-f6fd7f4b9b12` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/38a14527810142b7897180a1c2665ff5` | 12:40:08.786–12:40:41.881 | — |
| `drift` | 1 | `64f9903a-e64f-43e2-8ff4-674eba6e3708` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/902f03d7594b49bc9f59a0e131823bb6` | 12:38:55.648–12:39:29.015 | `eni-02f3731e50eab6fae` |
| `drift` | 2 | `4f3abbe3-e939-480f-9a55-612df8cd0d47` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/165f13943c204132bb6322fd148213f0` | 12:40:10.071–12:40:46.221 | — |
| `incremental` | 1 | `8330f8b2-31a9-448c-a21f-aa087c3526a6` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/8e89eb7b055a4eae9fc919a3ecd594ca` | 12:38:58.035–12:39:34.678 | `eni-067b873770eac5924` |
| `incremental` | 2 | `d572db4f-1851-4703-94f1-471b8c908e0a` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/08b41308389f40dba64b14f4fa200c55` | 12:40:08.645–12:40:42.117 | — |
| `periodic` | 1 | `87739f85-717e-4466-badc-4632789151f8` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/c7051194e8ee4bf084750297739aaad2` | 12:38:57.140–12:39:29.709 | `eni-0f6831f4a581f1ba8` |
| `periodic` | 2 | `4b1d5fb6-8368-4440-8c99-32f65ce8c4dd` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/f8547c7b12474d3d947f5d5a5dbdbf12` | 12:40:05.423–12:40:39.694 | — |

- **Paralelismo.** As quatro tasks iniciais rodaram juntas de 12:38:58.035 a
  12:39:29.015 (30,980 s). As quatro de retry se sobrepuseram por 29,623 s.
- **Rede.** As quatro ENIs iniciais estavam na sub-rede
  `subnet-0a3eef86260fb57e8` em `us-east-1a`, sem `Association.PublicIp`
  em `DescribeNetworkInterfaces`. O gateway S3 `vpce-0935fe0a9f6cf1d92`
  estava disponível. Os workers leram o snapshot e publicaram os registros de
  tentativa com permissões vinculadas ao endpoint. A correlação com registros
  individuais do log de acesso do bucket segue pendente.
- **Falha dos workers.** Os oito streams de `/continual-bench/batch` registraram
  uma `ValidationException` de `CreateModelPackage` com acesso negado ao ECR
  `continual-bench-ml`. O ECR confirmou o digest de inferência da release como
  `ACTIVE`. O bloqueio anterior na leitura do arquivo do modelo no S3 deu
  lugar à validação da imagem pelo SageMaker.
- **Reconciliação.** Os relatórios `reconciliation/20261109T154049Z.json` e
  `reconciliation/20261109T154050Z.json` confirmaram `clean: true`, zero jobs
  ativos e zero tasks em execução. As oito linhas de tentativa estão em
  `analytics/attempts/`.

### Correção de IAM para o registro de modelos

O CloudTrail confirmou oito negativas de `ecr:BatchGetImage` em
`continual-bench-ml` feitas pelo SageMaker em nome de `continual-bench-benchmark-job`.
A política inline `register-candidates` foi atualizada diretamente pelo AWS Core MCP
em 09/11/2026. A releitura confirmou que ela preservou as permissões anteriores e
acrescentou `ecr:BatchGetImage` e `ecr:DescribeImages` apenas em `continual-bench-ml`.
A simulação da role efetiva autorizou as duas ações nesse repositório e retornou
`implicitDeny` para ambas em `continual-bench-runtime`.

O PR #13 incorporou a mesma política ao CloudFormation, com o ARN importado da
stack persistente.

### Release do PR #13 e campanha `social-smoke-v4`

O pipeline `97b20860-260e-4634-a3a3-6f77ef202285` publicou a release
`a43d95ffe3deb864f96b17d706aac4dd5d6d2f86`. As stacks de runtime e de execução
ficaram em `UPDATE_COMPLETE`. A verificação de drift da role
`BenchmarkJobRole` retornou `IN_SYNC` após o deploy.
[PR #13](https://github.com/xxggabriel/residencia-ia/pull/13) · [Release no S3](https://s3.console.aws.amazon.com/s3/object/continual-bench-data-<ACCOUNT_ID>-us-east-1?region=us-east-1&prefix=releases/a43d95ffe3deb864f96b17d706aac4dd5d6d2f86/release.json)

O Docker Hub devolveu HTTP 429 em duas tentativas do Build. A terceira usou
temporariamente o espelho oficial do ECR Public com o mesmo digest da imagem
base. Os testes de imagem e o Build passaram, e o projeto voltou a usar
`infra/buildspecs/build.yml`. Os digests de runtime
`sha256:f2f1c0258f023f92d78438e057d78fba19c999ddfcab97774edc8192cb711f66`
e de inferência
`sha256:cfae3c9dad69f7aea2841d33e07cdcc7d846706e359cad4e77a5b41b86c79927`
foram confirmados no ECR.

O Step Functions executou o smoke de 13:12:38.233 a 13:14:20.707 (BRT) com o
snapshot social de 10.000 linhas. Terminou com `SUCCEEDED` e **4/4** unidades
concluídas na primeira tentativa. O Batch reportou `SUCCEEDED` nos quatro
jobs e o ECS reportou `STOPPED` com código 0 nas quatro tasks.
[Execução](https://us-east-1.console.aws.amazon.com/states/home?region=us-east-1#/v2/executions/details/arn:aws:states:us-east-1:<ACCOUNT_ID>:execution:continual-bench-benchmark:social-smoke-v4) · [Resumo no S3](https://s3.console.aws.amazon.com/s3/object/continual-bench-data-<ACCOUNT_ID>-us-east-1?region=us-east-1&prefix=campaigns/social-smoke-v4/summary.json)

| Política | Job AWS Batch | ARN da task ECS | Início–fim (BRT) | ENI verificada |
|---|---|---|---|---|
| `frozen` | `70f04396-5664-4c4b-9043-42fd1dcd211b` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/568dbc24ef5b4f85b44305d18423ff20` | 13:13:17.037–13:13:51.847 | `eni-026d0c72a7dd3da1c` |
| `drift` | `e83c508e-736a-42a6-8740-7ba1d4160b64` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/eeb998d6055e40ac95e3d0c8d41aead9` | 13:13:15.318–13:13:52.288 | `eni-0e1730357b7687464` |
| `incremental` | `ab0af3aa-eb00-4280-971b-0f4340026731` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/2c737565f5934b8683f9ed3f85b9a8e0` | 13:13:40.802–13:14:17.044 | `eni-0f0901132a470d4ce` |
| `periodic` | `4aeb1f3c-ce2a-48df-b071-b9f37791c6ae` | `arn:aws:ecs:us-east-1:<ACCOUNT_ID>:task/AWSBatch-ComputeEnvironment-V2PitMOOOjI5yHz8-be081166-2d6a-3955-9cda-f2cfbdaccd53/adeb56f8f05548ed920389f5e513ed76` | 13:13:14.275–13:13:48.207 | `eni-09f59e607fec2a62c` |

- Os intervalos `startedAt`–`stoppedAt` das quatro tasks tiveram uma janela
  comum de 13:13:40.802 a 13:13:48.207 (7,405 s).
- As quatro ENIs foram observadas sem IP público na sub-rede
  `subnet-0a3eef86260fb57e8`, em `us-east-1a`. A tabela de rotas
  `rtb-05ed9a446d42b1b77` aponta ao gateway S3 `vpce-0935fe0a9f6cf1d92`,
  exigido pela política da role para leitura e escrita dos artefatos do worker.
- Foram lidos os quatro `final.json` e verificados no S3 os quatro arquivos de
  modelo criptografados com KMS, as quatro linhas de tentativa e os 16 objetos
  analíticos. O resumo não registrou erros de consolidação. O carregador
  reproduziu 256 predições por modelo com diferença máxima de 0.
- O SageMaker retornou `Completed` e `PendingManualApproval` para a versão
  `1` de cada grupo `continual-bench-zia-social-{frozen,drift,incremental,periodic}-smoke`.
- O relatório `reconciliation/20261109T161420Z.json` registrou `clean: true`,
  sem jobs ativos ou tasks em execução. A validação do full segue pendente.

### Validações pendentes na AWS

- Testar cancelamento com jobs ativos, snapshot sem marcador e retry manual com sufixo.
- Preparar o snapshot full e confirmar memória, disco, quota e orçamento antes de
  iniciar a campanha full.
- Validar consultas Glue/Athena com resultados de uma campanha concluída,
  Kinesis/Firehose/Flink, Feature Store, perfis SageMaker,
  Quick Sight e `refresh-prices` na Price List API.

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
