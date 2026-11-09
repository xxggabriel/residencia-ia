# Módulos ativáveis

O perfil Core (rede, execução, observabilidade, analytics e billing) é o padrão. Os demais
módulos têm flag explícita em `infra/params/runtime-overrides.example.json`. Para ativar,
copie o arquivo para `s3://<artefatos>/config/runtime-parameters.json` e faça um merge, ou
reexecute o pipeline. A etapa de deploy aplica as flags no change set.

## Streaming (`EnableStreaming=true`)

- **Kinesis Data Streams** (`continual-bench-events`, 1 shard, KMS): `KinesisTransport`
  implementa a mesma porta `contract()`/`events()` do `KafkaTransport`. O modo streaming do
  avaliador binário passa a usá-lo sem mudar o protocolo. Cada evento leva `event_id` e `seq`.
  O consumidor restaura a ordem com um buffer limitado, porque retries do `PutRecords` podem
  reordenar registros. Ele descarta redeliveries verificadas, recusa lacunas e payloads
  alterados e grava o cursor (próximo `seq` e posição no shard) no S3. A retomada usa
  `AFTER_SEQUENCE_NUMBER` e só aceita cursor da mesma run.
- **Data Firehose** arquiva o stream em `archive/streams/` (GZIP, KMS).
- **Managed Service for Apache Flink** (`infra/flink/event_time_windows.py`, Flink 1.20):
  janelas por tempo do evento, watermark com atraso tolerado, eventos atrasados numa tabela
  separada e checkpoints a cada 60 s. A semântica é especificada e testada em
  `application.cloud.streaming.WindowState`, incluindo a restauração do estado.
  O CloudFormation cria a aplicação parada. Os comandos são:

```bash
scripts/aws/flink.sh start [--restore-latest]
scripts/aws/flink.sh status
scripts/aws/flink.sh stop
```

A campanha Zia de referência continua lendo o Parquet temporal preparado.

## Feature Store (`EnableFeatureStore=true`, online com `EnableOnlineStore=true`)

- Feature group offline `continual-bench-zia-social-edges` (Iceberg, KMS), com as features
  que o `prepare()` calcula apenas com datas anteriores.
- Glue ETL `continual-bench-materialize-features` materializa o snapshot de treino
  point-in-time como Parquet versionado e imutável em
  `features/materialized/<versão>/<job run>/`. O job falha se houver feature posterior ao
  instante da predição ou quebra de ordem.
- Glue Data Quality `continual-bench-prepared-social`: contagem, completude, domínio dos
  rótulos e faixas das features da tabela `prepared_social`.
- Smoke de escrita e leitura, com paridade e checagem point-in-time:

```bash
python scripts/aws/feature_store_smoke.py --group continual-bench-zia-social-edges --parquet <smoke.parquet> --online
aws glue start-job-run --job-name continual-bench-materialize-features
aws glue start-data-quality-ruleset-evaluation-run --data-source '{"GlueTable":{"DatabaseName":"continual_bench","TableName":"prepared_social"}}' \
  --role "arn:aws:iam::$ACCOUNT:role/continual-bench/continual-bench-glue" --ruleset-names continual-bench-prepared-social
```

Desative o online store ao fim do teste (`EnableOnlineStore=false`).

## ML avançado (`EnableMl=true`, endpoint com `EnableEndpoint=true`)

A imagem `ml` reaproveita as dependências da `runtime` e só muda o ponto de entrada
(`process`, `train`, `serve`). Perfis em `infra/sagemaker/profiles/`:

```bash
python scripts/aws/sagemaker_jobs.py processing --bucket $BUCKET --role $SM_ROLE --release latest
python scripts/aws/sagemaker_jobs.py training   --bucket $BUCKET --role $SM_ROLE --release latest
python scripts/aws/sagemaker_jobs.py tuning     --bucket $BUCKET --role $SM_ROLE --release latest
python scripts/aws/sagemaker_jobs.py inference  --bucket $BUCKET --role $SM_ROLE --release latest \
  --model-data s3://$BUCKET/campaigns/<id>/runs/<run>/attempts/<a>/model/model.tar.gz --delete-after
```

O tuning é uma execução separada. Ele minimiza `validation:log_loss` com no máximo 8 jobs
(2 em paralelo) num split de validação cronológico próprio, anterior ao holdout, que nunca é
lido. Cada run Zia registra no Model Registry (`continual-bench-zia-<dataset>-<política>`) um
candidato `PendingManualApproval`. O candidato referencia o snapshot, a run, o relatório, o
`model.tar.gz` no S3 e a imagem `ml` por digest. Antes de publicar, o worker confirma que o
`ZiaPredictor` reproduz as predições arquivadas do holdout. No build, a imagem `ml` passa pelo
contrato `/ping` e `/invocations`. Os checkpoints intermediários ficam em `models/` de cada
tentativa.

## BI (`EnableBi=true`, `QuickSightPrincipalArn=...`)

Pré-requisitos: assinatura Amazon Quick Sight Enterprise em us-east-1 e um usuário ou grupo
Quick Sight. A role de serviço do Quick Sight precisa de acesso ao Athena, ao bucket de dados
(incluindo `athena-results/`) e à chave KMS: Quick Sight → Manage → Security & permissions.
O módulo cria a fonte Athena, o dataset que junta métricas por política com estimativas de
custo por tentativa e a análise "policy comparison".
