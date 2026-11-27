# Logs dos workers no homelab

Os workers emitem eventos operacionais em JSON por linha. A fila SQLite no volume `event-spool` usa WAL. O serviço `log-forwarder` converte os eventos para GELF HTTP e os entrega ao Graylog. O Airflow 3.3.2 conserva o log nativo e acrescenta eventos do worker à aba **Logs**.

A integração observa as instâncias mapeadas de `benchmark_campaign.run_seed` quando a DAG de campanhas distribuídas está disponível e envia o campo `airflow_task` na submissão a `/campaign-seeds`. Esse contexto contém `dag_id`, `dag_run_id`, `task_id`, `map_index` e `try_number` e é retirado antes de calcular a identidade do job. O worker usa `job_attempt_id` para identificar a sua execução. Uma retentativa do Airflow que reaproveita um job recebe a associação de observadora. O `run_id` nos eventos do worker identifica o benchmark, e o `dag_run_id` identifica a execução do Airflow.

O contrato dos eventos inclui `schema_version`, `event_id`, `timestamp` UTC, `level`, `event_name`, `message`, `service` e `environment`. A busca usa o stream autorizado e a identidade da task. O leitor deduplica por ID, tolera indexação fora de ordem e aguarda os eventos finais por até cinco minutos. Se a leitura remota falhar, a aba exibe um aviso e continua com o log local. A fila retém até 20 mil eventos pendentes e stdout continua disponível como registro local.

## Configuração

O perfil `observability` do `compose.yaml` inclui Graylog, MongoDB, OpenSearch e o encaminhador. A interface do Graylog fica em loopback. O GELF e o armazenamento usam redes Docker. Use o mesmo nome de projeto ao atualizar um ambiente existente para reutilizar os volumes.

No `.env` privado do deploy, configure `COMPOSE_PROFILES=observability`, `BENCH_LOG_ENVIRONMENT=homelab`, `GRAYLOG_UI_PORT`, `GRAYLOG_PASSWORD_SECRET` e `GRAYLOG_ROOT_PASSWORD_SHA2`. O primeiro segredo é aleatório e o segundo é o SHA-256 da senha administrativa. Guarde a senha em `.secrets/graylog-admin` com permissões restritas e configure `BENCH_GRAYLOG_TOKEN_PATH=./.secrets/graylog-reader`. No projeto principal do homelab, a porta do Graylog é 19002.

```bash
docker compose --profile observability config --quiet
docker compose --profile observability up -d graylog
curl -fsS http://127.0.0.1:19002/api/system/lbstatus
python3 scripts/graylog/bootstrap.py --url http://127.0.0.1:19002 \
  --admin-password-file .secrets/graylog-admin \
  --token-file .secrets/graylog-reader \
  --environment homelab
```

O bootstrap configura o input GELF HTTP e cria uma conta de leitura restrita ao stream do projeto. Grave o `BENCH_GRAYLOG_STREAM` retornado no `.env` e selecione `BENCH_TASK_LOG_BACKEND=graylog`. Com o ambiente ocioso, reconstrua as imagens e atualize os serviços de aplicação conforme o procedimento de deploy do projeto. Para voltar à leitura local, use `BENCH_TASK_LOG_BACKEND=local` e recrie `airflow-api`. A coleta permanece disponível para reativar a leitura remota.

A [validação no projeto principal](observabilidade/continual-bench-2026-11-26/README.md) registra a implantação e a conferência após remover o piloto.
