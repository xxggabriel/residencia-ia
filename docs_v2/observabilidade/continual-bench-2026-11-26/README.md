# Graylog no continual-bench

Em 26/11/2026, o Compose consolidado foi aplicado no homelab em `/home/gabriel/residencia-ia-benchmark`, com o perfil `observability`. O Airflow 3.3.2 manteve o LocalExecutor e a porta 18081. O Graylog 7.1.0 ficou em `127.0.0.1:19002`, acessível por túnel SSH.

A atualização preservou os oito volumes existentes e os registros de 23 execuções e 639 instâncias de task do Airflow. Postgres, Kafka, MinIO e MLflow mantiveram os mesmos contêineres. O Compose criou quatro volumes para a observabilidade. A API continuou retornando as linhas nativas de uma execução anterior.

## Validação da implantação

A execução `manual__graylog-continual-bench-20261126` terminou em sucesso. Cada mapa exibiu os eventos do próprio job durante a execução na aba Logs. Ao final, o Graylog retornou cinco eventos únicos por job e a fila de encaminhamento estava vazia. O atraso observado entre a emissão e a primeira resposta incremental da API do Airflow foi de 2,45 a 3,40 s. [API do Airflow](airflow.jsonl) · [Busca no Graylog](graylog.jsonl) · [Estados finais das tasks](validation-tasks.csv)

| Seed | Mapa | Instância da task | Job | Aba Logs |
|---|---:|---|---|---|
| 71 | 0 | `01a218e4-d097-76a8-9142-d42206da65e5` | `2674d30c1e3dd9dd876cea15` | [Mapa 0](airflow-map-0.png) |
| 72 | 1 | `01a218e5-00a0-7f8c-a24d-697481e0ccef` | `ae3ba833bca8461804cfdd92` | [Mapa 1](airflow-map-1.png) |

## Validação após retirar o piloto

A retirada do `residencia-graylog` removeu os 11 volumes e o histórico operacional do piloto. No `continual-bench`, os 20 contêineres ativos, os 12 volumes e o histórico das 24 execuções que antecederam o novo teste foram preservados.

A execução `manual__graylog-pos-piloto-20261126` terminou em sucesso. O Graylog principal recebeu cinco eventos novos por job. As duas abas Logs exibiram os respectivos eventos durante a execução. A fila voltou a zero. O atraso observado até a primeira resposta incremental da API foi de 1,22 a 4,58 s. [API do Airflow](pos-piloto-airflow.jsonl) · [Busca no Graylog](pos-piloto-graylog.jsonl)

| Seed | Mapa | Instância da task | Job | Aba Logs |
|---|---:|---|---|---|
| 71 | 0 | `01a218eb-0bd1-78cb-99a6-596dca8800c9` | `835adfa8556090e669763aed` | [Mapa 0](pos-piloto-map-0.png) |
| 72 | 1 | `01a218eb-1d43-7208-a27e-48a11d2491c1` | `5338b68e498ba70d974eac1c` | [Mapa 1](pos-piloto-map-1.png) |

O homelab foi validado com a DAG e as rotas de campanhas distribuídas disponíveis no checkout implantado. As evidências de execução são daquele ambiente.
