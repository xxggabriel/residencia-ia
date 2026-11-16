# Registros e visualizações ZenML

## Configuração atual do homelab

O Docker Compose configura o projeto `default` e a stack `benchmark-minio-v2`.
A stack usa um orquestrador local e guarda os artefatos no bucket privado
`s3://zenml-artifacts`, no MinIO. O servidor lê por esse armazenamento os logs
das etapas e as visualizações de novos artefatos.

[scripts/bootstrap_zenml.py](../scripts/bootstrap_zenml.py) registra o armazenamento
S3 com o endpoint `S3_ENDPOINT_URL` e as referências `key` e `secret` ao segredo
ZenML `benchmark-minio-artifacts`. Antes do bootstrap, provisionar o bucket,
a identidade MinIO restrita e o segredo, conforme o
[guia operacional](benchmark-operations.md#bootstrap-e-recuperação) e o
[exemplo de política do bucket](zenml-minio-policy.example.json).
O bootstrap registra o armazenamento e a stack no projeto indicado.

Em **Continual Benchmark**, abrir a execução da pipeline `finite_benchmark`
para inspecionar o grafo e os logs por etapa. Para visualizar o conteúdo, selecionar
um artefato de uma execução gravada em `benchmark-minio-v2` e abrir **Visualization**.
O acesso a arquivos locais pelo servidor permanece desabilitado.

## Diagnóstico histórico de outubro de 2026

Em 26/10/2026, a execução `ba0deb5d-be2b-42cc-a57a-2703c7bb8b7f` usou o
armazenamento local `benchmark-local-v1`. O grafo aparecia no dashboard, e
as abas de logs e de visualização apresentavam o erro
`Files in a local artifact store cannot be accessed from the server.`
A restrição se aplicava à leitura pelo servidor mesmo com o volume montado.
Esses URIs históricos ainda apontam para o armazenamento local.

A execução de diagnóstico `d7b5d8fc-841b-41ec-98e2-3d53741bdfed` confirmou
a captura de logs por etapa e a existência dos JSONs de visualização. A validação
de 27/10/2026 registrou a operação de ponta a ponta com os artefatos das novas
execuções no MinIO e a publicação de experimentos no MLflow.

Os relatórios históricos podem ser consultados com
`git show f835ee8:reports/validation-20261027.md` e
`git show f835ee8:reports/zenml-observability-before.json`.
