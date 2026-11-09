# Plano da plataforma AWS gerenciada (registrado antes da implementação)

Data: 09/11/2026. Região: us-east-1. Branch de entrega: `main` (desenvolvimento em `Semana-09`).

## Estado inspecionado

- Não há `AGENTS.md` no repositório nem acima dele; a árvore Git estava limpa.
- Baseline: `pytest -q` com 585 testes aprovados.
- O protocolo Zia vive em `evaluation/zia.py` (`run`), com preparação em `datasets/zia.py`
  (`prepare`, `load_manifest`), predições em `evaluation/zia_predictions.py`, artefatos em
  `evaluation/zia_artifacts.py` e publicação MLflow em `evaluation/zia_publication.py`.
- A "matriz" Zia é dataset × `POLICIES` (DAG `benchmark_zia`); a matriz genérica é
  `application.matrix.ExperimentMatrix.plan()`.
- O avaliador Zia e o planejador não importam MLflow, ZenML, Kafka nem boto3.
- Não há AWS CLI nem credenciais nesta máquina: implantação real fica fora desta entrega.

## Fronteiras dos módulos

| Camada | Módulo novo | Responsabilidade |
|---|---|---|
| `infrastructure` | `object_store.py` | Armazenamento local com escrita condicional (fixtures e smoke local). |
| `infrastructure.aws` | `session`, `s3`, `batch`, `model_registry`, `kinesis`, `feature_store`, `pricing`, `athena`, `quotas`, `emf` | Adaptadores boto3. Nenhum SDK AWS em `core`, `models` ou `detectors`. |
| `infrastructure` | `stream_order.py` | Validação pura de ordem, duplicatas e lacunas para transportes. |
| `application.cloud` | `layout`, `contracts`, `sources`, `snapshots`, `planning`, `worker`, `records`, `lifecycle`, `costs`, `streaming`, `features` | Casos de uso: aquisição, preparação, plano, execução de uma run, publicação, reconciliação e custos. Recebem as portas por injeção. |
| `orchestration` | `aws_control.py`, `sagemaker.py` | Lambda de controle e ponto de entrada SageMaker (train/serve). |
| `cli` | subcomando `aws` | Entradas dos jobs Batch, CodeBuild e operação. |
| `infra/` | CloudFormation, ASL, buildspecs, Athena, Glue, Flink | Infraestrutura declarativa. |

## Ordem das entregas

1. Contratos, layout S3, armazenamento local e S3 com escrita condicional.
2. Aquisição por catálogo, preparação e manifesto de snapshot com marcador.
3. Plano de campanha (Zia e `ExperimentMatrix`), run_id determinístico, worker, publicação e
   registro de modelos.
4. Reconciliação, consolidação, Lambda de controle, eventos e alertas.
5. CloudFormation: persistente, CI/CD e raiz com módulos; ASL das duas state machines.
6. Módulos ativáveis: streaming, Feature Store, ML e BI. Custos: CUR 2.0, Budgets, anomalias,
   Price List e conciliação.
7. Testes, documentação operacional, validações e build das imagens.

## Verificações de aceite locais

- Suíte existente sem regressão e ruff limpo.
- Testes novos para contratos, layout, idempotência (escrita condicional, retry, reaproveitamento),
  reconciliação, permissões IAM, ASL, templates e protocolo (digests idênticos entre o caminho
  local e o caminho AWS com armazenamento local).
- Smoke local das quatro políticas em paralelo com fixtures.
- `cfn-lint` nos templates, validador ASL próprio e build `linux/amd64` das imagens.
- Implantação, aquisição real e campanha full somente com autorização e credenciais.
