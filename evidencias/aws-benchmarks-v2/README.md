# Evidências Complementares da AWS (Athena e S3)

Este diretório reúne os dados extraídos diretamente da conta AWS (`992382509130`, região `us-east-1`) para a campanha principal na AWS, complementando `evidencias/aws-benchmarks/`.

## Estrutura

- `athena/f109_model_versions.csv`: `885` versões de modelo (`initial`, `reset_retrain`, `continue`) com `start_seq`, `end_seq`, `train_samples`, `train_seconds`, `iterations` e `bytes` (`continual_bench.model_versions`).
- `athena/f109_cost_estimates.csv`: `133` tentativas de jobs Batch/Fargate com tempo de execução, fila e estimativa em USD (`continual_bench.cost_estimates`).
- `athena/f109_runs_uris.csv`: mapeamento dos `132` `report_uri` no S3 (`continual_bench.runs`).
- `s3-reports/alarms_by_run.csv`: todos os alarmes de drift (`seq`, `block_index`, `block_end_seq`, `detector`) extraídos dos `132` `report.json` no S3.
- `s3-reports/windows_full_28blocks.csv`: todos os `28` blocos do stream (`seed 42`) com métricas de qualidade, calibração e contagens de confusão.
- `s3-reports/retention_summary.csv`: `absolute_retention` e disponibilidade de `gem_bwt` / `post_learning_fm` nas `132` execuções.
- `s3-reports/confusion_matrices.csv`: matrizes de confusão no fluxo completo e no holdout nas `132` execuções.
- `s3-reports/class_metrics.csv`: precisão, revocação, F1, `pr_auc`, `roc_auc`, `ece` e suporte por classe.
- `s3-reports/retraining_matrices/*__accuracy_matrix.json`: matrizes de acurácia candidato x lote extraídas dos artefatos no S3 para `CARA` e `UPF`.
