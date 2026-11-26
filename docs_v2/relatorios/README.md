# Relatórios da Campanha Experimental na AWS

Este diretório reúne os relatórios técnicos da campanha experimental de avaliação de políticas de retreinamento em fluxos de dados (*data streams*), conduzida exclusivamente com base nas evidências auditadas da campanha na AWS. A análise combina os agregados verificados de [evidencias/aws-benchmarks/](../../evidencias/aws-benchmarks/README.md) com a fonte complementar organizada em [evidencias/aws-benchmarks-v2/](../../evidencias/aws-benchmarks-v2/README.md) (extrações diretas do Amazon Athena e extrações estruturadas dos 132 arquivos `report.json` do Amazon S3).

## Síntese Executiva da Campanha

A campanha na AWS avaliou políticas de adaptação e retreinamento em fluxo contínuo com o modelo `xgboost`, na revisão de código `f1092ee0232a6d16504b3a2211a037c41a02068a` (SHA-256 do código científico: `23997ee3d2f3048c2e726bb9bae97faaf034eab84638d5b643c4de614ecebe4c`, imagem de execução `sha256:4d9ed4a81942d8c2947683d1e8816bc64c623f458404d7580da82d71be362c93`).

| Dimensão da campanha | Valor verificado | Observação |
|---|---:|---|
| Streams integrais | 4 | `electricity`, `airlines`, `insects_abrupt_imbalanced`, `insects_incremental_abrupt_reoccurring_imbalanced` |
| Políticas avaliadas | 11 | `frozen`, `drift`, `drift_adwin`, `drift_ddm`, `incremental`, `periodic`, `cara`, `cara_cumulative`, `cara_periodic`, `upf`, `oracle` |
| Seeds computacionais | 3 | `42`, `43`, `44` |
| Unidades científicas | 132 | Matriz completa 4 x 11 x 3 (`succeeded_jobs = 132`, 132 arquivos `report.json` no S3) |
| Tentativas operacionais | 133 | Inclui 1 retomada de publicação de métricas (`historical_failed_publication_attempts_recovered = 1`), sem novo treinamento |
| Observações consumidas nas fontes | 1.392.014 | Soma de `source_rows` dos 4 streams |
| Registros de predição comparados | 27.561.666 | Total consolidado na comparação |
| Fatos métricos analíticos | 23.121 | Distribuídos em 188 combinações de partição e métrica |
| Registros de ajuste no inventário | 2.265 | 1.161 ajustes físicos (116.100 rodadas de boosting) e 1.104 reutilizações de evidência |
| Grade de recursos | 6.336 células | 6.168 fatos observados e 168 ausências explícitas preservadas como `NULL` |

## Fontes de Dados e Regras Metodológicas

As análises desta série integram três camadas complementares da mesma campanha na AWS:

- **Agregados auditados em disco (`evidencias/aws-benchmarks/`)**: tabelas consolidadas derivadas de consultas Athena validadas (`policy-summary.csv`, `quality-summary.csv`, `windows-summary.csv`, `holdout-summary.csv`, `resources-summary.csv`, `training-inventory.csv` e `papers/`).
- **Fonte complementar organizada (`evidencias/aws-benchmarks-v2/`)**: extrações diretas e versionadas com checksum (`SHA256SUMS`) do Amazon Athena (`athena/f109_model_versions.csv`, `athena/f109_cost_estimates.csv`, `athena/f109_runs_uris.csv`) e séries extraídas dos 132 arquivos `report.json` e artefatos no Amazon S3 (`s3-reports/alarms_by_run.csv`, `s3-reports/windows_full_28blocks.csv`, `s3-reports/retention_summary.csv`, `s3-reports/confusion_matrices.csv`, `s3-reports/class_metrics.csv`, `s3-reports/retraining_decisions.json` e `s3-reports/retraining_matrices/`).

A leitura e a interpretação dos resultados obedecem às seguintes regras de integridade científica:

1. **Separação estrita de denominadores e escopos (`metric_scope`)**:
   - A comparação principal ocorre na coorte comum pós-calibração (`common_postcalibration`), compartilhada por todas as políticas e seeds de cada stream, excluindo o prefixo de treino inicial, a calibração e o holdout.
   - Os diagnósticos de `holdout` (sufixo final `source_rows - holdout_start`), as métricas de avaliação contínua, as séries de 16 janelas alinhadas (`windows-summary.csv`) e os 28 blocos completos dos `report.json` possuem denominadores e escopos próprios. Nunca devem ser misturados com `common_postcalibration`, e a média aritmética de janelas ou blocos não substitui a métrica calculada sobre toda a coorte comum.
2. **Não agregação entre streams distintos**:
   - Os quatro streams não formam uma única população de teste; portanto, não se produz ranking único agregado entre datasets.
3. **Interpretação de variância entre seeds (`sd_ddof = 1`)**:
   - Os desvios-padrão reportados (`sample_sd`) são amostrais (`ddof=1`) entre três repetições computacionais (seeds `42`, `43`, `44`).
   - Barras de erro e valores de `sample_sd = 0` indicam estabilidade determinística da execução sob a mesma ordem de stream, e não constituem intervalos de confiança populacionais nem prova de generalização estatística.
4. **Preservação de ausências (`NULL`) e métricas específicas**:
   - Células ausentes permanecem `NULL` (nunca imputadas como zero). Em recursos, os 168 casos com `source_fact_present=False` permanecem explícitos.
   - `pr_auc` corresponde à precisão média (*average precision*) não interpolada em histogramas uniformes de 4.096 bins; `brier` e `brier_half_sum` equivalem no caso binário a `mean((y - p_1)^2)`.
   - `train_seconds` acompanha ajustes lógicos promovidos, enquanto `actual_total_fit_seconds` acompanha o trabalho físico da unidade (com `materialization_train_seconds` contido no total físico).
5. **Separação entre a campanha na AWS e os artigos de referência**:
   - Os valores publicados nos artigos base (`evidencias/aws-benchmarks/papers/`) possuem protocolos distintos (110 fatos de referência e 74 pares históricos) e não são agrupados numericamente com a campanha na AWS (`numerically_pooled_with_papers = false`), impedindo alegações de reprodução equivalente ou superioridade causal direta.

## Índice de Relatórios

1. [01-Desempenho-Preditivo-e-Holdout.md](01-Desempenho-Preditivo-e-Holdout.md): Análise de qualidade preditiva na coorte comum pós-calibração (`macro_f1`, `pr_auc`, `balanced_accuracy`, `log_loss`, `brier`), calibração probabilística, dinâmica temporal nas 16 janelas alinhadas, evolução ao longo dos 28 blocos completos dos `report.json` frente ao modelo `frozen` e contraste com o diagnóstico de `holdout`.
2. [02-Deteccao-de-Drift-e-Retreinamentos.md](02-Deteccao-de-Drift-e-Retreinamentos.md): Análise de alarmes de drift, contagem de retreinamentos, promoção e materialização de versões na campanha na AWS.
3. [03-Esquecimento-e-Retencao.md](03-Esquecimento-e-Retencao.md): Análise de retenção de desempenho, estabilidade entre transições de conceito e compromisso plasticidade-estabilidade sob as políticas avaliadas na campanha na AWS.
4. [04-Recursos-Retreinamentos-e-Custos.md](04-Recursos-Retreinamentos-e-Custos.md): Análise de tempo de treinamento lógico e físico, uso de memória, inventário de ajustes e materializações, reutilização na família CARA e custos operacionais na AWS (`continual_bench.cost_estimates`).
5. [05-Comparacao-com-Papers-Base.md](05-Comparacao-com-Papers-Base.md): Contextualização dos resultados com os artigos base (CARA e UPF) a partir de `evidencias/aws-benchmarks/papers/` e `evidencias/aws-benchmarks/metrics-by-seed.csv`, documentando diferenças metodológicas, limites de comparabilidade e o cotejo descritivo entre os valores publicados e a campanha na AWS.
6. [06-Matrizes-de-Confusao-e-Analise-por-Classe.md](06-Matrizes-de-Confusao-e-Analise-por-Classe.md): Análise de matrizes de confusão e métricas por classe (precisão, revocação, F1 e suporte) nos escopos de fluxo contínuo e holdout a partir de `evidencias/aws-benchmarks-v2/s3-reports/`.
7. [07-Analise-por-Regimes-e-Matrizes-de-Decisao.md](07-Analise-por-Regimes-e-Matrizes-de-Decisao.md): Análise por regimes operacionais com mapas de calor e matrizes de decisão (`M6..M14`) para seleção de políticas de retreinamento na campanha na AWS, baseada exclusivamente em `evidencias/aws-benchmarks-v2/`.
8. [08-Evolucao-por-Lote-e-Posicao-no-Fluxo.md](08-Evolucao-por-Lote-e-Posicao-no-Fluxo.md): Evolução por lote com intervalos de confiança de Wilson de 95% para os modelos candidatos (`M7..M23`) frente à trajetória do modelo em uso (`Em uso`) e mapas espaciais de acurácia, alarmes e atualizações por posição no fluxo (`INSECTS`), baseados exclusivamente em `evidencias/aws-benchmarks-v2/`.
