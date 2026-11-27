# Métricas

Esta página explica as métricas calculadas pelo `continual-bench`: como o catálogo está organizado, o que cada família mede, a unidade, a direção desejável, onde está o cálculo no código e as limitações de interpretação. O protocolo que produz cada população avaliada está em [Metodologia Experimental](Metodologia-Experimental).

## Organização do catálogo

O catálogo oficial é `configs/contracts/metric-definitions.json`, versão `residency-metrics-v1` (revisado em 2026-10-28). Ele é um inventário de definições e proveniência: a presença de uma métrica no catálogo não afirma que ela foi capturada ou publicada em uma execução específica.

| Bloco do JSON | Conteúdo |
|---|---|
| `definitions` | 198 entradas com família, fórmula, direção, unidade, escopos, pré-requisitos, motivos de indisponibilidade, fontes e limites |
| `families` | Nove famílias com escopos, pré-requisitos e limites comuns |
| `sources` | 34 fontes (código, notas de estudo, artigos e bibliotecas) |
| `availability_contract` | Regras de `null`, motivo e proibição de fabricar valores |
| `dynamic_definitions` | Séries por conceito: `retention_accuracy/concept_{concept}` e `retention_drop/concept_{concept}` |
| `benchmark_observations` | Extensão `benchmark-observations-v1`, publicada no namespace `observations_v1` do MLflow |

As 198 entradas incluem nomes canônicos e chaves nativas que são aliases uns dos outros (por exemplo, `scpe` e `cost_scpe`, `recovery_time` e `recovery_delay_mean_samples`). O mapa de aliases está em `src/continual_bench/evaluation/tracking.py`. A contagem, portanto, não representa 198 indicadores independentes.

Distribuição por família:

| Família (`family`) | Rótulo | Entradas |
|---|---|---:|
| `classification` | Qualidade preditiva binária | 32 |
| `continual` | Aprendizagem contínua por episódio | 8 |
| `drift` | Detecção e recuperação | 31 |
| `adaptation` | Benefício e custo da adaptação | 51 |
| `resources` | Recursos e eficiência | 44 |
| `regression` | Regressão e previsão de desempenho | 4 |
| `mlops` | Pipeline e overhead de plataforma | 8 |
| `cicd` | Qualidade e entrega de software | 10 |
| `business` | Indicadores econômicos de negócio | 10 |

Dois campos classificam cada entrada:

- `status`: `source_observed` (definição conferida no código indicado; 28 entradas, todas as chaves de `core/metrics.py`) ou `conditional_definition` (depende de dados e instrumentação adicionais; 170 entradas).
- `study_status`: `studied` (indicador explícito no material de estudo; 80), `operationalization` (fórmula escolhida para um conceito estudado; 29) ou `extension` (ampliação do benchmark; 89). Brier, log loss, AP, ECE/MCE, acurácia balanceada, MCC e kappa são extensões.

Versões e suplementos:

| Identificador | Papel | Código |
|---|---|---|
| `residency-metrics-v1` | Catálogo e perfil de tracking (`TrackingConfig.version`) | `config/benchmark.py`, `evaluation/tracking.py`, `infrastructure/metric_tracking.py` |
| `semana09-metrics-v1` | Relatório suplementar `task_evaluation-metrics.json` (matriz de tarefas e futuro próximo); não substitui as definições do catálogo | `evaluation/reporting.py`, `core/evaluation_metrics.py` |
| `benchmark-observations-v1` | Observações por run, ajuste, step e época | `evaluation/tracking_observations.py` |
| `system-metrics-v1` | Amostrador de CPU e memória do processo | `evaluation/system_metrics.py` |

Para a lista completa, consulte diretamente o JSON. Para a campanha AWS, `evidencias/aws-benchmarks/metric-dictionary.csv` traz 222 combinações de escopo e métrica com unidade, direção, definição e política de `null`.

## Implementações de cálculo

Há três acumuladores principais de qualidade. Eles diferem em detalhes que mudam os números:

| Acumulador | Usado por | Ranking (AUROC, AP) | Denominador vazio | Limiar |
|---|---|---|---|---|
| `Metrics` em `src/continual_bench/core/metrics.py` | motor binário por evento e lote, holdout offline, binário com políticas em lote | exato, com empates agrupados; armazena todos os escores (O(n)) | várias frações retornam 0 por compatibilidade; acurácia balanceada, AUROC, AP, MCC e kappa retornam `null` | 0,5 |
| `ClassificationMetrics` em `src/continual_bench/evaluation/zia_metrics.py` (com `models/classification_metrics.py`) | streams ordenados e Zia | histograma uniforme de 4.096 bins por classe, memória fixa e mesclável | `null` (função `ratio`) | por custo, ver [Metodologia Experimental](Metodologia-Experimental#calibração-e-limiar-de-decisão) |
| `metric_set` em `src/continual_bench/evaluation/scoring.py` | avaliação tipada e por épocas | não se aplica | conforme River | argmax |

O avaliador iterativo usa `evaluation/iterative_scores.py`, e as observações `observations_v1` usam `evaluation/tracking_observations.py` com convenções próprias de Macro-F1 (classe sem alvos nem predições recebe F1 zero).

## Qualidade preditiva

Classe positiva `y=1`. As fórmulas exatas estão no catálogo.

| Métrica | O que mede | Unidade | Direção | Limitações |
|---|---|---|---|---|
| `accuracy` | fração de acertos, `(TP+TN)/n` | fração | maximizar | sensível à prevalência; depende do limiar |
| `error_rate`, `errors` | erros de classificação | fração, exemplos | minimizar | idem |
| `precision`, `recall`, `specificity`, `negative_predictive_value` | qualidade por classe a partir da matriz de confusão | fração | maximizar | em `Metrics`, denominador vazio vira 0, o que não indica bom desempenho |
| `false_positive_rate`, `false_negative_rate` | taxas de erro por classe real | fração | minimizar | idem |
| `balanced_accuracy` | média de recall e especificidade (nos streams, média dos recalls das classes com suporte) | fração | maximizar | `null` sem as duas classes |
| `f1` | F1 da classe positiva, `2TP/(2TP+FP+FN)` | fração | maximizar | ignora verdadeiros negativos |
| `f1_macro` | média do F1 das duas classes | fração | maximizar | a convenção para classes ausentes difere entre acumuladores |
| `f1_weighted` | F1 por classe ponderado pelo suporte | fração | maximizar | dominado pela classe majoritária |
| `mcc` | coeficiente de correlação de Matthews | correlação em [-1, 1] | maximizar | `null` quando degenerado |
| `cohen_kappa` | concordância além do acaso, kappa de Cohen | coeficiente | maximizar | `null` quando degenerado |
| `roc_auc` | probabilidade de um positivo ter escore maior que um negativo, com meio crédito em empates | fração | maximizar | exige as duas classes; nos streams é aproximação por 4.096 bins |
| `average_precision` (`pr_auc` nos streams) | precisão média não interpolada por grupo de escores empatados | fração | maximizar | não é a área trapezoidal da curva precisão-recall; exige ao menos um positivo |
| `recall_at_fpr_0_01` | maior TPR observado com FPR até 0,01, sem interpolar | fração | maximizar | só em `observations_v1` e Zia |
| `positive_rate`, `predicted_positive_rate` | prevalência real e prevista | fração | descritiva | contexto para as demais |
| `n`, `tp`, `fp`, `tn`, `fn` | contagens | exemplos | descritiva | acompanham sempre os valores |
| `mean_window_accuracy` | média das acurácias das janelas | fração | maximizar | não substitui a métrica da coorte inteira |
| `fading_accuracy` | acurácia com decaimento exponencial, fator `fading_factor` (padrão 0,99) | fração | maximizar | `core/observability.py`; depende do fator |

### Probabilidades e calibração

| Métrica | O que mede | Unidade | Direção | Cálculo e limitações |
|---|---|---|---|---|
| `log_loss` | perda logarítmica média, em logaritmo natural | nats por exemplo | minimizar | probabilidade recortada em `[1e-15, 1-1e-15]` (`core/metrics.py`); nos streams, recorte inferior `1e-15` na probabilidade da classe observada |
| `brier` | erro quadrático médio da probabilidade positiva, `mean((y-p)^2)` | probabilidade ao quadrado | minimizar | nos streams binários equivale à metade da soma por classe; em multiclasse o acumulador usa a soma completa, e a exportação publica `brier_half_sum` para uma escala comum de 0 a 1 |
| `mse_probability` | igual ao Brier no caso binário | probabilidade ao quadrado | minimizar | não transforma a tarefa em regressão |
| `mae_probability` | erro absoluto médio da probabilidade | erro de probabilidade | minimizar | extensão |
| `ece` | erro de calibração esperado: média ponderada, em dez bins iguais, da diferença entre probabilidade positiva média e frequência positiva | diferença de probabilidade | minimizar | calibração da classe positiva, não da confiança da classe prevista; depende do binning e de `n` |
| `mce` | maior diferença entre bins não vazios | diferença de probabilidade | minimizar | instável com poucos exemplos por bin |

O projeto mede a calibração, mas não aplica recalibração (ver [Metodologia Experimental](Metodologia-Experimental#calibração-e-limiar-de-decisão)).

### Regressão

| Métrica | Unidade | Direção | Observação |
|---|---|---|---|
| `mae`, `rmse` (avaliação tipada, épocas, iterativo) | unidade do alvo | minimizar | River `MAE` e `RMSE` em `scoring.py`; no iterativo, soma dos quadrados dos resíduos sobre o total de pares |
| `r2` (iterativo) | adimensional | maximizar | `1 - SSE/SST` com soma centralizada de Welford; zero para alvo constante por convenção histórica |
| `mae`, `mse` canônicos do catálogo | unidade do alvo | minimizar | `null` na classificação binária, motivo `binary_probability_error_is_not_a_regression_task_metric` |

## Aprendizagem contínua: retenção e esquecimento

Seja `a[k,j]` a acurácia no probe do episódio `j` depois de concluído o episódio `k`. Há duas implementações com bases diferentes, mantidas com nomes distintos para não misturar definições.

| Métrica | Fórmula | Unidade | Direção | Código |
|---|---|---|---|---|
| `cl_average_accuracy` (AA) | média de `a[k,j]` para `j <= k` | fração | maximizar | `core/continual.py` |
| `cl_average_incremental_accuracy` (AIA) | média do histórico de AA | fração | maximizar | idem |
| `cl_forgetting` (FM) | média de `max(a[i,j], i < k) - a[k,j]`, usando todas as linhas anteriores, inclusive avaliações antes do aprendizado do episódio; sem truncar em zero | diferença de acurácia | minimizar | idem |
| `cl_backward_transfer` (BWT) | média de `a[k,j] - a[j,j]` para `j < k` | diferença de acurácia | maximizar | idem |
| `cl_intransigence` (IM) | `a_joint[k] - a[k,k]` com referência `joint_prefix_one_pass` | diferença de acurácia | minimizar | idem |
| `cl_forward_transfer` (FWT local) | média de `a[j,j] - a_task_only[j]`, após o aprendizado da tarefa | diferença de acurácia | maximizar | idem |
| `gem_backward_transfer` | média de `R[T,j] - R[j,j]` (equação 3 de GEM) | diferença de acurácia | maximizar | `core/evaluation_metrics.py` (`task_matrix_metrics`) |
| `gem_forward_transfer` | média de `R[j-1,j] - inicial[j]` (equação 4 de GEM); `null` sem baseline inicial | diferença de acurácia | maximizar | idem |
| `postlearning_forgetting` | média de `max(R[i,j], j <= i < T) - R[T,j]`, apenas linhas após o aprendizado | diferença de acurácia | minimizar | idem |
| `past_task_retention_accuracy` | média da acurácia final nas tarefas anteriores | fração | maximizar | idem |
| `retention_final_mean_accuracy`, `retention_final_mean_drop` | média da acurácia final e da queda desde o melhor probe, por conceito | fração, diferença | maximizar, minimizar | `evaluation/tracking.py` |
| `near_future_accuracy` | média das janelas completas de futuro próximo com modelo congelado | fração | maximizar | `core/evaluation_metrics.py` (`NearFutureEvaluation`) |

As fórmulas AA, AIA, FM, BWT, IM e FWT seguem *A Comprehensive Survey of Continual Learning: Theory, Method and Application* (2023); BWT e FWT de GEM seguem *Gradient Episodic Memory for Continual Learning* (2017); a acurácia de futuro próximo é uma adaptação operacional binária de *Rapid Adaptation in Online Continual Learning: Are We Evaluating It Right?* (2023), sem reproduzir seu benchmark. Ver [Referências](Referencias).

Limitações:

- As referências `task_only` e `joint_prefix_one_pass` usam o mesmo learner em uma passagem; não são oráculos convergidos.
- FM, BWT e FWT exigem mais de um episódio (`insufficient_tasks`). Sem probes independentes, os valores ficam `null` (`missing_probe`, `references_disabled`, `reference_budget_exceeded`).
- O placar prequential e a retenção em probes medem populações diferentes.
- A comparação de `near_future_accuracy` entre execuções exige `horizon`, `gap`, quantidade e offsets iguais. Padrão quando `tracking` é omitido: `horizon` 32, `gap` 1, quatro janelas (`DEFAULT_TRACKING` em `config/benchmark.py`).

## Detecção de drift e recuperação

| Métrica | O que mede | Unidade | Direção | Pré-requisito |
|---|---|---|---|---|
| `alarm_count` | alarmes emitidos | alarmes | descritiva | nenhum; zero alarmes não prova ausência de drift |
| `drift_tp`, `drift_fp`, `drift_fn` | associação entre alarmes e mudanças anotadas: primeiro alarme em `[mudança, min(mudança + tolerância, próxima mudança))` | mudanças, alarmes | maximizar TP, minimizar FP e FN | changepoints anotados; tolerância `detection_window` (padrão 500 exemplos) |
| `drift_precision`, `drift_recall`, `drift_f1` | qualidade da detecção | fração | maximizar | idem; acompanhamento incompleto é censurado |
| `drift_delay_mean_samples` | atraso médio entre mudança e primeiro alarme associado | exemplos | minimizar | idem |
| `drift_delay_release_mean` | atraso medido pela liberação do rótulo do alarme no relógio da fonte | unidades do relógio da fonte | minimizar | timestamps de replay são ordinais |
| `drift_false_alarm_rate` | alarmes na exposição estável divididos pelos exemplos fora de todas as janelas de detecção | alarmes por exemplo | minimizar | anotações |
| `drift_unmatched_alarms_per_1000_evaluated` (e o legado `drift_false_alarm_rate_per_1000`) | falsos positivos de associação por 1.000 exemplos avaliados | alarmes por 1.000 exemplos | minimizar | anotações |
| `drift_censored_changes`, `drift_complete_followup_changes`, `drift_observed_detections` | contagens de cobertura | mudanças | descritiva | anotações |
| `recovery_delay_mean_samples` | exemplos até a acurácia voltar ao alvo (acurácia pré-mudança menos `recovery_tolerance`, padrão 0,05) por `recovery_windows` janelas seguidas (padrão 2) | exemplos | minimizar | janela pré-mudança completa |
| `recovery_recovered_changes`, `recovery_censored_changes`, `recovery_unavailable_changes` | mudanças recuperadas, censuradas e sem alvo definido | mudanças | descritiva | idem |
| `samples_to_target_accuracy` | primeiro offset do fluxo em que a janela atinge `target_accuracy` (padrão 0,9) | offset da fonte | descritiva | não é iteração de treino |

Cálculo: `src/continual_bench/evaluation/tracking.py` (`detector_metrics`, `recovery_metrics`, `regime_metrics`). As métricas de detecção seguem a linha de *Learning with Drift Detection* (2004) e *Learning from Time-Changing Data with Adaptive Windowing* (2007) quanto a atraso e falsos alarmes; ver [Referências](Referencias) e [Detectores de Drift](Detectores-de-Drift).

Limitações:

- Electricity e Airlines não têm changepoints conhecidos; métricas dependentes de ground truth ficam `null` com `no_ground_truth_changepoints`.
- O avaliador de streams ordenados registra alarmes de ADWIN e DDM, mas não calcula precisão e recall de detecção contra as fronteiras publicadas de INSECTS, que ficam no manifesto da fonte.
- `drift_roc_auc` exige escores contínuos de drift e ensaios rotulados com e sem mudança; não é calculado e fica `null`.
- A probabilidade do classificador não substitui o escore do detector.

## Adaptação e custo

Estas métricas descrevem decisões de atualização e custos de cenário. A contabilidade de custos, preços e recursos está em [Custos e Recursos](Custos-e-Recursos).

| Métrica | O que mede | Unidade | Direção | Código |
|---|---|---|---|---|
| `update_count`, `retrain_count`, `updates_per_1000_evaluated` | atualizações implantadas após o modelo inicial e retreinamentos completos | atualizações | descritiva | `evaluation/tracking.py`, `evaluation/stream_benchmark.py` |
| `adaptation_realized_gain` | soma das diferenças de perda entre o modelo anterior congelado (shadow) e o atual nos mesmos eventos futuros | soma de perdas | maximizar | shadows coletados em `core/observability.py`, agregação em `evaluation/tracking.py`; exige `adaptation_shadows` |
| `adaptation_roi_mean`, `adaptation_roi_prediction_weighted`, `adaptation_roi_aggregate`, `adaptation_roi_global` | ganho dividido pelo custo adicional | redução de perda por unidade de custo | maximizar | `evaluation/tracking.py`; o global exige todas as atualizações precificadas e avaliadas |
| `cost_total` e componentes `cost_prediction_errors`, `cost_training_presentations`, `cost_update_events` | custo de cenário parametrizado | unidade de cenário | minimizar | `evaluation/costs.py` |
| `cost_scpe`, `retraining_mean_scpe` | excesso percentual de custo frente ao Oracle, `100*(custo - custo_oracle)/custo_oracle` | percentual | minimizar | `retraining/analysis.py` |
| `upf_total_cost` | `alpha * retreinamentos + soma das taxas de erro por bloco`; só para trajetórias `reset_retrain` | custo normalizado | minimizar | `evaluation/stream_benchmark.py`, `retraining/simulate.py` |
| `posthoc_update_event_cost` | idem, penalizando qualquer atualização | custo normalizado | minimizar | `evaluation/stream_benchmark.py` |
| `cost_sensitivity_auc` | integral trapezoidal desse custo sobre `upf_alphas` (padrão 0; 0,1; 1; 10) na mesma trajetória fixa | custo vezes alpha | descritiva | idem |
| `upf_paper_auc` | AUC do custo combinado com políticas executadas em vários alpha | custo vezes alpha | minimizar | `retraining/analysis.py`; não é AUROC |
| `retraining_*` | parâmetros e resultados da matriz CARA/UPF/Oracle (alpha, `c_hat`, retreinamentos, lotes offline e online, tempos) | vários | vários | `retraining/analysis.py` |

Limitações:

- Custo de cenário não é despesa real, lucro ou faturamento. Estimativas em USD e faturas pertencem a bases diferentes ([Custos e Recursos](Custos-e-Recursos)).
- `cost_sensitivity_auc` é sensibilidade pós-hoc de uma trajetória fixa; não reexecuta decisões e não equivale à AUC do artigo UPF.
- A média dos ROIs e a razão de somas de benefícios e custos são indicadores diferentes.
- SCPE e AUC de custo seguem *Cost-aware retraining for machine learning* (2024) e *When to retrain a machine learning model* (2025) apenas como definição; os protocolos locais diferem ([Metodologia Experimental](Metodologia-Experimental#comparação-com-artigos-publicados)).

## Recursos e esforço

| Métrica | O que mede | Unidade | Direção | Observação |
|---|---|---|---|---|
| `train_seconds`, `initial_train_seconds`, `update_train_seconds` | tempo de treino por escopo | segundos | minimizar | relógio monotônico |
| `prediction_seconds`, `prediction_latency_mean_seconds`, `prediction_latency_p50/p95/p99_seconds` | tempo de inferência no processo de avaliação | segundos por predição | minimizar | não é SLA de serviço |
| `prediction_throughput_per_second`, `postwarmup_train_samples_per_second` | vazão | predições ou apresentações por segundo | maximizar | idem |
| `training_presentations`, `training_unique_examples`, `training_repeated_presentations`, `training_used_post_warmup` | esforço de treino | apresentações, exemplos | descritiva | inclui ajuste inicial e replays |
| `process_peak_rss_bytes` | pico de memória residente do processo | bytes | minimizar | não é memória exclusiva do modelo |
| `sampled_rss_byte_seconds` | integral de amostras de RSS | byte vezes segundo | minimizar | `null` sem série de memória |
| `evaluation_compute_seconds`, `evaluation_attempt_wall_seconds`, `instrumentation_seconds`, `monitor_seconds` | tempos de avaliação, tentativa, instrumentação e monitoramento | segundos | minimizar | escopo do temporizador difere entre protocolos |
| `cl_probe_seconds`, `cl_overhead_seconds`, `cl_reference_train_seconds` | custo das referências e probes de aprendizagem contínua | segundos | minimizar | separados do custo da política |

Tempos medidos com tarefas concorrentes, hosts compartilhados ou CPUs diferentes não devem ser atribuídos apenas à política. Detalhes em [Custos e Recursos](Custos-e-Recursos).

## Famílias sem medição nos avaliadores

As famílias `mlops`, `cicd` e `business`, além de `energy_kwh`, `carbon_emissions`, `performance_forecast_mae`, `performance_forecast_bias`, `iterations_to_target`, `learning_speed_l99` e `learning_speed_l100` sem curva de referência, são definições condicionais. O tracking as publica como `null` com motivos como `no_energy_meter`, `requires_real_revenue_and_operating_costs`, `not_a_CI_build_execution` e `requires_compatible_baseline_curve_and_target` (`evaluation/tracking.py`). Estado: **Planejado** como medição; a definição existe, a captura não.

## Disponibilidade e valores ausentes

- Cada métrica escalar nova expõe valor ou `null`, mais motivo. O tracking nativo usa `status=available/unavailable`; a matriz de aprendizagem contínua usa `available=true/false`.
- NaN e infinito não são publicados. O MLflow recebe apenas valores finitos; os `null` e seus motivos ficam nos artefatos JSON.
- Zero válido e indisponibilidade são estados diferentes. Em `core/metrics.py`, algumas frações vazias retornam 0 por compatibilidade; as contagens devem acompanhar o valor.
- Séries podem existir sem resumo escalar (`series`, `availability`, `series_metadata`).

## Testes

| Arquivo | O que verifica |
|---|---|
| `tests/test_tracking_metrics.py` | matriz de confusão conhecida, calibração, ranking com empates, AP diferente de AUROC, valores vazios, MCC e kappa negativos, retomada por pickle |
| `tests/test_evaluation_metrics.py` | matriz de tarefas (BWT, FWT, forgetting), futuro próximo, validador temporal, detecção validada, versão `semana09-metrics-v1` |
| `tests/test_continual_metrics.py` | equações de forgetting, BWT e FWT, máximo do FM incluindo avaliações pré-aprendizado, orçamento de referências, indisponibilidade com motivo |
| `tests/test_forgetting_retention_coverage.py` | retenção e forgetting nos avaliadores binário e tipado, probes independentes |
| `tests/test_binary_regime_quality.py` | qualidade por regime, comparada com scikit-learn |
| `tests/test_binary_retraining_*.py` | disponibilidade e escores do binário com políticas em lote |
| `tests/test_stream_holdout_invariance.py` | invariância do aprendizado e das decisões aos rótulos do holdout |
| `tests/test_metric_publication.py`, `tests/test_metric_tracking_ordering.py` | linhagem, `null` e idempotência na publicação; ordem de conclusão das tags no MLflow |

Estado das métricas de `core/metrics.py`, matriz de tarefas, detecção e retenção: **Implementado** e **Validado** por esses testes. A disponibilidade em uma execução concreta deve ser conferida no relatório e no run correspondentes ([Tracking e Evidências](Tracking-e-Evidencias)).
