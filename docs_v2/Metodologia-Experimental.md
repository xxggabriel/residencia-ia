# Metodologia Experimental

Esta página descreve como o `continual-bench` avalia modelos, políticas de atualização e detectores de drift: os protocolos de avaliação disponíveis, a disponibilidade de rótulos, as populações de avaliação (coortes), o limiar de decisão, as seeds e os limites de comparação entre protocolos e com artigos publicados. As definições das métricas estão em [Métricas](Metricas).

## Princípios comuns

Todos os protocolos seguem as mesmas regras de causalidade, impostas no código e não apenas por convenção:

- **Prever antes de aprender.** Cada exemplo (ou bloco de exemplos) é previsto pelo modelo vigente antes que seu rótulo seja usado para treino, detecção ou decisão de atualização. Esse esquema é chamado de avaliação *prequential* (de *predictive sequential*) ou *test-then-train*, descrito em *On evaluating stream learning algorithms* (2013); ver [Referências](Referencias).
- **Ordem da fonte preservada.** O motor por evento rejeita eventos fora de ordem, `timestamp` não estritamente crescente ou `available_at` decrescente (`src/continual_bench/core/engine.py`, método `observe_batch`). O avaliador de streams ordenados rejeita blocos com timestamps fora de ordem (`src/continual_bench/evaluation/stream_benchmark.py`).
- **Rótulo só depois de liberado.** Cada evento carrega `available_at`, o instante em que seu rótulo fica disponível. O motor mantém uma fila de predições pendentes e só libera o rótulo quando `available_at` não ultrapassa o instante corrente (`Engine.release`).
- **Indisponível não é zero.** Métricas sem pré-requisito (classe ausente, sem anotação de drift, sem baseline) ficam `null` com motivo registrado. Ver [Métricas](Metricas#disponibilidade-e-valores-ausentes).

## Protocolos de avaliação

O código possui vários avaliadores. Eles respondem a perguntas diferentes e seus números não são intercambiáveis.

| Protocolo | Identificador no relatório | Implementação | Como funciona | Estado |
|---|---|---|---|---|
| Prequential por evento (streaming) | `event_prequential` (`mode: streaming`) | `evaluation/prequential.py`, `core/engine.py` | Lotes de tamanho 1; prediz, libera rótulos vencidos, monitora e talvez atualiza. Kafka real só com `--services` | Implementado; Validado por testes (`tests/test_forgetting_retention_coverage.py` e outros) |
| Prequential por lote (incremental) | `batch_prequential` (`mode: incremental`) | mesmos módulos, `batch_size` padrão 32 | Todas as predições do lote ocorrem antes de qualquer aprendizado dele; os rótulos são liberados no fim do lote conforme `available_at` | Implementado; Validado por testes |
| Binário incremental com políticas em lote (CARA, UPF, Oracle) | `batch_prequential` com política em lote | `evaluation/binary_retraining.py`, `retraining/` | Matriz de desempenho com no máximo 24 blocos contíguos; fase offline de `max(2, min(blocos - 2, blocos // 3))` blocos; decisões online nos blocos seguintes | Implementado; Validado por testes (`tests/test_binary_retraining_*.py`) |
| Holdout temporal offline | `temporal_holdout` (`mode: offline`) | `evaluation/holdout.py` | Treina no prefixo (`offline_train_fraction`, padrão 0,6), descarta do treino rótulos com `available_at` posterior ao início do teste e avalia o modelo congelado no sufixo. Exige política `none` | Implementado; Validado por testes |
| Streams ordenados por bloco, retenção só de métricas | `ordered-stream-batch-prequential-metrics-v2` | `evaluation/stream_benchmark.py`, `scripts/run_dataset_suite.py` | Primeiro bloco como aquecimento; blocos seguintes previstos antes do treino; rótulos disponíveis no fim do bloco; holdout temporal final de 10% (`HOLDOUT_FRACTION = 0.1`); sem gravação de pesos (`artifact_mode=metrics_only`) | Implementado; Validado por testes e pela campanha registrada em `evidencias/aws-benchmarks/README.md` |
| Streams ordenados com artefatos completos | `batch_prequential_with_final_temporal_holdout` | mesmo módulo, `artifact_mode=full` | Mesmo protocolo, com modelos e checkpoints persistidos | Implementado |
| Zia (fraude, e-commerce, rede social) | `zia-batch-prequential-holdout-v1` (constante do perfil AWS) | `evaluation/zia*.py`, `application/cloud/contracts.py` | Prequential por bloco com holdout temporal final e limiar de decisão por custo | Implementado; ver [Datasets](Datasets) |
| Avaliação iterativa por atualização | texto `prequential windows with separate chronological validation for each fit` | `evaluation/iterative.py` | A fonte é cortada em atualizações (uma por tarefa se houver até 20 tarefas contíguas, senão 10 blocos). Cada atualização é pontuada *test-then-train* e depois treinada por até 100 épocas com *early stopping* (paciência padrão 10, validação nos 10% cronológicos finais do bloco, restauração dos melhores pesos) | Implementado |
| Treino por épocas com holdout | `official_holdout` ou `purged_temporal_holdout` | `evaluation/epochs.py` | Passagens só de treino; após cada época, uma cópia congelada é avaliada no holdout. Sem partição oficial, o holdout é o sufixo de 20% e o treino exclui rótulos que só chegam depois do corte. O relatório declara "no prequential claim" | Implementado |
| Avaliação tipada (multiclasse e regressão) | campo `protocol` do relatório tipado | `evaluation/typed.py` | Prequential limitado para alvos não binários, com métricas River por tarefa | Implementado |
| Replicação CARA/UPF por lotes | `UPF-train-reference-v2-window-*` ou `UPF-train-reference-v3-full-source-*`, com sufixo `same-batch` ou `next-batch` | `evaluation/retraining_replication.py`, `evaluation/cara_replication.py` | 7 lotes offline e 8 online; holdout interno por lote; o sinal CARA usa referência de treino e consultas sem rótulos de teste | Implementado; Experimental para fins de comparação com os artigos |

O motor por evento é o único que controla o ciclo amostra a amostra. Ele recusa políticas em lote (`requires_batches`), e a configuração exige `mode: incremental` para CARA, UPF e Oracle (`src/continual_bench/config/benchmark.py`, `validate_protocol`). A descrição das políticas e detectores fica em [Modelos e Políticas](Modelos-e-Politicas) e [Detectores de Drift](Detectores-de-Drift).

### Aquecimento e drenagem

No motor por evento ou lote, os primeiros `warmup` exemplos (padrão 100) treinam o modelo inicial e ficam fora do placar. Ao final do fluxo, os rótulos ainda pendentes são drenados e pontuados (`Engine.finish`). No avaliador de streams ordenados, o primeiro bloco é o aquecimento.

### Portão de promoção

Com `promotion` configurado, um retreinamento `reset_retrain` reserva `validation_samples` exemplos da janela de replay como validação e só promove o candidato se a melhora mínima (`min_improvement`) for atingida (`core/engine.py`, `_admitted_training`). A configuração rejeita o portão no modo offline e em políticas `none` ou `online`.

## Disponibilidade e atraso de rótulos

| Situação | Regra no código | Observação |
|---|---|---|
| Fontes sintéticas, OpenML, manifestos locais e streams reais no motor binário | `available_at = timestamp + label_delay` (`datasets/synthetic.py`, `datasets/openml.py`, `datasets/local_manifest.py`, `datasets/real_streams.py`) | `label_delay` vem de `DatasetConfig` (padrão 0, máximo 10.000). Os manifestos registram `label_delay_is_simulated: true`: o atraso é simulado, não medido |
| Relógio simulado | `timestamp = epoch + sequência` nas fontes com relógio simulado | Atrasos ordinais de replay não são tempo de calendário |
| Prequential por lote | rótulos liberados no fim do lote, até `available_at <= timestamp` do último evento aceito | Pode divergir do prequential por evento |
| Binário com políticas em lote | cada candidato é ajustado no corte do fim do bloco com rótulos cujo `available_at` não ultrapassa o corte; rótulos pendentes continuam elegíveis em cortes posteriores | Implementado em `evaluation/binary_retraining.py`; ver a seção de protocolos históricos e correções |
| Holdout offline | rótulos do prefixo com `available_at >= início do teste` são excluídos do treino (intervalo `purged_label_delay` no relatório) | A configuração rejeita atraso que não deixa rótulos de treino em fontes `river_abrupt` |
| Streams ordenados | todos os rótulos de um bloco ficam disponíveis no fim desse bloco (`datasets/pinned_streams.py`, campo `label_availability`) | Convenção do protocolo, não atraso de rotulagem observado nos dados |

## Coortes de avaliação

Uma coorte é a população de exemplos sobre a qual uma métrica é calculada. Coortes diferentes respondem a perguntas diferentes e não devem ser misturadas em um único ranking.

| Coorte | Definição | Onde |
|---|---|---|
| Prequential | Exemplos após o aquecimento, pontuados antes do aprendizado | `report.metrics`, escopo `prequential` |
| Janelas | Blocos de `metric_window` exemplos (padrão 100) ou blocos do stream | `report.windows` |
| Holdout temporal final | Sufixo nunca usado para treino, calibração de políticas ou decisão | escopo `holdout` |
| Coorte comum pós-calibração | Faixa de posições compartilhada por todas as políticas e seeds de um stream: começa após a maior fase de calibração offline e termina antes do holdout | `scripts/verify_dataset_suite.py` (`calibration_end`, `holdout_start`); escopo `common_postcalibration` em `evidencias/aws-benchmarks/metric-dictionary.csv` |
| Probes de retenção | Exemplos reservados por episódio (gerador com semente separada em fontes sintéticas, `episode_probes` da fonte, ou reserva dentro do fluxo após o aquecimento), fora do treino, do detector e do placar prequential | `core/continual.py`, `evaluation/tracking.py` |
| Janelas de futuro próximo | Snapshot do modelo congelado antes de cada bloco; `gap` exemplos ignorados e `horizon` exemplos avaliados | `core/evaluation_metrics.py` (`NearFutureEvaluation`) |
| Por classe | Métricas one-vs-rest por classe | escopos `prequential_class:<i>` e `holdout_class:<i>` |

Na campanha registrada em `evidencias/aws-benchmarks/README.md`, a coorte comum tem, por execução, 27.182 exemplos em Electricity, 323.630 em Airlines, 213.165 em INSECTS abrupt imbalanced e 271.225 em INSECTS incremental abrupt reoccurring imbalanced. O aquecimento, a calibração e o holdout não entram nesse denominador.

## Calibração e limiar de decisão

O termo "calibração" aparece no projeto com dois sentidos distintos:

1. **Calibração de probabilidades.** Não há método de recalibração implementado: nenhum ajuste de Platt, regressão isotônica ou `CalibratedClassifierCV` existe em `src/`. O projeto apenas **mede** a calibração da probabilidade positiva por ECE e MCE com dez bins iguais (`core/metrics.py`, `_calibration`; `evaluation/zia_metrics.py`). Ver [Métricas](Metricas).
2. **Calibração offline das políticas.** É a fase inicial de blocos em que CARA, UPF e Oracle constroem a matriz de desempenho e ajustam seus parâmetros (`retraining/rules.py`, campo `offline_calibration_available`). CARA sem calibração offline válida fica inativo.

O limiar de decisão não é otimizado em dados de validação. Ele é fixo ou derivado de custos declarados:

| Avaliador | Regra | Origem |
|---|---|---|
| Motor binário (`core/metrics.py`) | classe positiva `y=1` quando `p >= 0.5` | constante no código |
| Streams ordenados e Zia | binário: `p_1 >= c_FP / (c_FP + c_FN)` (limiar de Bayes para escores calibrados); multiclasse: argmax, empates favorecem o menor ID | `evaluation/zia_metrics.py` (`decision_threshold`), `models/classification_metrics.py` (`decide`) |
| Perfil `configs/base/tracking.yaml` | custos 1 (FP) e 39 (FN), limiar 1/40 = 0,025 | usado por padrão pela CLI de `stream_benchmark.py` |
| Perfil `configs/base/tracking-dataset-suite.yaml` | custos 1 e 1, limiar 0,5 | suíte de streams ordenados |

A regra de Bayes pressupõe probabilidades calibradas, e nenhum passo garante essa calibração. O ECE e o MCE publicados permitem verificar o quanto essa hipótese se sustenta em cada execução.

## Seeds e repetições

| Contexto | Papel da seed | Interpretação |
|---|---|---|
| Fontes sintéticas | define o gerador e, portanto, a amostra | repetições geram fluxos diferentes |
| Fontes reais (Electricity, Airlines, INSECTS) | a ordem publicada é fixa; a seed entra na identidade do caso, na configuração de treino (`IterativeTrainingConfig.seed`, seed do XGBoost) e no Monte Carlo do UPF | repetições computacionais sobre a mesma população, não três amostras independentes |
| Modelos River (logística, Hoeffding Tree, kNN) | construção determinística | não há aleatoriedade inventada entre repetições |
| Suíte de streams ordenados e campanha AWS | seeds 42, 43 e 44 | média e desvio padrão amostral (`ddof=1`) por dataset, política e métrica (`scripts/verify_dataset_suite.py`) |
| Replicação UPF | dez seeds no harness histórico | desvio padrão populacional (`ddof=0`, `evaluation/retraining_replication.py`) |

As dispersões são descritivas. Seeds sobre o mesmo fluxo fixo e janelas sobrepostas não são amostras independentes de generalização, e as barras dos gráficos não são intervalos de confiança.

## Testes temporais

| Verificação | Implementação | Estado |
|---|---|---|
| Ordem e disponibilidade no motor | `Engine.observe_batch` e `Engine.release` | Implementado; Validado por testes |
| Validador de partições temporais: features e rótulos de treino antes do corte, teste a partir dele, IDs disjuntos, ordem estrita | `core/evaluation_metrics.py`, `validate_temporal_holdout` | Implementado; Validado em `tests/test_evaluation_metrics.py`. Não é chamado pelos avaliadores de produção: serve como contrato testado |
| Rótulos futuros do holdout não alteram aprendizado nem decisões | `tests/test_stream_holdout_invariance.py` | Validado |
| Probes independentes do treino e indisponíveis sem fonte independente | `tests/test_forgetting_retention_coverage.py` | Validado |
| Reconstrução das métricas por janela e partição a partir das predições | `scripts/verify_dataset_suite.py` | Implementado |

Os 194 testes dos arquivos `test_evaluation_metrics`, `test_continual_metrics`, `test_forgetting_retention_coverage`, `test_binary_regime_quality`, `test_binary_retraining_availability`, `test_binary_retraining_batch_scores`, `test_stream_holdout_invariance`, `test_metric_publication`, `test_metric_tracking_ordering` e `test_tracking_metrics` passam na árvore atual com `uv run --frozen pytest`. Detalhes da suíte em [Testes e CI/CD](Testes-e-CI-CD).

## Protocolos históricos e correções

Os protocolos mudaram à medida que foram encontrados problemas de causalidade ou de comparabilidade. Resultados de protocolos anteriores continuam identificados pelo protocolo original e não são reinterpretados com as regras novas.

| Fase | Protocolo | Limitação identificada | Mudança |
|---|---|---|---|
| Campanha `nonlinear-v5` (encerrada) | 867 configurações (289 por modelo: logística, Hoeffding Tree, kNN), 5 datasets (SEA, Agrawal e STAGGER com seeds 42 a 46; Electricity e Airlines com um prefixo cada), 3.000 exemplos, aquecimento 100, atraso simulado 3, lote 32, replay 256, 8 tarefas paralelas. Políticas: congelada, online, periódica, ADWIN e DDM; tetos de treino pós-aquecimento 2.900 e 5.800 (online 2.900, congelado 0); offline com `offline_train_fraction` 0,6. Manifesto `configs/history/campaign-nonlinear-v5.json` | O teto de apresentações de treino não iguala CPU, memória nem custo entre modelos; fontes reais usam um único prefixo e relógio simulado | Mantida como registro com identidade própria; dois casos Airlines/árvore/DDM foram reexecutados do início após correção de leitura (`recovery_revision` no manifesto: o `NominalSplitterClassif.cond_proba` do River criava entradas vazias na predição; probabilidades e contagens aprendidas inalteradas, sem mudar dataset, modelo, política, detector, orçamento nem atraso). Evidência pelo histórico Git: `git show f835ee8:reports/nonlinear-v5.md` |
| Semana 10, CARA histórico | Random Forest, 100 lotes (25 offline), divisão 90/10 por lote | Pré-processamento ajustado sobre o fluxo inteiro (transdutivo) | Mantido com identificador próprio; nenhuma variante causal é apresentada como reprodução |
| Semana 10, UPF histórico | XGBoost, 7 lotes offline e 8 online, divisão aleatória 80/20 dentro do lote, janelas sorteadas que podem se sobrepor | CARA adaptado à rota UPF usava rótulos do teste corrente no sinal de obsolescência | Variante `UPF-train-reference-v3-full-source`: referência de treino e consultas sem rótulos de teste; cobertura do arquivo inteiro em 15 blocos contíguos |
| Binário incremental com políticas em lote | perdas retrospectivas sem corte por disponibilidade | Uso de rótulos ainda não disponíveis no corte | Corte causal por `available_at`, `MatrixView` filtrada por exemplo, replay limitado às últimas N linhas liberadas. Seleção causal e limite de replay mudaram juntos, então a diferença frente ao histórico não pode ser atribuída só ao atraso de rótulos |
| Suíte de streams ordenados | `ordered-stream-batch-prequential-metrics-v2` | Comparação entre políticas com faixas diferentes | Coorte comum pós-calibração por stream, holdout final separado, retenção só de métricas |

O identificador `ordered-stream-batch-prequential-metrics-v2` é o único dessa família presente no código atual (`application/cloud/contracts.py`, `STREAM_METRICS_PROTOCOL`).

## Comparabilidade

Regras para ler resultados de protocolos diferentes:

- Um mesmo nome de dataset, política ou métrica não torna dois resultados comparáveis. É preciso que coincidam seleção dos eventos, fase de avaliação, denominador, modelo e regra de decisão.
- Diferenças numéricas entre protocolos não demonstram superioridade de uma política. Os contrastes de efeito de política devem ser pareados dentro do mesmo stream e da mesma seed.
- Streams distintos não formam uma única população de teste. Comparações entre datasets são descritivas.
- O Oracle é uma referência retrospectiva para seu objetivo formal, com acesso à matriz futura de erros. Não é um teto universal de F1, AP ou AUROC.
- O holdout offline não é comparável ao prequential.

### Comparação com artigos publicados

Os arquivos em `evidencias/aws-benchmarks/papers/` preservam 110 valores transcritos dos artigos *Cost-aware retraining for machine learning* (2024) e *When to retrain a machine learning model* (2025) e 74 pares com médias históricas locais. O arquivo `protocol-comparability.csv` lista dez dimensões em que artigos, execuções históricas e a campanha atual (chamada F109 nesses arquivos) diferem:

| Dimensão | Diferença principal |
|---|---|
| Origem e população | Fontes, codificações e recortes diferentes |
| Modelo | Random Forest e XGBoost com hiperparâmetros e versões distintos |
| Partição | Lotes, divisões internas e janelas diferentes; o CARA tem divergência documental entre a seção 7.2 e a Tabela 1 |
| Pré-processamento | Rota histórica transdutiva |
| Métricas publicadas | Os artigos publicam SCPE, acurácia das consultas e AUC de custo; a campanha atual publica F1 macro, AP, AUROC e calibração |
| Dispersão | Cinco seeds sem desvios (CARA), dez seeds com `±` de tipo não declarado (UPF), três seeds com `ddof=1` (atual) |

Consequências registradas em `papers/limitations.json`: a comparação é descritiva, não há reprodução idêntica, e não se calculam diferenças ou rankings entre a campanha atual e os valores dos artigos. A AUC de custo do UPF não é AUROC. A formulação local relativa do CARA no UPF não é a baseline absoluta do artigo UPF (seção A.12).

## Evidências de execução

| Campanha | Evidência na árvore atual |
|---|---|
| Suíte de streams ordenados na AWS: 4 streams, 11 políticas, 3 seeds, 132 unidades, revisão `f1092ee0232a6d16504b3a2211a037c41a02068a` | `evidencias/aws-benchmarks/README.md`, `manifest.json`, `validation.json` |
| Campanha corretiva (60 runs INSECTS e 24 ensaios UPF) | Histórica. O dossiê foi removido da árvore e só é recuperável pelo histórico Git (`git show a3aa3469a:docs/evidencias/benchmark-corrigido/README.md`) |
| Semana 10 (CARA/UPF e INSECTS) | `evidencias/semana-10/` |
| `nonlinear-v5` | Manifesto em `configs/history/campaign-nonlinear-v5.json`; relatório pelo histórico Git (`git show f835ee8:reports/nonlinear-v5.md`) |

Como verificar e reproduzir essas evidências: [Reprodutibilidade e Artefatos](Reprodutibilidade-e-Artefatos) e [Tracking e Evidências](Tracking-e-Evidencias). Limites gerais em [Limitações e Trabalhos Pendentes](Limitacoes-e-Trabalhos-Pendentes).
