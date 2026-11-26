# Modelos e Políticas

Esta página descreve os modelos de aprendizado disponíveis e as políticas que decidem quando atualizar um modelo: sem atualização, periódica, por alarme de drift, por desempenho, incremental, online, CARA, UPF e Oracle. Para cada política, explica a lógica de decisão como está no código, as entradas de custo e as diferenças em relação aos artigos de origem.

## Onde cada peça roda

O pacote tem três caminhos de avaliação que usam modelos e políticas de formas diferentes:

| Caminho | Código | Granularidade | Modelos | Políticas |
| --- | --- | --- | --- | --- |
| Motor binário por evento (`continual-bench run`, `matrix`) | `src/continual_bench/core/engine.py`, `evaluation/binary.py`, `evaluation/binary_retraining.py` | Uma predição por evento, rótulo liberado com atraso | Catálogo River (`models.MODELS`) | `none`, `online`, `periodic`, `drift`, `performance`; no modo `incremental`, também `cara`, `cara_cumulative`, `cara_periodic`, `upf`, `oracle` |
| Avaliador por blocos (streams ordenados e Zia) | `src/continual_bench/evaluation/stream_benchmark.py` | Blocos inteiros de linhas | XGBoost e regressão logística multinomial (`models/zia.py`) | `frozen`, `online`, `drift`, `drift_adwin`, `drift_ddm`, `incremental`, `periodic`, `cara`, `cara_cumulative`, `cara_periodic`, `upf`, `oracle` |
| Avaliadores tipado, por épocas e iterativo | `evaluation/typed.py`, `evaluation/epochs.py`, `evaluation/iterative.py` | Amostras de regressão e multiclasse (`typed.py`), além de tarefas binárias em `epochs.py` e `iterative.py` | Catálogo River | Tipado: `none`, `online`, `periodic` |

A arquitetura geral está em [Arquitetura](Arquitetura) e os protocolos de avaliação em [Metodologia Experimental](Metodologia-Experimental).

## Modelos

### Catálogo River

`MODELS` em `src/continual_bench/models/__init__.py` registra quatro famílias. Cada família tem um construtor por tipo de alvo; combinações sem construtor são rejeitadas na validação da configuração. Todos os modelos vêm do River 0.26.1.

| Família | Binária | Multiclasse | Regressão | Parâmetros padrão |
| --- | --- | --- | --- | --- |
| `logistic` | `LogisticRegression` com SGD | `OneVsRestClassifier(LogisticRegression)` | não | `learning_rate=0.05` |
| `linear_regression` | não | não | `LinearRegression` com SGD | `learning_rate=0.01` |
| `hoeffding_tree` | `HoeffdingTreeClassifier` | `HoeffdingTreeClassifier` | `HoeffdingTreeRegressor` | Binária: `grace_period=50`, `max_depth=8`, `leaf_prediction="nba"`, `max_size=8.0` MiB, `memory_estimate_period=100`. Multiclasse: `grace_period=25`, `max_depth=12`, `max_size=64`. Regressão: `grace_period=50`, `max_depth=12`, `max_size=64` |
| `knn` | `KNNClassifier` exato | `KNNClassifier` | `KNNRegressor` | `n_neighbors=11`, `window_size=256` (janela FIFO em `LazySearch`), `weighted=True` |

Detalhes de implementação:

- Logística e kNN usam um `StandardScaler` incremental e codificação one-hot para atributos não numéricos (`models/encoding.py`); `LinearRegressor` (`models/linear.py`) usa apenas `StandardScaler` sobre atributos numéricos. Em todos os casos, o escalonador só aprende em `learn`; a predição lê as estatísticas sem alterá-las.
- A árvore de Hoeffding consome atributos numéricos e nominais diretamente, sem escalonamento. O wrapper binário remove entradas vazias que a predição do River 0.26.1 cria nos observadores nominais, para que prever não altere o estado do modelo.
- Padrões por tarefa (`DEFAULT_MODELS`): `logistic` para binária, `hoeffding_tree` para multiclasse e `linear_regression` para regressão. O avaliador iterativo usa a árvore de Hoeffding com `grace_period=50`, `max_depth=8`, `max_size=8` como classificador.

**Estado:** Implementado. Validado por `tests/test_components.py` e `tests/test_extensions.py`; o modelo logístico binário é o padrão das matrizes binárias, e as matrizes INSECTS usam logística e árvore de Hoeffding.

### Modelos do avaliador por blocos

`src/continual_bench/models/zia.py` define dois modelos treinados por épocas com parada antecipada. O modelo é escolhido pelo número de classes: XGBoost para tarefas binárias e logística multinomial para multiclasse (`datasets/streams.py`, `default_model`).

| Modelo | Biblioteca | Padrões |
| --- | --- | --- |
| `xgboost` (`FraudOrReturnBooster`) | XGBoost 3.2.0 (extra `large-datasets`) | `objective="binary:logistic"`, `tree_method="hist"`, `max_depth=5`, `max_bin=128`, `eta=0.01`, `reg_lambda=0.001`, `seed=42`, 100 rodadas; `nthread` igual a `OMP_NUM_THREADS` |
| `multinomial_logistic` | NumPy e SciPy (`L-BFGS-B`) | `l2=0.001`, `learning_rate=0.01`, `seed=42`, até 100 iterações |

Com uma configuração iterativa (`config/iterative.py`), os padrões passam a ser `max_epochs=100`, `patience=100`, `learning_rate=0.01`, `l2_regularization=0.001`, `validation_fraction=0.1` e `eval_metric="validation:log_loss"`. Cada ajuste separa os últimos 10% da janela de treino para validação interna e restaura os melhores pesos. No XGBoost, uma "época" corresponde a uma rodada de boosting.

O peso da classe positiva (`scale_pos_weight`) só é injetado para os datasets Zia: 39 em Fraud e 10 em E-commerce. As fontes fixadas (Electricity, Airlines, INSECTS) não recebem peso de classe.

**Estado:** Implementado. Validado pela campanha AWS de 132 unidades com XGBoost (`evidencias/aws-benchmarks/README.md`) e pelos testes `tests/test_zia_*.py`. A logística multinomial é usada em Social, que está fora da campanha AWS atual.

### Outras bibliotecas

- **scikit-learn 1.9.1**: `ElasticNetCV`, usado como previsor do UPF (`retraining/forecaster.py`).
- **SciPy**: `dual_annealing`, usado só no benchmark de referência do CARA (`retraining/cara_reference.py`), e `minimize` na logística multinomial.

## Políticas do motor binário por evento

Implementação em `src/continual_bench/core/policies.py`. O motor prevê, libera o rótulo quando seu atraso vence, calcula o erro binário (`(p >= 0.5) != y`), atualiza o detector e pergunta à política o que fazer. O detector só sinaliza; a decisão é sempre da política.

Parâmetros de guarda (classe `PolicyConfig` em `config/benchmark.py`): `every=200`, `cooldown=100`, `min_samples=100`, `min_window=50`, `action="reset_retrain"`. `reset_retrain` recria modelo e pré-processamento e treina na janela de replay (padrão `replay_window=256` rótulos liberados); `learn_one` aplica uma atualização incremental com um exemplo.

| Política | Gatilho |
| --- | --- |
| `none` | Nunca atualiza; o modelo fica como saiu do aquecimento |
| `online` | Atualiza a cada rótulo liberado após o aquecimento, com `learn_one` (obrigatório). Ignora as guardas |
| `periodic` | Retreina quando `rótulos desde a última atualização >= every` |
| `drift` | Retreina quando o detector sinaliza drift |
| `performance` | Retreina quando a taxa de erro recente atinge `error_threshold` (padrão 0,3), depois de acumular `monitor_window` rótulos de avaliação (padrão `min_window`) |

Para `periodic`, `drift` e `performance`, a decisão passa antes pelas guardas, nesta ordem: ao menos `min_samples` rótulos e `min_window` exemplos na janela; pelo menos `cooldown` rótulos desde a última atualização. Com `reset_detector_on_retrain=True` (padrão), o detector é recriado após cada retreino. Uma promoção opcional (`promotion`) treina o candidato num prefixo da janela e só o promove se a acurácia num sufixo temporal superar a do modelo atual em `min_improvement`.

**Estado:** Implementado. Validado por `tests/test_task_policies.py` (gatilhos, guardas, atraso de rótulos, promoção, orçamento) e pelas matrizes binárias históricas da Semana 10.

## Políticas do avaliador por blocos

Implementação em `src/continual_bench/evaluation/stream_benchmark.py`. Depois de prever um bloco e liberar seus rótulos, o avaliador decide uma ação. Todos os retreinos usam no máximo os quatro blocos mais recentes (`REPLAY_BLOCKS = 4`). O bloco de holdout final nunca dispara ações.

| Política | Ação após cada bloco prequential |
| --- | --- |
| `frozen` | Nenhuma; mantém o modelo do aquecimento |
| `periodic` | Retreina do zero a cada quatro blocos, com os quatro mais recentes |
| `drift` | ADWIN recebe a taxa de erro do bloco (um valor por bloco, `clock=1`); um alarme retreina do zero |
| `drift_adwin` | ADWIN recebe o erro binário de cada exemplo (`delta=0.002`, `clock=32`); um ou mais alarmes no bloco geram um retreino |
| `drift_ddm` | DDM recebe o erro binário de cada exemplo (`warm_start=30`, `warning_threshold=2.0`, `drift_threshold=3.0`); alarmes geram um retreino |
| `incremental` | Continua o ajuste a partir do modelo vigente usando só o bloco atual (o XGBoost acrescenta rodadas; a logística parte dos pesos anteriores) |
| `online` | Atualização em minibatches: no XGBoost, renovação das folhas com o gradiente da perda logística do River (`minibatch_rows=8192`); na logística, SGD do River (`learning_rate=0.05`, `minibatch_rows=1024`) |
| `cara`, `cara_cumulative`, `cara_periodic`, `upf`, `oracle` | Decisão por matriz de desempenho (seções abaixo) |

O limiar de decisão binário é `p >= c_FP / (c_FP + c_FN)` com os custos do perfil de tracking (`configs/base/tracking.yaml` usa 1 e 39, limiar 0,025; `configs/base/tracking-dataset-suite.yaml` usa 1 e 1, limiar 0,5). Esse limiar também define o erro binário que alimenta ADWIN e DDM. Os detectores são descritos em [Detectores de Drift](Detectores-de-Drift).

**Estado:** Implementado. Validado na AWS para `frozen`, `drift`, `drift_adwin`, `drift_ddm`, `incremental`, `periodic`, `cara`, `cara_cumulative`, `cara_periodic`, `upf` e `oracle` (12 unidades por política em `evidencias/aws-benchmarks/cases.csv`). `online` está implementado, mas fica fora dos pedidos AWS da suíte.

## Políticas sensíveis a custo: CARA, UPF e Oracle

Implementação em `src/continual_bench/retraining/` (`matrix.py`, `rules.py`, `policies.py`, `schedule.py`, `forecaster.py`, `pricing.py`). Essas políticas decidem por lote a partir de uma **matriz de desempenho**, por isso o motor por evento as recusa: no caminho binário exigem `mode: incremental`.

### Matriz de desempenho

- O fluxo é dividido em `B` lotes. Para cada lote `i`, um modelo candidato é treinado na mesma janela de replay usada pelo `periodic` e avaliado em todos os lotes posteriores. `err[i, j]` é a taxa de erro 0-1 do modelo de `i` no lote `j > i`. São `B` ajustes e `B(B-1)/2` avaliações.
- Para o CARA, cada lote guarda 512 pontos rotulados e 100 consultas amostradas do próprio fluxo (`N_POINTS`, `N_QUERIES`).
- **Fase offline**: os primeiros `w` lotes, com `w = max(2, B // 3)` por padrão (configurável por `retraining.offline_blocks`). No caminho binário, `w = max(2, min(B - 2, B // 3))` e no máximo 24 lotes (`MAX_RETRAINING_BLOCKS`).
- **Fase online**: o modelo do último lote offline é implantado. Após observar o lote `t`, a política decide; um retreino passa a servir a partir do lote `t + 1`. Ao retreinar, o avaliador implanta exatamente o candidato da matriz (mesma janela e seed).
- A política só enxerga uma `MatrixView` com modelos treinados em lotes `<= t`, filtrada pela disponibilidade dos rótulos. Apenas o Oracle lê entradas futuras.
- Atributos ausentes nas amostras do CARA são preenchidos com a média da coluna no aquecimento ou zero (`warmup_column_mean_or_zero`).

### Entrada de custo

Todas usam o custo combinado `Ĉα = α × (número de retreinos) + Σ err[modelo implantado em t, t]`. `α` é o custo de um retreino relativo ao custo de errar um lote inteiro; equivale ao `κ` do CARA por consulta.

- No avaliador por blocos, `α` vem de `upf_alpha` do perfil de tracking (padrão 1,0).
- No caminho binário, a política aceita `alpha` ou `lambda`. Com `lambda`, `α = λ × custo_retreino_usd` (`retraining/pricing.py`). O custo por retreino vem de `custo_retreino_usd` explícito ou de um cenário Fargate On-Demand: 2 vCPU a US$ 0,04048 por vCPU-hora, 4 GB a US$ 0,004445 por GB-hora, preços de 2026-11-08 em us-east-1, cobrança mínima de 60 s no modo `dedicated_task`. As matrizes INSECTS usam `lambda: 10`. Esse valor é um cenário de preço, não faturamento observado; veja [Custos e Recursos](Custos-e-Recursos).

### CARA

Origem: *Cost-aware retraining for machine learning* (2024). Três regras (`rules.py`):

| Política | Regra | Parâmetro ajustado na fase offline |
| --- | --- | --- |
| `cara` (CARA-T) | Retreina se `Ψ̄(t, modelo implantado) >= τ` | `τ`, escolhido entre os valores de obsolescência observados (até 400 candidatos por quantis) e infinito |
| `cara_cumulative` (CARA-CT) | Soma `Ψ̄` desde o último retreino e retreina quando a soma atinge `τ_cum` | `τ_cum`, numa grade de 200 pontos entre 0 e a soma offline, mais infinito |
| `cara_periodic` (CARA-P) | Retreina nos lotes `t` com `(t - deslocamento) % período == 0` | Par período e deslocamento por busca exaustiva |

`Ψ̄` é a **obsolescência relativa** (staleness): a diferença entre a perda do modelo ponderada por similaridade nos dados do lote atual e nos dados do lote em que foi treinado, medida nas consultas do lote atual. A similaridade é um kernel RBF com `γ = 1 / (d × Var(X_offline))` sobre atributos brutos, e o valor é a média sobre as consultas. A otimização offline repete a regra sobre os lotes offline e fica com o parâmetro de menor custo; em empate, prefere o maior limiar. Se faltarem entradas na grade offline, o limiar vira infinito (nunca retreina), a menos que o usuário tenha fixado o parâmetro.

Adaptações em relação ao artigo:

- O retreino passa a valer no lote seguinte, não no próprio lote.
- A busca dos limiares usa grade e quantis em vez de *dual annealing*. O *dual annealing* está presente apenas no benchmark de referência (`retraining/cara_reference.py`, `scripts/replicate_cara.py`), que segue o repositório dos autores na revisão `ac2a600a3bbc5729087318087c9def96104c9023`.
- As consultas são amostradas do próprio fluxo de dados (100 por lote), e não de um conjunto fixo de consultas como nos experimentos com geradores do artigo; essa variante é chamada aqui de CARA-D. O `γ` do kernel é calculado dos dados offline.
- A comparação usa `>=`; o benchmark de referência usa a comparação estrita `>` dos autores.

O código também implementa `Markov` (linha de base do CARA, que retreina quando `Ψ̄ >= α`) e `StalenessOracle` (Oracle sobre a matriz de obsolescência, Algoritmos 2 e 3 do CARA). Os dois são usados nas análises e no benchmark de referência, não como políticas configuráveis.

### UPF

Origem: *When to retrain a machine learning model* (2025). Classe `UPF` em `rules.py`.

1. Um `ElasticNetCV` do scikit-learn prevê o erro de cada par (modelo `i`, lote `j`) a partir de `[i, j, j - i, z_shift]`, usando os erros já observados. `z_shift` é a distância L1 entre as médias dos atributos de dois lotes consecutivos. A escala é o desvio residual (mínimo 0,001).
2. Cada entrada futura é modelada como uma distribuição Beta com essa média (recortada em `[1e-3, 1 - 1e-3]`) e variância limitada a `min(std^2, 0.9 * mean * (1 - mean))`.
3. São sorteadas `samples` realizações (padrão 100) dos erros futuros. De trás para frente, cada estado escolhe uma única continuação para todas as amostras, comparando o quantil `δ` (padrão 0,95) dos custos de retreinar e de manter.
4. O UPF retreina se o quantil `δ` do custo de retreinar agora é estritamente menor que o de manter. Com `δ = 0,5` a regra compara medianas.

Parâmetros: `alpha`, `delta`, `samples`, `seed`. No avaliador por blocos em modo `metrics_only`, a seed do UPF é igualada à seed do modelo. Adaptações: o retreino passa a valer no lote seguinte no fluxo operacional; no replay do protocolo do artigo (`paper_timing`), a decisão do lote `t` usa o histórico até `t - 1`.

### Oracle

`oracle` calcula por programação dinâmica, em `O(B²)`, o cronograma de retreinos que minimiza `Ĉα` usando a matriz inteira, inclusive o futuro (`schedule.py`). Por ler lotes futuros, é uma **referência retrospectiva** (limite inferior do custo do seu objetivo), não uma política implantável. O objetivo padrão é o erro (`err`); `oracle_staleness` otimiza a obsolescência. O Oracle não é um teto universal de F1, AP ou AUROC.

**Estado das políticas por custo:** Implementado. Validado por `tests/test_retraining_algorithms.py` (programação dinâmica igual à busca exaustiva, ausência de vazamento do futuro, Oracle como limite inferior, comportamento do UPF com `α` baixo e alto), `tests/test_cara_reference.py`, `tests/test_matrix.py` e pela campanha AWS de 132 unidades. As comparações com os números publicados nos artigos são descritivas: protocolos, partições e modelos diferem (`evidencias/aws-benchmarks/papers/README.md`).

## Configuração

No caminho binário, a política é declarada no YAML da execução:

```yaml
mode: incremental
detector: {name: none}
policy: {name: upf, params: {lambda: 10.0, delta: 0.95, samples: 100}}
```

No avaliador por blocos, a política é um argumento posicional e os parâmetros vêm do perfil de tracking (`upf_alpha` e o bloco opcional `retraining` com `offline_blocks`, `mc_samples`, `delta`, `n_alphas`, `seed`, `comparison_rules`):

```bash
python -m continual_bench.evaluation.zia <raiz> <dataset> {smoke,full} <política> <saída>
```

Os comandos completos de execução estão em [Execução Local](Execucao-Local) e [Execução na AWS](Execucao-na-AWS). As métricas que cada política produz estão em [Métricas](Metricas); limitações em [Limitações e Trabalhos Pendentes](Limitacoes-e-Trabalhos-Pendentes). Referências bibliográficas em [Referências](Referencias).
