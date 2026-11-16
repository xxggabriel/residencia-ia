# Políticas de atualização com custo: CARA, UPF e Oracle

Substituem a antiga `cost_heuristic`, removida do código, dos campaigns e dos testes. Implementam:

- **CARA** (Mahadevan e Mathioudakis, *Cost-aware retraining for machine learning*, Knowledge-Based Systems 293, 111610, 2024): `cara` (CARA-T, limiar de obsolescência), `cara_cumulative` (CARA-CT) e `cara_periodic` (CARA-P). Os parâmetros são otimizados na fase offline.
- **UPF** (Regol et al., *When to retrain a machine learning model*, ICML 2025, arXiv 2505.14903): `upf`.
- **Oracle**: `oracle`, o escalonamento retrospectivamente ótimo por programação dinâmica (CARA, Algoritmos 2 e 3; UPF, seção 5). Usa lotes futuros, logo é um **limite inferior**, não uma política implantável.

Código em `src/continual_bench/retraining/`; as políticas se registram em `PolicyStrategy` (`retraining/policies.py`). Rodam no pipeline **batch-prequential Zia** (`evaluation/zia.py`). O motor por amostra (`core/engine.py`) as recusa com erro explícito, pois decidem por lote e dependem de uma matriz de desempenho.

## Protocolo

Os fluxos são divididos em `B` lotes (blocos do Zia). `err[i, j]` é a taxa de erro 0-1 do modelo treinado ao fim do lote `i` no lote `j > i`. O modelo de `i` usa a mesma janela de reposição do `periodic` (últimos `REPLAY_BLOCKS` blocos).

1. **Matriz de desempenho** (`retraining/matrix.py`). Uma passada pelo fluxo treina um modelo por lote e avalia cada modelo em cada lote posterior: `B` ajustes e `B(B-1)/2` predições. O rodízio guarda também 512 pontos e 100 queries por lote para a obsolescência do CARA.
2. **Fase offline**: os primeiros `w` lotes (`retraining.offline_blocks`, padrão `max(2, B // 3)`). O CARA otimiza `τ`, `τ_cum` ou `φ` repetindo o Algoritmo 1 sobre eles; o UPF não tem parâmetros offline, apenas histórico.
3. **Fase online**: do lote `w` ao fim. O modelo do lote `w-1` (o último offline) é implantado, como o `f_0` do UPF. Após observar o lote `t`, a política decide; um retreino passa a valer no lote `t+1`.
4. **Execução**: ao decidir *retrain*, o runner carrega o modelo já treinado para a matriz (mesma janela, mesmo seed), sem ajuste duplicado. O stream avaliado reproduz a matriz exatamente, e `verify_realised` falha o run se divergir.

**Desvio deliberado dos papers.** Os artigos avaliam o modelo retreinado em `t` no próprio lote `t` (UPF) ou cobram o retreino e obsolescência zero no mesmo lote (CARA). Isso exigiria avaliar num pedaço de `D_t` que também treinou o modelo. Aqui o retreino vale do lote seguinte (test-then-train), para todas as políticas e para o Oracle. Os números não são comparáveis célula a célula com as tabelas dos artigos.

**Sem vazamento do futuro.** A política só recebe uma `MatrixView` com os erros de modelos *treinados* em lotes `≤ t`. Um teste altera tudo o que seria observado depois de um lote e confere que as decisões anteriores não mudam. Só o Oracle lê entradas futuras.

## Algoritmos

**Custo** (UPF, eq. 3 e 24, com `eN = 1`): `Ĉα(θ) = α‖θ‖₁ + Σₜ err[modelo implantado em t, t]`. `α` é o custo de retreinar relativo ao de errar um lote inteiro; é o `κ` do CARA por query. Vem de `tracking.upf_alpha` (padrão 1.0).

**CARA** (`retraining/rules.py`, `matrix.staleness`). Obsolescência relativa `Ψ̄(t, t') = Ψ(Q_t, D_t, M_t') − Ψ(Q_t, D_t', M_t')`, com `Ψ(q, D, M) = 1/|D| Σ sim(q,x)·ℓ(M,x,y)`, kernel RBF com `γ = 1/(d · Var(X_offline))`, atributos brutos e média sobre as queries no pipeline.
- `cara`: retreina se `Ψ̄ ≥ τ`.
- `cara_cumulative`: soma `Ψ̄` desde o último retreino e retreina ao atingir `τ_cum`.
- `cara_periodic`: retreina a cada `φ` decisões.

A otimização offline é uma busca em grade sobre os valores observados (exata para `τ`, pois o custo é constante por partes), preferindo o maior limiar em empate. O artigo usa *dual annealing* para `τ` e `τ_cum`.

**UPF**. O `ElasticNetCV` do scikit-learn prevê o erro médio usando `[i, j, j−i, z_shift]`.
O termo `z_shift` é a distância L1 entre as médias dos atributos dos dois últimos lotes observados.
A média e a escala residual definem uma distribuição Beta (eq. 58–60). A cada decisão, o UPF amostra
`samples` (100) realizações dos erros futuros. De trás para frente, cada estado escolhe uma
continuação compartilhada por todas as amostras, comparando os quantis de custo de retreinar e
manter. O UPF retreina quando o quantil `δ` (0,95) do primeiro ramo é estritamente menor
(eq. 16–22). Com `δ = 0,5`, a comparação usa as medianas (ablação PF).

**Oracle**. DP em `O(B²)` sobre `err` (objetivo `error`, padrão) ou sobre `Ψ̄` (`objective="staleness"`, o Oracle do CARA). Os testes comparam com busca exaustiva.

## Métricas rastreadas

No `report.json`/MLflow (`tracking_v1/summary/<nome>`), só para estas políticas; as demais ficam `unavailable`:

| Métrica | Artigo | Significado |
|---|---|---|
| `retraining_c_hat` | UPF | `Ĉα(θ)` da fase online no `α` configurado |
| `upf_paper_auc` | UPF Tab. 1 | AUC de `Ĉα` em `[0, α_max]` (10 pontos, trapézio), política reexecutada por `α` |
| `retraining_alpha_max` | UPF §5 | menor `α` em que o Oracle não retreina: `max_r (E₀−E_r)/r` |
| `retraining_oracle_c_hat`, `retraining_oracle_retrains` | UPF, CARA | custo e retreinos do Oracle no mesmo `α` |
| `cost_scpe`, `retraining_mean_scpe` | CARA Tab. 2 | `100·(Ĉ−Ĉ_oracle)/Ĉ_oracle` no `α` do run e média na grade |
| `retraining_online_retrains` | CARA Tab. 4, UPF Tab. 2 | decisões de retreino online (sem o aquecimento) |
| `retraining_average_accuracy` | CARA Tab. 3, UPF Tab. 2 | acurácia média online (as queries são amostras dos lotes) |
| `performance_forecast_mae`, `performance_forecast_bias` | UPF §A.7 | erro e viés do previsor frente às entradas realizadas (só `upf`) |
| `retraining_lipschitz_L`, `retraining_retrain_upper_bound` | UPF Prop. 3.1 | `L` empírico e `r* ≤ T − √(α/L)` |
| `retraining_offline_seconds`, `retraining_online_decision_seconds` | UPF Tab. 6 | tempo offline e de uma decisão online |
| `retraining_matrix_fit_seconds` | — | custo de medir a matriz (não entra em `train_seconds`) |
| `retraining_offline_batches`, `retraining_online_batches` | — | `w` e `T` |

Séries (passo = índice da grade de `α`, ou lote online): `retraining/{cost,retrains,average_accuracy,scpe,strategy_cost}_vs_alpha/<política>` e `retraining/cumulative_strategy_cost/<política>`. O relatório traz ainda `retraining` com a matriz de desempenho, a de obsolescência, as curvas por política, a tabela `α` baixo/alto (0,1 e 0,9 de `α_max`), a robustez a `α` errado e os pares previsto×realizado do UPF. As definições estão em `configs/contracts/metric-definitions.json`.

`α_max`, a grade e a AUC usam a matriz do próprio run. A AUC do paper compara políticas na *mesma* grade; por isso `retraining.comparison_rules` (padrão: todas) é simulada em cada run, e `upf_paper_auc` é comparável entre runs do mesmo dataset.

## Gráficos

Gerados em todo run destas políticas, em PNG (matplotlib, `plots/retraining/*.png`, via `mlflow.log_figure`) e em SVG sem dependências (`artifacts/plots/retraining/*.svg` com os dados em `artifacts/data/retraining/`).

| Artefato | Figura do artigo |
|---|---|
| `retraining_problem` | UPF Fig. 1: acurácia de cada modelo `D_i` por lote, com o escalonamento do run |
| `cost_and_retrains_vs_alpha` | UPF Fig. 2: `Ĉα` e nº de retreinos × `α`, todas as políticas |
| `retrain_bounds` | UPF Fig. 4 / A.4: retas em `α = L(T−r)²` para `r* ≤ 0, 1, 2` |
| `wrong_alpha` | UPF Fig. 3: aumento % de `Ĉ` com `α` informado ≠ real (CARA e UPF) |
| `auc_table`, `low_high_alpha_table` | UPF Tab. 1 e 2 |
| `forecast_quality` | UPF §A.7: previsto × observado, MAE e viés |
| `cost_matrix` | CARA Fig. 3a, A.8, A.9: matriz de obsolescência com a estratégia e o Oracle |
| `cara_parameters` | CARA Fig. 3b: `τ*`, `τ_cum*`, `φ*` × `κ` |
| `strategy_retrains_accuracy` | CARA Fig. 4 a 7: custo da estratégia, retreinos e acurácia × `κ` |
| `cumulative_strategy_cost` | CARA Fig. B.10 |

As figuras de cenário único não têm as bandas de confiança dos artigos (10 repetições): cada run é um único fluxo.

## Uso

```bash
python -m continual_bench.evaluation.zia <root> social smoke upf out/social-upf
make aws-campaign-full-cost-aware   # cara, cara_cumulative, cara_periodic, upf e oracle em Social
```

Configuração opcional (ausente por padrão) em `tracking.retraining`: `offline_blocks`, `mc_samples`, `delta`, `n_alphas`, `seed`, `comparison_rules`.

## Limites

- Cada job repete a passada da matriz, que custa `B` ajustes e `O(B²)` predições; em `full` é a parte mais cara do run. Os campaigns principais (`campaign-*-full.json`) não incluem estas políticas; use `campaign-social-full-cost-aware.json`.
- Um dataset com menos de quatro blocos de fluxo falha com erro explícito. Em dados que o modelo já separa perfeitamente o Oracle custa 0, e `cost_scpe` fica indefinido.
- Resultados de smoke validam algoritmo e integração, não desempenho (veja [reliability.md](reliability.md)).

## Benchmark das Tabelas 1 e 2 do UPF

```bash
python scripts/replicate_upf.py gauss circles electricity airlines \
  --electricity ~/Downloads/electricity-normalized.arff \
  --airlines /path/to/airlines.csv --output /path/to/upf.json
```

O benchmark divide cada lote em treino e teste. O UPF decide antes de cada um dos oito lotes
online usando o histórico até o lote anterior. Se decidir retreinar, usa a partição de treino do
próprio lote e avalia a partição de teste. O Oracle usa a mesma matriz de erros e contabiliza o
erro de teste do modelo novo em cada retreino. Os sete lotes offline fornecem os 21 pares da
Tabela 4. Os erros na diagonal online entram no histórico após a avaliação de um modelo ativado.

O comando usa 10 sementes, XGBoost com 100 rodadas, `δ = 0,95`, 100 amostras de Monte Carlo e 10
pontos no intervalo de α da Tabela 4. O JSON contém a AUC, as curvas e as médias e desvios padrão
nos pontos de custo baixo e alto. Para Electricity, esses pontos são exatamente α = 0,1 e 0,9.
Use o CSV numérico do Scikit-Multiflow para preservar a codificação de Airplanes indicada em
A.6. Sem `--airlines`, o comando usa o OpenML ARFF do cache, que codifica as categorias em
outra ordem e permite comparar com as execuções locais anteriores.

O split aleatório de treino/teste (80/20), as sementes e o uso do índice do lote como `t`
são escolhas locais. A fórmula de Gauss tem um parêntese ausente no artigo e Circles usa
`ε = 1` por convenção local. O benchmark cobre quatro datasets e usa CARA e Frozen como
baselines locais, além do Oracle. Os números do pipeline prequential usam a temporização
operacional descrita acima.

## Benchmark do CARA com o protocolo dos autores

O benchmark em `scripts/replicate_cara.py` segue o repositório fornecido com o artigo, revisão
`ac2a600a3bbc5729087318087c9def96104c9023`. O fluxo do laboratório que compara CARA e UPF continua com a
contabilidade descrita acima. Os JSONs e gráficos de execuções antigas em `docs/results/` registram o
protocolo usado na época. Os JSONs novos incluem o campo `reference` com a revisão de referência.

| Código dos autores | Implementação do benchmark |
|---|---|
| `datasets/*.py`, `config/dataset/` | Geradores com seed zero fixa, CSVs originais de Electricity e Airlines, e Covertype binário escalado do LIBSVM. Airlines usa *one-hot* esparso e MaxAbsScaler. |
| `utils.get_online_offline_data_dict`, `qdm.py` | Fases de 25 e 75 lotes com modelos independentes, split de 10% por lote com seed zero, perda 0–1 e obsolescência relativa somada nas queries. RBF com `gamma=20` ou `1/n_features` em Airlines. |
| `algorithms/threshold.py`, `cum_threshold.py`, `markov.py` | Comparação estrita `>`. CARA-T e CARA-CT usam os limites dos autores no *dual annealing*. O benchmark fixa a seed do otimizador à do Random Forest. |
| `algorithms/periodic.py`, `dp.py` | Períodos inteiros, menor período em empate e fase preservada no início online. Oracle por programação dinâmica, com decisão também no último lote. |
| `find_max_retraining_cost.py`, `analyse.py` | Grade de 100 custos sobre o menor limite do Oracle **offline** entre as seeds, elevado ao próximo inteiro. A busca local delimita a fronteira por bisseção, pois a função usada com `fsolve` no original tem um platô. |
| `utils.py`, `plotting.py` | Custo inclui o treino inicial de cada fase. A contagem dos algoritmos CARA exclui esse treino. A acurácia usa a avaliação prequential do repositório, incluindo o recorte do deslocamento no último lote. O SCPE usa referência com sinal, exclui referência zero e preserva resultados iguais a zero. O resumo sobre a grade exclui `κ=0`. |

Execute com o extra `replication` do projeto e informe os arquivos de dados originais quando necessário.

```sh
python scripts/replicate_cara.py covcon-d circle-s electricity --seeds 5 \
  --electricity /path/to/elec.csv --output /path/to/cara.json
```

O CLI aceita `--airlines` para `airlines.csv` e `--covertype` para o arquivo
`covtype.libsvm.binary.scale`, também em `.bz2`. A opção `--kappa-max` define um teto explícito para
comparações na mesma grade. `--drift-backend skmultiflow` usa os detectores do ambiente original, quando
disponíveis. `--drift-backend river` oferece uma comparação com os detectores deste projeto, que têm
implementações diferentes. O experimento de classificação incremental dos apêndices usa outros modelos
e fica fora desse CLI.

Para gerar as comparações entre os resultados locais e os artigos, passe os JSONs de cada
benchmark ao script. Ele gera SCPE e acurácia para o CARA, AUC para o UPF e a comparação
da Tabela 2 quando o JSON de Electricity inclui os pontos de custo baixo e alto.

```sh
python scripts/plot_replication.py /path/to/cara.json /path/to/upf.json --output /path/to/plots
```
