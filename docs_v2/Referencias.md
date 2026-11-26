# Referências

Esta página reúne as obras, os datasets, as bibliotecas e os serviços que fundamentam o `continual-bench`, com o local do código em que cada um é usado. As demais páginas citam as obras pelo título e pelo ano e apontam para cá.

## Convenções

- **DOI** (Digital Object Identifier, identificador persistente de uma publicação) aponta para a versão publicada. Quando a obra também existe como preprint no arXiv (repositório de artigos ainda não revisados ou anteriores à versão final), o preprint aparece na linha "Preprint".
- Obras sem DOI trazem a página oficial (anais, editora ou autores). O rótulo **DOI não verificado** marca os casos em que não há identificador confirmado.
- "Uso no projeto" indica o caminho do código. "Adaptação" descreve diferenças entre a implementação e o método original, somente quando o código as torna explícitas.
- Os estados seguem os rótulos do projeto: **Implementado** (há código executável), **Validado** (há teste automatizado ou evidência registrada) e **Não implementado** (a obra é citada, mas o método não existe no código).

## Políticas de atualização

**Cost-aware retraining for machine learning** (2024). Ananth Mahadevan, Michael Mathioudakis. Knowledge-Based Systems, v. 293, artigo 111610. DOI: [10.1016/j.knosys.2024.111610](https://doi.org/10.1016/j.knosys.2024.111610). Página dos autores: [ananthmahadevan.com](https://ananthmahadevan.com/publication/cost-aware-retraining/).
- Uso no projeto: política CARA (Cost-Aware Retraining Algorithm) nas variantes CARA-T (limiar sobre a obsolescência), CARA-CT (limiar cumulativo) e CARA-P (periódica), além da linha de base Markov e do Oracle de obsolescência, em `src/continual_bench/retraining/rules.py` (`CaraT`, `CaraCT`, `CaraP`, `Markov`, `StalenessOracle`). Adaptadores por lote em `src/continual_bench/retraining/policies.py`. A rota de replicação do protocolo dos autores está em `src/continual_bench/retraining/cara_reference.py`, `src/continual_bench/evaluation/cara_replication.py` e `scripts/replicate_cara.py`. Valores publicados nas Tabelas 2 a 4 estão transcritos em `evidencias/aws-benchmarks/papers/published-reference-values.csv`. Estado: Validado (`tests/test_retraining_algorithms.py`, `tests/test_cara_reference.py`).
- Adaptação: no fluxo operacional (`rules.py`), o limiar `tau` é escolhido por busca entre os valores observados de obsolescência (amostrados em até 400 quantis) e `tau_cum` por uma grade de 200 pontos, em vez do *dual annealing* (otimização estocástica global) do artigo; a comparação é `>=`, o empate favorece o maior limiar e, por padrão, o modelo retreinado passa a servir no lote seguinte (a contabilidade do próprio lote existe com `same_batch=True`). A rota de replicação (`cara_reference.py`) usa `scipy.optimize.dual_annealing` e a comparação estrita `>`, como no código dos autores. O Oracle é calculado por programação dinâmica em `src/continual_bench/retraining/schedule.py`.

**When to retrain a machine learning model** (2025). Florence Regol, Leo Schwinn, Kyle Sprague, Mark Coates, Thomas Markovich. Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR v. 267, p. 51369-51404. Versão publicada sem DOI; página oficial: [proceedings.mlr.press/v267/regol25a](https://proceedings.mlr.press/v267/regol25a.html). Preprint: arXiv 2505.14903, DOI [10.48550/arXiv.2505.14903](https://doi.org/10.48550/arXiv.2505.14903).
- Uso no projeto: política UPF (Uncertainty-Performance Forecaster, previsor de desempenho com incerteza) em `src/continual_bench/retraining/rules.py` (`UPF`), previsor em `src/continual_bench/retraining/forecaster.py`, AUC de custo e análise de sensibilidade em `src/continual_bench/retraining/analysis.py`, replicação em `src/continual_bench/evaluation/retraining_replication.py` e `scripts/replicate_upf.py`, geradores Gauss e Circles em `src/continual_bench/datasets/retraining_benchmarks.py`. Valores das Tabelas 1 e 2 transcritos em `evidencias/aws-benchmarks/papers/published-reference-values.csv`. Estado: Validado (`tests/test_retraining_algorithms.py`).
- Adaptação: a média do erro futuro vem de um `ElasticNetCV` do scikit-learn sobre os atributos `[i, j, j - i, deslocamento]`; as entradas são modeladas como Beta com variância residual constante e a decisão compara quantis de Monte Carlo (padrões `delta = 0,95` e 100 amostras). O código oferece duas temporizações: a do artigo (decisão do lote `t` com histórico até `t - 1`) e a operacional (decisão após `t`, válida para `t + 1`). Em `retraining_benchmarks.py`, a regra Gauss é lida literalmente e Circles usa `eps = 1`, porque o apêndice A.6 do artigo não define `eps`.

**Towards cost-sensitive adaptation: When is it worth updating your predictive model?** (2015). Indrė Žliobaitė, Marcin Budka, Frederic Stahl. Neurocomputing, v. 150, p. 240-249. DOI: [10.1016/j.neucom.2014.05.084](https://doi.org/10.1016/j.neucom.2014.05.084).
- Uso no projeto: fonte primária (`roi_primary`) das métricas de ganho realizado e ROI de adaptação em `configs/contracts/metric-definitions.json`, calculadas em `src/continual_bench/evaluation/tracking.py`. Estado: Implementado.

As políticas de retreinamento periódico, por alarme de detector, por desempenho, de atualização online e sem retreinamento (`src/continual_bench/core/policies.py`: `PeriodicRetraining`, `DriftTriggeredRetraining`, `PerformanceTriggeredRetraining`, `OnlineUpdate`, `NoRetraining`) são linhas de base de engenharia; o repositório não associa a elas uma obra específica. O levantamento conceitual citado no contrato de métricas é *A survey on concept drift adaptation* (2014), listado em [Avaliação e métricas](#avaliação-e-métricas). Detalhes em [Modelos e Políticas](Modelos-e-Politicas).

## Detectores de drift

Todos os detectores implementados vêm do River 0.26.1 e são instanciados em `src/continual_bench/detectors/river.py` (`RiverDetector`). Eles recebem o erro binário de predição (0 ou 1) calculado quando o rótulo é liberado, e não os atributos brutos; entradas fora de {0, 1} são rejeitadas. Estado: Validado (`tests/test_task_policies.py`). Detalhes em [Detectores de Drift](Detectores-de-Drift).

**Learning from Time-Changing Data with Adaptive Windowing** (2007). Albert Bifet, Ricard Gavaldà. Proceedings of the 2007 SIAM International Conference on Data Mining, p. 443-448. DOI: [10.1137/1.9781611972771.42](https://doi.org/10.1137/1.9781611972771.42).
- Uso no projeto: `adwin` (`river.drift.ADWIN`) em `src/continual_bench/detectors/river.py`; ADWIN também é linha de base na replicação CARA (`src/continual_bench/evaluation/cara_replication.py`, com backend River ou scikit-multiflow).

**Learning with Drift Detection** (2004). João Gama, Pedro Medas, Gladys Castillo, Pedro Rodrigues. Advances in Artificial Intelligence, SBIA 2004, Lecture Notes in Computer Science, p. 286-295. DOI: [10.1007/978-3-540-28645-5_29](https://doi.org/10.1007/978-3-540-28645-5_29).
- Uso no projeto: `ddm` (`river.drift.binary.DDM`) em `src/continual_bench/detectors/river.py` e linha de base na replicação CARA.

**Early Drift Detection Method** (2006). Manuel Baena-García, José del Campo-Ávila, Raúl Fidalgo, Albert Bifet, Ricard Gavaldà, Rafael Morales-Bueno. Fourth International Workshop on Knowledge Discovery from Data Streams. DOI não verificado (anais de workshop sem DOI localizado); referência registrada no código do [River 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/drift/binary/eddm.py).
- Uso no projeto: `eddm` (`river.drift.binary.EDDM`) em `src/continual_bench/detectors/river.py`.

**Online and Non-Parametric Drift Detection Methods Based on Hoeffding's Bounds** (2015). Isvani Frías-Blanco, José del Campo-Ávila, Gonzalo Ramos-Jiménez, Rafael Morales-Bueno, Agustín Ortiz-Díaz, Yailé Caballero-Mota. IEEE Transactions on Knowledge and Data Engineering, v. 27, n. 3, p. 810-823. DOI: [10.1109/TKDE.2014.2345382](https://doi.org/10.1109/TKDE.2014.2345382).
- Uso no projeto: `hddm_a` e `hddm_w` (`river.drift.binary.HDDMA` e `HDDMW`) em `src/continual_bench/detectors/river.py`. O River data a obra como 2014 (publicação online); o volume impresso é de 2015.

**Continuous Inspection Schemes** (1954). E. S. Page. Biometrika, v. 41, n. 1/2, p. 100-115. DOI: [10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100).
- Uso no projeto: `page_hinkley` (`river.drift.PageHinkley`) em `src/continual_bench/detectors/river.py`.

**Reactive Soft Prototype Computing for Concept Drift Streams** (2020). Christoph Raab, Moritz Heusinger, Frank-Michael Schleif. Neurocomputing, v. 416, p. 340-351. DOI: [10.1016/j.neucom.2019.11.111](https://doi.org/10.1016/j.neucom.2019.11.111).
- Uso no projeto: `kswin` (`river.drift.KSWIN`, janela deslizante com teste de Kolmogorov-Smirnov) em `src/continual_bench/detectors/river.py`.
- Adaptação: o projeto fixa `seed = 42` por padrão e exige `0 < stat_size <= window_size / 2` e `0 < alpha < 1`. O detector recebe uma sequência escalar de erros binários, não os atributos do fluxo.

**Fast Hoeffding Drift Detection Method for Evolving Data Streams** (2016). Ali Pesaranghader, Herna L. Viktor. Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2016), Lecture Notes in Computer Science, p. 96-111. DOI: [10.1007/978-3-319-46227-1_7](https://doi.org/10.1007/978-3-319-46227-1_7).
- Uso no projeto: Não implementado. FHDDM aparece apenas como linha de base nos valores publicados do UPF (`evidencias/aws-benchmarks/papers/published-reference-values.csv`); `src/continual_bench/evaluation/retraining_replication.py` declara que as linhas de base ADWIN, KSWIN e FHDDM do artigo não são reproduzidas ali.

## Modelos

**Mining high-speed data streams** (2000). Pedro Domingos, Geoff Hulten. Proceedings of the Sixth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, p. 71-80. DOI: [10.1145/347090.347107](https://doi.org/10.1145/347090.347107).
- Uso no projeto: família `hoeffding_tree` (árvore de Hoeffding, que decide divisões com o limite de Hoeffding sobre uma amostra do fluxo) em `src/continual_bench/models/tree.py`, sobre `river.tree.HoeffdingTreeClassifier` e variantes multiclasse e de regressão. A documentação do River 0.26.1 cita *Mining time-changing data streams* (2001) como referência da classe.
- Adaptação: a árvore é limitada por padrão (`grace_period=50`, `max_depth=8`, `max_size=8.0`, predição de folha `nba`) e o wrapper corrige um efeito colateral do River 0.26.1, em que a predição podia criar estatísticas de classe nominais.

**Nearest neighbor pattern classification** (1967). T. Cover, P. Hart. IEEE Transactions on Information Theory, v. 13, n. 1, p. 21-27. DOI: [10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964).
- Uso no projeto: família `knn` em `src/continual_bench/models/neighbors.py`, sobre `river.neighbors.KNNClassifier` com `LazySearch`.
- Adaptação: a busca é exata, mas restrita a uma janela FIFO (padrões `window_size=256`, `n_neighbors=11`, votos ponderados), e não a todos os exemplos já vistos.

**XGBoost: A Scalable Tree Boosting System** (2016). Tianqi Chen, Carlos Guestrin. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, p. 785-794. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
- Uso no projeto: modelos dos benchmarks Zia em `src/continual_bench/models/zia.py` e modelo da replicação UPF (100 rodadas por ajuste) em `src/continual_bench/evaluation/retraining_replication.py`. Versão fixada: 3.2.0 (`pyproject.toml`, extra `large-datasets`).

**Random Forests** (2001). Leo Breiman. Machine Learning, v. 45, p. 5-32. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).
- Uso no projeto: `RandomForestClassifier` do scikit-learn na replicação CARA, em `src/continual_bench/evaluation/cara_replication.py`.

**Regularization and Variable Selection via the Elastic Net** (2005). Hui Zou, Trevor Hastie. Journal of the Royal Statistical Society Series B, v. 67, p. 301-320. DOI: [10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x).
- Uso no projeto: `ElasticNetCV` do scikit-learn como previsor de erro da política UPF, em `src/continual_bench/retraining/forecaster.py`.
- Adaptação: usa `cv = min(5, n)` e, quando o histórico tem menos de duas observações ou alvo constante, recai na média do erro com coeficientes nulos.

A regressão logística (`src/continual_bench/models/linear.py`) usa `river.linear_model.LogisticRegression` com otimizador SGD (gradiente descendente estocástico) e padronização incremental; o repositório não cita uma obra específica para ela. Documentação: [River LogisticRegression](https://riverml.xyz/0.26.1/api/linear-model/LogisticRegression/).

### Obras citadas na escolha dos modelos Zia

Estas obras fundamentam a escolha de modelo por tarefa descrita em [Modelos e Políticas](Modelos-e-Politicas). Elas usam outros datasets e não comprovam desempenho nos dados Zia.

**Predicting positive and negative links in online social networks** (2010). Jure Leskovec, Daniel Huttenlocher, Jon Kleinberg. Proceedings of the 19th International Conference on World Wide Web, p. 641-650. DOI: [10.1145/1772690.1772756](https://doi.org/10.1145/1772690.1772756). Preprint: arXiv 1003.2429, DOI [10.48550/arXiv.1003.2429](https://doi.org/10.48550/arXiv.1003.2429).
- Uso no projeto: motivação da regressão logística multinomial no benchmark `benchmark_zia_social` (`src/continual_bench/models/zia.py`).

**An extensive experimental comparison of machine and deep learning methods for credit and bank fraud detection** (2026). Edward K. Y. Yapp, Hui-Yuan Yeh. Finance Research Letters, v. 88, artigo 109190. DOI: [10.1016/j.frl.2025.109190](https://doi.org/10.1016/j.frl.2025.109190).
- Uso no projeto: motivação do XGBoost no benchmark `benchmark_zia_fraud`.

**Returns foresight: Explainable machine-learning models to predict and reduce product return propensity in omnichannel apparel retail** (2026). Xiao Chen, Muhammad Haron, Mammadova Simuzar Sultan. Journal of Retailing and Consumer Services, v. 92, artigo 104828. DOI: [10.1016/j.jretconser.2026.104828](https://doi.org/10.1016/j.jretconser.2026.104828).
- Uso no projeto: motivação do XGBoost no benchmark `benchmark_zia_ecommerce`.

## Avaliação e métricas

### Protocolo de avaliação

**Present Position and Potential Developments: Some Personal Views. Statistical Theory: The Prequential Approach** (1984). A. P. Dawid. Journal of the Royal Statistical Society, Series A, v. 147, n. 2, p. 278 em diante. DOI: [10.2307/2981683](https://doi.org/10.2307/2981683).
- Uso no projeto: origem do protocolo *prequential* (testar e depois treinar: avaliar cada exemplo antes de usá-lo no treino). O protocolo está em `src/continual_bench/evaluation/prequential.py`.
- Adaptação: no projeto, o rótulo pode chegar com atraso; a predição ocorre antes, e o monitoramento e a atualização só usam rótulos já liberados.

**On evaluating stream learning algorithms** (2013). João Gama, Raquel Sebastião, Pedro Pereira Rodrigues. Machine Learning, v. 90, p. 317-346. DOI: [10.1007/s10994-012-5320-9](https://doi.org/10.1007/s10994-012-5320-9).
- Uso no projeto: referência do erro no protocolo testar e depois treinar em fluxos, implementado em `src/continual_bench/evaluation/prequential.py`.

Veja [Metodologia Experimental](Metodologia-Experimental) e [Métricas](Metricas).

### Métricas de classificação e calibração

Implementadas em `src/continual_bench/core/metrics.py` e definidas em `configs/contracts/metric-definitions.json`. Estado: Validado (`tests/test_tracking_metrics.py`).

**A Coefficient of Agreement for Nominal Scales** (1960). Jacob Cohen. Educational and Psychological Measurement, v. 20, p. 37-46. DOI: [10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104).
- Uso no projeto: `cohen_kappa` (concordância entre predição e rótulo corrigida pelo acaso).

**Verification of Forecasts Expressed in Terms of Probability** (1950). Glenn W. Brier. Monthly Weather Review, v. 78, p. 1-3. DOI: [10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2).
- Uso no projeto: `brier` (erro quadrático médio da probabilidade prevista).

**Comparison of the predicted and observed secondary structure of T4 phage lysozyme** (1975). B. W. Matthews. Biochimica et Biophysica Acta, Protein Structure, v. 405, p. 442-451. DOI: [10.1016/0005-2795(75)90109-9](https://doi.org/10.1016/0005-2795(75)90109-9).
- Uso no projeto: `mcc` (coeficiente de correlação de Matthews).

**Obtaining Well Calibrated Probabilities Using Bayesian Binning** (2015). Mahdi Pakdaman Naeini, Gregory Cooper, Milos Hauskrecht. Proceedings of the AAAI Conference on Artificial Intelligence, v. 29. DOI: [10.1609/aaai.v29i1.9602](https://doi.org/10.1609/aaai.v29i1.9602).
- Uso no projeto: `ece` e `mce` (erros de calibração esperado e máximo).
- Adaptação: o projeto usa 10 faixas de probabilidade da classe positiva, como registrado no contrato; o binning bayesiano do artigo não é usado.

AUROC, precisão média e as demais métricas de contagem seguem a documentação do scikit-learn ([model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) e [calibration](https://scikit-learn.org/stable/modules/calibration.html)), citada no contrato como referência complementar.

### Métricas de aprendizado contínuo

**A Comprehensive Survey of Continual Learning: Theory, Method and Application** (2024). Liyuan Wang, Xingxing Zhang, Hang Su, Jun Zhu. IEEE Transactions on Pattern Analysis and Machine Intelligence, v. 46, p. 5362-5383. DOI: [10.1109/TPAMI.2024.3367329](https://doi.org/10.1109/TPAMI.2024.3367329). Preprint: arXiv 2302.00487 (o repositório cita a v3), DOI [10.48550/arXiv.2302.00487](https://doi.org/10.48550/arXiv.2302.00487).
- Uso no projeto: fonte (`cl_primary`) das métricas AA, AIA, FM, BWT, IM e FWT (acurácia média, acurácia média incremental, esquecimento, transferência para trás, intransigência e transferência para frente), em `src/continual_bench/core/continual.py`.

**Gradient Episodic Memory for Continual Learning** (2017). David Lopez-Paz, Marc'Aurelio Ranzato. Advances in Neural Information Processing Systems 30 (NeurIPS 2017). Versão publicada sem DOI; página oficial: [papers.nips.cc](https://papers.nips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html). Preprint: arXiv 1706.08840, DOI [10.48550/arXiv.1706.08840](https://doi.org/10.48550/arXiv.1706.08840).
- Uso no projeto: definições de BWT e FWT (equações 3 e 4), implementadas em `src/continual_bench/core/continual.py` (`cl_backward_transfer` e `cl_forward_transfer`). Descritas em [Métricas](Metricas).
- Adaptação: a FWT subtrai a acurácia de referência de uma tarefa avaliada antes do seu aprendizado (`single_accuracy`); quando essa referência não existe, a métrica fica indisponível com o motivo registrado (`src/continual_bench/core/continual.py`).

**Rapid Adaptation in Online Continual Learning: Are We Evaluating It Right?** (2023). Hasan Abed Al Kader Hammoud, Ameya Prabhu, Ser-Nam Lim, Philip H. S. Torr, Adel Bibi, Bernard Ghanem. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2023), p. 18806-18815. DOI: [10.1109/ICCV51070.2023.01728](https://doi.org/10.1109/ICCV51070.2023.01728). Preprint: arXiv 2305.09275, DOI [10.48550/arXiv.2305.09275](https://doi.org/10.48550/arXiv.2305.09275).
- Uso no projeto: acurácia no futuro próximo (*near-future accuracy*) em `src/continual_bench/core/evaluation_metrics.py`.
- Adaptação: versão operacional binária com intervalo e horizonte declarados, limitada a 1024 amostras e 128 janelas; não reproduz o benchmark visual do artigo.

**Same accuracy, twice as fast: continuous training surpasses retraining from scratch** (2025). Eli Verwimp, Guy Hacohen, Tinne Tuytelaars. Preprint arXiv 2502.21147, DOI [10.48550/arXiv.2502.21147](https://doi.org/10.48550/arXiv.2502.21147). Versão publicada não localizada.
- Uso no projeto: fonte (`speed_primary`) de iterações até o alvo, L99 e L100, em `src/continual_bench/evaluation/tracking.py`.

### Drift, custo e processo de MLOps

**A survey on concept drift adaptation** (2014). João Gama, Indrė Žliobaitė, Albert Bifet, Mykola Pechenizkiy, Abdelhamid Bouchachia. ACM Computing Surveys, v. 46, n. 4, p. 1-37. DOI: [10.1145/2523813](https://doi.org/10.1145/2523813).
- Uso no projeto: fonte primária (`gama_primary`) registrada em `configs/contracts/metric-definitions.json` para a avaliação de adaptação a drift.

**One or two things we know about concept drift: a survey on monitoring in evolving environments. Part A: detecting concept drift** (2024). Fabian Hinder, Valerie Vaquet, Barbara Hammer. Frontiers in Artificial Intelligence, v. 7, artigo 1330257. DOI: [10.3389/frai.2024.1330257](https://doi.org/10.3389/frai.2024.1330257).
- Uso no projeto: fonte (`hinder_primary`) das métricas de detecção (alarmes, verdadeiros e falsos positivos, falsos negativos, atraso), calculadas em `src/continual_bench/evaluation/tracking.py`.

**MLOps: A Taxonomy and a Methodology** (2022). Matteo Testi, Matteo Ballabio, Emanuele Frontoni, Giulio Iannello, Sara Moccia, Paolo Soda, Gennaro Vessio. IEEE Access, v. 10, p. 63606-63618. DOI: [10.1109/ACCESS.2022.3181730](https://doi.org/10.1109/ACCESS.2022.3181730).
- Uso no projeto: fonte (`mlops_primary`) das métricas de custo de cenário (infraestrutura, rotulagem, comunicação e oportunidade) no contrato de métricas.

**Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology** (2021). Stefan Studer, Thanh Binh Bui, Christian Drescher, Alexander Hanuschkin, Ludwig Winkler, Steven Peters, Klaus-Robert Müller. Machine Learning and Knowledge Extraction, v. 3, n. 2, p. 392-413. DOI: [10.3390/make3020020](https://doi.org/10.3390/make3020020).
- Uso no projeto: fonte (`crisp_primary`) dos mesmos componentes de custo de cenário.

**Challenges in Deploying Machine Learning: A Survey of Case Studies** (2022). Andrei Paleyes, Raoul-Gabriel Urma, Neil D. Lawrence. ACM Computing Surveys, v. 55, p. 1-29. DOI: [10.1145/3533378](https://doi.org/10.1145/3533378).
- Uso no projeto: citada apenas pelo título (fonte `week03`) nas métricas de tempo de pipeline e sobrecarga de plataforma do contrato. A identificação da obra foi feita pelo título.

**Practical Benchmark of Open-Source MLOps Platforms: Comparing MLflow, Metaflow and ZenML Across Model Type** (2025). Dan Gabriel Badea, Damian Monea, Lilia Sava. 24th RoEduNet Conference: Networking in Education and Research, p. 1-6. DOI: [10.1109/RoEduNet68395.2025.11208376](https://doi.org/10.1109/RoEduNet68395.2025.11208376).
- Uso no projeto: citada apenas pelo título curto (fonte `platform_notes`) nas métricas de sobrecarga de plataforma. O contrato informa que os números publicados não são medidas deste laboratório. A identificação da obra foi feita pelo título.

As métricas próprias dos artigos de política, como SCPE (erro percentual do custo da estratégia em relação ao Oracle, do CARA) e a AUC do custo combinado (do UPF, que não é AUROC), seguem *Cost-aware retraining for machine learning* (2024) e *When to retrain a machine learning model* (2025), listados em [Políticas de atualização](#políticas-de-atualização).

## Datasets

Os contratos de aquisição, licenças e hashes estão em `configs/contracts/dataset-catalog.json` e nos arquivos `*.lock.json` da mesma pasta. Detalhes em [Datasets](Datasets).

### Fluxos reais

**Electricity (ELEC2).** Origem: M. Harries, *SPLICE-2 Comparative Evaluation: Electricity Pricing* (1999), relatório técnico, University of New South Wales. DOI não verificado (relatório técnico sem DOI). Página oficial: [OpenML 151](https://www.openml.org/d/151), arquivo 2419.
- Uso no projeto: `RealSource` em `src/continual_bench/datasets/openml.py`, versão fixada em `configs/contracts/real-sources.lock.json`; também nas replicações CARA e UPF.

**Airlines.** Página oficial: [OpenML 1169](https://www.openml.org/d/1169), arquivo 66526. A descrição do OpenML credita Albert Bifet e Elena Ikonomovska e indica como fonte a competição Data Expo de 2009. DOI não verificado (o OpenML não indica artigo de origem para esta versão).
- Uso no projeto: `RealSource` em `src/continual_bench/datasets/openml.py`, lock em `configs/contracts/real-sources.lock.json`; replicações CARA e UPF (o UPF chama o dataset de *airplanes*).

**Challenges in benchmarking stream learning algorithms with real-world data** (2020). Vinicius M. A. Souza, Denis M. dos Reis, André G. Maletzke, Gustavo E. A. P. A. Batista. Data Mining and Knowledge Discovery, v. 34, p. 1805-1858. DOI: [10.1007/s10618-020-00698-5](https://doi.org/10.1007/s10618-020-00698-5). Preprint: arXiv 2005.00113, DOI [10.48550/arXiv.2005.00113](https://doi.org/10.48550/arXiv.2005.00113). Página oficial dos dados: [USP Data Stream Repository](https://sites.google.com/view/uspdsrepository).
- Uso no projeto: datasets INSECTS. `InsectsBinarySource` em `src/continual_bench/datasets/real_streams.py` (variantes `abrupt_imbalanced` e `incremental_abrupt_reoccurring_imbalanced`) e `LocalManifestSource` para `abrupt_balanced`, com locks em `configs/contracts/real-streams.lock.json`.
- Adaptação: a tarefa binária usa o rótulo original `2` como classe positiva e reúne as outras classes na negativa.

**Chemical gas sensor drift compensation using classifier ensembles** (2012). Alexander Vergara, Shankar Vembu, Tuba Ayhan, Margaret A. Ryan, Margie L. Homer, Ramón Huerta. Sensors and Actuators B: Chemical, v. 166-167, p. 320-329. DOI: [10.1016/j.snb.2012.01.074](https://doi.org/10.1016/j.snb.2012.01.074). Dataset: *Gas Sensor Array Drift Dataset*, UCI Machine Learning Repository, DOI [10.24432/C5RP6W](https://doi.org/10.24432/C5RP6W).
- Uso no projeto: loader `gas` em `src/continual_bench/datasets/real_streams.py`, lock em `configs/contracts/real-streams.lock.json`.

**Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables** (1999). Jock A. Blackard, Denis J. Dean. Computers and Electronics in Agriculture, v. 24, p. 131-151. DOI: [10.1016/S0168-1699(99)00046-0](https://doi.org/10.1016/S0168-1699(99)00046-0). Dataset: *Covertype*, UCI Machine Learning Repository, DOI [10.24432/C50K5N](https://doi.org/10.24432/C50K5N).
- Uso no projeto: somente na replicação CARA (`src/continual_bench/datasets/retraining_benchmarks.py`, `src/continual_bench/evaluation/cara_replication.py`, opção `--covertype` de `scripts/replicate_cara.py`), na versão binária escalada distribuída pelo LIBSVM.

### Geradores sintéticos

Os geradores vêm do River 0.26.1 e são instanciados em `src/continual_bench/datasets/synthetic.py`.

**A streaming ensemble algorithm (SEA) for large-scale classification** (2001). W. Nick Street, YongSeog Kim. Proceedings of the Seventh ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, p. 377-382. DOI: [10.1145/502512.502568](https://doi.org/10.1145/502512.502568).
- Uso no projeto: `sea` (`SyntheticSource`, `river.datasets.synth.SEA`).

**Database mining: a performance perspective** (1993). Rakesh Agrawal, Tomasz Imielinski, Arun Swami. IEEE Transactions on Knowledge and Data Engineering, v. 5, p. 914-925. DOI: [10.1109/69.250074](https://doi.org/10.1109/69.250074).
- Uso no projeto: `agrawal` (`SyntheticSource`, `river.datasets.synth.Agrawal`), com conceitos recorrentes.

**Incremental learning from noisy data** (1986). Jeffrey C. Schlimmer, Richard H. Granger. Machine Learning, v. 1, p. 317-354. DOI: [10.1007/BF00116895](https://doi.org/10.1007/BF00116895).
- Uso no projeto: `stagger` (`SyntheticSource`, `river.datasets.synth.STAGGER`).

**Mining time-changing data streams** (2001). Geoff Hulten, Laurie Spencer, Pedro Domingos. Proceedings of the Seventh ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, p. 97-106. DOI: [10.1145/502512.502529](https://doi.org/10.1145/502512.502529).
- Uso no projeto: `hyperplane` (`MovingSyntheticSource`, `river.datasets.synth.Hyperplane`); também é a referência que o River 0.26.1 associa ao `HoeffdingTreeClassifier`.

**Multivariate Adaptive Regression Splines** (1991). Jerome H. Friedman. The Annals of Statistics, v. 19, p. 1-67. DOI: [10.1214/aos/1176347963](https://doi.org/10.1214/aos/1176347963).
- Uso no projeto: função de Friedman que gera o alvo de `friedman_drift` (`FriedmanDriftSource`, `river.datasets.synth.FriedmanDrift`).

**Learning model trees from evolving data streams** (2011). Elena Ikonomovska, João Gama, Sašo Džeroski. Data Mining and Knowledge Discovery, v. 23, p. 128-168. DOI: [10.1007/s10618-010-0201-y](https://doi.org/10.1007/s10618-010-0201-y).
- Uso no projeto: modos de drift `gra`, `lea` e `gsg` de `friedman_drift`, em `src/continual_bench/datasets/synthetic.py`.

O gerador `rbf_drift` (`river.datasets.synth.RandomRBFDrift`) não traz referência bibliográfica na documentação do River 0.26.1 e o repositório não lhe associa uma obra. Os geradores Gauss e Circles seguem o apêndice A.6 de *When to retrain a machine learning model* (2025), e os geradores `covcon` e `circle` da replicação CARA seguem o repositório dos autores de *Cost-aware retraining for machine learning* (2024), revisão `ac2a600a3bbc5729087318087c9def96104c9023` (constante `REFERENCE` em `src/continual_bench/evaluation/cara_replication.py`).

### Imagens e visão temporal

**Gradient-based learning applied to document recognition** (1998). Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner. Proceedings of the IEEE, v. 86, p. 2278-2324. DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791). Página dos autores: [yann.lecun.com/exdb/mnist](https://yann.lecun.com/exdb/mnist/); os arquivos usados são os do espelho CVDF fixado em `configs/contracts/image-datasets.lock.json`.
- Uso no projeto: MNIST em `src/continual_bench/datasets/images.py`, base de `split_mnist` e `permuted_mnist`.

**Learning Multiple Layers of Features from Tiny Images** (2009). Alex Krizhevsky. Relatório técnico, University of Toronto. DOI não verificado (relatório técnico sem DOI). Fonte oficial: [relatório](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) e [página do CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html).
- Uso no projeto: CIFAR-10 e CIFAR-100 em `src/continual_bench/datasets/images.py` e `src/continual_bench/datasets/cifar_parquet.py`, base de `split_cifar10` e `split_cifar100`. A página do CIFAR solicita a citação deste relatório e não declara licença SPDX (`configs/contracts/image-datasets.lock.json`).

**Continual Learning Through Synaptic Intelligence** (2017). Friedemann Zenke, Ben Poole, Surya Ganguli. Proceedings of the 34th International Conference on Machine Learning, PMLR v. 70, p. 3987-3995. Versão publicada sem DOI; página oficial: [proceedings.mlr.press/v70/zenke17a](https://proceedings.mlr.press/v70/zenke17a.html).
- Uso no projeto: obra de origem usual do protocolo Split MNIST, implementado por `split_class_tasks` em `src/continual_bench/datasets/protocols.py`. O repositório não a cita; está aqui como fundamento do protocolo.

**An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks** (2013). Ian J. Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, Yoshua Bengio. Preprint arXiv 1312.6211, DOI [10.48550/arXiv.1312.6211](https://doi.org/10.48550/arXiv.1312.6211). Versão publicada não localizada.
- Uso no projeto: obra de origem usual do protocolo Permuted MNIST, implementado por `permuted_tasks` em `src/continual_bench/datasets/protocols.py`. O repositório não a cita; está aqui como fundamento do protocolo.

**CORe50: a New Dataset and Benchmark for Continuous Object Recognition** (2017). Vincenzo Lomonaco, Davide Maltoni. Proceedings of the 1st Annual Conference on Robot Learning, PMLR v. 78, p. 17-26. Versão publicada sem DOI; página oficial: [proceedings.mlr.press/v78/lomonaco17a](https://proceedings.mlr.press/v78/lomonaco17a.html). Preprint: arXiv 1705.03550, DOI [10.48550/arXiv.1705.03550](https://doi.org/10.48550/arXiv.1705.03550). Página dos dados: [vlomonaco.github.io/core50](https://vlomonaco.github.io/core50/).
- Uso no projeto: `core50` em `src/continual_bench/datasets/temporal_vision.py`, lock em `configs/contracts/temporal-vision.lock.json`.
- Adaptação: o manifesto registra o protocolo `CORe50 session-ordered NI diagnostic subset` (`src/continual_bench/datasets/temporal_vision.py`): um subconjunto diagnóstico do cenário NI (novas instâncias), ordenado por sessão, objeto e quadro, e não uma execução completa dos cenários NI, NC e NIC.

**The CLEAR Benchmark: Continual LEArning on Real-World Imagery** (2021). Zhiqiu Lin, Jia Shi, Deepak Pathak, Deva Ramanan. Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks. Versão publicada sem DOI; página oficial: [datasets-benchmarks-proceedings.neurips.cc](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/2838023a778dfaecdc212708f721b788-Abstract-round2.html). Preprint: arXiv 2201.06289, DOI [10.48550/arXiv.2201.06289](https://doi.org/10.48550/arXiv.2201.06289). Página dos dados: [clear-benchmark.github.io](https://clear-benchmark.github.io/).
- Uso no projeto: `clear10` e `clear100` em `src/continual_bench/datasets/temporal_vision.py`, com arquivos fixados em uma revisão do espelho no Hugging Face (`configs/contracts/temporal-vision.lock.json`).
- Adaptação: usa os splits oficiais da release de 2022; a avaliação *streaming* do artigo, com teste no período seguinte ao treino, não é reproduzida.

### Zia Data Labs

Datasets tabulares sintéticos publicados no Hugging Face, sem artigo associado. DOI não verificado (não há DOI). Páginas oficiais: [FreeFraudDetection50M](https://huggingface.co/datasets/ziadatalabs/FreeFraudDetection50M), [FreeSocialNetwork100M](https://huggingface.co/datasets/ziadatalabs/FreeSocialNetwork100M) e [FreeEcommerceOrders50M](https://huggingface.co/datasets/ziadatalabs/FreeEcommerceOrders50M).
- Uso no projeto: leitura e materialização em `src/continual_bench/datasets/zia.py` (PyArrow e DuckDB) e modelos em `src/continual_bench/models/zia.py`.

## Bibliotecas e serviços

### Bibliotecas

As versões vêm de `pyproject.toml`. SciPy e NumPy são dependências transitivas, sem versão fixada no `pyproject.toml`.

| Biblioteca | Versão | Documentação | Artigo | Uso no projeto |
| --- | --- | --- | --- | --- |
| River | 0.26.1 | [riverml.xyz/0.26.1](https://riverml.xyz/0.26.1/) | Montiel et al., *River: machine learning for streaming data in Python* (2021), Journal of Machine Learning Research, v. 22, p. 1-8, [página oficial](https://www.jmlr.org/papers/v22/20-1380.html); preprint DOI [10.48550/arXiv.2012.04740](https://doi.org/10.48550/arXiv.2012.04740) | Detectores (`src/continual_bench/detectors/river.py`), modelos (`src/continual_bench/models/`), geradores (`src/continual_bench/datasets/synthetic.py`) |
| scikit-learn | 1.9.1 | [scikit-learn.org/1.9](https://scikit-learn.org/1.9/) | Pedregosa et al., *Scikit-learn: Machine Learning in Python* (2011), Journal of Machine Learning Research, v. 12, p. 2825-2830, [página oficial](https://jmlr.org/papers/v12/pedregosa11a.html) | Métricas (`src/continual_bench/models/observations.py`), `ElasticNetCV` (`src/continual_bench/retraining/forecaster.py`), Random Forest (`src/continual_bench/evaluation/cara_replication.py`) |
| XGBoost | 3.2.0 | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/en/release_3.2.0/) | *XGBoost: A Scalable Tree Boosting System* (2016), DOI [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) | `src/continual_bench/models/zia.py`, `src/continual_bench/evaluation/retraining_replication.py` |
| SciPy | transitiva | [docs.scipy.org](https://docs.scipy.org/doc/scipy/) | Virtanen et al., *SciPy 1.0: fundamental algorithms for scientific computing in Python* (2020), Nature Methods, v. 17, p. 261-272, DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2) | `dual_annealing` em `src/continual_bench/retraining/cara_reference.py` |
| NumPy | transitiva | [numpy.org/doc](https://numpy.org/doc/) | Harris et al., *Array programming with NumPy* (2020), Nature, v. 585, p. 357-362, DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2) | Matrizes de desempenho e políticas (`src/continual_bench/retraining/`) |
| pandas | 2.3.3 | [pandas.pydata.org/docs](https://pandas.pydata.org/docs/) | McKinney, *Data Structures for Statistical Computing in Python* (2010), Proceedings of the 9th Python in Science Conference, p. 56-61, DOI [10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a) | Extras `services`, `aws` e `aws-mlflow`; serving MLflow (`src/continual_bench/models/serving.py`) |
| Matplotlib | 3.11.2 | [matplotlib.org](https://matplotlib.org/stable/) | Hunter, *Matplotlib: A 2D Graphics Environment* (2007), Computing in Science and Engineering, v. 9, p. 90-95, DOI [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55) | Extra `plots`; figuras em `src/continual_bench/infrastructure/mlflow_figures.py` |
| PyArrow | 26.0.1 | [arrow.apache.org/docs/python](https://arrow.apache.org/docs/python/) | Sem artigo | Parquet em `src/continual_bench/datasets/zia.py`, `pinned_streams.py`, `cifar_parquet.py` |
| DuckDB | 1.5.4 | [duckdb.org/docs](https://duckdb.org/docs/) | Sem artigo citado | Consultas sobre os snapshots Zia em `src/continual_bench/datasets/zia.py` |
| MLflow (`mlflow-skinny`) e `sagemaker-mlflow` | 3.17.2 e 0.5.0 | [mlflow.org/docs](https://mlflow.org/docs/latest/) | Sem artigo | Tracking em `src/continual_bench/infrastructure/` |
| ZenML | 0.98.0 | [docs.zenml.io](https://docs.zenml.io/) | Sem artigo | `src/continual_bench/orchestration/zenml_pipeline.py`, `zenml_steps.py` |
| Apache Airflow | 3.3.2 (imagem) | [airflow.apache.org](https://airflow.apache.org/docs/apache-airflow/3.3.2/) | Sem artigo | DAGs em `dags/` |
| Apache Kafka (`confluent-kafka`) | 4.3.1 (servidor) e 2.16.0 (cliente) | [kafka.apache.org](https://kafka.apache.org/documentation/), [cliente Python](https://docs.confluent.io/kafka-clients/python/current/overview.html) | Sem artigo | `KafkaTransport` em `src/continual_bench/infrastructure/services.py` |
| MinIO | commit `9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a` fixado em `docker/minio/Dockerfile` | [repositório](https://github.com/minio/minio) | Sem artigo | Armazenamento S3 local da pilha Docker Compose |
| Pydantic, PyYAML, Pillow, boto3 | 2.12.5, 6.0.3, 13.0.0, 1.43.117 | [Pydantic](https://docs.pydantic.dev/), [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation), [Pillow](https://pillow.readthedocs.io/), [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) | Sem artigo | Configuração (`src/continual_bench/config/`), imagens, clientes AWS (`src/continual_bench/infrastructure/aws/`) |
| scikit-multiflow | não fixada (opcional) | [scikit-multiflow.github.io](https://scikit-multiflow.github.io/) | Montiel et al., *Scikit-Multiflow: A Multi-output Streaming Framework* (2018), Journal of Machine Learning Research, v. 19, p. 1-5, [página oficial](https://jmlr.org/papers/v19/18-251.html) | Backend opcional `--drift-backend skmultiflow` em `src/continual_bench/evaluation/cara_replication.py` |
| MOA | não usada diretamente | [moa.cms.waikato.ac.nz](https://moa.cms.waikato.ac.nz/) | Bifet et al., *MOA: Massive Online Analysis* (2010), Journal of Machine Learning Research, v. 11, p. 1601-1604, [página oficial](https://jmlr.org/papers/v11/bifet10a.html) | Citada pelo River 0.26.1 como referência de HDDM e da árvore de Hoeffding |

### Serviços AWS

Clientes e templates usados pelo projeto. A operação está descrita em [Execução na AWS](Execucao-na-AWS) e os custos em [Custos e Recursos](Custos-e-Recursos).

| Serviço | Documentação | Uso no projeto |
| --- | --- | --- |
| Amazon S3 | [Guia](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html) | `src/continual_bench/infrastructure/aws/s3.py` |
| AWS Batch | [Guia](https://docs.aws.amazon.com/batch/latest/userguide/what-is-batch.html) | `src/continual_bench/infrastructure/aws/batch.py`, `quotas.py` |
| Amazon ECS e AWS Fargate | [Guia](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html) | `src/continual_bench/infrastructure/aws/batch.py`; preços em `configs/contracts/aws-prices.us-east-1.json` |
| Amazon ECR | [Guia](https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html) | `infra/cloudformation/persistent.yaml` |
| AWS Step Functions | [Guia](https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html) | `infra/statemachines/*.asl.json`, `src/continual_bench/orchestration/aws_cli.py` |
| Amazon SageMaker (Pipelines, MLflow gerenciado, Experiments, Model Registry, Feature Store) | [Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html), [MLflow](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html), [Experiments](https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html), [Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html), [Feature Store](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html) | `src/continual_bench/orchestration/sagemaker.py`, `src/continual_bench/infrastructure/aws/` (`mlflow_tracking.py`, `experiments.py`, `model_registry.py`, `feature_store.py`) |
| SageMaker Unified Studio e Amazon DataZone | [Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/what-is-sagemaker-unified-studio.html), [DataZone](https://docs.aws.amazon.com/datazone/latest/userguide/what-is-datazone.html) | Catálogo dos snapshots Zia; `src/continual_bench/orchestration/aws_cli.py`, `infra/cloudformation/studio*.yaml` |
| AWS Glue e Amazon Athena | [Glue](https://docs.aws.amazon.com/glue/latest/dg/what-is-glue.html), [Athena](https://docs.aws.amazon.com/athena/latest/ug/what-is.html) | `infra/glue/`, `infra/athena/`, `src/continual_bench/infrastructure/aws/athena.py` |
| Amazon Kinesis Data Streams e Managed Service for Apache Flink | [Kinesis](https://docs.aws.amazon.com/streams/latest/dev/introduction.html), [Flink](https://docs.aws.amazon.com/managed-flink/latest/java/what-is.html) | `src/continual_bench/infrastructure/aws/kinesis.py`, `infra/flink/` |
| Amazon CloudWatch e Amazon SNS | [CloudWatch](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/WhatIsCloudWatch.html), [SNS](https://docs.aws.amazon.com/sns/latest/dg/welcome.html) | `infra/cloudformation/modules/observability.yaml`, `src/continual_bench/orchestration/aws_control.py` |
| AWS CloudFormation | [Guia](https://docs.aws.amazon.com/cloudformation/) | `infra/cloudformation/` |
| AWS Price List e Service Quotas | [Price List](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/using-price-list-query-api.html), [Service Quotas](https://docs.aws.amazon.com/servicequotas/latest/userguide/intro.html) | `src/continual_bench/infrastructure/aws/pricing.py`, `quotas.py` |
| AWS Budgets e Data Exports | [Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html), [Data Exports](https://docs.aws.amazon.com/cur/latest/userguide/what-is-data-exports.html) | `infra/cloudformation/modules/billing.yaml`, consultas `infra/athena/cur_*.sql` |
