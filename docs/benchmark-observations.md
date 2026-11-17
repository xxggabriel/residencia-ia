# Observações por run e ajuste

O catálogo preserva as 198 definições de `residency-metrics-v1`. A entrada
`benchmark_observations` tem a revisão `benchmark-observations-v1` e descreve as
chaves adicionais. A revisão antiga continua identificável pelo hash do catálogo
retido em cada execução. A extensão não reprocessa runs concluídas.

## Inventário anterior à extensão

| Entrada e escopo | Chaves e fontes | Fórmula, unidade e granularidade | Publicação anterior |
| --- | --- | --- | --- |
| Binário offline, prequential por evento ou lote | `core.metrics.Metrics` alimenta `report.metrics`, `windows` e `tracking` | Acurácia = diagonal/N, `precision` = TP/(TP+FP), `recall` = TP/(TP+FN), `f1` positivo e `f1_macro` sobre as duas classes. Classe 1, limiar 0,5. AP não interpolada sobre escores empatados. Frações, por split/janela e run. | Publicação via Compose com `summary_*`, pontos por janela, `tracking_v1`; Batch e CLI `summary/*` e séries prefixadas pelo split. |
| Fluxo tipado | `evaluation.typed` e `scoring.metric_set` montam `metrics.{train,validation,test}` e `per_task` | River, `f1` macro sobre classes reais ou previstas, `accuracy`, `mae` e `rmse` sobre erros acumulados. Frações ou unidade do alvo. Avaliação antes do aprendizado, limitada à liberação de rótulos. | Publicador genérico `summary/<split>/*` e `per_task/*`. As janelas foram acrescentadas em `observations_v1`. |
| Benchmark iterativo | `evaluation.iterative` e `iterative_scores` produzem `summary`, `epoch_history`, `windows` e `class_metrics` | Acurácia e macro sobre classes observadas no alvo; AP agrupada por empate; MAE e RMSE calculados pelo escore daquele escopo. Curvas de treino sobre probe e validação sobre sufixo interno. | Métricas por época com eixo global, `training_round/*`, `summary/*`, `class_metrics/*`, curvas por janela. |
| Benchmark por épocas | `evaluation.epochs` produz `epochs[].metrics` e `training_history` | `scoring.metric_set` sobre a cópia congelada avaliada a cada época em holdout. Acurácia, F1, MAE/RMSE, além de contagens e durações em segundos. A série anterior de steps usa a época como coordenada. | `epoch_tracking.publish_epochs` via Compose; séries `holdout/epochs/*`, `compute/epochs/*` e `costs/epochs/*` na matriz genérica. |
| Zia offline e online | `evaluation.zia_metrics.ClassificationMetrics` em `metrics`, `holdout`, `windows`, matrizes e tabelas por classe; `models.zia` no treino | Frações derivadas de confusão, positivo de índice 1 e limiar de custo, argmax em Social. AP não interpolada e AUROC pelos 4.096 bins fixos em [0,1]. `recall_at_fpr_0_01` maximiza TPR dentre os pontos com FPR elegível. | `summary_*`, `tracking_v1/summary/*`, `tracking_v1/*`, `holdout_*` via Compose; escopos com `/` no SDK e REST gerenciados. |
| SageMaker Training/AMT | `orchestration.sagemaker.train` e loop nativo | Perda e classificação ao final da rodada nativa, split de validação interno e checkpoint escolhido pelo critério configurado. | Experiments, CloudWatch, séries legadas e runs filhas quando MLflow está configurado. |
| Matrizes CARA/UPF e avaliação contínua | `retraining.matrix`, `core.continual`, `core.evaluation_metrics` | Matriz cross-checkpoint de qualidade, ganho e custo segundo protocolo. BWT GEM, FM após aprendizado e forgetting histórico têm baselines diferentes. A matriz de decisão estritamente futura não estabelece retenção. | Relatórios de replicação e seus dados, `tracking_v1`, `task_evaluation-metrics.json` e relatórios/plots de política conforme o protocolo. |
| Alarmes | `evaluation.tracking.detector_metrics` e `validated_detection` | Primeiro alarme elegível por mudança anotada com tolerância em exemplos; atraso, precisão, recall e F1; follow-up incompleto censurado. Contagem de alarmes independe de anotações. | `tracking_v1/summary/*`, artefatos de drift e relatório suplementar. |
| Recursos e custo | `evaluation.costs`, `evaluation.system_metrics`, `core.observability` | Durações monotônicas e RSS por processo ou amostragem identificada. Preços, cenário de operação, volumes e latências no escopo de cada ledger. | `system/*`, `costs/total/*`, `costs/aws/*`, chaves históricas em `tracking_v1` e relatórios. |

O catálogo detalha fórmulas, unidades e pré-requisitos das 198 chaves legadas.
A comparação com a versão anterior confirmou as mesmas definições no JSON
e encontrou todas as chaves legadas nos cinco casos publicados no MLflow.
[Catálogo de métricas](../configs/contracts/metric-definitions.json) O relatório
versionado mantém a separação prequential, treino, validação, teste e holdout.

## Séries e dados

| Granularidade | No MLflow | Artefato ou metadado |
| --- | --- | --- |
| Run | `observations_v1/quality/<split>/*`, `windows/<split>/*`, `effort/*`, `resources/*`, `retention/*`, `drift/*` no mesmo namespace | `tracking-observations/observations.json`, `overview/`, `plots-manifest.json`, `mlflow-lineage.json`. Linhagem inclui dataset, versão, modelo, política, seed e ambiente quando admitidos. |
| Ajuste | Mesma hierarquia de `mlflow.parentRunId` e `cb-fit-id`. Resumos por split, esforço e recursos no step 0. | `training-history.json` no pai. Dados e plots em `tracking-observations/` na filha, checkpoint ou segmento com hash no histórico. Status do ajuste e status de promoção são independentes. |
| Step | `observations_v1/step/*`, contador local iniciando em 1 | Tipo de operação explícito. Lote, apresentações acumuladas, duração, perda nativa e predições observadas quando definidas. As séries originais continuam disponíveis com seus eixos. |
| Época | `observations_v1/epoch/*`, contador local iniciando em 1, com intervalo inclusivo de steps quando registrado | Métricas agregadas por split e metadados de última época, melhor época e checkpoint, quando selecionados. Aprendizado online sem épocas expõe `observed_epoch_count=unavailable`. |

Matriz de confusão, classes, suporte, curvas de ranking, resíduos e dados de
retenção ficam em JSON. As janelas genéricas podem registrar a confusão em
triplas esparsas de índices e contagens. Os escopos de run e ajuste mantêm a
matriz densa quando observada. Na classificação, a classe sem suporte recebe recall
nulo. Macro-F1 inclui as classes-alvo declaradas e atribui F1 zero à classe sem
alvos nem predições. Predições de sentinelas contam no denominador da acurácia e
nos erros da classe real. AP usa escores positivos finitos. Os adaptadores tipados e iterativos
registram quantas predições tinham escore disponível e incluem as demais
predições liberadas na avaliação por decisão. Em Zia, a aproximação por bins agrupa escores da mesma faixa como
empates e mantém o acumulador limitado a 4.096 bins por classe. RMSE usa a soma
dos quadrados dos resíduos e o total de pares avaliados. AP, Macro-F1 e RMSE são agregados sobre as observações ou estatísticas
suficientes de cada escopo.

O relógio e o escopo do denominador acompanham a vazão. O RSS de
`resource.getrusage(RUSAGE_SELF)` é o pico do processo desde o seu início.
Medições de `psutil` representam amostragem onde o runtime a disponibiliza. O
custo usa a moeda e o método declarados. Estimativa de cenário, estimativa em
USD e fatura pertencem a bases de atribuição diferentes. `null` e o motivo
ficam no documento, e o MLflow recebe apenas os valores escalares finitos.

## Comparabilidade

- `observations_v1/.../macro_f1` fixa a convenção de classes ausentes. Para
  duas classes e três rótulos negativos classificados corretamente, o Macro-F1
  com universo fixo é 0,5, contra 1,0 quando a classe ausente é omitida.
  Predições e confusão permanecem iguais. Compare o histórico em sua série
  original, ou use só runs com o novo contrato. O exemplo de referência pode
  ser calculado com `f1_score(..., labels=[0, 1], average="macro")`.
- AP de Zia e AP exata mantêm metodologias distintas. Os scores 0,9 e 0,90001
  caem no mesmo bin. Se apenas o maior for positivo, AP exata é 1,0 e AP
  histogramada é 0,5. Ambas agrupam seus respectivos empates.
- Retenção temporal sobre o mesmo holdout reservado mede qualidade nas mesmas
  células para checkpoints comparados. BWT de GEM e FM pós-aprendizado exigem
  uma diagonal de aprendizado válida. O forgetting histórico pode incluir
  avaliações anteriores ao aprendizado.
- Para timestamps fornecidos pela fonte, `observations_v1/drift/release_delay_mean`
  usa `alarm.released_at - change.timestamp` em matches com follow-up completo.
  Na referência com mudança no seq 2 e timestamp 110 e alarme liberado no
  timestamp 125, o valor é 15 unidades do relógio da fonte. A série legada
  contém 123 no mesmo caso se o epoch estiver em zero, pois supõe
  `timestamp = epoch + seq`. O atraso ordinal continua 2 exemplos. Runs
  históricas precisam dos timestamps originais para recálculo da nova série.

- A chave `r2` no fluxo iterativo usa a soma centralizada de Welford. A
  fórmula é 1 − SSE/SST, com a convenção histórica de zero para um alvo
  constante. Com alvos [10⁹, 10⁹+1, 10⁹+2] e predições [10⁹, 10⁹+1, 10⁹+1],
  o cálculo anterior perde a variância por cancelamento numérico e resulta em
  zero. O revisado dá 0,5, igual a `sklearn.metrics.r2_score`. Compare a
  revisão do ambiente nas runs antigas, especialmente em séries com alvo
  grande e pequena dispersão.
- O scorer iterativo anterior inferia binariedade pelo conjunto de rótulos
  presentes no recorte. Em um problema multiclasse, uma janela contendo só
  `0` e `1` podia produzir `pr_auc`. A chave legada permanece disponível.
  `observations_v1/.../average_precision` exige classificação binária
  declarada e classe positiva explícita.
- Sem perfil de preços AWS, Zia mantém os custos de cenário e marca a
  estimativa USD como indisponível. O caminho anterior não concluía nessas
  condições, pois tentava ler os preços do perfil nulo.

## Runtime e plots

Os workers usam as versões fixadas de River e scikit-learn. O Matplotlib gera
os PNGs com backend Agg. O HTML interativo autocontido usa os dados dos painéis
e funciona sem CDN. O catálogo, os PNGs, o HTML e os dados de origem têm
caminhos próprios sob `tracking-observations/`. `plots-manifest.json` registra
ausências, escopos, duração da renderização e redução de pontos desenhados. Os dados completos
observados continuam disponíveis nas respectivas fontes. Os gráficos legados
continuam nos caminhos `plots/` e `data/`. O extra `plots` habilita os PNGs
em uma instalação CLI que não inclua o extra `services`. As imagens para
Docker Compose e AWS incluem a versão fixada do Matplotlib.

O diagnóstico temporal do Zia usa os primeiros 256 exemplos de cada um de
até 12 períodos de 30 dias em UTC no holdout. Avalia até 16 checkpoints
promovidos, incluindo o primeiro e o último. Todos os checkpoints são avaliados na mesma seleção. A retenção é a média das acurácias dos períodos. O
orçamento de diagnóstico e as posições indisponíveis ficam no artefato.
Checkpoints de candidatos da matriz que existiram apenas no diretório
temporário ficam identificados por hash e tamanho. O histórico de promoção
continua ligado aos modelos persistidos.


## Cobertura validada

| Caminho | Evidência |
| --- | --- |
| Zia Fraud, E-commerce e Social | 36 execuções smoke a partir dos Parquets preparados, três datasets por 12 políticas. Casos de referência para suportes ausentes, empate de escores e minibatch parcial. |
| Zia Fraud incremental, no MLflow | Run principal (experimento `30`, run `94d400c9412142ceac55473365109f26`) com 10 filhas (inicial e 9 ajustes). Os nove primeiros ajustes tiveram três steps e três épocas. Conferidos early stopping, restauração da melhor época, checksums dos dez checkpoints e republicação depois de falha após um lote aceito. |
| Zia Social frozen, no MLflow | Run principal (experimento `29`, run `179b9666dc29419c922ab106d11a23cd`). Macro-F1, acurácia e matriz de confusão reconstituídos do Parquet de predições. |
| Zia Fraud online, no MLflow | Run principal (experimento `30`, run `9e028a7505d04da2943dbd71b9c7cd7c`) com ajuste inicial e nove atualizações. As atualizações têm steps e disponibilidade explícita de épocas. |
| Regressão genérica por épocas e fluxo tipado online, no MLflow | Épocas (experimento `32`, run `804bc2f9abfc44d69cde855d78669307`) com 231 steps e três épocas. Online (experimento `31`, run `12aebd9a977d4cc389dbe4e8e9425f01`) com 1.004 steps no ajuste online, dados completos preservados e redução aplicada no desenho. MAE/RMSE refeitos das predições nos dois protocolos. |
| CARA e UPF | Testes de matriz e retenção reservada. CARA no MLflow (experimento `33`, run `f6ad752d406b4f0ba93756e4a7986ae7`) com quatro candidatas e republicação estável. As variações simuladas de política e parâmetros compartilham os candidatos da matriz. |
| AWS Batch e SageMaker | Imagem AWS runtime construída com os hashes do lock. Smoke do Zia Social gerou PNGs. Os testes de integração exercitaram a publicação gerenciada com adaptadores simulados. Jobs e serviços gerenciados não foram acionados nesta validação. |

Nas runs do MLflow, os artefatos principais ficam em
`tracking-observations/overview/` e o manifesto em
`tracking-observations/plots-manifest.json`. Cada filha tem seu próprio
`tracking-observations/data.json` e os gráficos aplicáveis. Os cinco HTMLs
de visão geral abriram no Firefox e permitiram alternar séries e consultar
detalhes. O recorte de escala executou 10 milhões de predições no caminho de
avaliação da fraude em 40 blocos nativos, sob limite de memória de 2 GiB, com
pico de RSS do processo de 485.257.216 bytes. Esse recorte mediu o avaliador
e excluiu o arquivo de predições e a publicação remota.

A visão geral, a timeline e o explorador são publicados na run principal.
Os caminhos e o comportamento da prévia estão na seção
[Páginas de retreinamento em Artifacts](mlflow-tracking.md#páginas-de-retreinamento-em-artifacts).
