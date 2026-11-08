# Métricas e contratos da Semana 09

Foi acrescentado o suplemento `semana09-metrics-v1`, preservando as definições e
identidades da Semana 08. O módulo `core/evaluation_metrics.py` acrescentou avaliação de horizontes futuros
com snapshots, validação temporal e um resumo de matrizes de tarefas. A montagem
do relatório e a validação do matcher existente ficaram em
`evaluation/reporting.py`, preservando o domínio sem dependências de I/O
ou da camada de aplicação.

## Cobertura e fontes

| Componente | Fonte e versão | Implementação | Validação |
| --- | --- | --- | --- |
| Qualidade binária, calibração e probabilidades | `core/metrics.py`, contrato `residency-metrics-v1` | Existente; precisão, recall, F1, AUROC/AP, Brier, log loss e calibração | Suíte existente |
| Detecção com anotações | `evaluation/tracking.py::detector_metrics`, contrato v1 | Reutilizada; suplemento validou offsets, unicidade, ordem e tolerância | `test_evaluation_metrics.py` cobriu matches, alarmes repetidos, censura e ausência de ground truth |
| Recuperação e recursos | `evaluation/tracking.py` e `core/observability.py`, contrato v1 | Existente; suplemento referenciou o ledger sem recalcular custos | Suíte existente e teste do suplemento |
| BWT e FWT de GEM | [Lopez-Paz e Ranzato, arXiv 1706.08840v6, equações 3 e 4](https://arxiv.org/html/1706.08840v6) | Novo resumo de matriz; FWT exigiu baseline anterior ao aprendizado | Testes com matrizes calculáveis, células ausentes e transferência positiva |
| Near-future accuracy | [Hammoud et al., arXiv 2305.09275v1, seção 3.5](https://arxiv.org/html/2305.09275v1) | Adaptação operacional binária com gap e horizonte declarados; sem reprodução do benchmark visual | Testes de snapshot, atraso, replay, falha, retomada e limites |
| Separação temporal | Contrato local `validate_temporal_holdout` | Novo validador de disponibilidade e IDs nas partições | Testes de leakage por ID e por rótulos tardios |
| Objetivo de retreinamento UPF | [Regol et al., ICML 2025](https://proceedings.mlr.press/v267/regol25a.html), [arXiv 2505.14903v1, seção 5](https://arxiv.org/html/2505.14903v1) | Mantida distinção entre custo de cenário, sensibilidade pós-hoc e campanha UPF | Nenhuma alegação de reprodução ou AUC UPF |

Os resultados pass/fail dependem da execução no homelab e foram registrados no
relatório principal; a presença dos testes nesta tabela descreveu sua cobertura.

## Detecção e ausência de anotações

`validated_detection` exigiu changepoints únicos e ordenados, alarmes únicos por
offset e tolerância positiva. Reutilizou o matching da Semana 08: primeiro alarme
no intervalo `[mudança, min(mudança + tolerância, próxima mudança))`, com um
verdadeiro positivo por mudança. Alarmes adicionais permaneceram sem match.
Follow-up incompleto foi censurado e seus alarmes foram excluídos da contagem de
F1; o relatório manteve o número de mudanças censuradas.

`changepoints=None` significou ausência de ground truth. Nessa situação,
precision/recall/F1 de detecção permaneceram `null`. Uma lista vazia representou
um controle explicitamente anotado sem mudanças. Alarmes sobre erro preditivo
indicaram mudança de performance; não estabeleceram isoladamente mudança em
`P(X)` ou em `P(Y|X)`.

## Matrizes de tarefas

Para uma matriz `R[i,j]` de acurácias de teste após a tarefa `i`, o resumo calculou:

- Acurácia média das tarefas já concluídas, com peso igual por tarefa.
- BWT: média de `R[T,j] - R[j,j]`, para tarefas anteriores a `T`.
- FWT de GEM: média de `R[j-1,j] - baseline_inicial[j]`. Sem baseline inicial,
  essa métrica permaneceu `null`.
- Forgetting após aprendizado: média de `max(R[i,j], j <= i < T) - R[T,j]`.
  Ganhos posteriores puderam produzir valores negativos; não houve clipping.
- Retenção: média da acurácia final nas tarefas anteriores, sem divisão por
  acurácia inicial.

Células necessárias ausentes propagaram `null`. A definição histórica de
`cl_forward_transfer` da Semana 08 permaneceu preservada; seu baseline de modelo
treinado apenas na tarefa não equivale ao baseline de inicialização de GEM.
Também foi preservado o forgetting histórico que inclui probes anteriores ao
aprendizado da tarefa. Os nomes distintos impediram misturar essas definições.

## Horizontes futuros com snapshot

`NearFutureEvaluation` capturou o modelo antes da primeira feature de cada bloco.
Um bloco consistiu em `gap` exemplos ignorados, seguido por `horizon` exemplos de
avaliação. O snapshot permaneceu congelado enquanto o modelo principal continuou
seu fluxo de aprendizado. Cada previsão recebeu fingerprint das features e
metadados; rótulos foram pontuados apenas no hook de liberação e após `available_at`.

O coletor aceitou até 128 janelas de até 1024 exemplos cada. Replay de previsões e
rótulos idênticos foi idempotente; conflitos foram rejeitados. Falhas na previsão
não avançaram o watermark nem registraram janelas parciais. O estado pôde ser
incluído no checkpoint existente por pickle. A inferência diagnóstica usou cópias
do snapshot para isolar modelos cujo método de predição modifica estado; seu
custo pertenceu ao overhead de avaliação, não à latência de produção.

`near_future_accuracy` foi a média das janelas completas. Janelas incompletas
mantiveram `observed_accuracy` descritiva, contagem de rótulos e motivo de
indisponibilidade. O agregado excluiu janelas incompletas explicitamente; a
comparação entre execuções exigiu horizonte, gap, quantidade e offsets iguais.

O paper motivou afastar a avaliação dos exemplos imediatamente seguintes para
reduzir correlação de rótulos. O valor adequado de gap dependeu do dataset e deve
ser fixado com dados de desenvolvimento. `gap=0` serviu como controle imediato;
nem esse caso nem um gap arbitrário garantiram remoção de correlações. A
implementação não reproduziu as arquiteturas, datasets e orçamento do paper.
Também não implementou o previsor probabilístico ou a decisão UPF.

## Integração e limites

O observer chamou `before_prediction(event, model, labeled, prediction_seconds)`
antes do aprendizado, e `after_label(event, probability, trace, released_at,
model, evaluated)` ao liberar os rótulos. A quantidade de rótulos treinados não
pôde ultrapassar o offset da previsão. A garantia de que o modelo fornecido foi
treinado apenas nesse prefixo permaneceu responsabilidade do engine.

`application.evaluation_reporting.supplemental_report(report, directory,
near_future=collector)` gerou
`task_evaluation-metrics.json` com escrita atômica. Sem o coletor, a acurácia futura
permaneceu indisponível: um ledger prequential comum não reconstruiu modelos
congelados retroativamente. O resumo reutilizou tempo, memória, frequência e
custo do tracking existente. O ledger de candidatos contou tentativas treinadas,
promovidas e rejeitadas, incluindo amostras, duração total e tempo gasto nos
candidatos rejeitados. Custo de cenário não foi convertido em dinheiro,
energia ou carbono sem medição e precificação.

O validador temporal exigiu features e rótulos de treino anteriores ao cutoff,
features de teste a partir dele, ordem crescente e IDs disjuntos. Duplicatas de
conteúdo, grupos de pacientes/usuários/objetos, transformações de imagem e ordem
temporal natural exigiram verificações específicas nos adaptadores e manifestos.
