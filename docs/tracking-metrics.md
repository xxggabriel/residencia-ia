# Contrato de métricas e rastreamento da residência

O catálogo [metric-definitions.json](../configs/metric-definitions.json), versão
`residency-metrics-v1`, reúne o inventário das Semanas 01–07 e as extensões do
benchmark. Cada definição contém família, fórmula, direção, unidade, escopos,
pré-requisitos, motivos de indisponibilidade e fontes observadas. O catálogo inclui
nomes canônicos e chaves nativas/aliases; sua contagem não representa indicadores
científicos independentes. As referências
bibliográficas não provam que um valor foi calculado por uma execução.

Este documento descreve o contrato metodológico. A disponibilidade efetiva deve
ser comprovada pelo relatório, artefatos e metadados de cada run. A inclusão de uma
métrica no catálogo não afirma que sua captura, execução ou publicação foi
concluída. `source_observed` identifica uma definição conferida no código indicado;
`conditional_definition` identifica uma definição que depende de entradas e
instrumentação adicionais.

## Evidência de estudo e extensões

| Semanas | Conteúdo confirmado | Fonte observada |
| --- | --- | --- |
| 01 | Treinamento, inferência, latência, requisições/s e custo, com discussão qualitativa | [Relatório Semana 01](https://chatgpt.com/space/page_0a66d6fa6aa48191a06c7aa67465bc07) |
| 02–03 | Tempo de entrega/implantação, compilação e testes, cobertura, violações de padrões, recursos e manutenção | [Anotações CI/CD](https://chatgpt.com/space/page_4a9cd39a6bf881918e2dbcc26ca76e0e), [Challenges in Deploying ML](https://chatgpt.com/space/page_50f90e6a0aa481918086a99b090b2e44), [Dívida técnica](https://chatgpt.com/space/page_a1d9502a2d9c8191b4c217f67aed6bea) |
| 04 | AA, AIA, FM, BWT, IM e FWT | [Anotações do survey](https://chatgpt.com/space/page_6abe3d0632588191a28c719a34dfbddb), seção C; [survey primário](https://arxiv.org/pdf/2302.00487v3), seção 2.3 |
| 05 | Desempenho recente, falsos alarmes, atraso, recuperação, processamento e memória | [Resumo Semana 05](https://chatgpt.com/space/page_36f6a28c3e2081918bfde273ef4cea4d) |
| 06 | Discriminação dos escores de drift, ganho realizado, ROI, MAE, SCPE, custo total, AUC de custo e eficiência até qualidade-alvo | [Resumo Semana 06](https://chatgpt.com/space/page_9ffeb147edb08191865236f6f36b72cb) e fontes específicas abaixo |
| 07 | Precisão, recall, F1, MSE, KPIs, recursos, tempo até disponibilização e overhead de plataforma | [Taxonomia MLOps](https://chatgpt.com/space/page_3209b6b9044881919b6bf68809d3e41a), [CRISP-ML(Q)](https://chatgpt.com/space/page_9948a1d301c88191ab5b8521f2c05edb), [benchmark de plataformas](https://chatgpt.com/space/page_f4f90f99da2c819195f41a22c8af8a4c) |

O JSON distingue `studied` (indicador explícito nas fontes), `operationalization`
(fórmula operacional escolhida para um conceito estudado) e `extension` (ampliação
do benchmark). Não foi confirmada nas notas examinadas a adoção específica de
Brier, log-loss, AP, ECE/MCE, balanced accuracy, MCC ou kappa. Essas métricas são
extensões, mesmo quando úteis e já presentes no código. A anotação de Importance
Weighted Cross Validation (`page_27f258a9e1dc819188d4ddf6c3a63664`) estava vazia e
não foi usada como evidência. Não se atribui às fontes o estudo das quatro métricas
DORA como um conjunto.

## Qualidade preditiva binária

A fonte executável é [Metrics](../src/continual_bench/domain/metrics.py). Suas 28
chaves cobrem contagens, qualidade por classe, agregações de F1, probabilidades,
ranking e calibração. As fórmulas individuais estão no catálogo.

| Grupo | Chaves |
| --- | --- |
| Contagens | `n`, `tp`, `fp`, `tn`, `fn`, `errors` |
| Desempenho | `accuracy`, `error_rate`, `precision`, `recall`, `specificity`, `false_positive_rate`, `false_negative_rate`, `negative_predictive_value`, `balanced_accuracy`, `f1`, `f1_macro`, `f1_weighted`, `mcc`, `cohen_kappa` |
| Distribuição | `positive_rate`, `predicted_positive_rate` |
| Probabilidades e ranking | `brier`, `log_loss`, `roc_auc`, `average_precision` |
| Calibração | `ece`, `mce` |

A classe positiva é `y=1`, com decisão em `p>=0.5`. F1 sem sufixo é o F1 positivo;
macro F1 inclui as duas classes, e o ponderado usa seus suportes. AUROC considera
meio crédito nos empates; AP é não interpolada, por grupos de escores iguais,
e não a área trapezoidal de precision-recall. AUROC exige ambas as classes;
AP exige ao menos um positivo. O armazenamento de escores é O(n), com ordenação
O(n log n); registrar o tamanho do escopo é parte da interpretação.

ECE/MCE usam dez bins iguais e comparam frequência positiva e probabilidade
positiva média. Não são calibração da confiança da classe prevista. Os resultados
dependem do binning e da quantidade de exemplos.

Por compatibilidade, várias frações de confusão vazias e métricas em `n=0` usam
zero no formato existente. Esse zero não demonstra bom desempenho. As contagens e
a disponibilidade devem acompanhar os valores; balanced accuracy, AUROC, AP, MCC
e kappa podem ser `null` conforme seus pré-requisitos. A avaliação é temporal:
prever antes de aprender e incorporar o rótulo somente quando liberado.

## Aprendizagem contínua por episódio

Seja `a[k,j]` a acurácia do probe do episódio `j` após concluir o episódio `k`.
As fórmulas são as das equações 1–7 do [survey estudado](https://arxiv.org/pdf/2302.00487v3):

| Indicador | Fórmula |
| --- | --- |
| AA | `mean(j<=k, a[k,j])` |
| AIA | `mean(i<=k, AA[i])` |
| FM | `mean(j<k, max(i=1..k-1, a[i,j])-a[k,j])`, sem truncar em zero |
| BWT | `mean(j<k, a[k,j]-a[j,j])` |
| IM | `a_joint[k]-a[k,k]` |
| FWT | `mean(j=2..k, a[j,j]-a_task_only[j])` |

A matriz usa episódios ordenados, inclusive episódios distintos de um conceito
recorrente, e avalia todos os probes sintéticos declarados em cada linha. Avaliações
de episódios futuros são diagnósticas, sem alimentar treino ou decisão. Probes por
conceito agregados não substituem essa matriz. O máximo do FM usa **todas** as linhas
anteriores (`max_all_prior_rows`), incluindo avaliações antes do aprendizado do
episódio. Isso explicita uma ambiguidade da fonte: ela define inicialmente a matriz
triangular, mas sua equação de FM usa todas as linhas anteriores. O máximo aqui é
amostrado; não é um máximo contínuo. AA
pondera episódios igualmente, não suas quantidades de eventos. AIA considera o
histórico de AA, não a média acumulada de acertos do fluxo. FM, BWT e FWT exigem
mais de um episódio. FWT neste contrato é **após o aprendizado da tarefa**, e não
transferência zero-shot para tarefa ainda não aprendida.

[ContinualEvaluation](../src/continual_bench/domain/continual.py) define referências
operacionais `task_only` e `joint_prefix_one_pass`: o mesmo learner treinado em
uma passagem, no range do episódio ou no prefixo de rótulos já liberados. Essas
referências não são oráculos conjuntos convergidos. A comparação deve expor modelo,
range `[start,end)`, versão do dataset, apresentações de treino, cap e tempo de cada
referência. Os custos das referências ficam separados dos custos da política
principal. O cap próprio é por modelo de referência; não equivale a limite
cumulativo global de avaliação. Referências não autorizam alterar os orçamentos
experimentais das políticas.

Sem probes independentes ou referências adequadas, registrar `null` com motivo,
por exemplo `missing_probe`, `references_disabled`, `reference_budget_exceeded`
ou `insufficient_tasks`. Não preencher baselines ausentes com zero. Não inferir
fronteiras ou conceitos de Electricity/Airlines a partir de alarmes do próprio
detector e tratá-los como verdade independente.

## Drift e recuperação

Os estudos de [Gama e adaptação](https://chatgpt.com/space/page_36f6a28c3e2081918bfde273ef4cea4d)
e [Hinder e detecção](https://chatgpt.com/space/page_4216cd117bdc8191b015023c5f8c9415)
fundamentam falsos positivos/negativos, atraso, discriminação de escores e recursos.
A contagem de alarmes é observável sem ground truth. Precisão, recall, falso alarme
por exposição e atraso exigem associação explícita entre alarmes e mudanças:
tolerância, tratamento de duplicados, fronteiras de transição e horizonte observado.
O deadline é `min(change+tolerance,next_change)`, sem truncá-lo artificialmente ao
fim observado. Se a execução termina antes do deadline, a mudança e todos os
alarmes na sua janela são excluídos de TP/FP/FN, precisão, recall e atraso de
followup completo. Alarmes já observados nessas janelas são preservados nos detalhes
e em `drift_observed_detections`; não desaparecem da evidência. Uma janela completa
sem alarme constitui FN. Mudanças anteriores ao início avaliado também não possuem
followup completo.

`drift_false_alarm_rate` é alarmes na exposição estável divididos pelos exemplos
avaliados fora de **todas** as janelas de detecção, em `alarms/example`. Duplicatas
dentro de uma janela completa podem ser FP de associação, mas não são alarmes na
exposição estável. A extensão `drift_unmatched_alarms_per_1000_evaluated` usa FP de
associação por 1.000 exemplos avaliados e permanece separada. A chave legado
`drift_false_alarm_rate_per_1000` segue essa extensão, não a taxa estável canônica.

Atraso precisa identificar o relógio: evento de origem, disponibilidade do rótulo
ou processamento. Timestamps ordinais de replay não são tempos de calendário.
Electricity e Airlines têm cronologia e changepoints reais desconhecidos neste
contrato; taxas dependentes dessa verdade são N/A. Zero alarmes não demonstra
zero drift. Atraso não observado permanece N/A, nunca atraso zero. Uma mudança
com followup completo sem alarme é FN; followup incompleto é censura, e ausência
de baseline de recuperação é indisponibilidade, conforme as contagens separadas.

A ROC-AUC do detector exige escores e rótulos independentes de cenários com e sem
mudança. Não usar a probabilidade predita pelo classificador como substituto do
escore do detector. Recuperação exige alvo, tolerância, janela e persistência
predefinidos; a fonte não fornece um limiar universal para este benchmark. Uma
mudança sem janela anterior completa não tem alvo definido: entra em
`recovery_unavailable_changes`. Somente mudanças com alvo definido e recuperação
não observada entram em `recovery_censored_changes`.

## Ganho, custos e eficiência

| Fonte estudada | Contrato e limite |
| --- | --- |
| [Cost-sensitive adaptation](https://chatgpt.com/space/page_e910e26aed6481918627e8c2e014760c) | Ganho é a soma das diferenças de perdas do modelo anterior congelado e do atual, nos mesmos eventos futuros até a próxima atualização. ROI divide esse ganho pelo custo adicional; zero no denominador é indefinido. O custo da própria avaliação é separado do denominador conforme o artigo. |
| [Cara](https://chatgpt.com/space/page_7093b08d28008191a176b16b6bf9c28e) | SCPE compara custo da estratégia com Oracle retrospectivo no mesmo objetivo. Não substituir Oracle pela melhor configuração observada. Obsolescência ponderada por consultas é um objetivo substituto, não benefício real do modelo novo. |
| [UPF](https://chatgpt.com/space/page_eeec4651e54c8191ab556c46fd2ffbff) | Custo normalizado combina erros por lote e retreinamentos ponderados por alpha. A AUC do artigo exige políticas avaliadas em vários alpha, grade comum e faixa definida pelo Oracle. |
| [Same accuracy twice as fast](https://chatgpt.com/space/page_0e6c9b1c85c8819181ac107a1281f117) | L99/L100 comparam iterações até qualidade-alvo com uma baseline fornecida. L99 usa 99% da qualidade da baseline; alvo não atingido é censura, não razão zero. |

O shadow para ganho realizado é opt-in e acrescenta inferência, memória e
possivelmente serialização. Sua ausência torna ganho/ROI pareado N/A. Comparar
janelas diferentes antes e depois não produz o contrafactual exigido pelo ROI.
Neste contrato o ganho usa o modelo imediatamente anterior de cada atualização.
`adaptation_roi_aggregate` descreve somente intervalos shadow elegíveis, e expõe
`adaptation_roi_eligible_updates`, `adaptation_roi_omitted_updates`,
`adaptation_roi_eligible_cost` e `adaptation_roi_omitted_priced_cost`. O custo omitido
só é numérico quando todos seus componentes exigidos são conhecidos.
`adaptation_roi_global` exige que **todas** as atualizações pós-warmup tenham custo
precificado e horizonte futuro avaliado, com identidade/range de treino compatíveis.
Se a última atualização não tem previsões futuras, o global é N/A, mesmo que o
agregado do subconjunto elegível exista. Nenhum desses ROIs inclui o treinamento
inicial como se fosse uma adaptação pareada. Média ponderada dos ROIs e razão de somas de
benefícios/custos são indicadores diferentes.

Custo parametrizado por erro, treino ou operação é um **cenário**. Não é despesa
real em R$, lucro, ROS ou economia comprovada. As unidades devem ser compatíveis.
Computação, rotulagem, comunicação e oportunidade exigem fontes próprias. Sem
receita e despesas reais, os KPIs econômicos permanecem N/A.

Uma grade pós-hoc aplicada à **mesma trajetória fixa** mede sensibilidade do custo.
Ela não reexecuta decisões para cada alpha, não reproduz o UPF e não constitui a
AUC do paper. Sem a campanha e a faixa apropriadas, usar a definição distinta
`posthoc_cost_sensitivity`, mantendo `upf_paper_cost_auc` indisponível. A definição
canônica é uma **série** por índice da grade, com `alpha_grid`, base de custos e
unidade em `series_metadata`, sem alias escalar de AUC. `cost_sensitivity_auc` é a
extensão escalar de integração dessa grade; `posthoc_update_event_cost` penaliza
qualquer atualização, incluindo `learn_one`. Já `upf_total_cost` só se aplica a
trajetórias de `reset_retrain`, excluindo políticas online ou mistas. Seu escopo
continua sendo a trajetória fixa e as janelas observadas, sem reproduzir o paper.

MAE/MSE de um alvo numérico e MAE/viés de um previsor de desempenho são objetos
diferentes. `mae` e `mse` canônicos permanecem N/A nesta classificação binária.
As extensões `mae_probability` e `mse_probability` calculam erros das probabilidades;
a segunda coincide com Brier, sem transformar a tarefa em regressão. As métricas
do previsor UPF exigem previsões futuras registradas antes
do resultado e um previsor efetivo; a classificação binária sozinha não as produz.

Para os indicadores científicos `iterations_to_target`, `l99` e `l100`, a
referência deve fornecer `curve_scope=training_iteration_fixed_test`,
`method_curve`, `baseline_curve`, `baseline_target` e `test_dataset_sha256`, além
da linhagem da referência. Os passos devem ser inteiros não negativos, estritamente
crescentes; as acurácias, números finitos em [0,1] ou `null` explícito; o alvo deve
ser finito em [0,1]. A identificação do teste fixo e a evidência têm hashes SHA256
hexadecimais. A URI identifica a evidência fornecida; o cálculo não a baixa nem
certifica que o conteúdo externo corresponde ao hash informado.

Referências com `curve_scope=window_end_source_offset_including_warmup` produzem
apenas `prequential_samples_ratio_l99` e `prequential_samples_ratio_l100`.
`samples_to_target_accuracy` também mede offset do fluxo. Esses diagnósticos
operacionais não são iterações de treinamento com o mesmo teste fixo e não são
aliases dos indicadores científicos; a qualidade e a dificuldade do fluxo podem
mudar no tempo.

## Medições de recursos, pipeline e CI/CD

Tempo de treino, monitoramento, probes, referências e inferência devem ter escopos
explícitos. Latência e throughput exigem relógio medido e fronteiras da operação.
Guardar hardware, dependências, concorrência e tamanho do fluxo. Oito tasks em
paralelo podem disputar CPU, memória e I/O; diferenças de tempo não devem ser
atribuídas apenas à política.

Peak RSS não é RAM-tempo nem memória exclusiva do modelo. RAM-tempo exige amostras
intervalares de RSS e uma regra de integração declarada. Sem série de memória,
registrar N/A; `peak RSS * duração` não pode ser apresentado como integral medida.
O custo das referências, shadows e cópias de avaliação precisa permanecer visível.

Lead time, duração completa de pipeline, cobertura, violações, releases e overhead
de plataforma dependem de telemetria específica. Overhead exige uma execução de
referência equivalente. A existência de MLflow/Airflow não fornece esses números.
Os multiplicadores do estudo de plataformas não são medições deste laboratório.

## Disponibilidade e linhagem

Cada indicador escalar novo deve expor valor numérico ou `null`, mais motivo e
disponibilidade: o tracking nativo usa `status=available/unavailable`, e a matriz
CL usa `available=true/false`. Uma definição de série pode estar somente em
`series`, `availability` e `series_metadata`, sem resumo escalar artificial.
Exemplos de motivos: `no_ground_truth`, `missing_oracle`, `missing_financial_source`,
`missing_efficiency_baseline`, `missing_drift_scores`, `missing_performance_forecaster`,
`missing_memory_time_series`, `missing_alpha_campaign` e `missing_coverage_report`.
Não publicar NaN ou infinito como alternativa a N/A. Zero válido e indisponibilidade
são estados diferentes.

Ligar cada resultado à versão do dataset, range/split ordenado, versão do modelo,
contrato de fonte/código, configuração, seed, ambiente e escopo. Guardar a matriz
CL e os metadados das referências como artefatos. `continual-probes.json` preserva
probes por episódio e `task_ends`, fonte/dataset e código; seu SHA256 próprio,
quantidade de linhas e protocolo constam de `probe_dataset`. Ele permite
reconstituir os testes das matrizes e referências, inclusive recorrências; uma métrica escalar isolada não
reconstitui sua linhagem. A retomada por checkpoint precisa preservar matrizes,
contadores e referências, respeitando os guards de fonte e configuração existentes.

O catálogo é uma fonte de definições e proveniência. A confirmação de captura e
publicação requer verificações da implementação e evidência da execução; esta
entrega documental não constitui nova campanha nem relatório de resultados.
