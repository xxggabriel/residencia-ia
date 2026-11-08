# Políticas detectores e promoção da Semana 09

Foram preservados os controles sem atualização, periódico, por alarme e online. Foram acrescentados os gatilhos `performance` e `cost_heuristic`, com configuração declarativa em `policy.params`, e a promoção opcional por validação temporal. A implementação foi preparada para testes funcionais; não constitui campanha comparativa nem reprodução dos resultados dos artigos.

## Contratos e decisões

O engine recebeu exclusivamente erros binários de predições feitas antes da aprendizagem, após a liberação do rótulo. `recent_error` foi calculado em janela limitada (`monitor_window`, padrão máximo entre `metric_window` e `min_window`) de rótulos de avaliação liberados. Os rótulos de warm-up não foram incluídos. Todas as políticas de retreinamento preservaram tamanho mínimo da janela, cooldown e orçamento de exemplos treinados. A política online continuou atualizando cada exemplo após warm-up, conforme contrato anterior.

A política `performance` acionou uma decisão quando a taxa da janela atingiu `error_threshold` (padrão 0,3) após completar `monitor_window` rótulos (ou `min_window` se não configurado). O limiar deve ser fixado antes da campanha, sem ajuste nos resultados de teste.

A política `cost_heuristic` comparou `max(0, recent_error - target_error) * horizon * error_cost` com `fixed_train_cost + window_size * sample_train_cost`. A atualização exigiu benefício projetado estritamente maior. Os padrões foram erro alvo 0,1, horizonte 100, custo por erro 1, custo fixo de treino 1 e custo por exemplo 0. São unidades abstratas do cenário; não representam automaticamente dinheiro ou energia. O erro alvo representa uma hipótese de recuperação, não uma estimativa garantida do candidato.

## Relação com CARA e UPF

| Fonte e versão | Relação implementada | Diferenças e estado |
| --- | --- | --- |
| Mahadevan e Mathioudakis, CARA, Knowledge Based Systems 293, 111610, 2024; [paper original](https://ananthmahadevan.com/publication/cost-aware-retraining/cost-aware-retraining.pdf) | Referência para separar custo de obsolescência e custo de treino | `cost_heuristic` não implementou a ponderação de erros pela vizinhança das queries, matriz retrospectiva de custos ou otimização offline dos parâmetros de CARA T, CT ou P. Não é reprodução CARA. |
| Regol et al., ICML 2025, PMLR 267, 51369 a 51404; [publicação original](https://proceedings.mlr.press/v267/regol25a.html) | Referência para explicitar o compromisso entre perda preditiva e atualização | A heurística não implementou previsão probabilística da evolução do desempenho com incerteza do UPF. Reprodução fiel e sua calibração permanecem pendentes. O endpoint do PDF indicado pela publicação respondeu ao leitor com MIME não suportado; o resumo original foi consultado. |

A implementação da heurística usa somente informação disponível no instante da decisão. Não consultou changepoints, rótulos futuros ou métricas do modelo treinado futuramente.

## Detectores

Foram mantidos ADWIN e DDM e integrados EDDM, HDDMA, HDDMW, Page Hinkley e KSWIN, todos através do River **0.26.1**, já fixado no projeto. A implementação original foi reutilizada, sem reescrever os algoritmos. KSWIN recebeu seed padrão 42 e validação da relação entre janela de referência e janela estatística.

O sinal neste engine é **performance drift**, uma mudança no erro preditivo binário. Um alarme não prova sozinho mudança em P(Y|X); covariate drift em P(X) exige monitoramento de atributos com contrato e controles próprios. KSWIN aceita uma sequência escalar, mas aqui recebe o erro, e não atributos brutos. Os wrappers rejeitaram entradas fora de 0 e 1.

Fontes canônicas da versão implementada:

- [Exportações de detectores River 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/drift/__init__.py).
- [Detectores binários River 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/drift/binary/__init__.py).
- [EDDM e referência original de 2006](https://github.com/online-ml/river/blob/0.26.1/river/drift/binary/eddm.py).
- [Page Hinkley e referências originais](https://github.com/online-ml/river/blob/0.26.1/river/drift/page_hinkley.py).
- [KSWIN e referência Raab et al. 2020](https://github.com/online-ml/river/blob/0.26.1/river/drift/kswin.py).

## Treinamento e promoção

Sem `promotion`, cada treinamento concluído continuou promovendo o novo modelo. Com `promotion.validation_samples`, o engine reservou o sufixo temporal da janela liberada para comparar o candidato com o incumbente, e treinou o candidato somente no prefixo, respeitando o orçamento. O candidato foi promovido quando o ganho de acurácia foi estritamente maior que `promotion.min_improvement`.

O holdout de cada candidato não integrou seu treino. O incumbente pode ter aprendido parte desses exemplos em uma tentativa anterior; este gate é uma regra operacional conservadora sobre dados históricos disponíveis, não uma avaliação independente de generalização. O desempenho do benchmark continuou vindo de predições prequential anteriores ao rótulo. Comparar candidatos em um holdout histórico não autoriza reutilizar esse holdout para declarar superioridade.

`report.candidates` registrou candidatos treinados, aceitos e rejeitados, exemplos, duração e validação. `report.updates` registrou somente atualizações promovidas. Todos os candidatos concluídos consumiram orçamento e iniciaram cooldown, inclusive os rejeitados. A rejeição preservou o incumbente e o detector. Falha de treino não substituiu o incumbente; recuperação operacional deve retomar o último checkpoint confirmado, como no fluxo existente. Checkpoints anteriores receberam defaults compatíveis para os campos novos; configurações históricas omitiram os novos campos vazios na identidade.

## Verificação

Os testes acrescentados cobriram gatilhos, ponto de equilíbrio de custo, rejeição de configurações inválidas, atraso de rótulos, separação treino e validação, rejeição e consumo de orçamento, checkpoint e replay idempotente, falha de treinamento e contratos/reprodutibilidade dos detectores. Executar os testes somente no homelab e consultar o relatório da Semana 09 para os comandos e resultados efetivamente observados.
