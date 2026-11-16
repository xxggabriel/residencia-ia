# Políticas, detectores e promoção da Semana 09

O motor binário por amostra oferece os controles sem atualização, periódico, por alarme e online, além do gatilho `performance`, configurado em `policy.params`. A promoção opcional compara um candidato com o modelo vigente em uma janela de validação temporal.

## Contratos e decisões

O motor recebe erros binários de predições feitas antes da aprendizagem, depois da liberação do rótulo. `recent_error` usa os rótulos de avaliação liberados em uma janela de tamanho `monitor_window`, cujo padrão é o maior valor entre `metric_window` e `min_window`. Os rótulos de aquecimento ficam fora dessa janela. O retreino aplica janela mínima, cooldown e orçamento. A política online atualiza a cada rótulo liberado depois do aquecimento, e a configuração exige orçamento compatível com esse comportamento.

A política `performance` aciona uma decisão quando a taxa de erro da janela atinge `error_threshold` (padrão 0,3), após acumular ao menos `monitor_window` rótulos, ou `min_window` quando `monitor_window` é omitido. Fixe o limiar antes da campanha.


## Relação com CARA e UPF

| Fonte e versão | Implementação | Diferenças e estado |
| --- | --- | --- |
| Mahadevan e Mathioudakis, CARA, Knowledge Based Systems 293, 111610, 2024; [paper original](https://ananthmahadevan.com/publication/cost-aware-retraining/cost-aware-retraining.pdf) | `cara`, `cara_cumulative`, `cara_periodic` e `oracle` (DP dos Algoritmos 2 e 3) | Retreino vale do lote seguinte; busca em grade no lugar de *dual annealing*; queries amostradas do próprio fluxo (CARA-D). |
| Regol et al., ICML 2025, PMLR 267, 51369 a 51404; [publicação original](https://proceedings.mlr.press/v267/regol25a.html), arXiv 2505.14903 | `upf` (ElasticNetCV, Beta, quantil Monte Carlo) e `oracle` | Retreino vale a partir do lote seguinte no fluxo operacional. A recorrência compara quantis e escolhe uma continuação por estado. |

As políticas por custo usam a matriz de desempenho do avaliador Zia por lotes.
O motor binário por amostra continua usando `performance` e os demais gatilhos
descritos nesta página. Os Oracles usam dados futuros para servir de referência;
CARA e UPF decidem a partir da informação disponível em `MatrixView`. Os protocolos
de referência dos artigos e a comparação em Electricity têm comandos próprios.
[Políticas por custo e benchmarks de referência](cost-aware-retraining.md).

## Detectores

ADWIN, DDM, EDDM, HDDMA, HDDMW, Page Hinkley e KSWIN usam as implementações do River **0.26.1**, fixado no projeto. KSWIN usa seed padrão 42 e valida a relação entre a janela de referência e a janela estatística.

O sinal neste engine é **performance drift**, uma mudança no erro preditivo binário. Um alarme não prova sozinho mudança em P(Y|X); covariate drift em P(X) exige monitoramento de atributos com contrato e controles próprios. KSWIN aceita uma sequência escalar, mas aqui recebe o erro, e não atributos brutos. Os wrappers rejeitaram entradas fora de 0 e 1.

Fontes canônicas da versão implementada:

- [Exportações de detectores River 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/drift/__init__.py).
- [Detectores binários River 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/drift/binary/__init__.py).
- [EDDM e referência original de 2006](https://github.com/online-ml/river/blob/0.26.1/river/drift/binary/eddm.py).
- [Page Hinkley e referências originais](https://github.com/online-ml/river/blob/0.26.1/river/drift/page_hinkley.py).
- [KSWIN e referência Raab et al. 2020](https://github.com/online-ml/river/blob/0.26.1/river/drift/kswin.py).

## Treinamento e promoção

Sem `promotion`, cada treinamento concluído promove o novo modelo. Com `promotion.validation_samples`, o motor reserva o sufixo temporal da janela liberada para comparar o candidato com o incumbente. O candidato é treinado no prefixo, dentro do orçamento, e é promovido quando o ganho de acurácia supera `promotion.min_improvement`.

O holdout de cada candidato não integrou seu treino. O incumbente pode ter aprendido parte desses exemplos em uma tentativa anterior; este gate é uma regra operacional conservadora sobre dados históricos disponíveis, não uma avaliação independente de generalização. O desempenho do benchmark continuou vindo de predições prequential anteriores ao rótulo. Comparar candidatos em um holdout histórico não autoriza reutilizar esse holdout para declarar superioridade.

`report.candidates` registrou candidatos treinados, aceitos e rejeitados, exemplos, duração e validação. `report.updates` registrou somente atualizações promovidas. Todos os candidatos concluídos consumiram orçamento e iniciaram cooldown, inclusive os rejeitados. A rejeição preservou o incumbente e o detector. Falha de treino não substituiu o incumbente; recuperação operacional deve retomar o último checkpoint confirmado, como no fluxo existente. Checkpoints anteriores receberam defaults compatíveis para os campos novos; configurações históricas omitiram os novos campos vazios na identidade.

## Verificação

Os testes cubrem gatilhos, configurações inválidas, atraso de rótulos, separação entre treino e validação, rejeição e consumo de orçamento, retomada por checkpoint, replay idempotente, falha de treinamento e reprodutibilidade dos detectores. O [relatório da Semana 09](task-report.md) registra os comandos e resultados da primeira rodada no homelab.
