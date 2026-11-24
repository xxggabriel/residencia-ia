# Evidências da campanha de correção metodológica

Este conjunto reúne os resultados de uma campanha posterior de avaliação, com disponibilidade temporal de rótulos corrigida, referência de treino separada do teste e retenção exclusiva de métricas. O registro histórico permanece identificado por sua própria revisão e por suas condições de execução. Os resultados atuais não completam retroativamente atividades anteriores da residência.

A campanha terminou em 24/11/2026, às 05:11:02 UTC, com 60 de 60 casos da matriz concluídos e oito trabalhos de comparação concluídos, totalizando 24 ensaios. As sementes foram 42, 43 e 44. O [recibo de conclusão](validation/final-campaign-completion.json) registra as contagens e o encerramento dos processos.

A exportação integral terminou com código de saída zero e `manifest.complete=true`. A tabela comparável usa os agregados finais dessa exportação. A auditoria da publicação nativa e a conferência independente do pacote exportado foram concluídas, sem falhas nos respectivos escopos.

## Cobertura dos dados

As fontes reais foram consumidas integralmente. Gauss e Circles foram gerados separadamente para cada semente, com o número de exemplos indicado abaixo.

| Dataset | Linhas por fonte ou realização | Uso nesta campanha |
| --- | ---: | --- |
| INSECTS abrupt imbalanced | 355.275 | Matriz binária incremental |
| INSECTS incremental abrupt reoccurring imbalanced | 452.044 | Matriz binária incremental |
| Electricity | 45.312 | UPF individual e comparações por pares |
| Airlines | 539.383 | UPF individual e comparações por pares |
| Gauss | 75.000 | UPF individual |
| Circles | 75.000 | UPF individual |

O [plano da campanha](campaign-plan.json) registra os hashes das quatro fontes reais. Os resultados individuais das comparações preservam a origem, a semente, as faixas de linhas e a cobertura por bloco das realizações utilizadas.

Consumir todas as linhas não significa usar todas elas para treinar ou pontuar uma mesma métrica. No binário, há aquecimento, calibração offline, atraso de rótulos e exemplos reservados para avaliação. Nas comparações UPF e por pares, cada fonte é dividida em 15 blocos contíguos, com a cauda preservada. Os sete primeiros blocos formam a fase offline; os oito seguintes formam a fase online. Cada bloco tem uma divisão interna de 80% para treino e 20% para holdout, determinada pela semente. Treino, holdout, população pontuada e população comum entre políticas têm denominadores próprios.

A cobertura integral desses 15 blocos está registrada inclusive em Electricity, cujos blocos têm 3.020 ou 3.021 linhas, e em Airlines, com 35.958 ou 35.959 linhas. Gauss e Circles têm 5.000 linhas em cada bloco. Os valores exatos das partições de treino e holdout estão nos resultados de cada ensaio.

## O que foi executado

A matriz tem 20 configurações antes das repetições: dois fluxos INSECTS, dois modelos e cinco estratégias. Cada configuração foi executada com as três sementes, formando os 60 casos. Os modelos são regressão logística e Hoeffding Tree. As estratégias são modelo congelado após o aquecimento, retreinamento acionado por ADWIN, CARA, UPF e Oracle.

A atualização Online por rótulo ficou deliberadamente para uma próxima rodada. São 12 casos adiados, correspondentes a dois fluxos, dois modelos e três sementes. Eles não fazem parte do denominador de 60 casos concluídos. A expressão fase online, usada nos protocolos por blocos, designa o período de avaliação e não a presença dessa estratégia adiada.

Os oito trabalhos adicionais se dividem em quatro comparações individuais UPF, sobre Gauss, Circles, Electricity e Airlines, e quatro comparações por pares, sobre Electricity e Airlines nos protocolos `same-batch` e `next-batch`. Cada trabalho contém três ensaios, um por semente. Portanto, oito trabalhos correspondem a 24 ensaios, sem acrescentar 24 casos ao denominador da matriz.

No protocolo `same-batch`, o candidato é ajustado com a partição de treino do bloco corrente e avaliado no holdout desse bloco. No `next-batch`, a implantação ocorre no bloco seguinte. As informações disponíveis e o momento da decisão precisam acompanhar qualquer comparação entre esses protocolos. Eles não devem ser reunidos em uma única classificação de desempenho.

## Disponibilidade temporal e papel das referências

No avaliador binário por blocos, as predições antecedem o treino. Um candidato só recebe rótulos cujo `available_at` já foi alcançado no corte da decisão. O replay reúne até o limite configurado de exemplos liberados, respeitando a ordem dos eventos e as reservas de avaliação. As observações entregues às políticas obedecem ao mesmo corte. Uma perda ainda indisponível não vira erro zero.

Essa regra vale também para a calibração offline. Quando não há informação suficiente para calibrar CARA ou ajustar o histórico do UPF, o relatório conserva a indisponibilidade e seu motivo.

O Oracle usa perdas futuras para escolher a agenda e é uma referência retrospectiva. Seus candidatos de treino continuam sujeitos aos cortes de disponibilidade e à regra de implantação da comparação. Ele não representa uma política prospectiva pronta para operação.

A variante `UPF-train-reference-v3-full-source` usa os arquivos integrais e separa a referência de treino dos rótulos de teste. Essa extensão tem identidade própria. Não é apresentada como reprodução equivalente da AUC publicada, pois a cobertura, a representação dos dados e outras escolhas do protocolo local diferem do resultado de referência.

## Como ler as métricas

As métricas do escopo original de cada execução permanecem nos relatórios. Para comparar estratégias da matriz, o exportador também recalcula as métricas sobre a interseção dos IDs de eventos disponíveis entre os casos de cada combinação de dataset e modelo. Esse escopo aparece como `paired_event_intersection`.

A tabela por semente registra a quantidade de eventos pontuados, o primeiro e o último índice de sequência. O manifesto registra a população comum e seu hash. Esses denominadores devem acompanhar F1 macro, precisão média e AUROC. Não se deve comparar diretamente o resumo original de duas execuções que pontuaram populações diferentes.

F1 macro considera as duas classes binárias, com limiar de decisão de 0,5. A precisão média, identificada por `average_precision`, usa ranking exato com os empates agrupados e não corresponde à integração trapezoidal de uma curva precisão-revocação. `quality.roc_auc` representa a área sob a curva ROC das predições no escopo indicado.

A AUC do objetivo UPF é outra quantidade: resume custo ou erro objetivo ao longo da grade de preços de retreinamento. Ela não é uma AUROC de classificação. Acurácia média por bloco e acurácia agrupada por quantidade de exemplos também permanecem distintas quando os blocos têm tamanhos diferentes.

No agregado de INSECTS, o desvio padrão é amostral, com `ddof=1`. Nas comparações UPF e por pares, a dispersão segue o contrato populacional, com `ddof=0`. Os resultados nas sementes de um mesmo fluxo real fixo não constituem três amostras independentes de generalização. As dispersões são descritivas e não sustentam, por si, significância estatística ou superioridade.

O `n` dos agregados conta valores disponíveis, e não apenas sementes planejadas. As métricas primárias apresentadas nas tabelas abaixo têm `n=3`. Entre as métricas auxiliares, seis grupos de `strategy_scpe` do trabalho `upf-circles`, em `alpha=0`, têm `n=1` e `missing=2`. Também há grupos com `n=0`, preservados como indisponíveis. Essas situações não foram preenchidas com zero; a leitura das demais métricas precisa considerar `n`, `missing` e o motivo de indisponibilidade do respectivo escopo.

### Resultados sobre a população comum

Os valores abaixo foram transcritos do [agregado final](aggregate-metrics.csv), no escopo `paired_event_intersection`. Cada célula apresenta média ± desvio padrão amostral das sementes 42, 43 e 44, com `n=3`, `missing=0` e `ddof=1`. As métricas estão na escala de 0 a 1 e foram arredondadas a seis casas somente para apresentação. O CSV mantém a precisão disponível.

#### INSECTS abrupt imbalanced, regressão logística

População comum por execução: 236.587 eventos. Primeiro índice de sequência: 118.418; último: 355.274. Os 15 casos esperados desse grupo, cinco estratégias em três sementes, estavam disponíveis.

| Estratégia | F1 macro | Precisão média | AUROC |
| --- | ---: | ---: | ---: |
| Congelado | 0,770795 ± 0,000000 | 0,648242 ± 0,000000 | 0,850498 ± 0,000000 |
| ADWIN | 0,805521 ± 0,000000 | 0,766331 ± 0,000000 | 0,910571 ± 0,000000 |
| CARA | 0,740947 ± 0,025808 | 0,636261 ± 0,040935 | 0,849205 ± 0,018200 |
| UPF | 0,678545 ± 0,165666 | 0,543044 ± 0,240707 | 0,726177 ± 0,221334 |
| Oracle retrospectivo | 0,803506 ± 0,000000 | 0,739202 ± 0,000000 | 0,893939 ± 0,000000 |

#### INSECTS abrupt imbalanced, Hoeffding Tree

População comum por execução: 236.587 eventos. Primeiro índice de sequência: 118.418; último: 355.274. Os 15 casos esperados desse grupo, cinco estratégias em três sementes, estavam disponíveis.

| Estratégia | F1 macro | Precisão média | AUROC |
| --- | ---: | ---: | ---: |
| Congelado | 0,720611 ± 0,000000 | 0,619391 ± 0,000000 | 0,844846 ± 0,000000 |
| ADWIN | 0,794465 ± 0,000000 | 0,677805 ± 0,000000 | 0,870497 ± 0,000000 |
| CARA | 0,757740 ± 0,011081 | 0,636671 ± 0,013742 | 0,853235 ± 0,011609 |
| UPF | 0,781660 ± 0,006132 | 0,664195 ± 0,015953 | 0,870727 ± 0,002760 |
| Oracle retrospectivo | 0,802708 ± 0,000000 | 0,699724 ± 0,000000 | 0,886377 ± 0,000000 |

#### INSECTS incremental abrupt reoccurring imbalanced, regressão logística

População comum por execução: 301.228 eventos. Primeiro índice de sequência: 150.689; último: 452.043. Os 15 casos esperados desse grupo, cinco estratégias em três sementes, estavam disponíveis.

| Estratégia | F1 macro | Precisão média | AUROC |
| --- | ---: | ---: | ---: |
| Congelado | 0,531917 ± 0,000000 | 0,314314 ± 0,000000 | 0,591513 ± 0,000000 |
| ADWIN | 0,782234 ± 0,000000 | 0,709110 ± 0,000000 | 0,886200 ± 0,000000 |
| CARA | 0,701986 ± 0,021833 | 0,571580 ± 0,022944 | 0,801535 ± 0,024757 |
| UPF | 0,688533 ± 0,057258 | 0,537659 ± 0,071993 | 0,774182 ± 0,064913 |
| Oracle retrospectivo | 0,753313 ± 0,000000 | 0,649867 ± 0,000000 | 0,846185 ± 0,000000 |

#### INSECTS incremental abrupt reoccurring imbalanced, Hoeffding Tree

População comum por execução: 301.228 eventos. Primeiro índice de sequência: 150.689; último: 452.043. Os 15 casos esperados desse grupo, cinco estratégias em três sementes, estavam disponíveis.

| Estratégia | F1 macro | Precisão média | AUROC |
| --- | ---: | ---: | ---: |
| Congelado | 0,387375 ± 0,000000 | 0,188420 ± 0,000000 | 0,279240 ± 0,000000 |
| ADWIN | 0,791363 ± 0,000000 | 0,667911 ± 0,000000 | 0,871998 ± 0,000000 |
| CARA | 0,734580 ± 0,003966 | 0,567983 ± 0,010811 | 0,827389 ± 0,003609 |
| UPF | 0,708103 ± 0,017778 | 0,526474 ± 0,022999 | 0,794431 ± 0,013232 |
| Oracle retrospectivo | 0,753620 ± 0,000000 | 0,566570 ± 0,000000 | 0,820359 ± 0,000000 |

As faixas de sequência não implicam intervalos sem lacunas: o denominador é a quantidade de IDs efetivamente comuns. Igualar os eventos pontuados não iguala o histórico, o orçamento ou os instantes de treino das estratégias. O Oracle minimiza o objetivo formal no espaço de candidatos e agendas da sua comparação; ele não é um limite universal para F1 macro, precisão média ou AUROC. Valores de desvio padrão iguais a zero nestas três execuções tampouco demonstram ausência de variabilidade em novos fluxos.

## Revisões do cálculo e do pós-processamento

| Etapa | Revisão | O que identifica |
| --- | --- | --- |
| Avaliação executada | [`98a6dd5`](https://github.com/xxggabriel/residencia-ia/tree/98a6dd5a676f46bf1df337de82b33fd0e554c085) | Modelos, decisões, métricas e resultados científicos da campanha |
| Exportação e organização das tabelas | [`f43c724`](https://github.com/xxggabriel/residencia-ia/tree/f43c7249559ad21d38ee2968502c61fdfed30a24) | Serialização, compressão e particionamento dos arquivos de evidência |
| Verificação e publicação dos tempos registrados | [`07aa7f5`](https://github.com/xxggabriel/residencia-ia/tree/07aa7f5acdd30101c446b733b134f82b172df352) | Validação dos recibos e derivação do tempo de corpo de trabalho a partir dos horários existentes |

A revisão do exportador ou do helper não substitui a revisão que gerou os resultados. O pós-processamento não executa novamente treino, inferência ou decisões para preencher o dossiê.

O ambiente da campanha foi Linux x86_64 com Python 3.12.13, River 0.26.1, NumPy 2.5.4, scikit-learn 1.9.1 e XGBoost 3.2.0. A [identidade da imagem](validation/runtime-image-98a6dd5.json) e a [relação de distribuições](validation/runtime-distributions-98a6dd5.json) permitem conferir os demais componentes. Os limites de threads de BLAS e bibliotecas correlatas foram mantidos em 1. O aprendiz XGBoost declara separadamente `nthread=2` nos resultados UPF; esse parâmetro não deve ser confundido com a variável de ambiente do BLAS.

O [runtime portátil posterior](validation/portable-runtime-07aa7f5.json) passou no selftest x86_64, nas três interfaces CLI e na conferência dos hashes dos scripts, com rede desativada. Essas verificações de empacotamento têm recibos próprios e não são novas execuções da campanha integral, nem comprovação de uma implantação na AWS.

## Recursos, tempos e custos

O cenário monetário usado para obter o preço de referência das decisões considera Fargate com 2 vCPU, 4 GiB de memória, 50 GiB de armazenamento efêmero, franquia de 20 GiB e cobrança mínima de 60 segundos por ação dedicada. Para CARA, UPF e Oracle da matriz, `lambda=10` multiplica esse custo nominal para resolver `alpha`. Esse cenário não mede os recursos efetivamente utilizados e não representa uma fatura.

A construção das matrizes ajusta candidatos que servem à análise de várias agendas. A quantidade e o tempo desses ajustes não são a quantidade e o tempo dos retreinos escolhidos por uma política. Os indicadores de construção, implantação selecionada e objetivo formal precisam conservar seus escopos, inclusive quando o experimento calcula candidatos que não chegam a ser implantados.

Na execução, o controlador manteve até quatro trabalhos simultâneos em um container limitado a 8 CPU e 12 GiB. O controlador executou 52 casos da matriz e os oito trabalhos UPF ou por pares. Outros oito casos da matriz foram executados por dois trabalhadores auxiliares no perfil de 2 CPU e 8 GiB, em duas fases de seis e dois casos. As fases auxiliares não se sobrepuseram. O teto combinado foi de seis trabalhos simultâneos. Nos dois perfis, CPU e memória são limites agregados do respectivo container e são compartilhados pelos processos ativos. Eles não representam uma reserva individual de 8 CPU e 12 GiB para cada caso concorrente do controlador. A [primeira fase](validation/execution-scheduling/campaign-auxiliary-execution.json) e a [segunda fase](validation/execution-scheduling/campaign-auxiliary-execution-second.json) preservam as seleções e os horários.

Os limites reais de CPU e memória e os tempos medidos não são entradas das decisões de CARA, UPF, Oracle ou ADWIN nessa matriz. O preço usado nas agendas é o cenário nominal configurado. Há também um orçamento interno da Hoeffding Tree, fixo em 8 MiB e verificado a cada 100 exemplos, que limita o tamanho do modelo e não depende de o container ter 8 ou 12 GiB. A [inspeção do contrato de recursos](validation/resource-profile-policy-audit.json) registra os localizadores de código e as ressalvas.

Essa inspeção estática não demonstra igualdade numérica entre perfis. Tempos, throughput, uso de memória e estimativas baseadas em duração precisam manter a identificação do perfil e da concorrência. A execução compartilhada não permite atribuir isoladamente diferenças de eficiência às políticas.

Os relógios também têm escopos distintos. O `compute_seconds` registrado pelo avaliador, o `wall_seconds_this_attempt` de uma tentativa e o tempo `corrective_campaign/job_body_wall_seconds` não são intercambiáveis. Os dois primeiros também são intervalos de tempo decorrido nos pontos de instrumentação correspondentes; seus nomes não os transformam em medições de tempo de CPU. O último é a diferença entre os horários registrados no corpo do trabalho: inclui materialização, avaliação, diagnósticos e publicação. Ele exclui inicialização do processo, importações e validações anteriores ao início desse intervalo, além do hash final e da gravação do recibo após seu término. Não é tempo de CPU.

Cada um dos oito trabalhos UPF ou por pares contém os três ensaios e trabalho compartilhado de publicação. Dividir seu tempo por três não produz um tempo individual medido por ensaio. O [manifesto dos tempos](validation/execution-times/corrective-execution-times-manifest.json) e a [tabela de tempos](validation/execution-times/corrective-execution-times.csv) preservam esse escopo sem preencher relógios históricos ausentes. Foram registradas 68 linhas, uma para cada caso ou trabalho concluído, e a [conferência da publicação dos tempos](validation/execution-times/corrective-execution-times-tracking.json) confirmou os 60 pontos associados aos casos da matriz. Esses intervalos pertencem ao corpo da tentativa bem-sucedida registrada; não reconstroem tentativas anteriores interrompidas nem demonstram, sozinhos, que houve treino novo em vez de reutilização de evidência.

O `max_seconds` interno é verificado no laço sequencial principal, usado por `none` e ADWIN. A rota por blocos de CARA, UPF e Oracle não faz essa verificação. A análise suplementar e outras etapas do caso também não ficam cobertas por um prazo global desse campo. O launcher corretivo não aplica timeout ao subprocesso. Um limite de duração do processo inteiro depende de supervisão externa. Interrupções e falhas de memória impedem a conclusão da unidade; não autorizam apresentar resultados parciais como cobertura integral.

## Organização e integridade dos arquivos

O [manifesto científico](manifest.json) é o ponto de entrada para contagens, escopos, população comum, relatórios, previsões e arquivos tabulares. Ele preserva os bytes produzidos pelo exportador e o contrato `table_files`, sem incorporar depois o README ou as provas complementares. Os nomes físicos das tabelas devem ser obtidos desse campo, pois uma tabela lógica pode estar em arquivo único, em gzip ou dividida em partes comprimidas.

O [manifesto da distribuição](package-manifest.json) reúne o inventário completo dos arquivos distribuídos, incluindo este README e o conteúdo de `validation/`, além das proveniências do código e do pós-processamento. Seu inventário evita uma referência de hash ao próprio arquivo. O [guia das verificações](validation/README.md) orienta a leitura das provas. O `validation/index.json` é preservado como índice do controle inicial, sem ser apresentado como lista completa das verificações adicionadas posteriormente.

| Conteúdo lógico | Uso |
| --- | --- |
| `coverage.csv` | Situação e identidade dos 60 casos da matriz |
| `intentional-exclusions.csv` | Exclusões deliberadas, incluindo os 12 casos Online adiados |
| `paired-by-seed.csv` | Métricas da população comum e denominadores por semente |
| `aggregate-metrics.csv` | Médias e desvios dos escopos originais e da população comum |
| `replication-by-seed.csv` e `replication-aggregate.csv` | Séries e agregados dos trabalhos UPF e por pares |
| `metrics-long.csv` | Valores escalares, séries e matrizes exportados dos relatórios |
| `vector-index.csv` | Localização dos vetores densos preservados integralmente nos JSONs |
| `reports/`, `replications/` e `predictions/` | Relatórios completos e ledgers de predição, com compressão gzip quando indicada |
| `validation/` | Provas de execução, consistência, repetibilidade e limites das verificações |

Quando `table_files` trouxer `format=csv+gzip-shards`, as partes devem ser lidas na ordem de `shards`. Cada parte repete o cabeçalho e fica abaixo de 64 MiB comprimidos. Para reconstruir a tabela lógica, conserve somente o primeiro cabeçalho e mantenha a ordem dos registros. As faixas `source_row_range` usam índices de dados a partir de zero, com fim exclusivo, sem contar os cabeçalhos repetidos.

Os vetores densos não são transformados em milhões de linhas repetidas nos CSVs. O catálogo informa o arquivo, o `json_pointer`, o tipo de conteúdo, o comprimento e o hash do vetor. Seus valores continuam completos no JSON correspondente, que pode ser descomprimido sem reexecutar o experimento. Índice é uma forma de acesso à evidência, não uma redução do seu conteúdo.

Os hashes têm objetos diferentes. `source_report_sha256` identifica os bytes do relatório original; `retained_report_sha256` identifica os bytes gzip retidos após a normalização de informações de máquina e conexões. O hash do vetor é calculado sobre sua representação JSON canônica, e não sobre o arquivo gzip nem sobre os bytes originais do relatório. Um hash normalizado não deve ser tratado como o SHA dos bytes brutos. Os relatórios preservam o conteúdo científico numérico, enquanto informações privadas de conexão e caminhos de máquina são removidas.

O [arquivo de checksums](SHA256SUMS) cobre a distribuição completa, incluindo o manifesto da distribuição, e permite conferir os bytes distribuídos. A representação gzip conserva o conteúdo integral de cada payload normalizado. A presença de um índice ou de partes CSV não substitui o manifesto como contrato de leitura.

Nenhum peso, checkpoint ou estado persistente de treino foi incluído. Os modelos permaneceram em memória durante os cálculos. Uma unidade interrompida reinicia sua avaliação; uma publicação interrompida pode reutilizar a evidência já concluída após conferir sua identidade e seus hashes.

## Verificações disponíveis e seus limites

A [prova de repetibilidade](validation/reproducibility/reproducibility-98a6dd5.json) compara 30 pares de execuções de smoke e repetição em processos separados com a revisão executada. Esses pares usam SEA com 1.200 exemplos, dois modelos, cinco estratégias e as três sementes. Houve igualdade exata dos campos científicos comparados, incluindo os 27.600 pares de registros do ledger. Tempos, recursos e custos dependentes de duração foram excluídos da igualdade. Esse controle não é uma repetição integral dos 60 casos INSECTS.

Os [seis controles integrais](validation/reproducibility/full-controls-98a6dd5.json) comparam casos INSECTS de regressão logística entre a revisão de controle e `98a6dd5`, com 1.654.130 pares de registros do ledger. A adição correta da AUROC por regime foi conferida contra a área da curva ROC antiga. O controle não cobre Hoeffding Tree e não demonstra, sozinho, repetibilidade de todas as configurações, sementes ou perfis de recursos da campanha final.

A [verificação dos resultados UPF e por pares](validation/upf/prova-auditoria-upf-98a6dd5.json) realizou 91.742 verificações em oito trabalhos e 24 ensaios, sem divergências dentro dos critérios registrados. Foram conferidos cobertura, partições, denominadores, agendas, custos, oráculos, métricas derivadas das matrizes de confusão e agregados. Isso não equivale a recalcular todas as métricas probabilísticas.

Uma [verificação adicional das curvas retidas](validation/upf/probability-curve-consistency.json) fez 8.880 verificações. Nos 360 holdouts dos candidatos, recuperou a precisão média e reconstruiu 359 áreas ROC; um holdout tinha uma única classe, deixando a AUROC indefinida. O exame de 2.712 linhas de política e preço identificou métricas agrupadas armazenadas somente como escalares. Sem as probabilidades ou o ranking agrupado, AP e AUROC dessas agendas não puderam ser reconstruídas. Também não foram reconstruídos Brier, log loss ou erros de calibração. Curvas de candidatos em blocos distintos não permitem recuperar esses valores simplesmente por média ou concatenação.

Os seis processos do [microbenchmark formal](validation/microbenchmark/batch-scores-formal-summary.json) e o [recibo de ativação do caminho nativo](validation/microbenchmark/native-scorer-activation.json) têm escopos próprios. Eles verificam uma operação de pontuação e suas condições controladas. Seus tempos não medem o pipeline de ponta a ponta e não devem ser divulgados como aceleração da campanha integral.

A [verificação de código antes da publicação](validation/postprocessing-checks.json), na revisão `07aa7f5`, registra 1.238 testes aprovados e 1.503 avisos em 83,66 segundos, além das verificações de formato e lint. Os avisos já integravam a execução de referência. Também foram registrados 50 testes focados no pós-processamento. O [preflight com os resultados reais](validation/execution-times/real-replications-preflight.json) conferiu os oito trabalhos, os 24 ensaios e 48 relatórios nativos, sendo 24 de candidatos e 24 de políticas. Testes, conclusão do cálculo e verificação do tracking são evidências diferentes. Um `recorded_metric_points` ausente ou nulo no catálogo não equivale a zero pontos de métricas nativas: a contagem deve ser conferida no escopo e no backend efetivamente auditados.

A [auditoria da publicação nativa](validation/mlflow/native-mlflow-verification-summary.json) conferiu 108 execuções principais finalizadas e 1.032 ajustes filhos, com 20.912.990 registros de métricas nativas. No escopo auditado, não houve pontos ausentes, duplicações de coordenadas, divergências de valores ou pontos extras. A maior série tinha 450.852 pontos; 2.304 séries ultrapassaram 250 pontos. A conferência também confirmou os 60 tempos derivados da revisão `07aa7f5`.

Foram verificados os 108 arquivos ZIP publicados e os hashes de seus 10.968 membros, que somam 24.081.968.589 bytes descomprimidos. Não foram encontrados modelos, checkpoints ou dados brutos com os nomes proibidos nos membros verificados. Os 108 diretórios de origem não continham estado persistente de modelo, e os 1.032 ajustes filhos tinham zero objetos de artefato. O [relatório detalhado](validation/mlflow/final98-native-mlflow-audit.json) delimita a lista de execuções consultadas e a verificação dos arquivos conhecidos. Esse resultado comprova a integridade do material conferido, sem transformar a inspeção em nova avaliação de inferência ou prova de ausência de todo viés metodológico.

A [conferência independente da exportação](validation/export-qa/qa-evidence.json) foi aprovada, com `complete=true` e nenhuma falha. Ela verificou os 60 casos, os oito trabalhos e 24 ensaios, a correspondência exata dos 68 payloads normalizados com suas fontes, os 5.084 vetores catalogados e 19.337.586 linhas de ledger comparadas aos registros originais. Também conferiu as populações comuns, os cálculos dos agregados, as indisponibilidades, os hashes e os contratos das partes CSV. Essa conferência valida a transformação e a integridade da evidência exportada; não é nova execução de modelos.

A [conferência visual final](validation/export-qa/qa-figuras-final.json) abriu individualmente os 12 PNGs, sendo quatro figuras F1 das populações comuns e oito figuras de custo UPF. Não foram identificados defeitos visuais materiais no escopo registrado. As figuras de custo continuam separadas das métricas probabilísticas de classificação.

## Relação com o histórico

A correção simultânea da disponibilidade de rótulos e do replay, a adoção de fontes integrais e a mudança de populações avaliadas impedem atribuir qualquer diferença histórica a um único fator. Uma comparação posterior deve explicitar protocolo, revisão, dados, seed, população pontuada, recursos e escopo da métrica.

O [resumo de recuperação histórica](validation/historical-recovery.json) registra correspondência de SHA-256 para uma das três fontes do manifesto, o ARFF de Electricity. Essa correspondência identifica o conteúdo, sem comprovar isoladamente qual cópia foi lida por um processo histórico. Foram conferidos três recibos existentes contra os hashes de seus relatórios; eles pertencem a campanhas distintas e não comprovam execução ou implantação na AWS, nem completam os comprovantes específicos da tarefa #83 ou das execuções históricas INSECTS.

Os dois CSVs históricos e o launcher específico continuam não recuperados no escopo inspecionado. As ações [#101](https://github.com/xxggabriel/tarefas/issues/101) e [#83](https://github.com/xxggabriel/tarefas/issues/83) permanecem abertas. Também não foram recuperados, nessa etapa, o comando histórico exato e todo o contexto de hardware e concorrência. Enquanto essas lacunas não forem resolvidas e verificadas, esta campanha não comprova equivalência com aqueles resultados nem substitui os arquivos ausentes.
