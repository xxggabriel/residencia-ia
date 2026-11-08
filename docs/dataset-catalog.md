# Datasets e protocolos da Semana 09

> **Atualização em 08/11/2026:** Stream51 foi removido do projeto (catálogo, adaptador, lock e DAG), porque a origem oficial `klab.cis.rit.edu` continuou sem resolver DNS e não há cópia oficial disponível. O catálogo e o gate de admissão passaram a exigir 17 fontes; os registros abaixo preservam o estado de 04/11/2026.

Foram implementados geradores com drift explícito, admissão de dados locais por manifesto e transformações de protocolo sobre amostras preparadas. O arquivo `configs/dataset-catalog.json` registrou apenas os candidatos nomeados no pedido e no relatório. Ele não substituiu o catálogo de 47 registros da planilha indisponível e não representou uma campanha executada.

## Contratos e fronteiras

`MovingSyntheticSource` produziu os `Event` binários já consumidos pelo motor. `FriedmanDriftSource` e fontes locais multiclasse/regressão produziram `TaskSample` por `samples_iter()`. Nesses casos, `events()` rejeitou explicitamente o uso no motor binário, preservando os alvos. Não foi feita conversão de seis espécies ou gases para `bool`, nem de alvos Friedman para classes.

`TaskSample` preservou ID, sequência, timestamp, disponibilidade de rótulo, features, alvo, task ID, split e grupo. As funções de protocolo não treinaram modelos. Foi entregue um avaliador tipado separado do motor binário para regressão e amostras multiclasse preparadas, descrito abaixo.

## Matriz de implementação

| Componente | Fonte e versão | Implementação | Testes novos | Limite |
| --- | --- | --- | --- | --- |
| SEA, AGRAWAL e STAGGER | River 0.26.1, `SyntheticSource` existente | Conceitos em segmentos, com recorrência AGRAWAL | Suíte existente | Controles sintéticos |
| Rotating Hyperplane | [River Hyperplane 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/datasets/synth/hyper_plane.py), Hulten et al. 2001 | Nativa, features móveis e magnitude positiva obrigatórias | Replay, seeds, alvos, rejeição de gerador estático | Movimento contínuo; sem pontos discretos de drift |
| Random RBF Drift | [River RandomRBFDrift 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/datasets/synth/random_rbf.py) | Nativa, velocidade e centroides móveis positivos obrigatórios, duas classes | Replay, seeds, parâmetros e alvos | Movimento contínuo; sem F1 de detector por pontos inventados |
| Friedman Drift | [River FriedmanDrift 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/datasets/synth/friedman.py), Ikonomovska et al. 2011 | Nativa, modos `gra`, `lea`, `gsg`, posições dentro do recorte; avaliador LinearRegression | Replay, alvo contínuo, posições, transições, treino/avaliação e rejeição binária | Baselines tipadas; sem políticas de custo/drift de regressão |
| INSECTS | [Repositório dos autores](https://sites.google.com/view/uspdsrepository), Souza et al. 2020 | CSV/ARFF local, variante e classes declaradas, sem download | Fixtures de schema, hashes, ordem, licenças e alvos | Variante real e anotações ainda precisam ser admitidas |
| Elec2 | OpenML 151 v1, `RealSource` existente | Prefixo com hash fixado, relógio simulado | Suíte existente | Ordem publicada; sem drift verdadeiro anotado |
| Gas Sensor Array Drift | [UCI 224](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset), atualização batch10 de 14/10/2013 | Leitura LIBSVM local em ordem dos lotes e preservação de classes | Fixture multiclasse com dois lotes | Dados reais não baixados; direitos precisam de revisão |
| Split MNIST/CIFAR | [MNIST](https://yann.lecun.com/exdb/mnist/), [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) | Partição de classes declarada sobre amostras preparadas | Preservação de classes/splits, IDs duplicados e tarefas | Não é loader de arquivos IDX/pickle/imagens brutas |
| Permuted MNIST | Protocolo artificial, permutação fixada pela seed | Mesma permutação por tarefa em todos os splits, tarefa zero identidade | Replay, transformação, preservação de entrada/split | Features precisam de ordem explícita; sem cronologia natural |
| CORe50 | [Autores, CoRL 2017](https://vlomonaco.github.io/core50/) | Grupos/sessões explícitos com preservação de splits e ordem interna | Ordem por grupo e IDs | Definir cenário NI/NC/NIC e run oficial antes de executar |
| Stream51 | [Implementação dos autores](https://github.com/tyler-hayes/Stream-51) | Grupos e ordem oficiais fornecidos por manifesto preparado | Contratos temporais genéricos | Preparar frames e protocolo de novidade; não reproduziu métricas originais |
| CLEAR10/CLEAR100 | [Autores, NeurIPS 2021 e releases 2022](https://clear-benchmark.github.io/) | Buckets temporais e splits declarados | Contratos temporais genéricos | Selecionar versão e streaming/iid; preservar direitos por imagem |

Os testes acima foram entregues para execução inicial exclusivamente no homelab; os resultados observados ficaram no relatório de validação consolidado. Fixtures pequenas não constituíram homologação com os datasets originais.

## Admissão local

Foi criado `LocalDatasetManifest` estrito, sem campos desconhecidos. Ele exigiu nome, variante, versão, fonte HTTPS, licença e sua referência, confirmação explícita de acesso, formato, arquivos relativos com SHA-256 e cardinalidade, schema, target, classes e descrição da ordem. Para tarefas binárias, exigiu exatamente dois rótulos e o rótulo positivo. Campos de alvo, tempo, split e grupo foram excluídos das features para prevenir leakage.

O adaptador verificou todos os hashes antes de emitir amostras e preservou a ordem dos arquivos e das linhas. `validate_all()` conferiu integralmente schema, cardinalidades e ordem temporal. `samples_iter()` leu um prefixo limitado e não validou o sufixo; antes de uma campanha, executar `validate_all()` explicitamente. A origem e o hash do arquivo bruto continuaram necessários mesmo após conversão para CSV/JSONL; registrar a transformação em `preparation` e manter o manifesto do original junto ao manifesto preparado.

Não houve escrita em cache, download, extração de arquivo nem descoberta automática de credenciais. O cache foi a própria pasta do manifesto; caminhos e symlinks que escapassem dela foram rejeitados. A identidade da variante dependeu do conteúdo do manifesto, recorte, relógio e atraso. A seed foi registrada como ignorada em fontes reais, pois elas não foram embaralhadas.

Os formatos suportados foram CSV com cabeçalho único, ARFF denso, LIBSVM com todos os campos declarados e JSONL com um objeto plano por linha. Missing values, features não finitas, classes não declaradas e schemas divergentes foram rejeitados. Para Gas, declarar features `"1"` a `"128"`, classes `"1"` a `"6"` e lotes na ordem 1 a 10. O parser não completou features omitidas com zero. Esse comportamento exigiu um manifesto fiel à representação escolhida.

Timestamps de origem, quando declarados, foram inteiros não negativos e não decrescentes. Sem eles, utilizou-se relógio ordinal simulado, nunca descrito como calendário de coleta. O motor binário exigiu timestamps estritamente crescentes e um fluxo sem splits de teste misturados. O atraso de rótulos foi explicitamente simulado em ambos os casos.

O repositório USP declarou CC BY 4.0 e ressalvou direitos próprios de fontes externas. A página UCI de Gas apresentou uma restrição histórica a pesquisa no texto descritivo; a revisão de direitos deve resolver a licença aplicável ao arquivo adquirido, sem inferir autorização pela disponibilidade pública. Nenhum aceite foi feito por esta implementação.

## Preparação e execução futura

1. Obter a planilha e reconciliar os registros e variantes selecionados com esta matriz.
2. Selecionar a fonte oficial e registrar versão imutável, licença, recorte, hashes dos arquivos brutos e transformados, orçamento e aprovação de acesso.
3. Preparar arquivos locais sem treinamento de normalizadores no teste ou futuro. Preservar splits oficiais, sessões, vídeos e buckets; não separar frames correlacionados aleatoriamente entre treino e teste.
4. Criar um manifesto `LocalDatasetManifest`; executar `LocalManifestSource(...).validate_all()` no homelab e arquivar o recibo.
5. Usar `split_class_tasks`, `permuted_tasks` ou `temporal_tasks` com ordens oficiais versionadas. Registrar task ID disponível ou oculto, passagens, replay, pré-treino e orçamento.
6. Executar primeiro um smoke de poucas amostras. Executar avaliações por tarefa com modelo compatível, forgetting/retention e near-future accuracy somente após fixar o protocolo.

CORe50 separou sessões oficiais de treino/teste; o site dos autores descreveu sessões 3, 7 e 10 reservadas ao teste. Stream51 incluiu ordenações por instância e por classe/instância e avaliação de novidade; a ordenação deve acompanhar o recorte. CLEAR distinguiu avaliação iid de streaming no futuro próximo e publicou releases distintas. As transformações implementadas preservaram esses metadados quando fornecidos, sem inventar um protocolo oficial ausente.

Covertype não recebeu cronologia natural. CGLM/CLOC/BudgetCL e CLAD permaneceram candidatos sujeitos a definição de escopo, licença e recursos; não receberam loaders fictícios. Downloads ou campanhas de grande volume permaneceram pendentes.

## Avaliador tipado executável

`evaluation/typed.py` implementou `run_typed(source, root, *, task=None, policy="online", warmup=20, every=50, window=128, max_samples=1000, max_seconds=60, training_budget=None)`. O tipo veio de `source.manifest()["task"]` e foi conferido contra o tipo solicitado. Foram aceitas regressão e classificação multiclasse, com amostras de `samples_iter()`.

O baseline de regressão utilizou `StandardScaler` e `LinearRegression` nativos do River 0.26.1; o scaler foi atualizado somente na aprendizagem. A classificação utilizou `HoeffdingTreeClassifier` com profundidade e memória limitadas. As métricas foram MAE/RMSE e acurácia/MacroF1, respectivamente. A primeira predição multiclasse, antes de qualquer aprendizagem, foi registrada como `__untrained__` e contada como erro nas partições avaliadas.

Foram implementadas as políticas congelada após warm-up, online e periódica por contagem de rótulos de treino liberados. O replay foi limitado por janela e o orçamento de treino excluiu o warm-up. Na política periódica, um candidato foi treinado em uma instância nova e promovido automaticamente como baseline. O avaliador não implementou políticas de drift/custo nem promoção por validação para esses tipos de tarefa.

Cada predição precedeu a liberação de rótulos no timestamp corrente; rótulos atrasados não alteraram predições já feitas. Somente exemplos com split `train` foram aprendidos, inclusive no warm-up e no replay. Teste e validação produziram métricas separadas e nunca alimentaram treino. A avaliação foi prequential intercalada: o modelo pôde aprender rótulos anteriores de treino entre duas predições de teste. Ela não foi descrita como um holdout temporal com modelo congelado. A contagem de rótulos disponíveis em cada predição foi persistida para auditoria. No fim do prefixo houve drenagem explícita dos rótulos pendentes.

Foram persistidos `samples.jsonl`, `predictions.json` e `report.json`, com identidade derivada de configuração, dados, fonte, ambiente e código. Execuções completas foram idempotentes; uma falha antes do relatório final exigiu reinício do prefixo, sem checkpoint parcial. O prefixo foi limitado a 10 mil amostras, 64 MiB e 600 segundos no máximo; os defaults foram menores. Foram registrados tempos de treino/predição e pico RSS do processo inteiro, incluindo dependências e harness. Os tempos medidos não compuseram o digest de predições.

Os resultados por task ID foram métricas prequential por tarefa. Eles não constituíram uma matriz de acurácia após cada tarefa e não sustentaram cálculo de forgetting. Protocolos CL que reagruparam ou repetiram os dados precisaram fornecer uma sequência de replay contígua e um relógio explicitamente artificial, preservando IDs e os metadados originais em sua preparação. Não se afirmou execução de redes de visão, treino de CORe50/Stream51/CLEAR ou reprodução de resultados dos autores.
