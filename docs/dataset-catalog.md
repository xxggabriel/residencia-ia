# Datasets e protocolos de avaliação

O [registro executável de datasets](../src/continual_bench/datasets/catalog.py) define as 17 fontes e variantes com preparação por épocas e orienta o gate de admissão da DAG `benchmark_task_suite`. Cada entrada indica o tipo de alvo, o adaptador prequential e o loader de preparação. `split_cifar` e `clear` também estão disponíveis como aliases para manifestos locais.

O [contrato de candidatos da Semana 09](../configs/contracts/dataset-catalog.json) preserva o recorte da pesquisa original. O [relatório de preparação e execução](task-dataset-execution.md) registra os testes com dados reais e a remoção do Stream51 em 08/11/2026 devido à indisponibilidade da origem oficial.

## Contratos e fronteiras

`MovingSyntheticSource` produz os `Event` binários consumidos pelo motor. `FriedmanDriftSource` e fontes locais multiclasse ou de regressão produzem `TaskSample` por `samples_iter()`. O método `events()` admite apenas fontes binárias e rejeita os alvos multiclasse e de regressão.

`TaskSample` preserva ID, sequência, timestamp, disponibilidade de rótulo, features, alvo, task ID, split e grupo. As funções de protocolo preparam as amostras. O avaliador tipado treina e avalia modelos para regressão e classificação multiclasse.

## Matriz de implementação

| Componente | Fonte e versão | Implementação | Verificação | Limite |
| --- | --- | --- | --- | --- |
| SEA, AGRAWAL e STAGGER | River 0.26.1, `SyntheticSource` existente | Conceitos em segmentos, com recorrência AGRAWAL | Suíte existente | Controles sintéticos |
| Rotating Hyperplane | [River Hyperplane 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/datasets/synth/hyper_plane.py), Hulten et al. 2001 | Nativa, features móveis e magnitude positiva obrigatórias | Replay, seeds, alvos, rejeição de gerador estático | Movimento contínuo; sem pontos discretos de drift |
| Random RBF Drift | [River RandomRBFDrift 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/datasets/synth/random_rbf.py) | Nativa, velocidade e centroides móveis positivos obrigatórios, duas classes | Replay, seeds, parâmetros e alvos | Movimento contínuo; sem F1 de detector por pontos inventados |
| Friedman Drift | [River FriedmanDrift 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/datasets/synth/friedman.py), Ikonomovska et al. 2011 | Nativa, modos `gra`, `lea`, `gsg`, posições dentro do recorte; avaliador LinearRegression | Replay, alvo contínuo, posições, transições, treino/avaliação e rejeição binária | Baselines tipadas; sem políticas de custo/drift de regressão |
| INSECTS | Variante `abrupt_balanced`, Souza et al. 2020 | Loader CSV com ordem e classes originais e aquisição explícita | Fixtures de schema, hashes e ordem. Smoke com dados reais no homelab | Prefixo da variante selecionada. [Contrato e proveniência](real-streams.md) |
| Elec2 e Airlines | OpenML, versões do [lock](../configs/contracts/real-sources.lock.json) | `RealSource`, prefixo de cache verificado com relógio simulado | Suíte de fontes e smoke no homelab | Ordem publicada sem pontos de drift anotados |
| Gas Sensor Array Drift | UCI 224, atualização batch10 de 14/10/2013 | Loader ZIP/LIBSVM com lotes em ordem e classes originais | Fixtures do parser e smoke com dados reais no homelab | O recorte pode abranger só parte do primeiro lote. [Contrato e proveniência](real-streams.md) |
| Split MNIST, Split CIFAR10 e Split CIFAR100 | MNIST IDX e revisões Parquet CIFAR fixadas | Agrupamento por classe sobre imagens dos splits oficiais | Fixtures e smoke com dados reais no homelab | Prefixos dos splits, com cobertura parcial de classes. [Loaders de imagens](image-datasets.md) |
| Permuted MNIST | MNIST IDX, permutação fixada pela seed | Duas tarefas com IDs disjuntos e transformação consistente entre treino e teste | Replay, transformação e preservação de split | Protocolo com orçamento global de exemplos e relógio artificial. [Loaders de imagens](image-datasets.md) |
| CORe50 | Arquivo original 128×128 | Loader ZIP parcial com sessões oficiais e ordem interna | Fixtures e smoke com dados reais no homelab | Recorte diagnóstico NI. [Loaders temporais](temporal-vision.md) |
| CLEAR10 e CLEAR100 | Release de julho de 2022 | Loader ZIP parcial com buckets e splits oficiais | Fixtures e smoke com dados reais no homelab | Splits da release e direitos por imagem. [Loaders temporais](temporal-vision.md) |

Os resultados da validação inicial estão no [relatório de execução](task-dataset-execution.md). As evidências com dados reais referem-se ao recorte de 100 exemplos e duas épocas.

## Admissão local

O esquema `LocalDatasetManifest` exige nome, variante, versão, fonte HTTPS, licença e sua referência, aprovação de acesso, formato, arquivos relativos com SHA-256 e cardinalidade, features, alvo e descrição da ordem. Para classificação exige o vocabulário de classes. Tarefas binárias exigem exatamente dois rótulos e a indicação do positivo. Campos de alvo, tempo, split e grupo ficam fora das features para prevenir leakage.

O adaptador verifica os hashes na abertura e preserva a ordem dos arquivos e das linhas. `samples_iter()` lê um prefixo limitado. Execute `validate_all()` para conferir schema, cardinalidade e ordem temporal de todo o conteúdo. Quando houver conversão para CSV ou JSONL, registre a transformação em `preparation` e guarde a origem e o SHA-256 do arquivo bruto com o manifesto preparado.

O adaptador lê arquivos existentes na pasta do manifesto. Rejeita caminhos absolutos e links simbólicos que escapem da pasta. O conteúdo do manifesto, o recorte, o relógio e o atraso determinam a identidade da variante. A seed é registrada para auditoria e a fonte mantém a ordem original.

Os formatos suportados foram CSV com cabeçalho único, ARFF denso, LIBSVM com todos os campos declarados e JSONL com um objeto plano por linha. Missing values, features não finitas, classes não declaradas e schemas divergentes foram rejeitados. Para Gas, declarar features `"1"` a `"128"`, classes `"1"` a `"6"` e lotes na ordem 1 a 10. O parser não completou features omitidas com zero. Esse comportamento exigiu um manifesto fiel à representação escolhida.

Timestamps de origem, quando declarados, foram inteiros não negativos e não decrescentes. Sem eles, utilizou-se relógio ordinal simulado, nunca descrito como calendário de coleta. O motor binário exigiu timestamps estritamente crescentes e um fluxo sem splits de teste misturados. O atraso de rótulos foi explicitamente simulado em ambos os casos.

O repositório USP declarou CC BY 4.0 e ressalvou direitos próprios de fontes externas. A página UCI de Gas apresentou uma restrição histórica a pesquisa no texto descritivo; a revisão de direitos deve resolver a licença aplicável ao arquivo adquirido, sem inferir autorização pela disponibilidade pública. Nenhum aceite foi feito por esta implementação.

## Admissão de uma nova fonte por manifesto

1. Selecionar a fonte oficial e registrar versão imutável, licença, recorte, hashes dos arquivos brutos e transformados, orçamento e aprovação de acesso.
2. Preparar os arquivos com normalizadores ajustados somente no treino. Preservar splits oficiais, sessões, vídeos e buckets. Manter frames correlacionados dentro do mesmo split.
3. Criar um manifesto `LocalDatasetManifest`, executar `LocalManifestSource(...).validate_all()` e arquivar o recibo.
4. Quando necessário, usar `split_class_tasks`, `permuted_tasks` ou `temporal_tasks` com ordens versionadas. Registrar se o task ID está disponível, o número de passagens, o replay, o pré-treino e o orçamento.
5. Executar primeiro um smoke de poucas amostras. Fixar o protocolo antes de medir retenção, forgetting ou acurácia em um horizonte futuro.

Covertype, CGLM, CLOC, BudgetCL e CLAD ficaram no levantamento bibliográfico, fora da lista de fontes admitidas pela DAG.

## Avaliador tipado executável

O [avaliador tipado](../src/continual_bench/evaluation/typed.py) executa `run_typed` sobre amostras de `samples_iter()` para regressão e classificação multiclasse. O parâmetro `task` deve corresponder à declaração do manifesto. `model` seleciona uma família compatível no [registro de modelos](../src/continual_bench/models/__init__.py). O parâmetro `costs` contabiliza um cenário de custo e `progress` recebe métricas intermediárias.

O modelo padrão de regressão combina `StandardScaler`, atualizado durante a aprendizagem, com `LinearRegression` do River 0.26.1. O de classificação é `HoeffdingTreeClassifier` com profundidade e memória limitadas. As métricas são MAE e RMSE na regressão, acurácia e MacroF1 na classificação. Uma predição multiclasse ausente vira `__untrained__` e conta como erro quando a amostra participa da avaliação.

O avaliador oferece as políticas `none` (congelada após warm-up), `online` e `periodic` (por contagem de rótulos de treino liberados). O replay é limitado por janela e o orçamento de treino exclui o warm-up. Na política periódica, o aprendiz é reinicializado e treinado com a janela de replay.

Cada predição precedeu a liberação de rótulos no timestamp corrente; rótulos atrasados não alteraram predições já feitas. Somente exemplos com split `train` foram aprendidos, inclusive no warm-up e no replay. Teste e validação produziram métricas separadas e nunca alimentaram treino. A avaliação foi prequential intercalada: o modelo pôde aprender rótulos anteriores de treino entre duas predições de teste. Ela não foi descrita como um holdout temporal com modelo congelado. A contagem de rótulos disponíveis em cada predição foi persistida para auditoria. No fim do prefixo houve drenagem explícita dos rótulos pendentes.

O avaliador persiste `samples.jsonl`, `predictions.json`, `report.json` e artefatos de métricas e custos, com identidade derivada de configuração, dados, fonte, ambiente e código. Execuções completas são idempotentes. Uma interrupção antes do relatório final exige reiniciar o prefixo. Os limites são 10 mil amostras, 64 MiB de materialização e 600 segundos, com padrões de 1.000 amostras e 60 segundos. O relatório inclui tempos de treino e predição e o pico RSS do processo. O digest usa os registros de predição.

Os resultados por task ID registram métricas prequential calculadas à medida que chegam as amostras de cada tarefa. O cálculo de forgetting requer um protocolo com avaliação após cada tarefa. Protocolos que reagrupam ou repetem os dados devem fornecer uma sequência de replay contígua e um relógio explicitamente artificial, preservando os IDs e os metadados de origem.

## Catálogo dos snapshots tabulares na AWS

Os snapshots Social, Fraud e E-commerce do benchmark Zia são publicados no catálogo do SageMaker Unified Studio. Cada ativo identifica o dataset, a revisão da fonte, a versão da preparação e o escopo `full` ou `smoke`. O formulário de metadados referencia o manifesto e o Parquet no S3, com versão e hashes SHA-256.

A automação reage aos marcadores de conclusão dos snapshots e aos manifestos finais de campanhas AWS Batch Zia. Uma avaliação elegível precisa ter a identidade e os hashes do snapshot correspondentes, além de um run ativo, concluído e com métricas no MLflow. O catálogo reúne os links para os runs, e cada run vinculado traz a tag `cb-catalog-asset-url`. O [guia de operações AWS](aws/operations.md#publicação-automática-no-catálogo) descreve o caminho no Studio, a implantação e a recuperação de eventos.
