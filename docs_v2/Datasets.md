# Datasets

Esta página descreve os conjuntos de dados que o pacote `continual-bench` sabe abrir: origem, versão fixada, tipo de tarefa, preparação, divisão temporal e limites de amostras. A referência é o código em `src/continual_bench/datasets/` e os contratos em `configs/contracts/`.

## Visão geral

O pacote tem dois registros de dados que se complementam:

- **Catálogo executável** (`src/continual_bench/datasets/catalog.py`): declara cada fonte com tipo de alvo (`binary`, `multiclass` ou `regression`), família e duas fábricas, uma para o fluxo prequential e outra para a preparação limitada por épocas. Prequential significa que cada exemplo é primeiro usado para testar o modelo e só depois, quando o rótulo é liberado, pode ser usado para treino.
- **Registro de streams ordenados** (`src/continual_bench/datasets/streams.py`): converte fontes em pares `<dataset>/<escopo>.parquet` e `<dataset>/<escopo>.json` (manifesto com contagem de linhas e SHA-256) para o avaliador por blocos. Ele resolve um nome na ordem Zia, fontes reais fixadas e, por fim, o catálogo.

O comando `uv run continual-bench components` lista as entradas do catálogo com tarefa e família.

### O gate de 17 entradas

`DATASETS` em `catalog.py` reúne as entradas que têm carregador limitado para épocas. São 17:

| Família | Entradas | Tarefa |
| --- | --- | --- |
| `river_abrupt` | `sea`, `agrawal`, `stagger` | binária |
| `river_moving` | `hyperplane`, `rbf_drift` | binária |
| `river_regression` | `friedman_drift` | regressão |
| `openml` | `electricity`, `airlines` | binária |
| `real_stream` | `insects`, `gas` | multiclasse |
| `image` | `split_mnist`, `permuted_mnist`, `split_cifar10`, `split_cifar100` | multiclasse |
| `temporal_vision` | `core50`, `clear10`, `clear100` | multiclasse |

Fora do gate ficam quatro entradas com `prepare_kind="none"`: as duas variantes binárias desbalanceadas de INSECTS (família `usp_stream`) e os aliases `split_cifar` e `clear`, que apenas apontam para manifestos locais cuja tarefa é declarada na admissão. O protocolo de admissão por épocas e o seu gate estão em [Metodologia Experimental](Metodologia-Experimental).

`load_dataset` aceita de 10 a 10.000 exemplos por fonte e, para as fontes que encapsulam um fluxo, aplica atraso de rótulo de 3 unidades (`PREPARED_LABEL_DELAY`) e relógio iniciado em `1735689600` (`EPOCH`).

**Estado:** Implementado. Validado por `tests/test_components.py`, `tests/test_image_datasets.py`, `tests/test_temporal_vision.py`, `tests/test_real_stream_datasets.py`, `tests/test_insects_datasets.py` e `tests/test_real_sources.py`. Em 4 de novembro de 2026, um smoke com 100 exemplos originais distintos e duas épocas por fonte aprovou as 17 fontes atuais (Stream51, então a 18ª, ficou bloqueado). A consolidação, com o hash dos fontes `c12041fa90c65452d7abe64e66d6f0c3bdfd2a6aafd9c79254224d63f7912d30`, está no histórico Git: `git show f835ee8:reports/semana09-datasets/consolidation.json`.

## Geradores sintéticos

Todos usam os geradores do River 0.26.1 (`river.datasets.synth`, licença BSD-3-Clause da implementação) e um relógio ordinal simulado (`timestamp = epoch + sequência`). Implementação em `src/continual_bench/datasets/synthetic.py`.

| Entrada | Construção | Mudanças de conceito |
| --- | --- | --- |
| `sea` | `synth.SEA`, variantes 0, 1, 2 | Três segmentos iguais, fronteiras em `samples // 3` e `2 * samples // 3`; cada segmento usa a seed `seed + i` |
| `stagger` | `synth.STAGGER`, funções 0, 1, 2 | Mesmos três segmentos |
| `agrawal` | `synth.Agrawal`, funções 0, 1, 0 | Três segmentos com recorrência do primeiro conceito |
| `hyperplane` | `synth.Hyperplane`, padrões `n_features=10`, `n_drift_features=2`, `mag_change=0.001`, `noise_percentage=0.05`, `sigma=0.1` | Movimento contínuo, sem pontos discretos (`changepoints=None`) |
| `rbf_drift` | `synth.RandomRBFDrift`, duas classes, `n_features=10`, `n_centroids=20`, `n_drift_centroids=10`, `change_speed=0.001` | Movimento contínuo, sem pontos discretos |
| `friedman_drift` | `synth.FriedmanDrift`, modos `gra`, `lea` (três posições) e `gsg` (com janela de transição) | Posições padrão em frações iguais do recorte; o novo regime começa em `seq = posição` |

O wrapper de Hyperplane e RBF recusa configurações estáticas (magnitude ou velocidade de mudança nulas). FriedmanDrift é regressão: `events()` lança erro e o motor binário rejeita a fonte. Os geradores abruptos expõem *probes* (amostras de referência por conceito, com seeds independentes) para métricas de retenção.

As referências de origem citadas na documentação são *Mining Time-Changing Data Streams* (2001), para Hyperplane, e *Learning model trees from evolving data streams* (2011), para FriedmanDrift. Veja [Referências](Referencias).

## Fontes tabulares reais

### Electricity e Airlines (OpenML)

Fixadas em `src/continual_bench/datasets/openml.py` e `configs/contracts/real-sources.lock.json`. Não há download implícito: a preparação usa `scripts/prepare_real_sources.py`.

| Fonte | OpenML (id, versão) | Linhas | Bytes | SHA-256 do ARFF | Alvo e classe positiva |
| --- | --- | ---: | ---: | --- | --- |
| `electricity` | 151, v1 | 45.312 | 3.092.013 | `2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2` | `class`, positivo `UP` |
| `airlines` | 1169, v1 | 539.383 | 19.168.157 | `c7683a90b40c2cabb1ddc950413da8efa231c058327b8c95511c40af13516c14` | `Delay`, positivo `1` |

A admissão confere SHA-256, colunas, tipos nominais, categorias do alvo e contagem de linhas. O cabeçalho de Electricity declara normalização e substituição de ausentes feitas pelo publicador, sem escopo de ajuste documentado; Airlines não recebe normalização adicional no adaptador. A licença registrada é o rótulo do catálogo OpenML (`Public`), sem inferir uma licença SPDX. As fontes não têm fronteiras de drift anotadas.

### INSECTS balanceado e Gas Sensor Array Drift

Implementação em `src/continual_bench/datasets/real_streams.py`, função `prepare_real_stream(name, cache, limit=100, seed=42, *, download=False)`; contrato em `configs/contracts/real-streams.lock.json`.

| Fonte | Origem e versão | Contrato |
| --- | --- | --- |
| `insects` | Arquivo `abrupt_balanced.csv` do USP Data Stream Repository, mapeado pelo River 0.26.1 | 52.848 linhas, 33 atributos, 6 classes, 14.151.769 bytes |
| `gas` | UCI 224, atualização do lote 10 de 14/10/2013 | 13.910 linhas, 128 atributos, 6 classes, dez lotes na ordem 1 a 10; ZIP limitado a 16 MiB e conteúdo descomprimido a 64 MiB |

O lock não traz SHA-256 publicado para essas duas fontes: a primeira aquisição calcula o hash do arquivo inteiro e o fixa no recibo local `acquisition.json`. A seleção é o prefixo publicado, sem embaralhamento; a seed é registrada, mas ignorada. Gas registra o lote como `task_id`. Nenhuma das duas declara pontos de drift (`changepoints=None`). Licenças registradas: CC BY 4.0 do repositório USP, com ressalva sobre fontes externas; para Gas, a página UCI indica CC BY 4.0, mas seu texto descritivo menciona restrição a uso em pesquisa; o código registra as duas informações (`license_note` em `real_streams.py`) e nenhum aceite de termos é feito automaticamente.

### INSECTS binário desbalanceado

Duas variantes do mesmo repositório USP, usadas no benchmark binário e na suíte de streams (classe `InsectsBinarySource`). O rótulo original `2` vira positivo (`1`) e as outras cinco classes viram negativo (`0`).

| Entrada | Arquivo | Linhas | SHA-256 | Fronteiras anotadas (índice inicial do novo regime) |
| --- | --- | ---: | --- | --- |
| `insects_abrupt_imbalanced` | `abrupt_imbalanced.csv` | 355.275 | `985a5b46c2cb0e42e48a78d5001b57a79ed64fa69f9e5b0362a34cfb552dbc22` | 83859, 128651, 182320, 242883, 268380 |
| `insects_incremental_abrupt_reoccurring_imbalanced` | `incremental_abrupt_imbalanced.csv` | 452.044 | `d07e22c0521755f2452d96bb1c7068b26a281be2a627e32f21a4c0c6cfbfd9bc` | 150683, 301365 |

As fronteiras e a distribuição de classes por regime estão em `real_streams.py` e no lock. Elas servem só para avaliação e auditoria: não entram como atributo, task ID ou sinal para as políticas. A origem das fronteiras é a Tabela 2 de *Challenges in Benchmarking Stream Learning Algorithms with Real-world Data* (2020).

### Snapshots Zia Data Labs

Três datasets sintéticos publicados no Hugging Face pela Zia Data Labs, fixados por revisão e SHA-256 em `src/continual_bench/datasets/zia.py`. O manifesto gerado pela preparação registra a licença `CC-BY-NC-4.0` (`zia.py`, campo `license`); confira os termos vigentes na página de cada dataset no Hugging Face antes de redistribuir.

| Entrada | Repositório e revisão | Linhas | Classes | Modelo padrão |
| --- | --- | ---: | --- | --- |
| `fraud` | `FreeFraudDetection50M`, `8d2baf07…` | 50.000.000 | `legitimate`, `fraud` | XGBoost (`scale_pos_weight=39`) |
| `social` | `FreeSocialNetwork100M`, `8518bf04…` | 100.000.000 arestas (3.000.000 nós) | `follow`, `friend`, `block`, `mute` | Regressão logística multinomial |
| `ecommerce` | `FreeEcommerceOrders50M`, `d4f02e6b…` | 50.000.000 | `other_status`, `returned` | XGBoost (`scale_pos_weight=10`) |

A preparação (`scripts/prepare_zia.py <root> {fraud,social,ecommerce} {smoke,full} [--download]`) ordena por timestamp e, em empates, por ID. Em `fraud` e `ecommerce`, os atributos numéricos são gerados em SQL do DuckDB, e categorias viram códigos inteiros por hash. Em `social`, o DuckDB apenas projeta e ordena `event_id`, `event_time`, `label`, `src` e `dst`, enquanto os 7 atributos (`same_region`, `source_age`, `target_age`, `source_in`, `source_out`, `target_in`, `target_out`) são calculados em Python/NumPy (`social_features()`): os graus contam apenas arestas de datas estritamente anteriores (com transformação `np.log1p(count) / 10`), e as idades das contas são divididas por 3.650 dias. O escopo `smoke` usa as primeiras 10.000 linhas físicas da fonte.

**Estado:** Implementado. A suíte AWS atual não inclui Fraud, Social e E-commerce (veja abaixo).

## Imagens

### Split MNIST, Permuted MNIST e CIFAR

- MNIST usa os quatro arquivos IDX originais, fixados por tamanho e SHA-256 em `configs/contracts/image-datasets.lock.json` (11.594.722 bytes ao todo) com base nos checksums do TensorFlow Datasets. Implementação em `src/continual_bench/datasets/images.py`.
- Split CIFAR10 e Split CIFAR100 usam, no catálogo, os espelhos Parquet `uoft-cs/cifar10` (commit `0b2714987fa478483af9968de7c934580d0bb9a2`) e `uoft-cs/cifar100` (commit `aadb3af77e9048adbea6b47c21a81e47dd092ae5`), lidos por intervalos HTTP em `src/continual_bench/datasets/cifar_parquet.py`. Os hashes LFS completos ficam como referência não verificada, porque a leitura é parcial; o código confere o SHA-256 de cada intervalo e de cada imagem selecionada. Os carregadores binários oficiais de CIFAR continuam disponíveis em `images.py` para seleção explícita.

| Entrada | Protocolo |
| --- | --- |
| `split_mnist`, `split_cifar10` | Classes agrupadas em cinco tarefas de duas classes (`class_incremental`) |
| `split_cifar100` | Dez tarefas de dez classes; alvo fino preservado |
| `permuted_mnist` | `disjoint_budgeted_permuted_mnist`: imagens distintas alternadas entre uma tarefa identidade e outra com permutação de pixels fixada pela seed. Difere do protocolo usual, que repete o conjunto inteiro em cada tarefa |

O recorte seleciona 80% de imagens do treino oficial e 20% do teste oficial, sem embaralhar e sem mover imagens entre divisões. Pixels são divididos por 255 nas dimensões originais. O relógio é ordinal e artificial. Licenças: MNIST e CIFAR não declaram licença SPDX; o lock registra o texto da origem (CIFAR pede a citação de Krizhevsky 2009) e o espelho Parquet é registrado como `unknown`.

### CORe50, CLEAR10 e CLEAR100

Implementação em `src/continual_bench/datasets/temporal_vision.py`; contrato em `configs/contracts/temporal-vision.lock.json`.

| Entrada | Origem | Regra implementada |
| --- | --- | --- |
| `core50` | Arquivo original `core50_128x128.zip` dos autores (MD5 publicado `745f3373fed08d69343f1058ee559e13`, usado como referência) | Classe por objeto; treino nas sessões 1, 2, 4, 5, 6, 8, 9, 11 e teste nas sessões 3, 7, 10; ordem sessão, objeto, quadro |
| `clear10`, `clear100` | Release de julho de 2022, espelho Hugging Face na revisão `b518a845a98f1c913497ab98a19727cb65b74e65` | Buckets, classes e divisões oficiais; BACKGROUND é mantido quando presente |

O leitor usa intervalos HTTP sobre o ZIP e exige respostas 206. Cada imagem é convertida para RGB, redimensionada para 16 por 16 com interpolação bilinear e dividida por 255 (768 atributos). Não há extrator pré-treinado. O carregador por amostras aceita de 10 a 1.000 imagens; `prepare_temporal_arrays`, usado pelo benchmark iterativo, aceita até 10.000. CORe50 gera apenas um subconjunto diagnóstico do cenário NI, não um protocolo NI/NC/NIC completo.

## Manifestos locais

`LocalManifestSource` (`src/continual_bench/datasets/local_manifest.py`) admite dados próprios descritos por um `LocalDatasetManifest`: nome, variante, versão, fonte HTTPS, licença, aprovação de acesso, formato (CSV, ARFF denso, LIBSVM ou JSONL), arquivos com SHA-256 e cardinalidade, atributos, alvo e ordem. Hashes são conferidos na abertura, caminhos absolutos e links simbólicos que saem da pasta são rejeitados, e campos de alvo, tempo, divisão e grupo não podem ser atributos. `validate_all()` confere o conteúdo inteiro. As funções `split_class_tasks`, `permuted_tasks` e `temporal_tasks` (`src/continual_bench/datasets/protocols.py`) reorganizam amostras preparadas em tarefas.

## Streams ordenados para o avaliador por blocos

O avaliador por blocos (`src/continual_bench/evaluation/stream_benchmark.py`, também acessível como `continual_bench.evaluation.zia`) consome Parquets com as colunas `event_time`, `label` e atributos `float32`. Três provedores produzem esses arquivos:

| Provedor | Datasets | Tamanho de bloco | Modelo |
| --- | --- | --- | --- |
| `ZiaStreamProvider` | `fraud`, `social`, `ecommerce` | 250.000 linhas (`full`), 1.000 (`smoke`) | Declarado no contrato Zia |
| `PinnedRealStreamProvider` | `electricity`, `airlines` e as duas variantes INSECTS binárias | `max(1000, ceil(0,9 × linhas / 24))` em `full`; 200 em `smoke` (prefixo de 2.000 linhas) | XGBoost |
| `CatalogStreamProvider` | Demais entradas binárias e multiclasse do catálogo | 1.000 (`full`, 10.000 exemplos), 200 (`smoke`, 2.000 exemplos) | XGBoost se binária, logística multinomial se multiclasse |

`PinnedRealStreamProvider` grava a fonte inteira no escopo `full`, sem ajustar transformações: atributos nominais viram o índice da categoria no cabeçalho ARFF e numéricos são convertidos para `float32`. O manifesto declara que o relógio é ordinal simulado e que não se conhecem tempos reais de evento nem de disponibilidade de rótulos.

### Como os blocos são formados

1. O arquivo é lido em ordem, em blocos do tamanho declarado; exemplos com o mesmo timestamp ficam sempre no mesmo bloco (`chronological_blocks`). Os últimos `int(0,1 × linhas)` exemplos formam o holdout temporal (`HOLDOUT_FRACTION = 0.1`), avaliado com o modelo vigente e nunca usado para treino ou decisões.
2. O primeiro bloco treina o modelo inicial (aquecimento).
3. Cada bloco seguinte é previsto antes de seus rótulos serem liberados; os rótulos ficam disponíveis ao fim do bloco. Esse atraso é uma convenção do protocolo, não um atraso medido na fonte.
4. Retreinos usam no máximo os quatro blocos mais recentes (`REPLAY_BLOCKS = 4`).

Na campanha AWS registrada em `evidencias/aws-benchmarks/cases.csv`, os tamanhos efetivos foram:

| Stream | Linhas | Linhas por bloco | Linhas no holdout |
| --- | ---: | ---: | ---: |
| Electricity | 45.312 | 1.700 | 4.531 |
| Airlines | 539.383 | 20.227 | 53.938 |
| INSECTS abrupt imbalanced | 355.275 | 13.323 | 35.527 |
| INSECTS incremental abrupt reoccurring imbalanced | 452.044 | 16.952 | 45.204 |

As faixas de comparação comum de cada stream estão em [Metodologia Experimental](Metodologia-Experimental).

## O que foi executado com cada fonte

| Uso | Fontes | Estado e evidência |
| --- | --- | --- |
| Smoke de 100 exemplos e duas épocas (4/11/2026) | As 17 entradas do gate | Validado: `git show f835ee8:reports/semana09-datasets/consolidation.json` |
| Suíte de streams ordenados na AWS, 132 unidades | Electricity, Airlines e as duas variantes INSECTS binárias | Validado: `evidencias/aws-benchmarks/` (4 streams, 11 políticas, seeds 42, 43 e 44) |
| Matriz binária INSECTS da Semana 10 | Variantes INSECTS binárias | Parcial: 66 de 72 unidades arquivadas (`evidencias/semana-10/insects/`) |
| Benchmarks Zia | Fraud, Social, E-commerce | Implementado; fora dos pedidos AWS atuais |
| Suporte local da suíte com sete streams | As quatro acima mais Fraud, E-commerce, Social | Implementado (`scripts/run_dataset_suite.py`) |

## Limitações

- O relógio de todas as fontes preparadas é ordinal e simulado; não representa o calendário de coleta.
- Os recortes de 100 exemplos não cobrem todas as classes, lotes ou tarefas das fontes grandes; servem para validar carga e contrato, não desempenho.
- INSECTS balanceado e Gas não têm hash publicado pelos autores; o hash fixado é o observado na primeira aquisição.
- Os hashes completos dos Parquets CIFAR não são verificados pela leitura parcial.
- Stream51 foi removido do catálogo em 08/11/2026 porque a origem oficial não resolvia DNS. Covertype, CGLM, CLOC, BudgetCL e CLAD aparecem só no levantamento bibliográfico (**Planejado**, sem carregador).
- Detalhes adicionais em [Limitações e Trabalhos Pendentes](Limitacoes-e-Trabalhos-Pendentes).

Para modelos e políticas aplicados a esses dados, veja [Modelos e Políticas](Modelos-e-Politicas); para os detectores, [Detectores de Drift](Detectores-de-Drift).
