# continual-bench

O `continual-bench` é um pipeline de benchmark para políticas de treinamento e atualização de modelos e para detectores de drift. Ele foi desenvolvido para responder, com o mesmo código e sob protocolos explícitos, a uma pergunta prática: quando os dados mudam ao longo do tempo, qual estratégia de atualização mantém o modelo útil com menos esforço computacional e menor custo?

Para isso, o repositório executa fluxos de dados ordenados no tempo, aplica políticas como retreino periódico, retreino por alarme de detector, atualização incremental e políticas sensíveis a custo (CARA, UPF e Oracle), e registra desempenho preditivo, adaptação, retenção, recursos computacionais e custos de cada execução.

A documentação completa está na [Wiki do projeto](docs/Home.md).

## Visão geral

Um *drift* é uma mudança na distribuição dos dados ou na relação entre atributos e rótulo. O benchmark simula a operação de um modelo nesse cenário: cada exemplo (ou bloco de exemplos) é previsto antes de o rótulo ficar disponível, os rótulos chegam com atraso, um detector pode sinalizar mudanças e uma política decide se e como atualizar o modelo. As métricas são calculadas sobre faixas de avaliação comuns a todas as políticas, para que as comparações usem o mesmo denominador.

Todo o ciclo contínuo roda no núcleo Python (`src/continual_bench`). Serviços externos são opcionais e têm papéis delimitados: MLflow registra execuções, MinIO ou S3 guardam artefatos, Kafka transporta eventos, ZenML e Airflow disparam execuções finitas, e o perfil AWS escala campanhas com Step Functions e AWS Batch.

## Principais capacidades

As capacidades abaixo têm código executável e testes automatizados ou evidências de execução registradas no repositório.

- **Três caminhos de avaliação**: motor binário por evento com modelos do River, avaliador por blocos com XGBoost e regressão logística multinomial, e avaliadores tipados para regressão, multiclasse e treino por épocas.
- **Políticas de atualização**: sem atualização, online, periódica, por alarme de drift, por desempenho, incremental, CARA (três variantes), UPF e Oracle retrospectivo.
- **Detectores de drift**: ADWIN e DDM usados em campanhas; EDDM, HDDM-A, HDDM-W, Page-Hinkley e KSWIN implementados e cobertos por testes de contrato.
- **Datasets**: geradores sintéticos com drift, fontes reais com versão e hash fixados (Electricity, Airlines, INSECTS, Gas Sensor Array Drift) e fontes de imagem para aprendizado contínuo.
- **Protocolos com disponibilidade temporal de rótulos**: testar e depois treinar (por evento, por lote e por blocos), holdout temporal e avaliação por épocas.
- **Métricas** de qualidade, calibração, retenção, adaptação, custo e recursos, descritas em um catálogo versionado (`configs/contracts/metric-definitions.json`).
- **Custos em escopos separados**: custo de cenário parametrizado, estimativa de infraestrutura e alocação, sem misturá-los com faturamento observado.
- **Reprodutibilidade**: identidade de execução por hash da configuração, locks de dependências e de dados, checkpoints retomáveis e pacotes de evidência com `SHA256SUMS`.
- **Execução local, Docker Compose e AWS** com o mesmo código do avaliador. A campanha AWS registrada cobre 4 streams, 11 políticas e 3 seeds (`evidencias/aws-benchmarks/`).

Os testes rápidos e os *smokes* validam algoritmo, protocolo e integração, não desempenho. Nenhum resultado do repositório é uma alegação de estado da arte, e diferenças entre protocolos não provam superioridade de um método. As lacunas conhecidas estão em [Limitações e trabalhos pendentes](docs/Limitacoes-e-Trabalhos-Pendentes.md).

## Requisitos

- Python 3.12 (o projeto declara `>=3.12,<3.13`).
- [uv](https://docs.astral.sh/uv/) para instalar as dependências fixadas em `uv.lock`.
- Docker com Compose, apenas para o perfil com serviços.
- Conta AWS e GitHub Actions configurado com OIDC, apenas para o perfil AWS. Esse perfil provisiona recursos que geram custos.

## Primeiros passos

Instale as dependências de desenvolvimento e rode os testes rápidos:

```bash
uv sync --frozen --extra dev
```

```bash
make test
```

Liste os componentes disponíveis e execute uma configuração de exemplo:

```bash
uv run --frozen continual-bench components
```

```bash
uv run --frozen continual-bench run configs/runs/streaming.yaml
```

Execute o smoke local, que percorre três fontes sintéticas nos três modos de avaliação com políticas de referência e grava o resumo em `state/smoke/`:

```bash
make smoke
```

Para datasets reais, imagens, XGBoost e serviços, instale os extras correspondentes (`services`, `datasets`, `large-datasets`, `plots`). Os detalhes estão em [Instalação e configuração](docs/Instalacao-e-Configuracao.md) e [Execução local](docs/Execucao-Local.md).

## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| `src/continual_bench/` | Pacote Python: núcleo, configuração, datasets, modelos, detectores, políticas, avaliação, aplicação, infraestrutura e orquestração |
| `configs/` | Configurações de execução, matrizes, campanhas, contratos de métricas e locks de dados |
| `tests/` | Testes unitários e de integração |
| `scripts/` | Execução de campanhas, preparação de dados, exportação, verificação e figuras |
| `dags/` | DAGs manuais do Airflow |
| `infra/`, `docker/`, `compose*.yaml` | CloudFormation, Step Functions, Glue, Athena, SageMaker, imagens e perfis Docker Compose |
| `notebooks/` | Notebook e módulos que reconstroem figuras a partir das evidências |
| `evidencias/` | Pacotes de evidência versionados com manifestos e checksums |
| `docs/` | Documentação do projeto, organizada como Wiki |

## Documentação

| Tema | Página |
|---|---|
| Propósito, escopo e estado das capacidades | [Visão geral](docs/Visao-Geral.md) |
| Camadas e fluxo de uma execução | [Arquitetura](docs/Arquitetura.md) |
| Ambiente, dependências e variáveis | [Instalação e configuração](docs/Instalacao-e-Configuracao.md) |
| CLI, smoke, matrizes e Docker Compose | [Execução local](docs/Execucao-Local.md) |
| Perfil gerenciado na AWS | [Execução na AWS](docs/Execucao-na-AWS.md) |
| Fontes de dados, versões e divisões temporais | [Datasets](docs/Datasets.md) |
| Modelos e políticas de atualização | [Modelos e políticas](docs/Modelos-e-Politicas.md) |
| Detectores de drift | [Detectores de drift](docs/Detectores-de-Drift.md) |
| Protocolos, rótulos, calibração e coortes | [Metodologia experimental](docs/Metodologia-Experimental.md) |
| Definição, unidade e direção das métricas | [Métricas](docs/Metricas.md) |
| Escopos de custo e recursos computacionais | [Custos e recursos](docs/Custos-e-Recursos.md) |
| Seeds, identidade, locks e checksums | [Reprodutibilidade e artefatos](docs/Reprodutibilidade-e-Artefatos.md) |
| MLflow, evidências, notebooks e figuras | [Tracking e evidências](docs/Tracking-e-Evidencias.md) |
| Testes, CI e deploy | [Testes e CI/CD](docs/Testes-e-CI-CD.md) |
| Acompanhamento e recuperação de falhas | [Operação e diagnóstico](docs/Operacao-e-Diagnostico.md) |
| O que não está coberto | [Limitações e trabalhos pendentes](docs/Limitacoes-e-Trabalhos-Pendentes.md) |
| Artigos, datasets e bibliotecas | [Referências](docs/Referencias.md) |

Os links internos das páginas em `docs/` usam o formato do GitHub Wiki (sem a extensão `.md`). Eles funcionam quando a pasta é publicada como Wiki; na navegação pelo repositório, abra as páginas pelo índice acima ou pela [página inicial](docs/Home.md).
