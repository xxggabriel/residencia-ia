# Arquitetura modular do `continual_bench`

O pacote combina **datasets × modelos × políticas de retreinamento × detectores de drift**.
Os catálogos, avaliadores e casos de uso mantêm rastreáveis as configurações e as métricas das
comparações entre estratégias.

## Camadas e regra de dependência

```
cli ─┐
orchestration (servidores HTTP de jobs, pipelines ZenML)
     │
application (casos de uso: runner, campanha, matriz, fábrica de componentes, prontidão)
     │
evaluation (protocolos: holdout temporal, prequential, typed, épocas; tracking; custos)
     │
datasets · models · detectors (adaptadores das bibliotecas de ML, registrados em catálogos)
     │
core (contratos, tipos de tarefa, políticas, motor, métricas) ── sem dependências externas
```

`config` (esquemas Pydantic) e `infrastructure` (armazenamento local, versões de modelo,
Kafka/S3/MLflow) são usados pelas camadas acima de `core`. Uma camada só importa camadas
abaixo dela; `core` não importa River, Pydantic, Kafka, ZenML, MinIO nem MLflow.

| Pacote | Responsabilidade | Módulos principais |
|---|---|---|
| `core` | Contratos (`Event`, `TaskSample`, `Model`, `Estimator`, `Detector`, `Policy`), estratégias de tipo de tarefa, políticas, motor prequential, métricas | `contracts`, `tasks`, `registry`, `policies`, `engine`, `metrics`, `continual`, `evaluation_metrics`, `observability` |
| `config` | Configurações declarativas e validadas | `benchmark` (`Config`), `typed` (`TypedConfig`), `epochs` (`EpochConfig`) |
| `datasets` | Catálogo único de datasets e adaptadores | `catalog`, `synthetic`, `openml`, `arff`, `local_manifest`, `protocols`, `real_streams`, `images`, `cifar_parquet`, `temporal_vision`, `base` |
| `models` | Famílias de modelos com um construtor por tipo de tarefa | `__init__` (catálogo), `linear`, `tree`, `neighbors`, `encoding`, `serving` |
| `detectors` | Catálogo de detectores de drift | `__init__`, `river` |
| `evaluation` | Protocolos de avaliação, derivação de métricas, métricas de sistema e gráficos | `binary`, `holdout`, `prequential`, `typed`, `epochs`, `tracking`, `reporting`, `costs`, `scoring`, `environment`, `system_metrics`, `artifacts` |
| `application` | Casos de uso | `runner`, `components`, `matrix`, `campaign`, `readiness`, `plugin_identity`, `cloud/*` (perfil AWS) |
| `infrastructure` | E/S externa | `storage`, `versioning`, `services`, `catalog`, `metric_tracking`, `epoch_tracking`, `run_artifacts`, `object_store`, `stream_order`, `aws/*` (boto3) |
| `orchestration` | Pontos de entrada que agendam trabalho finito | `server`, `epoch_server`, `zenml_pipeline`, `zenml_steps`, `epoch_pipeline`, `aws_control`, `aws_cli`, `sagemaker` |

## Padrões de projeto

| Padrão | Onde | Para quê |
|---|---|---|
| **Registry / Catálogo** | `core.registry.Registry`; `datasets.catalog.CATALOG`, `models.MODELS`, `detectors.DETECTORS` | Uma declaração por componente. As fábricas são strings `módulo:atributo` resolvidas sob demanda, então validar uma configuração nunca importa River ou PyArrow. |
| **Abstract Factory** | `models.ModelFamily`; `application.components.ComponentRegistry` | Uma família (ex.: `knn`) fornece um construtor para cada tipo de tarefa (`binary`, `multiclass`, `regression`). O `ComponentRegistry` cria fonte, modelo, detector e política a partir da configuração. |
| **Strategy** | `core.tasks` (`Binary`, `Multiclass`, `Regression`); `core.policies`; `evaluation.binary.protocol_for` | Regras de alvo, decisão de retreinamento e protocolo de avaliação são objetos intercambiáveis em vez de cadeias de `if`. |
| **Template Method** | `core.policies.GuardedPolicy.decide`; `core.engine.Engine.release` | Guardas comuns (`min_samples`, `min_window`, `cooldown`) e o ciclo predição → liberação → monitoramento → atualização ficam fixos; as subclasses/etapas só definem o gatilho. |
| **Adapter** | `datasets.base.EventSamples`; `models.encoding.EncodedEstimator` | Um fluxo binário vira um fluxo de duas classes para os avaliadores typed/épocas; qualquer estimador River ganha a mesma codificação (escala incremental + one-hot). |
| **Builder** | `application.matrix.ExperimentMatrix.plan` | Expande o produto cartesiano em casos validados e registra cada combinação descartada com o motivo. |
| **Command** | `cli.COMMANDS` | Cada subcomando é um handler registrado; novos comandos não alteram o `main`. |
| **Dependency Injection** | `run_typed(..., model_factory=...)`, `run_epochs(..., model_factory=...)`, `BinaryRun` | Testes e extensões trocam o modelo ou a fonte sem monkeypatch. |

## Datasets × modelos: compatibilidade

Cada dataset declara o tipo de alvo no catálogo (`binary`, `multiclass`, `regression`, ou
`None` quando o manifesto local o define na admissão). Cada família de modelo declara os tipos
que suporta:

| Família | binary | multiclass | regression |
|---|:-:|:-:|:-:|
| `logistic` | ✓ | ✓ (one-vs-rest) | |
| `linear_regression` | | | ✓ |
| `hoeffding_tree` | ✓ | ✓ | ✓ |
| `knn` | ✓ | ✓ | ✓ |

| Avaliador | Tipos de alvo | Observação |
|---|---|---|
| Binário (`Config`: streaming, incremental, offline) | binary | Motor com detector + política. Datasets ou modelos incompatíveis são rejeitados na validação. |
| Typed prequential (`TypedConfig`) | multiclass, regression | Fluxos binários são adaptados para as classes `"0"`/`"1"` sem perda de informação. |
| Épocas (`EpochConfig`) | multiclass, regression | Mesmo adaptador; holdout oficial ou sufixo temporal purgado. |

`continual-bench components` lista o catálogo efetivo (incluindo plugins).

## Matriz de experimentos

```bash
continual-bench matrix configs/matrices/binary-comparison.yaml --dry-run   # plano + combinações descartadas
continual-bench matrix configs/matrices/binary-comparison.yaml --state state/matrices
```

O resultado fica em `state/matrices/<identidade>/`: `matrix.json` (diário retomável, com status
de cada caso) e `summary.json` (uma linha por caso com dataset, semente, modelo, modo, detector,
política e métricas finais). Uma nova execução reaproveita os casos concluídos; cada caso
continua idempotente pela identidade da sua configuração. Veja também
`configs/matrices/any-model.yaml` para o avaliador typed com regressão. Na AWS, o mesmo plano
alimenta o Step Functions e gera uma unidade Batch por caso. Cada tentativa recebe um run no
MLflow. Consulte a campanha `infra/examples/campaign-binary-matrix.json` e o
[guia do perfil AWS](aws/operations.md#campanhas-no-sagemaker-studio).

## Como estender

- **Dataset**: implementar `events()` (binário) ou `samples_iter()` (typed) + `manifest()` e
  adicionar um `DatasetSpec` em `datasets/catalog.py` (ou `registry.register("source", nome,
  fábrica, task=...)` por plugin).
- **Modelo**: adicionar uma `ModelFamily` em `models/__init__.py` com um construtor por tarefa
  (ou `registry.register("model", nome, fábrica, tasks=("regression",))` por plugin). Construtores
  binários seguem `predict(x) -> probabilidade`/`learn(x, y)`; os demais, `predict_one`/`learn_one`.
- **Política**: subclasse de `GuardedPolicy` implementando `triggered(context)` e registro em
  `core.policies.POLICIES`.
- **Detector**: `Spec` em `detectors/__init__.py`; recebe somente o erro binário do rótulo liberado.

`examples/custom_plugin.py` registra uma política e um modelo de regressão por plugin.

## Compatibilidade e reprodutibilidade

- A identidade de execução vem da configuração validada. Checkpoints também validam o contrato
  de código, dependências, fonte e transporte antes de retomar.
- Modelos registrados pelo publicador local carregam o código empacotado. Os scripts de
  recuperação leem o `loader_module` registrado no `MLmodel`.
- Registro da refatoração de 08/11/2026. A comparação no homelab conferiu igualdade de digests
  de predição e métricas em 100 casos, sendo 91 binários, 3 typed e 6 de épocas. Esse recorte
  descreve a equivalência entre as duas revisões verificadas naquela data.

## Configurações

Os arquivos de `configs/` seguem a estrutura base → sobrescrita descrita em
[configs/README.md](../configs/README.md): `base/` concentra os padrões compartilhados; `runs/`,
`typed/`, `campaigns/` e `matrices/` herdam deles com `extends`; `contracts/` guarda locks e
definições de métricas versionados; `history/` guarda os manifestos congelados das campanhas
executadas. `continual_bench.config.load_model(schema, caminho)` resolve a herança e valida o
resultado.

## Perfil AWS gerenciado

`application.cloud` implementa aquisição, preparação, plano, execução de uma run, publicação,
reconciliação e custos do perfil AWS. Os adaptadores boto3 ficam em `infrastructure.aws` e
são injetados nas fronteiras de I/O. O perfil reaproveita o avaliador Zia, os geradores de
artefatos e o planejador de matrizes. Sua execução usa Step Functions, Batch e S3. A entrega
do runtime usa GitHub Actions com OIDC.

A stack complementar `studio.yaml` expõe a preparação, a campanha Batch e a verificação do
resumo no SageMaker Pipelines. A stack `studio-native.yaml` encadeia Processing, Training e
Tuning para o baseline Social. Os adaptadores em `infrastructure.aws` enviam métricas e
metadados ao MLflow gerenciado por REST/SigV4. O estado do workflow e o estado de cada job
são consultáveis no serviço executor correspondente.

A stack `studio-catalog.yaml` conecta eventos S3 a uma fila FIFO e a uma função Lambda.
O marcador do snapshot abastece o catálogo do SageMaker Unified Studio. O manifesto final
de uma avaliação Batch Zia vincula o run MLflow ativo ao snapshot confirmado. A preparação,
o treino e a avaliação tabular continuam nos mesmos pipelines e avaliadores. Veja
[infraestrutura e operação na AWS](aws/README.md) e [tracking](mlflow-tracking.md).
