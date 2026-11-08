# Refatoração modular — validação no homelab (08/11/2026)

Execuções no servidor `gabriel@192.168.100.27`, diretório isolado
`~/refactor-modular-20261108`, imagem `continual-bench-datasets:semana09-v2`
(`sha256:b5fbee1129c29a2140087d5649471cc84f1a0081f7ffcb63d148011c38f1afb8`), contêiner
`--network none --read-only`, `PYTHONHASHSEED=0`. Nenhum serviço da stack foi usado.

| Verificação | Antes | Depois |
|---|---|---|
| `pytest -q` (sem `services`/`benchmark`) | 490 passed | 527 passed (490 + 37 novos) |
| `ruff check src tests dags examples scripts` | — | All checks passed |
| Import de todos os módulos na imagem `continual-bench-worker:0.1.0` | — | 66 módulos, 0 falhas; todos os nomes importados pelos scripts resolvidos |
| Equivalência golden (100 casos) | baseline | 0 diferenças; 100/100 digests de predição idênticos |

## Equivalência golden

`golden_capture.py` executa, sem serviços, 91 configurações binárias (sea, agrawal, stagger,
hyperplane, rbf_drift × logistic, hoeffding_tree, knn × streaming/drift-ADWIN, streaming/online,
incremental/periodic-DDM, incremental/performance, streaming/cost_heuristic e offline, com
tracking e avaliação futura; mais um caso com portão de promoção e orçamento), 3 execuções typed
(friedman_drift × none/online/periodic) e 6 de épocas. Remove apenas campos voláteis (tempos,
memória, ambiente, caminhos, hashes de código e identidades derivadas deles).

`golden_diff.py` compara os relatórios campo a campo. Ignora somente valores terminados em
`sha256`, que embutem o `code_sha256` (que muda em qualquer alteração de código) ou tempos de
predição (o ledger de predições registra `prediction_seconds`). Com essa regra: 0 casos
divergentes; os `prediction_digest` (hash encadeado de IDs de eventos e probabilidades) são
idênticos nos 100 casos.

```text
golden-baseline.json  7430b0dd629ac36fd30c24e4ae9afa838a2e1ee167c3ebbc81f011d6638134d6
golden-final.json     8d4486395a74254d2d99af69335985ac2191aa6fb3ecbedef9b83652a68a9dc1
cases=100 differing=0 digest_equal=100
```

## Não validado nesta etapa

- Testes `-m services` (Kafka, MinIO, MLflow e ZenML reais) não foram executados: escrevem na
  stack em operação. Os módulos correspondentes foram verificados por import na imagem do worker.
- As imagens Docker não foram reconstruídas; os testes montam o código-fonte por volume.
