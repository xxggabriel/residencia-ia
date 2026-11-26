# Detectores de Drift

Esta página descreve os detectores de drift disponíveis, seus parâmetros, a biblioteca de origem, o sinal que monitoram e como um alarme chega à política de atualização. Drift (deriva) é a mudança, ao longo do tempo, na distribuição dos dados ou na relação entre atributos e rótulo.

## Princípio de funcionamento

Os detectores apenas sinalizam. Quem decide atualizar o modelo é a política (veja [Modelos e Políticas](Modelos-e-Politicas)). Todos os detectores do projeto monitoram o **erro preditivo**, calculado depois que o rótulo de um exemplo é liberado. Por isso, o que detectam é *performance drift*: uma mudança na taxa de erro. Um alarme não prova sozinho uma mudança em P(Y|X), e o projeto não monitora a distribuição dos atributos P(X). Um detector multivariado de atributos não está implementado.

## Catálogo

`DETECTORS` em `src/continual_bench/detectors/__init__.py` registra `none` e sete detectores do River 0.26.1 (`src/continual_bench/detectors/river.py`). O sinal declarado no catálogo é `released_label_binary_prediction_error`.

| Nome na configuração | Classe River | Parâmetros padrão (River 0.26.1) | Artigo de origem citado pelo River |
| --- | --- | --- | --- |
| `none` | `NullDetector` (do projeto) | nunca sinaliza | |
| `adwin` | `drift.ADWIN` | `delta=0.002`, `clock=32`, `max_buckets=5`, `min_window_length=5`, `grace_period=10` | *Learning from Time-Changing Data with Adaptive Windowing* (2007) |
| `ddm` | `drift.binary.DDM` | `warm_start=30`, `warning_threshold=2.0`, `drift_threshold=3.0` | *Learning with Drift Detection* (2004) |
| `eddm` | `drift.binary.EDDM` | `warm_start=30`, `alpha=0.95`, `beta=0.9` | *Early Drift Detection Method* (2006) |
| `hddm_a` | `drift.binary.HDDMA` | `drift_confidence=0.001`, `warning_confidence=0.005`, `two_sided_test=False` | *Online and Non-Parametric Drift Detection Methods Based on Hoeffding's Bounds* (2015) |
| `hddm_w` | `drift.binary.HDDMW` | `drift_confidence=0.001`, `warning_confidence=0.005`, `lambda_val=0.05`, `two_sided_test=False` | Mesmo artigo do HDDM-A |
| `page_hinkley` | `drift.PageHinkley` | `min_instances=30`, `delta=0.005`, `threshold=50.0`, `alpha=0.9999`, `mode="both"` | *Continuous Inspection Schemes* (1954) |
| `kswin` | `drift.KSWIN` | `alpha=0.005`, `window_size=100`, `stat_size=30`, `seed=42` (o projeto fixa a seed) | *Reactive Soft Prototype Computing for Concept Drift Streams* (2020) |

Os valores padrão foram conferidos nas assinaturas do River 0.26.1 instalado pelo `uv.lock`. Qualquer parâmetro do construtor pode ser passado em `detector.params`, por exemplo `detector: {name: adwin, params: {delta: 0.002}}` em `configs/runs/streaming.yaml`.

Validações próprias do projeto:

- O wrapper `RiverDetector.update` aceita apenas 0 ou 1 e rejeita qualquer outro valor (inclusive atributos brutos).
- KSWIN exige `0 < stat_size <= window_size / 2` e `0 < alpha < 1`.
- Só a saída `drift_detected` é usada. Os estados de alerta (*warning*) de DDM, EDDM e HDDM não acionam nada.

As referências completas estão em [Referências](Referencias).

## Sinal monitorado e disparo da política

### Motor binário por evento

No motor (`src/continual_bench/core/engine.py`, método `_monitor`), para cada rótulo liberado após o aquecimento:

1. O erro binário é `int((p >= 0.5) != y)`, onde `p` é a probabilidade prevista antes de o rótulo existir.
2. O detector recebe esse erro e devolve `True` se detectou drift.
3. A política recebe um `DecisionContext` com o número de rótulos, o tamanho da janela, o alarme e a taxa de erro recente.
4. A política `drift` retreina quando há alarme, desde que as guardas `min_samples`, `min_window` e `cooldown` permitam. As demais políticas ignoram o alarme.
5. Com `reset_detector_on_retrain: true` (padrão), um novo detector é criado após cada retreino.

Todo alarme é registrado em `report.alarms` com o identificador do evento e a decisão tomada, mesmo quando a política recusa o retreino.

### Avaliador por blocos

O avaliador por blocos (`src/continual_bench/evaluation/stream_benchmark.py`) não usa o catálogo `DETECTORS`: instancia `ADWIN` e `DDM` do River diretamente, com parâmetros fixos no código, e os alimenta depois de prever cada bloco prequential.

| Política | Detector e parâmetros | Sinal | Efeito |
| --- | --- | --- | --- |
| `drift` | ADWIN, `delta=0.002`, `clock=1` | `1 - acurácia` do bloco, um valor contínuo por bloco | Alarme retreina do zero com os quatro blocos mais recentes |
| `drift_adwin` | ADWIN, `delta=0.002`, `clock=32` | Erro binário de cada exemplo do bloco | Um ou mais alarmes no bloco geram um único retreino |
| `drift_ddm` | DDM, `warm_start=30`, `warning_threshold=2.0`, `drift_threshold=3.0` | Erro binário de cada exemplo do bloco | Um ou mais alarmes no bloco geram um único retreino |

O erro binário usa o limiar de decisão do perfil de tracking (`p >= c_FP / (c_FP + c_FN)`); em multiclasse, a classe prevista. Nesse caminho o detector não é recriado após o retreino: o estado interno segue o comportamento do River, que se reinicia ao detectar drift. Blocos de holdout não alimentam o detector. Cada alarme é registrado com posição, bloco, detector e sinal.

## Diferenças em relação aos artigos originais

- **Sinal**: os métodos foram propostos para monitorar o erro de um classificador, mas aqui o erro é o do rótulo liberado com atraso (no motor por evento) ou é avaliado bloco a bloco (no avaliador por blocos). Na política `drift` do avaliador por blocos, ADWIN recebe uma taxa por bloco, não uma sequência de erros por exemplo.
- **Frequência de verificação do ADWIN**: o River verifica cortes a cada `clock` atualizações (padrão 32), não a cada exemplo.
- **Alertas**: DDM, EDDM e HDDM produzem nível de alerta nos artigos e no River; o projeto usa apenas a detecção.
- **Page-Hinkley** opera com `mode="both"` no padrão do River, então uma queda acentuada do erro também pode gerar alarme.
- **KSWIN** é um teste de Kolmogorov-Smirnov sobre uma janela deslizante; aqui recebe a sequência de erros 0 e 1, não atributos, e usa seed fixa 42 para a amostragem da janela.
- Os detectores e suas implementações são os do River 0.26.1, que podem diferir das implementações originais dos autores. O benchmark de referência do CARA oferece `--drift-backend skmultiflow` para usar os detectores do ambiente original, quando disponíveis, e `--drift-backend river` para os do projeto (`scripts/replicate_cara.py`).

## Estado

| Detector | Estado | Evidência |
| --- | --- | --- |
| ADWIN | Validado | Testes de contrato em `tests/test_task_policies.py`; usado nas matrizes INSECTS (`configs/matrices/insects-*.yaml`) e nas políticas `drift` e `drift_adwin` da campanha AWS de 132 unidades (`evidencias/aws-benchmarks/cases.csv`) |
| DDM | Validado | Testes de contrato; política `drift_ddm` da campanha AWS; `configs/runs/incremental.yaml` |
| EDDM, HDDM-A, HDDM-W, KSWIN | Experimental | Implementados e cobertos por testes de contrato (erro binário, serialização com pickle); KSWIN tem teste de seed e janela. Não há campanha registrada com esses detectores |
| Page-Hinkley | Experimental | Testes de contrato e teste de detecção de aumento controlado do erro; sem campanha registrada |
| Detector de drift em atributos P(X) | Pendente | Não implementado |
| Análise de sensibilidade dos parâmetros dos detectores | Pendente | Nenhuma matriz em `configs/matrices/` varia os parâmetros dos detectores |

As métricas de detecção (atraso, falsos alarmes, correspondência com fronteiras anotadas) e suas condições de cálculo estão em [Métricas](Metricas). As fontes com fronteiras anotadas estão em [Datasets](Datasets).
