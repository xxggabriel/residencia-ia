# Rastreamento no MLflow: métricas, métricas de sistema, gráficos e dados

Os avaliadores geram métricas, gráficos e dados conforme o protocolo e as entradas disponíveis.
No homelab, os publicadores registram no MLflow as métricas e os artefatos produzidos.
No perfil AWS, o MLflow gerenciado recebe as campanhas Zia, as matrizes e os jobs
nativos do SageMaker.

## Métricas coletadas por padrão

Sem a seção `tracking`, a execução binária usa o perfil completo `DEFAULT_TRACKING`: métricas
`residency-metrics-v1`, referências de aprendizado contínuo, *shadows* de adaptação (modelo anterior
× novo após cada atualização) e janelas de futuro próximo com modelo congelado
(`task_evaluation`: horizonte 32, gap 1, até 4 janelas). Uma seção `tracking` explícita continua
valendo integralmente, então campanhas antigas mantêm identidade e comportamento.

No MLflow, cada run binário recebe:

- métricas por janela (`accuracy`, `f1`, `brier`, …, com `step = end_seq`);
- resumos `summary_*` e todas as séries e métricas finais `tracking_v1/*`;
- métricas de sistema `system/*`;
- artefatos `plots/` e `data/`, além do relatório, manifesto, catálogo e modelo registrado.

Os runs de épocas recebem as métricas por época, os custos, `compute/training_seconds` e
`compute/prediction_seconds`, e também `system/*`, `plots/` e `data/`.

## Métricas de sistema (aba *System metrics*)

Durante cada tentativa de avaliação, `SystemMetricsSampler` lê o container, o host e o processo em segundo
plano: na partida, a cada `BENCH_SYSTEM_METRICS_INTERVAL` segundos (padrão 0,5) e no fim. Os nomes
seguem a convenção do MLflow e são publicados com o prefixo `system/` e o timestamp real de cada
leitura:

| Métrica | Origem |
|---|---|
| `system/container_cpu_usage_vcpus`, `system/container_cpu_limit_vcpus`, `system/container_cpu_utilization_percentage` | CPU acumulada do cgroup v2 por intervalo, em vCPUs e em porcentagem da cota |
| `system/container_memory_usage_megabytes`, `system/container_memory_limit_megabytes`, `system/container_memory_usage_percentage` | memória contabilizada no cgroup v2, em MiB e em porcentagem do limite |
| `system/cpu_utilization_percentage`, `system/system_memory_usage_*` | CPU e memória do host quando o respectivo limite de cgroup está indisponível |
| `system/process_cpu_utilization_percentage`, `system/process_memory_rss_megabytes`, `system/process_threads` | CPU, RSS e número de threads do processo executor, em séries separadas |
| `system/disk_usage_percentage`, `system/disk_usage_megabytes`, `system/disk_available_megabytes` | disco do diretório de estado |
| `system/network_receive_megabytes`, `system/network_transmit_megabytes` | rede acumulada desde a partida |
| `system/gpu_<i>_utilization_percentage`, `system/gpu_<i>_memory_usage_*` | GPUs NVIDIA, quando `pynvml` e o driver existem |

A primeira leitura inclui a cota de CPU, quando disponível. O uso depende da diferença em
`cpu.stat:usage_usec` entre duas leituras e do tempo decorrido. Nos workers padrão, 100% equivale
a 1 vCPU ou 2.048 MiB de RAM, conforme a série. Amostras curtas de CPU podem passar de 100%
devido ao alinhamento com o período da cota. A memória inclui cache contabilizado em
`memory.current`. A quantidade de threads é independente da cota de CPU.
Leituras longas são limitadas a 3.600 amostras: ao atingir o limite, metade é descartada e o
intervalo dobra. O psutil já vem na imagem do worker. Sem ele, a leitura do cgroup e do RSS via
procfs continua disponível. A amostragem só observa: não
altera predições, decisões nem digests. O escopo é a tentativa de avaliação no processo, incluindo
a derivação das métricas e excluindo a geração de gráficos e a publicação.

## Gráficos e dados (artefatos)

No diretório da execução (`state/.../<run_key>/artifacts/`) e na raiz dos artefatos do run no
MLflow:

```
plots/<grupo>/<nome>.svg     um gráfico por figura (SVG, exibido pelo visualizador do MLflow)
data/<grupo>/<nome>.json     exatamente os dados desenhados naquele gráfico
data/<dump>.json             dumps completos: summary/, series/, events/, system/, …
data/index.json              todas as figuras com caminhos de plot e dados
```

| Grupo | Execução binária | Typed | Épocas |
|---|---|---|---|
| `performance` | acurácia (com alarmes, mudanças e adaptações), F1, precisão/recall, taxas de erro, MCC/kappa, balanço de classes, confusão, acurácia com fading | métricas por partição, por tarefa, erro/MAE móvel | métricas no holdout por época |
| `probabilistic` | Brier/log loss, calibração ECE/MCE, ROC AUC/AP | — | — |
| `drift` | linha do tempo de alarmes × mudanças × adaptações, atraso de recuperação | — | — |
| `adaptation` | apresentações de treino acumuladas, holdout de promoção, orçamento | apresentações acumuladas | — |
| `continual` | métricas CL por tarefa, matriz de acurácia, retenção por conceito | — | — |
| `near_future` | acurácia do modelo congelado por janela | — | — |
| `costs` | componentes do custo de cenário, sensibilidade a alpha | componentes do custo | custo por época e total |
| `summary` | barras com o valor final de cada família de métricas | — | — |
| `compute` | — | — | tempo de treino e predição por época |
| `system` | CPU, memória, disco, rede, threads e GPU | idem | idem |

Os gráficos são gerados sem dependências externas (`evaluation/artifacts/svg.py`): têm qualidade
vetorial para relatórios e são determinísticos para os mesmos dados. Para outra visualização,
basta ler o JSON correspondente em `data/`.

## Republicação e limites

A publicação é idempotente: pontos `system/*` já existentes são conferidos e não duplicados, e os
artefatos de mesmo caminho são sobrescritos com o mesmo conteúdo. A marca
`tracking_v1_status=complete` continua sendo a última escrita antes de encerrar o run.

A CLI local de avaliação typed e de matriz grava os resultados em diretórios de estado.
O tracking AWS de matrizes é configurado no worker do Batch e aceita casos binários, typed e
de épocas.

## Dashboards detalhados com `mlflow.log_figure()`

Além dos SVGs acima, cada run publicado grava dashboards em matplotlib (PNG) com
`mlflow.log_figure()`, em `plots/<grupo>/` — o visualizador de artefatos do MLflow os exibe direto.
O código está em `infrastructure/mlflow_figures.py`; precisa do extra `plots` (já incluso em
`services` e `aws-mlflow`). Sem matplotlib, os SVGs continuam sendo publicados e nada falha.

| Artefato | O que mostra |
|---|---|
| `performance/dashboard.png` | acurácia, F1, precisão/recall/especificidade e MCC/kappa por janela, com média móvel e linhas de alarme, mudança de conceito e adaptação |
| `performance/confusion_matrix.png`, `confusion_timeline.png` | matriz agregada (contagens e normalizada por classe real) e composição TP/TN/FP/FN por janela |
| `performance/error_tradeoff.png` | trade-off FPR × FNR por janela, colorido pelo avanço do fluxo |
| `probabilistic/dashboard.png` | Brier, log loss, ECE/MCE e ROC AUC/AP |
| `drift/timeline.png`, `recovery_delay.png` | desempenho com faixa de eventos; atraso de recuperação por mudança |
| `adaptation/updates.png` | treino acumulado × orçamento e portão de promoção (candidato × incumbente) |
| `continual/accuracy_matrix.png`, `cl_metrics.png` | matriz de acurácia anotada e métricas de aprendizado contínuo |
| `costs/breakdown.png` | componentes do custo e sensibilidade a alpha |
| `summary/final_metrics.png` | métricas finais, prequential × holdout |
| `performance/epochs.png` | métricas e tempo por época (execuções de épocas) |
| `system/resources.png` | CPU, memória, rede, disco e threads |

Cada gráfico só é gerado quando o relatório tem os dados correspondentes. As figuras são
registradas em `publish_run_artifacts` (todos os fluxos locais) e no publicador SageMaker abaixo.

## MLflow no SageMaker Studio

Abra **Applications → MLflow → continual-bench-mlflow → Open MLflow → default →
Experiments** ou obtenha uma URL de acesso com `make aws-mlflow-ui`. O item **Experiments**
do menu principal do Studio depende de um MLflow Tracking Server. O runtime publica no
**MLflow App** serverless.

| Execução | Experimento | Filtro e conteúdo |
|---|---|---|
| Campanhas Zia no Batch | `<sha-curto>-continual-bench-zia-experiments` | `cb-campaign-id`. Relatório final, janelas, holdout e métricas de sistema. |
| Matriz no Batch | `continual-bench-matrix` | `cb-campaign-id`. Uma run por tentativa, com `cb-unit-id`, `cb-batch-job-id`, séries `progress/*` e resumo `final/*`. |
| Processing, Training e trials de Tuning | `continual-bench-native` | `cb-pipeline-execution-id`. Uma run por job ou trial, com progresso da preparação ou métricas de validação por época. |

Os pipelines mostram etapas e estado da execução no Studio. Os detalhes de orquestração e
cancelamento ficam em [operação na AWS](aws/operations.md#campanhas-no-sagemaker-studio).
O catálogo tabular do SageMaker Unified Studio vincula os snapshots Social, Fraud
e E-commerce aos runs Batch Zia que têm manifesto final compatível. No MLflow,
`cb-catalog-asset-url` aponta para o ativo e `cb-catalog-snapshot-sha256` identifica
o manifesto. Veja [publicação automática](aws/operations.md#publicação-automática-no-catálogo).

### Publicação Zia via SDK opcional

O publicador padrão usa HTTP REST com autenticação SigV4. Para Zia, com
`MLFLOW_TRACKING_URI` apontando para um ARN do SageMaker, o método
`SageMakerMlflowTracker.publish_run_report` também pode usar o plugin `sagemaker-mlflow`
quando instalado. Esse caminho do SDK suporta:

- **Tracking:** experimento, run retomável por `zia_run_key`, params, tags, métricas por janela/holdout/sistema;
- **Plots:** os dashboards acima, via `mlflow.log_figure()`;
- **Datasets:** `mlflow.log_input` com digest sha256 da fonte, então cada revisão é um input distinto;
- **Modelos:** modelo final como `pyfunc` (`model_pyfunc/`) e uma versão no Model Registry por checkpoint,
  com aliases `initial`, `step-<seq>`, `final` e `current` (o MLflow reserva `latest`), mais o modelo
  de família que compara políticas lado a lado;
- **Artefatos:** diretório do run espelhado na raiz do run.

A imagem AWS padrão inclui o cliente REST/SigV4. O SDK é um extra:

```bash
pip install ".[aws-mlflow]"   # mlflow-skinny, sagemaker-mlflow, matplotlib
```

Se o SDK falhar, o tracker registra o motivo em `api_errors` e continua pelo cliente REST/SigV4.
Republicar o mesmo `run_id` usa a identidade do run e das versões registradas para retomada.
Os dashboards em PNG dependem de matplotlib no ambiente publicador.

O caminho REST/SigV4 foi usado na validação das integrações com o app AWS.
O caminho opcional do SDK foi exercitado com o file store local. Consulte o escopo
em [validação na AWS](aws/validation.md).
