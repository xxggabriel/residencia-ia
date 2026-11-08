# Rastreamento no MLflow: métricas, métricas de sistema, gráficos e dados

Toda execução coleta **todas as métricas por padrão**. Ao final de cada avaliação, gera localmente
gráficos segmentados por tema e os JSON com os dados de cada gráfico. Quando a execução é publicada
no MLflow, esses arquivos e as métricas de sistema vão para o run.

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

O avaliador typed gera os mesmos artefatos e métricas de sistema localmente, mas ainda não tem um
publicador MLflow. A matriz (`continual-bench matrix`) executa sem serviços: os artefatos ficam em
cada diretório de caso.
