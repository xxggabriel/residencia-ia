# Benchmarks temporais Zia Data Labs

Cada dataset tem uma DAG com quatro runs. O algoritmo, os dados, os limites e o
protocolo ficam fixos dentro de cada DAG. As fontes são sintéticas, publicadas
pela Zia Data Labs sob CC BY-NC 4.0. As revisões e os SHA-256 estão fixados no
[contrato de entrada](../src/continual_bench/datasets/zia.py).

| DAG | Fonte e volume | Alvo | Algoritmo e evidência |
|---|---|---|---|
| `benchmark_zia_fraud` | [FreeFraudDetection50M](https://huggingface.co/datasets/ziadatalabs/FreeFraudDetection50M), 50 milhões | Fraude em pagamento | XGBoost de histogramas em CPU. O [estudo comparativo de fraude](https://www.sciencedirect.com/science/article/pii/S1544612325024390) encontrou XGBoost entre os métodos com melhor desempenho. |
| `benchmark_zia_social` | [FreeSocialNetwork100M](https://huggingface.co/datasets/ziadatalabs/FreeSocialNetwork100M), 100 milhões de arestas e 3 milhões de nós | Tipo da conexão observada, entre follow, friend, block e mute | Regressão logística multinomial. O [estudo de Leskovec et al.](https://arxiv.org/abs/1003.2429) demonstrou regressão logística com atributos estruturais para o problema relacionado de classificar o sinal de arestas. |
| `benchmark_zia_ecommerce` | [FreeEcommerceOrders50M](https://huggingface.co/datasets/ziadatalabs/FreeEcommerceOrders50M), 50 milhões | Status registrado como devolvido, frente a todos os demais status | XGBoost de histogramas em CPU. O [estudo de devoluções em varejo](https://www.sciencedirect.com/science/article/pii/S0969698926001086) obteve AUROC de 0,879 em 254 mil exemplos de holdout. |

As referências orientam a escolha por tarefa e usam outros datasets. A
superioridade de um algoritmo nestes três arquivos exigiria uma avaliação
comparativa adicional. Em e-commerce, status intermediários entram na classe
`other_status` porque a fonte fornece um único status por pedido. Esse alvo
representa o status observado e depende da maturação dos pedidos.

## Protocolo

1. Preparar os Parquets no homelab, ordenados por timestamp e, em empates, ID.
   A preparação usa ordenação externa em disco. Os identificadores, rótulos
   auxiliares de fraude, peso de interação social e graus finais ficam fora das
   features. Os graus sociais contam apenas arestas de datas anteriores. As
   tabelas de nós fornecem data de criação e igualdade de região.
2. Consumir todos os registros de eventos em lotes-alvo de 250 mil linhas no
   escopo `full` e mil linhas no escopo `smoke`. Empates temporais ficam no
   mesmo bloco. O primeiro bloco serve para ajuste inicial. Os demais são
   avaliados antes de qualquer atualização. Os últimos cerca de 10% formam
   holdout temporal com modelo congelado, separado das métricas prequenciais.
   A fonte fornece data do evento, sem confirmação do atraso dos rótulos.
   No protocolo, os rótulos do bloco ficam disponíveis após a predição.
3. Aplicar exatamente uma das políticas a cada run.

| Política | Atualização após predição |
|---|---|
| `frozen` | Modelo inicial congelado. |
| `periodic` | Reinicia o ajuste a cada quatro blocos e usa os quatro mais recentes. |
| `drift` | ADWIN sobre taxa de erro por bloco. Reinicia o ajuste em alarmes, com até quatro blocos recentes. |
| `incremental` | Continua o ajuste a cada bloco a partir do modelo atual. XGBoost acrescenta uma rodada. A logística usa os pesos anteriores como ponto inicial do otimizador. |

## Telemetria

O MLflow recebe resumos prequenciais, holdout, séries por bloco, métricas por
classe, explicações dos modelos e os modelos serializados de todas as
atualizações. Acurácia, acurácia balanceada, matriz de confusão, F1, precisão,
recall, log loss, Brier, calibração e custos seguem o escopo registrado em
`report.json`. AP e AUROC usam histogramas de 4.096 bins para manter memória
constante. Em fraude e devoluções, `recall_at_fpr_0_01` informa a recuperação
até 1% de falsos positivos. A rede social registra métricas um-contra-resto
e agregações macro. Métricas do catálogo que dependem de drift anotado,
contrafactuais, medições de CI ou custos de negócio têm disponibilidade
explícita no relatório.

Os custos `aws_cost_*` usam os tempos de treino e predição e o tamanho de
todos os modelos publicados. `aws_evaluation_cost_*` usa o tempo de
avaliação incluindo leitura dos dados, métricas e serialização. Ambos usam
o perfil AWS de [tracking](../configs/base/tracking.yaml). O sistema coleta
CPU e memória do cgroup e threads do processo em séries distintas. A
preparação externa e a publicação ficam fora desses tempos. As DAGs usam
Airflow e os workers existentes. O transporte é leitura local de Parquet,
e os artefatos do MLflow são persistidos no MinIO do stack.

## Operação

No homelab, preparar os datasets com Python 3.12 e `uv sync --locked --extra
datasets --extra large-datasets` ou usar a imagem do worker para preparar os
arquivos já baixados. O parâmetro `--download` exige `curl` e deve ser usado
no servidor que guarda os originais.

```bash
root="$HOME/.cache/continual-bench/zia-v1"
for dataset in fraud social ecommerce; do
  .venv/bin/python scripts/prepare_zia.py "$root" "$dataset" smoke --download
done
```

Copiar para o Mac somente `smoke.parquet` e `smoke.json` por dataset.
Cada amostra contém as primeiras 10 mil linhas físicas da fonte, depois
ordenadas e transformadas no homelab. A amostra valida o fluxo funcional.
Com os extras instalados, executar localmente:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m continual_bench.evaluation.zia \
  state/zia-v1 fraud smoke incremental state/zia-smoke/fraud/incremental
```

Depois do teste local, repetir a preparação no homelab com `full` e iniciar
o stack com `compose.zia.yaml` após os arquivos Compose do ambiente.
Exportar `BENCH_ZIA_DATA_DIR` com o caminho dos Parquets e
`BENCH_GIT_REVISION` com a revisão implantada. O volume de dados fica
somente leitura nos oito workers, cada um com 1 vCPU e 2 GiB. Ativar as
três DAGs e dispará-las manualmente com `scope: full` e um nome de campanha
distinto. O nome do experimento segue
`continual-bench/zia-<dataset>-<scope>-<campaign>`. O worker mantém
`progress.json`, `report.json` e `receipt.json` no estado da run. Uma
publicação incompleta pode ser repetida usando a mesma campanha. Uma
interrupção anterior ao relatório final pode refazer o treinamento.
