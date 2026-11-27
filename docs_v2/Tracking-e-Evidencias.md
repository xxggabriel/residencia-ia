# Tracking e Evidências

Esta página descreve o que o `continual-bench` registra no MLflow e no ZenML, localmente e na AWS, quais scripts fazem backfill e verificação desses registros, e o que contém cada pacote de evidência versionado em `evidencias/`. A última parte orienta a leitura cautelosa das tabelas e figuras.

## Visão geral

| Capacidade | Onde está | Estado |
| --- | --- | --- |
| Publicação no MLflow local (Docker Compose) com `--services` | `src/continual_bench/infrastructure/services.py`, `metric_tracking.py`, `run_artifacts.py` | **Implementado**; testes de integração exigem serviços reais (marcador `services`) |
| Publicação no MLflow gerenciado do SageMaker (REST/SigV4 e SDK `sagemaker-mlflow`) | `src/continual_bench/infrastructure/aws/mlflow_tracking.py` | **Implementado**; evidência de uso real nos experimentos 403 a 408 e 430 citados em `evidencias/semana-10/insects/consolidacao.md` |
| Camada comum de tracking (lotes, reconciliação, runs filhas, uploads paralelos) | `src/continual_bench/infrastructure/tracking/` | **Validado**: `tests/test_tracking_package.py` |
| Métricas de sistema amostradas | `src/continual_bench/evaluation/system_metrics.py` | **Validado**: `tests/test_run_artifacts.py` |
| Runs filhas por ajuste com seed | `src/continual_bench/infrastructure/training_tracking.py` | **Validado**: `tests/test_training_seed_tracking.py` |
| Pipeline ZenML `finite_benchmark` | `src/continual_bench/orchestration/zenml_pipeline.py`, `zenml_steps.py` | **Implementado** |
| Backfill e verificação de registros | `scripts/backfill_*.py`, `scripts/verify_*.py` | **Implementado**; `tests/test_tracking_backfill.py` e `tests/test_tracking_verification.py` cobrem parte da lógica |
| Pacotes de evidência da Semana 10 e da campanha F109 | `evidencias/` | **Validado** por checksums; ver [Reprodutibilidade e Artefatos](Reprodutibilidade-e-Artefatos#conferir-checksums) |

## MLflow local

O MLflow é um servidor de rastreamento de experimentos: guarda parâmetros, métricas com passo (`step`) e artefatos de cada execução (run). No perfil Docker Compose, a publicação é opcional e só acontece com `--services` nos subcomandos `run`, `typed`, `epoch-job` e `matrix` (confira com `continual-bench run --help`). O worker iterativo publica quando `MLFLOW_TRACKING_URI` está definido.

Fluxo de `publish()` em `src/continual_bench/infrastructure/services.py`, no modo padrão (`artifact_mode: full`):

1. Sob um lock por `run_key`, envia ao MinIO `report.json`, `manifest.json`, `checkpoint.bin` e `events.jsonl`.
2. Procura no experimento um run com a tag `run_key`. Se houver mais de um, falha e exige reconciliação manual; se não houver, cria um run com nome `dataset-modelo-modo-detector-política-seed`.
3. Registra parâmetros como `seed`, `policy`, `detector`, `dataset_sha256`, `code_sha256`, `samples` e os resumos `summary_*`.
4. Grava `receipt.json` com o identificador do run e as URIs dos objetos.

O nome do experimento é `mlflow_experiment` quando a configuração o declara; caso contrário, é o nome da estratégia (`experiment_name` em `src/continual_bench/infrastructure/catalog.py`).

No modo `metrics_only`, a publicação passa por `publish_local_benchmark` (`src/continual_bench/infrastructure/benchmark_tracking.py`), não registra modelo e envia ao MinIO os arquivos da execução exceto `events.jsonl` e `receipt.json`.

A publicação é idempotente: antes de acrescentar pontos, o publicador compara o histórico existente de cada métrica (`missing_metric_points` em `src/continual_bench/infrastructure/tracking/reconcile.py`) e não duplica pontos já gravados. Conflitos de valor falham, sem atalho pelo último valor.

## Campanhas CARA e UPF

Com o tracking habilitado e uma URI do MLflow configurada, a campanha cria ou recupera a run principal antes de construir as matrizes. Cada seed abre uma run filha `matrix-seed-<seed>` em `RUNNING` quando começa a construir a matriz. Ao concluir, a run recebe a duração, o tempo de ajuste por lote e o artefato `matrix_summary.json` antes de mudar para `FINISHED`.

A avaliação abre runs por política e seed. Cada avaliação de seed publica as métricas, os arquivos de resumo, regimes e linha do tempo, e os eventos de retreino disponíveis antes do encerramento. A grade de avaliação calculada a partir do conjunto das matrizes continua aguardando todas as seeds. Os agregados por campanha e política, junto da comparação com os artigos, são publicados na consolidação.

O fluxo local e o worker usado pela DAG `benchmark_campaign` seguem esse ciclo. Falhas de execução capturadas encerram as runs afetadas com `FAILED`.

## MLflow gerenciado no SageMaker

Na AWS, os workers publicam no MLflow gerenciado do SageMaker. Há dois caminhos em `src/continual_bench/infrastructure/aws/mlflow_tracking.py`:

- cliente REST com autenticação SigV4 (assinatura de requisições AWS), usado por padrão;
- plugin `sagemaker-mlflow` (extra `aws-mlflow`), que acrescenta `mlflow.log_input` com o digest da fonte, `mlflow.log_figure`, modelo `pyfunc` e versões no Model Registry. Se o SDK falhar, o motivo vai para `api_errors` e a publicação continua pelo REST.

O lock `requirements-aws.lock` fixa `sagemaker-mlflow==0.5.0`, e `docker/aws/Dockerfile` instala esse lock; portanto a imagem AWS inclui o SDK.

O nome do experimento é montado pela propriedade `experiment_name` do tracker: SHA curto da revisão como prefixo, nome da stack (`BENCH_STACK`, padrão `continual-bench`), o segmento `zia` e a política da unidade. Jobs nativos de Processing, Training e Tuning usam `<stack>-native` (`src/continual_bench/infrastructure/aws/native_tracking.py`). Filtros por campanha usam tags `cb-*`, como `cb-campaign-id`. Operação e acesso à interface estão em [Execução na AWS](Execucao-na-AWS) e [Operação e Diagnóstico](Operacao-e-Diagnostico).

## O que é registrado

| Categoria | Chaves ou caminhos no MLflow | Observações |
| --- | --- | --- |
| Métricas por janela | `accuracy`, `f1`, `brier` e outras, com `step = end_seq` | Avaliação binária |
| Resumos | `summary_*` (parâmetros) e `tracking_v1/*` (métricas) | `tracking_v1_status=complete` é a última escrita antes de encerrar o run |
| Observações por escopo | `observations_v1/quality/<split>/*`, `windows/*`, `effort/*`, `resources/*`, `retention/*`, `drift/*`, `step/*`, `epoch/*` | Linhagem em `tracking-observations/mlflow-lineage.json` |
| Métricas de sistema | `system/*` (CPU e memória do cgroup, processo, disco, rede, GPU quando houver) | Ver abaixo |
| Runs filhas | Uma por ajuste, ligadas por `mlflow.parentRunId`, com tags `cb-fit-id`, `cb-fit-action`, `cb-step-kind` e, quando conhecida, `cb-seed` | Step local começa em 1; `global_step` segue a ordem no run pai |
| Artefatos | `plots/<grupo>/*.svg`, `data/<grupo>/*.json`, `data/index.json`, `training-history.json`, `tracking-observations/`, `dashboard.html`, PNGs de `mlflow.log_figure()` | PNGs exigem matplotlib (extra `plots`); sem ele, os SVGs continuam sendo publicados |

As definições das métricas estão em [Métricas](Metricas) e no catálogo `configs/contracts/metric-definitions.json`. A presença de uma chave no catálogo não prova que ela foi capturada em uma execução: a referência é o relatório e o run.

### Métricas de sistema

`SystemMetricsSampler` lê o processo, o cgroup v2 e o host em uma thread separada: na partida, a cada `BENCH_SYSTEM_METRICS_INTERVAL` segundos e no fim. O intervalo padrão é **5,0 segundos** (`DEFAULT_INTERVAL` em `src/continual_bench/evaluation/system_metrics.py`, confirmado por `tests/test_run_artifacts.py`). A série é limitada a 3.600 amostras: ao atingir o limite, metade é descartada e o intervalo dobra. No modo `metrics_only`, o runner desativa esse limite. GPUs NVIDIA só aparecem quando `pynvml` e o driver existem. A amostragem observa e não altera predições nem decisões.

### Desempenho da publicação

`src/continual_bench/infrastructure/tracking/` concentra a lógica compartilhada entre o cliente MLflow local e o REST da AWS:

| Módulo | Função |
| --- | --- |
| `batching.py` | Respeita os limites do `runs/log-batch`: 1.000 métricas, 100 parâmetros, 100 tags e 1.000 entidades por requisição |
| `reconcile.py` | Busca históricos em paralelo para reconciliação idempotente |
| `rounds.py` | Publica runs filhas em paralelo |
| `artifacts.py` | Planeja o layout uma vez e envia arquivos com 8 workers (`DEFAULT_UPLOAD_WORKERS`) |
| `async_writer.py` | Progresso assíncrono fora da thread de treino |
| `backends.py` | Portas `MlflowClientBackend` e `RestBackend`; concorrência `BENCH_MLFLOW_PUBLISH_CONCURRENCY`, padrão 4 |

Os ganhos de tempo dessas otimizações foram medidos em testes com latência simulada. Não há medição com chamadas reais ao MLflow gerenciado na AWS registrada no repositório.

## ZenML

Com `--zenml`, a CLI executa o pipeline `finite_benchmark` (`src/continual_bench/orchestration/zenml_steps.py`), com as etapas `materialize`, `benchmark`, `audit_drift_and_retraining` e `persist`, sem cache e com logs por etapa. No modo de serviços, `ZENML_STORE_URL` precisa ser uma URL HTTP remota.

No Docker Compose, o projeto é `default` e a stack `benchmark-minio-v2` grava artefatos em `s3://zenml-artifacts` no MinIO (`compose.yaml`, `scripts/bootstrap_zenml.py`). Fora do Compose, o padrão é o projeto `continual-benchmark` e uma stack local `benchmark-local-v1`, cujos arquivos o servidor ZenML não consegue exibir. O ZenML não é usado no perfil AWS.

## Backfill e verificação

Os scripts abaixo leem execuções já concluídas e não treinam modelos. Vários assumem caminhos do container worker (`/app`, `/state`).

| Script | Finalidade | Entradas principais |
| --- | --- | --- |
| `scripts/backfill_training_history.py` | Inspeciona ou recupera o histórico por ajuste de runs antigas; com `--apply`, grava evidência derivada | fontes locais, `s3://.../final.json` ou `--run-id`; `--output-dir` |
| `scripts/backfill_tracking_metrics.py` | Enriquece runs existentes com `tracking_v1` sem substituir relatórios | `--manifest` (padrão `configs/history/campaign-nonlinear-v5.json`), `--state`, `--publish` |
| `scripts/verify_tracking_metrics.py` | Auditoria somente leitura de sidecars e históricos do MLflow | `--manifest`, `--state`, `--output` |
| `scripts/backfill_zia_artifacts.py` | Completa experimentos Zia a partir de relatórios retidos | IDs de experimento |
| `scripts/verify_dataset_suite.py` | Reconstrói escores e coortes pareadas a partir das predições; `--native-tracking` confere o MLflow | `--output` |
| `scripts/verify_versioned_campaign.py`, `collect_controlled_campaign.py`, `collect_versioned_references.py`, `summarize_versioned_campaign.py` | Coleta e auditoria das campanhas `controlled-v3` e `versioned-v4` dentro do worker | manifestos em `configs/history/` |
| `scripts/export_corrective_evidence.py`, `export_dataset_epoch_evidence.py`, `export_task_evidence.py` | Exportam evidência somente de métricas, sem copiar dados brutos ou modelos | `--state`, `--output` |

Exemplo de inspeção sem efeito colateral:

```bash
python scripts/backfill_training_history.py /caminho/dos/resultados
python scripts/backfill_training_history.py --apply --output-dir /caminho/novo /caminho/dos/resultados
```

## Pacotes de evidência

As evidências vigentes são as listadas abaixo. O dossiê da campanha corretiva (`evidencias/benchmark-corrigido/`) foi removido da árvore (commit `8e1a4415e`) e só pode ser consultado pelo histórico Git.

| Pacote | Origem | Conteúdo | Estado |
| --- | --- | --- | --- |
| `evidencias/aws-benchmarks/` | Campanha F109 na AWS, revisão `f1092ee0232a6d16504b3a2211a037c41a02068a`, protocolo `ordered-stream-batch-prequential-metrics-v2`, modelo XGBoost; resultados de consultas Athena identificadas em `provenance.json` | CSVs das consultas 12, 13b, 14b e 16, tabelas por seed e resumos, `manifest.json`, `provenance.json`, `validation.json`, `metric-dictionary.csv` | 132 unidades completas (4 streams × 11 políticas × 3 seeds) e 133 tentativas operacionais, uma delas a retomada de uma publicação sem novo treino (`provenance.json`) |
| `evidencias/aws-benchmarks/papers/` | Valores transcritos dos artigos CARA e UPF e médias dos harnesses históricos | 110 fatos de referência e 74 pares históricos, parâmetros, comparabilidade de protocolo e limitações | Separado da F109; não combinado numericamente com ela (`manifest.json`, campo `papers`) |
| `evidencias/semana-10/insects/` | Campanha `insects-72-v1` no AWS Batch com tracking no SageMaker MLflow; experimentos 403 a 408 (revisão `ffe0b89`) e 430 (revisão `31656c4`); consulta de 18/11/2026 (`consolidacao.md`) | `runs.csv`, `metricas-mlflow.csv` (24.723 registros), médias, métricas pareadas, regimes, detector ADWIN, janelas, 9 figuras, PDF e `SHA256SUMS` | 66 de 72 execuções completas (`dados/cobertura.csv`); faltam as seis de Online com Hoeffding (duas configurações × três seeds) |
| `evidencias/semana-10/cara-upf/` | Execuções locais dos protocolos dos artigos em 17/11/2026, base `178939e26bead00c58c211668d2c7b84bf668aed` (`dados/manifesto-execucao.json`) | JSONs comprimidos de UPF (10 seeds), CARA (5 seeds), comparação pareada (10 seeds), trajetórias, 6 figuras e `SHA256SUMS` | Completo para o escopo declarado; `git_revision` registrado como `uncommitted` |
| `reports/semana-10-narrativa/` | Material local da publicação das figuras da Semana 10 | Corpo do PR, recibos do GitHub e um `mlflow.db` de testes | Não versionado (`reports/` está no `.gitignore`) |

Os quatro CSVs nomeados por identificador de consulta foram recodificados das linhas salvas da API Athena; o `provenance.json` registra que isso não afirma identidade de bytes com o CSV original no S3. Também registra `native_mlflow_histories_requeried: false` e `canonical_full_validation_claimed: false`. O hash do catálogo de métricas registrado na F109 coincide com o de `configs/contracts/metric-definitions.json` na árvore atual.

A F109 cobre Electricity, Airlines e duas variantes binárias de INSECTS. Datasets, políticas e detectores estão descritos em [Datasets](Datasets), [Modelos e Políticas](Modelos-e-Politicas) e [Detectores de Drift](Detectores-de-Drift).

## Como ler os resultados com cautela

**Coortes diferentes não se misturam.** Na F109, a comparação principal usa `metric_scope = common_postcalibration`: a mesma faixa de eventos após a calibração e antes do holdout, compartilhada por todas as políticas e seeds de cada stream (limites em `cases.csv` e no README do pacote). Os escopos `prequential`, `holdout` e por classe continuam nos CSVs e não devem ser comparados com a coorte comum. No histórico INSECTS, as métricas pareadas usam a interseção dos eventos dos 33 runs de cada fluxo (236.587 eventos no abrupto e 301.228 no recorrente, `dados/metricas-pareadas-por-seed.csv`); `medias.csv` usa o recorte original de cada configuração.

**Streams não formam uma população única.** Cada stream é um problema distinto. Não se deve fazer um ranking agregado entre streams nem tirar médias de métricas de streams diferentes.

**Seeds não são amostras independentes.** Os desvios-padrão usam divisor `n - 1` (`sd_ddof=1`) entre três repetições computacionais sobre o mesmo fluxo ordenado. As barras dos gráficos não são intervalos de confiança. Na F109, as métricas de qualidade são idênticas entre as seeds 42, 43 e 44: em `quality-summary.csv`, `macro_f1` e `pr_auc` têm `sample_sd = 0` em todas as 44 combinações de stream e política. Por exemplo, Electricity com `frozen` tem `macro_f1` médio 0,6836 e desvio 0. Os únicos desvios diferentes de zero nas métricas de qualidade são de ordem 1e-14 (`log_loss`, `brier`), compatíveis com arredondamento numérico. A variação entre seeds aparece nos recursos (tempos e memória), não na qualidade. Portanto, essa campanha não fornece estimativa de incerteza da qualidade.

**Recursos são descritivos.** As 132 avaliações rodaram em oito modelos de CPU diferentes (`hardware.csv`) sob a mesma reserva de 2 vCPU e 4 GiB. Políticas derivadas reutilizam matrizes de candidatos do CARA, então o tempo de uma unidade consumidora não equivale ao custo de uma execução independente. `train_seconds` acompanha ajustes lógicos promovidos; `actual_total_fit_seconds`, o trabalho físico. Custos de cenário ou de tabela de preços não são fatura observada; ver [Custos e Recursos](Custos-e-Recursos).

**Ausência não é zero.** Células vazias são NULL. Em `resources-by-seed.csv`, 168 fatos ausentes estão marcados com `source_fact_present=False`, e os arquivos não permitem distinguir se cada ausência é métrica indefinida, inaplicável ou não publicada.

**Unidades e definições importam.** ROC AUC e AP usam histogramas de 4.096 bins; `pr_auc` é precisão média não interpolada. A AUC do UPF (`upf_paper_auc`) integra custo combinado sobre a grade de α, deve ser minimizada e não é AUROC. O SCPE é excesso percentual em relação ao Oracle do protocolo CARA.

**Comparação com artigos é descritiva.** Os arquivos de `aws-benchmarks/papers/` registram que mesmo nome de métrica e de dataset não torna os protocolos equivalentes. `protocol-comparability.csv` lista dez dimensões em que artigos, harnesses históricos e F109 diferem (origem, modelo, partição, pré-processamento, objetivo, disponibilidade de rótulos, coorte, métricas e dispersão). Pontos que exigem atenção, conforme `papers/limitations.json`:

- F1 e AP da F109 não correspondem a SCPE, acurácia ou AUC de custo publicados; não se calculam diferenças ou rankings entre F109 e os artigos.
- O harness histórico UPF usa desvio populacional (`ddof=0`) com dez seeds; o valor após ± nos artigos foi transcrito sem atribuir tipo estatístico.
- O harness histórico CARA ajustou o pré-processamento no fluxo inteiro (convenção transdutiva) e não preservou desvios por seed.
- A rota histórica que adaptou CARA ao protocolo UPF usou rótulos do teste corrente na obsolescência.
- O artigo CARA tem uma divergência entre a Tabela 1 e a seção 7.2 sobre o número de lotes online, mantida explícita.

As fontes são *Cost-aware retraining for machine learning* (2024) e *When to retrain a machine learning model* (2025); o histórico INSECTS usa os fluxos descritos em *Challenges in benchmarking stream learning algorithms with real-world data* (2020). Ver [Referências](Referencias).

**Execuções incompletas ficam visíveis.** No INSECTS, as duas configurações Online com Hoeffding aparecem só em um diagnóstico parcial (`dados/janelas-parciais-online-hoeffding.csv`), com janelas anteriores ao primeiro drift anotado, e não entram nas comparações finais.

## Limitações

- Os experimentos AWS são nomeados por `<sha>-<stack>-zia-<política>` (`src/continual_bench/infrastructure/aws/mlflow_tracking.py`); confirme o nome real na interface do MLflow antes de montar consultas.
- Os carregadores do notebook não têm teste automatizado; a validação registrada em `validation.json` aponta para um script de regressão fora do versionamento.
- O termo F109 identifica a campanha nos arquivos, mas não há documento no repositório que defina sua origem.

Veja também [Reprodutibilidade e Artefatos](Reprodutibilidade-e-Artefatos), [Metodologia Experimental](Metodologia-Experimental) e [Limitações e Trabalhos Pendentes](Limitacoes-e-Trabalhos-Pendentes).
