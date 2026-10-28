# Laboratório de aprendizado contínuo — Semana 08

Fluxo de processamento desacoplado para experimentos com mudança de conceito (drift) e atualização de modelos. Apache Airflow coordena **tarefas com execução finita**; ZenML (projeto independente, não Apache) organiza quatro etapas e seus artefatos; Kafka transporta eventos; MinIO guarda dados e pontos de recuperação; MLflow rastreia experimentos. Somente o núcleo Python controla o ciclo por amostra/lote. O agendador do ZenML não controla o fluxo contínuo de dados.

O projeto original foi confirmado pelo chat Codex `01a1789a-e60c-77d0-895d-caa196f9d2c5`: `/Users/gabrieloliveira/residencia-ia`, ramificação `Semana-08`, base `ca10670518ec9739f08a305aa344b6f4fe044e3e`. Código enviado ao servidor do laboratório doméstico: `/home/gabriel/residencia-ia-benchmark`.

## Começar sem serviços

Requer Python 3.12. Dependências do serviço executor (worker) fixadas em `uv.lock` e exportadas com hashes em `requirements.lock`. Airflow usa um ambiente Python separado.

```bash
uv sync --frozen --extra dev --extra services
PYTHONPATH=. uv run pytest -q
uv run continual-bench run configs/streaming.yaml
uv run continual-bench run configs/incremental.yaml
uv run continual-bench run configs/offline.yaml
uv run continual-bench smoke --state state/smoke --samples 600 --baselines
```

O último comando executa **21 experimentos limitados**: três fontes × dois protocolos de avaliação antes do treino (prequential) × três modelos de referência, mais três avaliações com separação temporal entre treino e teste. O modo local de processamento contínuo lê JSONL, sem Kafka, e o relatório identifica `transport: local_jsonl`. Isso valida o algoritmo e o protocolo e não valida serviços reais. Para comparações maiores, variar a semente aleatória, o tamanho da amostra, o tamanho do lote, o detector e a política em configurações versionadas. O teste rápido não deve ser apresentado como uma avaliação de desempenho ou uma demonstração do estado da arte.

Avaliação comparativa ampliada, executada separadamente: `uv run python scripts/benchmark_matrix.py --samples 30000 --seeds 42 43 44`. Executa 63 configurações em subprocessos separados e salva `matrix-summary.json`; acrescentar `--services --zenml` somente com o conjunto de serviços ativo. Não é executado automaticamente pelo pytest ou pelo teste rápido.

## Conjuntos de dados e protocolos

| Fonte | Variante | Mudança de conceito sintética conhecida | Justificativa |
|---|---|---|---|
| SEA | River 0.26.1, variantes 0→1→2, sem ruído | índices `N//3` e `2*N//3` | Classificação numérica leve |
| Agrawal | funções 0→1→0, sem balanceamento/perturbação | mesmos limites, com recorrência | Regras distintas e conceitos recorrentes |
| STAGGER | funções 0→1→2, sem balanceamento | mesmos limites | Atributos categóricos e mudança de regra |

Os três geradores sintéticos River têm código BSD-3-Clause. Foram integradas também duas fontes reais fixadas por versão e SHA-256:

| Fonte real | Variante admitida | Linhas brutas | Alvo positivo |
|---|---|---:|---|
| Electricity | OpenML 151, versão 1 | 45312 | `class=UP` |
| Airlines | OpenML 1169, versão 1 | 539383 | `Delay=1` |

O catálogo OpenML declara licença `Public` para ambas; preservamos esse rótulo, sem convertê-lo em uma licença SPDX. Os ARFF completos são validados antes do pipeline e armazenados com hash. O adaptador usa um prefixo na ordem publicada, sem embaralhar, com timestamps e atraso de rótulo simulados; a cronologia original e o atraso real não foram estabelecidos. Electricity foi normalizado/imputado pelo publicador, cujo escopo de ajuste não está documentado. Não há pontos de mudança conhecidos nem probes sintéticos para essas fontes. Consulte [fontes fixadas](configs/real-sources.lock.json) e [operação](docs/benchmark-operations.md#datasets-reais-e-campanha-versionada-v4).

Cada manifesto registra fonte e versão, semente aleatória, SHA-256 do JSONL, limites dos conceitos e tipo de timestamp. Os timestamps são **simulados**, `epoch + sequência` em segundos; não representam medições do mundo real. IDs estáveis derivam de versão da fonte e sequência. A ordem original nunca é embaralhada. Cada segmento usa um gerador com semente aleatória explícita. Os conjuntos de diagnóstico fixos (probes) usam outra semente, nunca são usados no treino e não são fornecidos ao detector nem à política.

- **Processamento contínuo (streaming)**: teste antes do treino por evento; Kafka real somente com `--services`. Um tópico/partição por configuração preserva a ordem. A ingestão é simulada sem espera de relógio; timestamps permanecem no evento.
- **Processamento incremental**: todas as predições do lote ocorrem antes de qualquer treino daquele lote. Rótulos são liberados no fim do lote, de acordo com `available_at`. Esse protocolo pode ter resultados diferentes do streaming.
- **Processamento em lote estático (offline)**: treino no prefixo temporal, avaliação congelada no sufixo. Remove do treino o intervalo de rótulos ainda indisponíveis no início do teste. A política deve ser `none`. Não é comparável diretamente com a avaliação antes do treino (prequential).

Na avaliação prequential, `warmup` é o trecho inicial usado no aquecimento: cada política o utiliza uma única vez para treinar. Esse trecho é excluído das métricas. Depois dele, o detector recebe somente o erro de uma predição feita antes da disponibilidade do rótulo. Ao final, há drenagem explícita dos rótulos pendentes: a avaliação considera horizonte `último timestamp + atraso`. Fontes novas devem manter `timestamp` estritamente crescente e `available_at` não decrescente; atrasos fora de ordem exigem outro adaptador de liberação, não uma mudança silenciosa de protocolo.

## Experimentos e extensão

`domain/` contém classes de dados, contratos, métricas, decisões e motor; não importa River, Kafka, ZenML, MinIO, MLflow ou Pydantic. `application/` valida configurações, monta componentes por injeção e coordena tarefas. `adapters/` implementa ferramentas concretas. As estratégias são substituíveis no registro de componentes sem modificar o motor (DIP/OCP); contratos têm responsabilidades separadas, sem classe de serviço onipotente.

| Componente | Implementações iniciais | Interface |
|---|---|---|
| Modelo | regressão logística River + padronização numérica com StandardScaler + codificação categórica indicadora (one-hot) | `predict(x) -> probabilidade`, `learn(x,y)` |
| Detector | ADWIN, DDM, nenhum | `update(erro_rotulado) -> alarme` |
| Política | nenhuma, periódica, por mudança de conceito, online | `decide(contexto) -> ação/motivo`, `updated(contagem)` |
| Fonte | SEA, Agrawal, STAGGER, Electricity, Airlines | eventos ordenados + manifesto + conjuntos de diagnóstico opcionais |
| Persistência | ponto de recuperação local com gravação atômica, S3 MinIO, MLflow remoto | interfaces em `domain/contracts.py` |

Detector **não** decide retreinamento. As políticas de retreinamento impõem `cooldown`, `min_samples`, `min_window` e frequência. A política `online` atualiza a cada rótulo disponível após o aquecimento, sem esses guardas de retreinamento. `reset_retrain` cria modelo/preprocessador novos e treina do zero na janela limitada de rótulos já disponíveis; **não** reprocessa todo o histórico. `learn_one` atualiza o modelo existente só com o exemplo liberado. `reset_detector_on_retrain` torna explícito o reinício do detector. O modelo de referência `none` congela modelo/preprocessador após o aquecimento.

Para adicionar uma extensão (plugin), implementar o contrato, registrar uma função de criação importável no nível do módulo e declarar `plugins: [modulo:register]`. Ver `examples/custom_plugin.py`; passar os testes de contrato, serialização, preservação de ordem e retomada. As extensões executam código Python confiável e devem ser instaladas pelo operador. O ponto de recuperação guarda o hash do pacote completo (incluindo recursos); auxiliares de outros pacotes devem ser declarados em `plugin_packages: [nome_do_pacote]`, inclusive importações tardias. Extensões fora de um pacote são tratadas como módulos únicos e exigem declaração explícita dos auxiliares. A API de tarefas rejeita extensões enviadas por HTTP. Para uma fonte real, incluir manifesto com licença, URL, versão e soma de verificação, esquema e significado temporal; `concept=None` e sem pontos de mudança nem conjuntos de diagnóstico inventados. Não baixar ou ordenar dados silenciosamente.

`examples/temporal_csv.py` fornece um adaptador executável de referência, sem dados reais incluídos: registrar `examples.temporal_csv:register`, escolher `dataset.name: temporal_csv` e declarar em `dataset.params` path, SHA-256, license_id, source_url, timestamp_column, label_column e feature_columns. Exige timestamps inteiros UTC já ordenados e variável-alvo binária (0/1); impede o uso da variável-alvo e do instante do evento como atributos de entrada. Não adquire nem licencia conjuntos de dados pelo usuário. Em execução local, instalar o módulo ou usar `PYTHONPATH=.`; o contêiner inclui os exemplos no caminho de busca de módulos.

## Iniciar no servidor do laboratório doméstico

**Atualização em 27/10/2026:** MLflow corrigido com senha exclusiva e limites explícitos de threads; worker com prontidão autenticada. Airflow disponibiliza `benchmark_streaming`, `benchmark_incremental`, `benchmark_offline`, `benchmark_strategies`, `benchmark_full`, `benchmark_controlled` e `benchmark_versioned`, além do smoke. As DAGs são manuais e parametrizadas. ZenML usa o projeto `default`, identificado como **Continual Benchmark**, porque o painel OSS 0.98.0 não exibe projetos adicionais. A stack dedicada `benchmark-minio-v2` armazena novos logs e artefatos no bucket privado `zenml-artifacts`; a leitura de logs e visualizações pela API do servidor foi validada. Execuções antigas com armazenamento local são preservadas. Consulte [Operação do benchmark](docs/benchmark-operations.md) para executar o fluxo completo e [validação de 27/10](reports/validation-20261027.md) para a evidência atual. Usar `-f compose.yaml -f compose.homelab.yaml` nesta implantação.

O diagnóstico inicial encontrou 28 CPUs lógicas, 30 GiB RAM, 762 GiB livres em `/`, Docker 29.1.3 e Compose 2.40.3, sem contêineres existentes. Limites do conjunto de serviços ficam no Compose, com fila de 16 tarefas e execução de até oito avaliações comparativas simultâneas. Nenhum Docker socket é montado.

1. Manter o arquivo de ambiente privado fora do Git no servidor. `.env.example` contém somente valores de exemplo. MinIO exige usuário de pelo menos 3 caracteres e senha de pelo menos 8. MLflow 3.17.2 exige senha de pelo menos 12 em `MLFLOW_PASSWORD`, independente das outras interfaces. O arquivo precisa das credenciais das interfaces web, MinIO e Postgres, Fernet/JWT do Airflow e segredo de sessão Flask do MLflow.
2. Validar estrutura sem imprimir valores: carregar o arquivo aprovado no ambiente e executar `python3 scripts/preflight.py`. Não executar `docker compose config` sem `--quiet` usando credenciais reais.
3. Construir e iniciar somente este conjunto de serviços:

```bash
cd /home/gabriel/residencia-ia-benchmark
docker compose --env-file /caminho/privado/benchmark.env config --quiet
docker compose --env-file /caminho/privado/benchmark.env build
docker compose --env-file /caminho/privado/benchmark.env up -d --wait --wait-timeout 300
docker compose --env-file /caminho/privado/benchmark.env ps
```

O comando `up` inicializa os buckets de benchmark e MLflow idempotentemente, migra apenas o Postgres deste conjunto de serviços e cria o usuário Airflow aprovado. Não remover volumes. Não reutilizar diretórios/volumes de outros sistemas. `docker compose stop` interrompe apenas este conjunto de serviços, quando solicitado; não usar `down -v` em dados persistentes.

As únicas portas publicadas vinculam exclusivamente ao IP privado aprovado `192.168.100.27`:

| Interface do notebook | URL |
|---|---|
| Apache Airflow | `http://192.168.100.27:18081` |
| ZenML | `http://192.168.100.27:18082` |
| MLflow | `http://192.168.100.27:15000` |
| Console MinIO | `http://192.168.100.27:19001` |

Kafka, Postgres, API S3 e API de tarefas ficam somente na rede Docker. Nenhuma publicação à internet, túnel obrigatório ou alteração de firewall/SSH/VPN faz parte da implantação. Login/senhas são os valores aprovados no arquivo externo. Nunca copie esse arquivo para o Git, relatórios ou registros de execução. A lista de hosts permitidos do MLflow é explícita; autenticação básica é separada dessa proteção. O console MinIO é separado da API S3 interna.

## Validar serviços reais

```bash
docker compose --env-file /caminho/privado/benchmark.env exec worker \
  pytest -q -m services tests/test_services.py
docker compose --env-file /caminho/privado/benchmark.env exec worker \
  continual-bench smoke --state /state/full-smoke --samples 600 --services --zenml --baselines
docker compose --env-file /caminho/privado/benchmark.env exec airflow-scheduler \
  airflow dags list-import-errors
docker compose --env-file /caminho/privado/benchmark.env exec airflow-scheduler \
  airflow dags test benchmark_smoke
```

O teste de integração exige Kafka, MinIO, MLflow e ZenML reais; verifica quinze combinações de dataset/protocolo, além da proteção de origem do painel, objetos baixados e comparados por SHA-256, rastreamento remoto, URI `mlflow-artifacts:`, reprocessamento sem métricas duplicadas. Depois, acionar `benchmark_smoke` na interface ou com `airflow dags trigger benchmark_smoke` valida o agendador em operação. A DAG tem uma checagem inicial `services_ready` e nove tarefas manuais com execução finita, `schedule=None`, limite de execução, novas tentativas e submissão idempotente. Não dizer que simulações ou execução local validaram esses serviços.

Em cada `state/<run_key>/`: configuração, JSONL, manifesto, ponto de recuperação, relatório e, após publicação real, `receipt.json` com identificador MLflow e URIs S3. Artefatos e logs dos quatro passos ZenML usam `s3://zenml-artifacts`, com endpoint MinIO explícito e segredo ZenML de uma identidade restrita a esse bucket. O bootstrap é `python scripts/bootstrap_zenml.py`, após provisionar o bucket/identidade/segredo conforme [docs/benchmark-operations.md](docs/benchmark-operations.md). O servidor lê os novos logs e visualizações sem liberar acesso a arquivos locais. O volume local e as execuções históricas foram preservados; seus URIs não foram migrados.

## Métricas, falhas e limites

Registramos acurácia, F1 binário, pontuação de Brier e perda logarítmica por janela de rótulos liberados e no agregado; alarmes com IDs/tempo de disponibilidade; decisões e duração/tamanho de treino; tempo total e tempo de processamento e pico RSS do processo. RSS representa o maior valor de memória residente registrado pelo processo, não memória exclusiva do modelo; o servidor isola cada tarefa em subprocesso. O comando `smoke` da interface de linha de comando executa várias configurações no mesmo processo e seus valores de RSS não devem ser comparados como consumo isolado.

Conjuntos de diagnóstico sintéticos medem desempenho em conceitos já vistos e queda em relação ao melhor resultado anterior nesses conjuntos. São diagnósticos de retenção; não são avaliação independente depois de repetidas consultas, não controlam políticas e não estabelecem causalidade de esquecimento. Mudanças de conceito e alarmes **não são** medida de esquecimento. Não há alegação de estado da arte, nem ponto de mudança real inventado. Consultar `docs/research.md` e `docs/reliability.md`.

Separação de testes: `pytest` rápido não usa serviços nem avaliação comparativa completa; `-m services` é integração externa; experimentos maiores são comandos explícitos. Consulte `reports/validation.md` para a evidência efetivamente executada e os bloqueios, em vez de inferir que a configuração foi implantada.

MinIO comunitário está arquivado e sem manutenção ativa. Este laboratório fixa a versão de segurança de outubro de 2025 pelo commit oficial, compila o código-fonte e conserva licença AGPL. Não usa imagem `latest`, produto pago ou substituto silencioso. Isso não equivale a um serviço mantido para produção; dependências, cópias de segurança e acesso precisam de revisão antes de outro uso.

## Online corrigido e campanha controlada

A DAG `benchmark_controlled` executa a campanha `controlled-v3`: cinco sementes (42–46),
3000 amostras por configuração e tetos de 2900/5800 apresentações de exemplos ao treino após o warmup.
São 261 execuções, incluindo offline e controles que reproduzem os resultados anteriores.
`online` passa a executar 2900 atualizações individuais por fluxo, inclusive os rótulos drenados ao final.
O controle histórico usa outro nome e ação periódica esparsa, preservando a evidência anterior.

O Airflow admite oito tarefas `evaluate_and_publish` simultâneas e o worker executa oito subprocessos
com configurações SDK ZenML isoladas. O contador real de concorrência é exposto em `/health`.
O teto de treino é medido em chamadas de aprendizado, incluindo replays; não equivale a segundos
de CPU nem força os detectores a gastar todo o orçamento. Detalhes em
[Operação do benchmark](docs/benchmark-operations.md#campanha-controlada-v3).

Manifesto: [campaign-controlled-v3.json](configs/campaign-controlled-v3.json).
A comparação reproduzível usa `scripts/compare_controlled_campaign.py` e preserva os protocolos
separados, a semente 42 histórica e a dispersão entre as cinco sementes.


## Datasets e modelos versionados

`benchmark_versioned` executa `versioned-v4`: 255 configurações sintéticas (cinco sementes)
e 34 configurações reais (um prefixo fixo de Electricity e outro de Airlines), com oito tarefas
`evaluate_and_publish` paralelas. O manifesto [campaign-versioned-v4.json](configs/campaign-versioned-v4.json)
fixa 3000 exemplos, atraso 3 e os orçamentos da v3. Sementes não criam repetições independentes
dos datasets reais; o adaptador registra explicitamente `seed_ignored`.

A matriz v4 usa cinco experimentos MLflow: congelado, online, periódico, drift/ADWIN e drift/DDM.
Cada run registra DatasetInput, origem/licença/versão/hash dos dados, configuração, ambiente,
métricas, hash do código e modelo. O Model Registry armazena o modelo final de cada run,
com pré-processamento, assinatura e dependências para inferência via `models:/nome/versão`.

O primeiro ajuste e cada atualização posterior têm estados serializados e versões imutáveis
no MinIO, com cadeia de pais e seleção dos dados de treino. Para online, isso inclui cada rótulo
aprendido; o índice `model-versions.json` permite recuperar qualquer estado. Os estados intermediários
ficam no catálogo por hash; o Registry recebe uma versão final por run. O dataset bruto, os eventos
utilizados e cada seleção de treino são rastreáveis pelos mesmos hashes. Os experimentos históricos
em `continual-bench-v1` permanecem disponíveis.

Validação completa: `scripts/verify_versioned_campaign.py`, executado dentro do worker.

Evidência da campanha completa de 27/10/2026: [versioned-v4.md](reports/versioned-v4.md).

Análise das políticas e detectores nos cinco datasets: [versioned-v4-analysis.md](reports/versioned-v4-analysis.md).

## Modelos não lineares e campanha v5

A DAG `benchmark_nonlinear` compara logística, Hoeffding Tree e kNN com os mesmos 289 casos
por família da v4: cinco datasets, cinco sementes sintéticas, prefixos reais fixos e tetos de
2900/5800 apresentações de treino após o aquecimento. São 867 configurações, com até oito
`evaluate_and_publish` simultâneas. O treino inicial e cada atualização preservam modelo e
seleção dos dados; o MLflow registra família/parâmetros e mantém os experimentos por política/detector.

Consulte o [contrato e parâmetros](docs/nonlinear-v5-protocol.md), o
[manifesto por run](configs/campaign-nonlinear-v5.json) e a
[comparação com a campanha v4](reports/nonlinear-v5.md). A análise distingue qualidade preditiva,
probabilidades, apresentações de treino e tamanho dos artefatos; os tetos não igualam o custo de CPU.

## Tracking ampliado das métricas da residência

O [catálogo versionado](configs/metric-definitions.json) e o
[contrato de métricas](docs/tracking-metrics.md) distinguem medidas calculadas, definições
condicionais e evidências ausentes. As novas runs registram classificação, ranking/calibração,
retenção e transferência por episódio, drift/recuperação, custo configurável e recursos.
`tracking_v1/summary/*` contém os agregados no MLflow; `tracking_v1/*` contém as séries.
O artefato `tracking/residency-metrics-v1/<hash>/payload.json` traz definições, escopos e
motivos de indisponibilidade. Ausência de dados não vira zero.

A DAG `benchmark_metrics_smoke` verifica oito combinações dos cinco datasets e três modelos;
`benchmark_metrics` prepara a matriz completa com uma identidade nova, preservando a v5.
Ambas mantêm o limite de oito tarefas `evaluate_and_publish`. O parâmetro `campaign` permite
nomear a campanha e `tracking` configura os coeficientes e a instrumentação. Referências e
comparações com cópias do modelo anterior acrescentam custo de avaliação, medido separadamente.
O perfil [tracking-metrics-profile.json](configs/tracking-metrics-profile.json) usa unidades
sintéticas de cenário, sem representar preços observados. O exemplo
[metrics-example.yaml](configs/metrics-example.yaml) executa localmente com:

```bash
uv run continual-bench run configs/metrics-example.yaml --state state/metrics
```

Para enriquecer campanhas históricas, `scripts/backfill_tracking_metrics.py` gera um sidecar
com proveniência e acrescenta somente métricas deriváveis às runs existentes. Não chama o
publisher de modelos nem altera relatórios, checkpoints, dados ou versões originais. AUROC,
calibração e referências não gravadas permanecem indisponíveis nessas runs.
