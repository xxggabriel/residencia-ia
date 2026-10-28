# Operação do benchmark — Semana 08

## Interfaces

- Airflow: <http://192.168.100.27:18081>
- ZenML: <http://192.168.100.27:18082>
- MLflow: <http://192.168.100.27:15000>

MLflow usa `WEB_USER` e a senha exclusiva `MLFLOW_PASSWORD`, no arquivo privado
`/home/gabriel/.config/continual-bench/benchmark.env`. A cópia local fica em `.env`.
Esses arquivos têm modo 600 e não integram o Git. As outras interfaces mantêm suas credenciais.

## Executar no Airflow

Todas as DAGs usam `schedule=None`: ativá-las permite disparos manuais, sem iniciar experimentos periódicos.

| DAG | Configurações por semente | Escopo |
| --- | ---: | --- |
| `benchmark_full` | 33 | Todos os três protocolos e estratégias |
| `benchmark_streaming` | 15 | Avaliação prequential por evento, com Kafka |
| `benchmark_incremental` | 15 | Avaliação prequential por lote |
| `benchmark_offline` | 3 | Holdout temporal congelado |
| `benchmark_strategies` | 30 | Comparação das estratégias nos dois modos prequential |
| `benchmark_smoke` | 9 | Verificação rápida original |
| `benchmark_controlled` | 51 + controles | Campanha v3, 261 configurações no padrão |
| `benchmark_versioned` | 51 sintéticas + 34 reais fixas | Campanha v4, 289 configurações no padrão |

Em **Trigger DAG**, configure `samples` (300–100000, padrão 3000), `seeds`
(1–10 sementes distintas, padrão `[42]`) e `label_delay` (0–100, padrão 3).
Os três datasets são SEA, Agrawal e STAGGER. Cada protocolo prequential compara:
modelo congelado, atualização periódica, atualização online a cada rótulo disponível,
retreinamento por ADWIN e retreinamento por DDM. O modo offline permanece congelado.

Fluxo Airflow: `services_ready → plan → evaluate_and_publish [mapeado] → consolidate_results`.
Cada tarefa mapeada envia uma configuração validada ao worker e aguarda publicação completa.
O resultado consolidado aparece no retorno/XCom da última tarefa e em seu log, com os IDs MLflow.
As configurações são determinísticas: repetir a mesma matriz reaproveita trabalhos concluídos;
alterar uma configuração gera outra identidade. DAGs de protocolos e a DAG completa compartilham
identidades para evitar repetir o mesmo experimento. Falhas não são convertidas em sucesso.

O worker executa até oito avaliações em subprocessos isolados; Airflow admite até oito tarefas por DAG e oito globalmente, com timeout e retry.
Para uma campanha com configurações novas, o tempo inclui captura dos logs ZenML e publicação autenticada.

## ZenML

A instalação usa ZenML OSS 0.98.0. Seu SDK aceita o projeto adicional `continual-benchmark`,
criado durante a correção, mas o próprio SDK informa que o dashboard OSS só exibe `default`.
Por isso, a operação final usa o projeto `default`, com display name **Continual Benchmark**
e descrição específica. O projeto adicional e as execuções de diagnóstico nele foram preservados;
não foi contratada uma edição paga nem contornada essa limitação.

A pipeline `finite_benchmark` tem quatro passos visíveis:

1. `materialize`: gera dados e manifesto reproduzíveis.
2. `benchmark`: realiza predição, liberação dos rótulos, detecção de drift e decisões de atualização.
3. `audit_drift_and_retraining`: verifica completude, modelo congelado, alarmes e cooldown.
4. `persist`: publica relatório/checkpoint/dados no MinIO e métricas/artefatos no MLflow.

O ciclo temporal permanece no motor Python. Dividi-lo em tarefas independentes de detector e treino
por evento alteraria a ordem prequential. O passo de auditoria inspeciona decisões efetivamente tomadas;
os pontos de mudança sintéticos nunca são usados como gatilho de retreinamento.

A stack `benchmark-minio-v2` usa um orquestrador local no worker e armazenamento S3 privado.
Logs e visualizações novos são lidos pelo servidor no MinIO. O SDK `s3fs==2026.9.0` foi acrescentado
ao lock; `boto3` passou a 1.43.117 para compatibilidade com `aiobotocore==3.10.0`.

## Bootstrap e recuperação

O arquivo Compose exige `MLFLOW_PASSWORD` explícito. Não usa a senha das demais interfaces como fallback.
`scripts/preflight.py` rejeita senha ausente, curta ou de exemplo. MLflow limita threads BLAS/OMP
para não esgotar o limite de processos do contêiner. A origem exata `http://192.168.100.27:15000`
é permitida em `--cors-allowed-origins`: somente `--allowed-hosts` não libera as chamadas POST do painel.
Uma configuração incorreta de origem faz a UI apresentar `INTERNAL_ERROR`, mesmo com a API Python funcional.

Antes do bootstrap ZenML, devem existir:

- bucket privado `zenml-artifacts`;
- identidade MinIO `zenml-benchmark`, restrita a esse bucket;
- segredo ZenML `benchmark-minio-artifacts`, com campos `key` e `secret`.

O backup privado da identidade está em `/home/gabriel/.config/continual-bench/zenml-minio.json`, modo 600.
O segredo é referenciado pelo armazenamento, sem credenciais literais na configuração da stack.
A política efetiva concede localização/listagem do bucket e leitura/escrita/remoção de seus objetos,
incluindo operações multipart. Não concede acesso aos buckets `benchmarks` e `mlflow`.

Depois de provisionar esses recursos, executar no diretório remoto:

```bash
docker compose --env-file /home/gabriel/.config/continual-bench/benchmark.env \
  -f compose.yaml -f compose.homelab.yaml exec -T worker python scripts/bootstrap_zenml.py
```

O script é idempotente e registra projeto/armazenamento/stack. Não cria credenciais administrativas.
Volumes, execuções anteriores, checkpoints e os URIs locais históricos são preservados.
O servidor continua com a restrição de acesso local habilitada.

## Resultados e limites

As execuções históricas usam `continual-bench-v1`. As novas são separadas por política/detector
conforme a seção v4 abaixo, com nomes por dataset, modo, detector, política e semente. Registra métricas por janela, métricas agregadas `summary_*`, parâmetros, hash do código,
hash dos dados, contagens de alarmes/atualizações e artefatos JSON. Relatórios completos também estão
em `s3://benchmarks/<run_key>/report.json`; os recibos ligam cada execução ao MLflow e aos objetos.

Streaming, incremental e offline mantêm protocolos próprios; a consolidação não mistura suas métricas.
A campanha padrão é uma validação funcional com dados sintéticos e uma semente, não uma conclusão
estatística sobre superioridade das estratégias. Para pesquisa, ampliar amostras/sementes e revisar
os demais parâmetros de cada configuração em experimentos controlados.

## Campanha controlada v3

A DAG manual `benchmark_controlled` usa cinco sementes (42–46), 3000 amostras e atraso de rótulo 3.
O manifesto executável é `configs/campaign-controlled-v3.json`: 261 configurações únicas
(240 prequential, 15 offline e seis controles que reproduzem o antigo online esparso na semente 42).
O online agora aprende cada rótulo disponível após o aquecimento, sem cooldown, incluindo a drenagem final.
No incremental, todas as predições do lote continuam anteriores às atualizações daquele lote.

`training_budget` limita chamadas de treino após o warmup, contando reapresentações da janela.
Os tetos são 2900 e 5800, sem incluir os 100 exemplos comuns de aquecimento. O online precisa
de 2900 chamadas e tem um único braço; o congelado consome zero. Periódico/ADWIN/DDM têm os dois tetos.
Tetos comuns não significam consumo idêntico: drift não recebe retreinamentos forçados para gastar orçamento.
Ao atingir o teto, o modelo permanece congelado, mas métricas e detector continuam recebendo rótulos.
O último retreinamento pode usar somente o sufixo mais recente que cabe no orçamento, respeitando min_window.
Orçamentos incompatíveis com atualização online a cada rótulo são rejeitados.

Os controles `controlled-v3-bridge` usam política periódica de 100 rótulos com ação `learn_one`,
reproduzindo as 29 atualizações do antigo online para validar a comparação histórica sem reintroduzir o erro.
As execuções antigas e seus checkpoints permanecem intactos. Cada novo subprocesso tem seu próprio
`ZENML_CONFIG_PATH`; configuração SDK mutável não é compartilhada entre os oito processos.

### Resultado verificado da campanha v3

Executada em 27/10/2026, de 14:40:54 a 14:57:13 (America/Sao_Paulo): 261 configurações
e 264 tarefas Airflow concluídas, sem retries. Foram registrados oito evaluate_and_publish e oito
execuções do worker ativos simultaneamente em 223 das 474 amostras do monitor.
Os 261 registros MLflow FINISHED e relatórios S3 foram conferidos; os 33 controles históricos
reproduziram métricas, dados e, quando disponível, o digest das predições.

Com cinco sementes, o online corrigido liderou a acurácia média nos seis cenários de dataset/protocolo.
Com 2900 apresentações pós-warmup, teve também as melhores médias de F1, Brier e log loss
entre os braços prequential dessa campanha. O periódico consumiu exatamente 2900 apresentações
no teto menor e 3584 no teto de 5800, contra 2900 do online.
O teto maior preservou vantagens específicas: F1 no STAGGER e log loss em Agrawal; no incremental
Agrawal, também Brier. Não existe dominância do online em todas as métricas com orçamento maior.

Nos detectores, ambos os tetos produziram predições idênticas nos 60 pares correspondentes.
O consumo máximo observado foi 768 apresentações: o orçamento não restringiu esses braços.
DDM emitiu três alarmes antes da primeira mudança em cada protocolo; ADWIN, zero.
Esses prefixos não substituem um ensaio estacionário completo para calibrar falsos alarmes.

Agrawal permanece fraco: acurácia online média de 58.04% (streaming) e 57.67% (incremental).
A referência retrospectiva constante sempre-positivo tem acurácia 57.73% e F1 73.20%; o F1 online
ficou em 66.52% e 66.42%. Essa referência foi calculada a posteriori a partir dos rótulos e não
é um braço pré-especificado nem uma execução MLflow adicional.

Comparação: [controlled-v3-comparison.json](../reports/controlled-v3-comparison.json).
Validação e ambiente: [controlled-v3-validation.json](../reports/controlled-v3-validation.json) e
[controlled-v3-environment.json](../reports/controlled-v3-environment.json).
Scripts de coleta/verificação: `scripts/collect_controlled_campaign.py` (executar no worker) e
`scripts/compare_controlled_campaign.py` (análise local). O cache bruto verificado fica em
`reports/generated/controlled-v3-inputs.json`, ignorado pelo Git.
A dispersão reportada é desvio-padrão entre cinco sementes, não intervalo de confiança.
Os limites são apresentações de exemplos, não segundos de CPU nem orçamentos efetivamente iguais
para os detectores. As mudanças de concorrência, quotas e RPCs impedem atribuir ganho de tempo
exclusivamente à paralelização.


## Datasets reais e campanha versionada v4

A DAG `benchmark_versioned` usa cinco sementes 42–46 nos sintéticos e um único prefixo de cada
fonte real. No padrão são 289 configurações: 255 sintéticas e 17 por fonte real, incluindo
streaming/Kafka, incremental e offline. `configs/campaign-versioned-v4.json` fixa as configurações
executadas, identidades e SHA do código. O argumento `samples` limita os exemplos usados;
esta campanha usa 3000, sem alegar execução de todas as linhas brutas. Nenhum resultado real
é artificialmente repetido com cinco sementes ignoradas pelo adaptador.

Antes de executar o pipeline, adquirir e admitir as fontes completas explicitamente:

```bash
docker compose --env-file /home/gabriel/.config/continual-bench/benchmark.env \
  -f compose.yaml -f compose.homelab.yaml exec -T worker \
  python scripts/prepare_real_sources.py --directory /state/datasets
```

O script verifica SHA-256, número de linhas, esquema, categorias, valores finitos
e ausência de missing nas duas fontes. Se os arquivos já existirem, verifica-os novamente.
O pipeline somente lê os ARFF admitidos em `BENCH_DATASETS`; não baixa dados implicitamente.
As versões e URLs constam em `configs/real-sources.lock.json` e são conferidas com os pins do adaptador.

- Electricity: OpenML 151 v1, 45312 linhas, 8 atributos; positivo `UP`.
- Airlines: OpenML 1169 v1, 539383 linhas, 7 atributos; positivo `1`.

Ambos preservam o rótulo de licença `Public` informado no OpenML. A ordem é a publicada,
sem garantia adicional de cronologia. O eixo temporal e a disponibilidade dos rótulos são simulados.
Electricity declara Normalize/ReplaceMissingValues no cabeçalho, sem documentar em qual trecho
foram ajustados; essa incerteza impede afirmar ausência de vazamento no pré-processamento de origem.
Não se atribuem pontos de mudança conhecidos ou probes a esses datasets.

### Experimentos e metadados MLflow

Cada combinação de estratégia tem experimento próprio:

| Política | Detector | Experimento |
|---|---|---|
| congelado | nenhum | `continual-bench/policy-none/detector-none` |
| online | nenhum | `continual-bench/policy-online/detector-none` |
| periódico | nenhum | `continual-bench/policy-periodic/detector-none` |
| drift | ADWIN | `continual-bench/policy-drift/detector-adwin` |
| drift | DDM | `continual-bench/policy-drift/detector-ddm` |

Para visualizar somente a campanha completa, filtrar `tags.campaign = 'versioned-v4'`;
os testes integrados têm outro nome de campanha e ficam preservados para auditoria.

As runs incluem DatasetInput nativo do MLflow, dataset, versão da fonte, licença, digest dos eventos,
semente e seu significado, protocolo, configuração completa, ambiente/dependências, código,
orçamento/consumo de treino, métricas por janela, alarmes e atualizações. Os artefatos preservam
`dataset-catalog.json`, `catalog-links.json`, `model-metadata.json`, `model-versions.json`, configuração,
ambiente, relatório e manifesto. O recibo liga os IDs MLflow aos URIs MinIO e ao Model Registry.

### Versionamento e recuperação

O bucket `benchmarks` mantém objetos imutáveis: ARFF bruto por SHA, eventos JSONL por SHA,
catálogo dos dados e seleções de treino, código completo por hash, diário binário de estados e índice.
A publicação rejeita uma chave imutável com conteúdo diferente. Checkpoints guardam o offset do diário;
a retomada descarta somente a cauda não confirmada, preservando a cadeia de versões.

O primeiro ajuste e cada ação concluída de atualização geram uma versão com hash do estado,
versão pai, versão dos dados, intervalo de treino `[início,fim)`, disponibilidade/liberação do rótulo
e SHA do código. O aquecimento conta como um ajuste inicial; cada atualização online posterior
conta como uma versão. O estado inclui o pré-processador. Os estados intermediários são agrupados
em um diário e índice por run para evitar milhares de chamadas individuais ao Registry.
O carregamento confere o hash do estado e a identidade completa da versão; quando o resumo
`model-metadata.json` está presente, confere também o hash do índice.

Cada modelo final é também uma versão nativa do Model Registry, na família
`continual-bench-{dataset}-{mode}-{policy}-{detector}`. A versão referencia o run_id e os hashes do
modelo/dataset/código. `mlflow.pyfunc.load_model("models:/nome/versão")` recarrega o pacote com
assinatura, pré-processamento e dependências. Para estados intermediários, baixar o diário e índice
a partir dos URIs do recibo e usar `continual_bench.adapters.versioning.load_version` com o version_id.
Baixar também `model-metadata.json` permite conferir o checksum integral do índice.
Apenas artefatos próprios e autenticados devem ser desserializados.

Auditoria após a conclusão:

```bash
docker compose --env-file /home/gabriel/.config/continual-bench/benchmark.env \
  -f compose.yaml -f compose.homelab.yaml exec -T worker \
  python scripts/verify_versioned_campaign.py > versioned-v4-inputs.json
```

A auditoria verifica todas as runs, DatasetInputs, experimentos, Registry, conteúdo S3, cadeia e
checksums de cada estado, intervalos de treino, liberação dos rótulos e recarregamento de um modelo
por dataset pelo Registry em processo novo, usando o código embarcado. Esse teste não recria
um ambiente de dependências do zero e cobre uma entrada de referência por dataset.
`--limit` serve apenas para diagnóstico parcial e não certifica a campanha.

## Campanha não linear v5

`benchmark_nonlinear` usa a mesma matriz de dados, sementes e orçamentos da v4 para cada uma
das três famílias: logística, `hoeffding_tree` e `knn`. O manifesto está em
`configs/campaign-nonlinear-v5.json`; filtrar `tags.campaign = 'nonlinear-v5'` no MLflow.
Oito tarefas mapeadas `evaluate_and_publish` e oito subprocessos do worker são os limites de
concorrência; o número real de tarefas simultâneas fica no registro operacional da campanha.

O [protocolo](nonlinear-v5-protocol.md) fixa hiperparâmetros e representação. Os diários guardam
cada estado treinado: árvore/online/Airlines gerou cerca de10GiB em um único diário no preflight.
Não dimensionar armazenamento somente pelo tamanho do modelo final. O auditor verifica os
blobs em streaming, com hashes por estado e conteúdo completo do diário.

Em uma recuperação por alteração de código, não reutilizar checkpoint de outra revisão.
Preservar o diretório da tentativa, registrar o override por configuração e limpar somente
as tarefas Airflow afetadas. A v5 preservou duas falhas de cache nominal do River e refez esses
casos com a correção; os demais casos mantiveram sua revisão original. Conferir a evidência
[de recuperação](../reports/nonlinear-v5-recovery.json) e o
[relatório comparativo](../reports/nonlinear-v5.md).
