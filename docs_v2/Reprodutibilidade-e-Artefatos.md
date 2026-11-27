# Reprodutibilidade e Artefatos

Esta página descreve como o `continual-bench` identifica cada execução, como as seeds são definidas e propagadas, como dados, modelos e contratos são versionados e onde ficam os artefatos gerados. Também mostra como conferir checksums e reconstruir localmente as figuras e tabelas das evidências versionadas, sem treinar modelos nem acessar serviços externos.

## Visão geral

| Capacidade | Onde está | Estado |
| --- | --- | --- |
| Identidade de execução (`run_key`) derivada da configuração | `src/continual_bench/config/benchmark.py` (`Config.identity`) | **Validado**: `tests/test_run_identity.py` |
| Contrato de retomada (configuração, dados, código, pacotes, plugins, transporte) | `src/continual_bench/application/runner.py` (`resume_contract`) | **Validado**: `tests/test_run_identity.py` |
| Versionamento de modelos com diário e checksum | `src/continual_bench/infrastructure/versioning.py` (`ModelVersions`, `load_version`) | **Validado**: `tests/test_versioning.py` |
| Modo `metrics_only` sem estado de modelo persistido | `src/continual_bench/infrastructure/artifact_policy.py` | **Implementado**; a campanha F109 registra verificações desse modo em `evidencias/aws-benchmarks/training-inventory.csv` |
| Locks de datasets e impressão digital da suíte | `configs/contracts/*.lock.json`, `src/continual_bench/evaluation/environment.py` (`dataset_fingerprint`) | **Implementado** |
| Pacotes de evidência com manifesto e SHA-256 | `evidencias/aws-benchmarks/manifest.json`, `evidencias/semana-10/*/SHA256SUMS` | **Validado**: checksums conferem na árvore atual (comandos abaixo) |
| Reconstrução das 44 figuras da Semana 10 a partir dos CSVs | `notebooks/figuras_benchmarks_semana_10.ipynb` e módulos `notebooks/benchmark_*.py` | **Experimental**: funciona localmente, mas os carregadores não têm teste automatizado em `tests/` |

## Identidade de execução

Cada configuração validada (`Config`) produz um identificador determinístico, o `run_key`: os primeiros 24 caracteres hexadecimais do SHA-256 do JSON da configuração, serializado com chaves ordenadas e sem espaços (`Config.identity` em `src/continual_bench/config/benchmark.py`). Qualquer mudança de parâmetro, inclusive de seed, gera outro `run_key` e, portanto, outro diretório de estado.

O `run_key` sozinho não basta para reutilizar um resultado. Antes de retomar um checkpoint ou devolver um relatório já concluído, o runner compara um **contrato de retomada** (`resume_contract` em `src/continual_bench/application/runner.py`) que reúne:

- a configuração completa;
- o SHA-256 do arquivo de eventos materializado (`events.jsonl`);
- a versão do Python e as versões dos pacotes relevantes;
- o `code_sha256`, um hash de todos os arquivos `.py` do pacote;
- o valor de `PYTHONHASHSEED` (registrado como `random` quando não definido);
- o SHA-256 do catálogo de métricas `configs/contracts/metric-definitions.json`;
- a identidade dos plugins carregados;
- o transporte de eventos (arquivo local ou, no modo `streaming`, o contrato do transporte, por exemplo Kafka).

Se algum item diverge, a retomada falha com erro explícito em vez de misturar estados. Os testes em `tests/test_run_identity.py` cobrem, por exemplo, a recusa de reutilizar um cache local quando o transporte muda e a invalidação quando um módulo auxiliar de plugin é alterado.

O ambiente de execução (`environment()` em `src/continual_bench/evaluation/environment.py`) também registra `git_revision`. Localmente, esse campo vale `uncommitted`, a menos que a variável `BENCH_GIT_REVISION` seja definida. A imagem AWS grava a revisão na construção (`docker/aws/Dockerfile`). Por isso, o vínculo confiável com o código é o `code_sha256`, e não o campo `git_revision` de execuções locais.

## Seeds

Uma **seed** é o valor inicial de um gerador pseudoaleatório. No projeto, ela é declarada em um único lugar e propagada aos componentes que a usam.

| Ponto | Comportamento | Fonte |
| --- | --- | --- |
| Declaração | `dataset.seed`, inteiro entre 0 e 2^31 - 1, padrão 42 | `src/continual_bench/config/benchmark.py` (`DatasetConfig`) |
| Geradores sintéticos | Cada gerador usa `seed + deslocamento` fixo por função ou tarefa | `src/continual_bench/datasets/synthetic.py` |
| Matrizes de casos | Cada seed da matriz gera um caso com `dataset.seed` próprio; a política UPF recebe a mesma seed quando `policy_seed_from_dataset` está ativo | `src/continual_bench/application/matrix.py` |
| Matriz de retreinamento (CARA, UPF, Oracle) | `compute_binary_retraining(..., seed=config.dataset.seed)` | `src/continual_bench/application/runner.py` |
| Worker AWS | A seed do pedido é copiada para a configuração iterativa e para a seção `retraining` | `src/continual_bench/application/cloud/worker.py` |
| Modelos Zia (XGBoost) | Parâmetro `seed`, padrão 42, sobrescrito pela seed da configuração iterativa | `src/continual_bench/models/zia.py` |
| Runs filhas por ajuste no MLflow | Param `seed` e tag `cb-seed` quando o ajuste conhece sua seed; ausência preservada em históricos antigos | `src/continual_bench/infrastructure/training_tracking.py`, **Validado** em `tests/test_training_seed_tracking.py` |

**Cuidado ao interpretar seeds.** Uma seed só produz variação quando algum componente aleatório a consome. Em fluxos reais com ordem fixa e modelos determinísticos, seeds diferentes podem gerar exatamente o mesmo resultado. Os manifestos históricos registram isso explicitamente: `configs/history/campaign-versioned-v4.json` declara que a seed é ignorada para dados reais ("not independent repeats"), e `configs/history/campaign-nonlinear-v5.json` declara que as seeds 42 a 46 só afetam fontes sintéticas. Na campanha F109, as métricas de qualidade são idênticas entre as seeds 42, 43 e 44 (ver [Tracking e Evidências](Tracking-e-Evidencias)).

## Dados, contratos e locks

Ao preparar uma execução, `prepare()` (`src/continual_bench/application/runner.py`) materializa a fonte em `state/<run_key>/events.jsonl`, limitada a 512 MiB, e grava `manifest.json` com o SHA-256 desses bytes. Esse hash é usado como `dataset_version`. Se uma execução existente encontrar uma fonte com outro hash, ela falha com a mensagem `source changed for an existing run; choose a new config/name`. Na avaliação, o hash do `events.jsonl` é conferido de novo antes de começar.

Os contratos versionados ficam em `configs/contracts/`:

| Arquivo | Função |
| --- | --- |
| `metric-definitions.json` | Catálogo `residency-metrics-v1` com 198 definições; seu hash entra no contrato de retomada |
| `real-streams.lock.json` | Admissão de INSECTS e Gas; a primeira aquisição autorizada registra o SHA-256 bruto e as aberturas seguintes rejeitam mudanças |
| `real-sources.lock.json`, `image-datasets.lock.json`, `temporal-vision.lock.json` | Locks de fontes reais e de imagens |
| `dataset-catalog.json` | Levantamento de candidatos a datasets |
| `aws-prices.us-east-1.json` | Tabela de preços usada em estimativas de cenário, que não é faturamento observado |

A função `dataset_fingerprint()` combina os hashes de `image-datasets.lock.json`, `real-streams.lock.json` e `temporal-vision.lock.json` com o `code_sha256` e as versões de pacotes. Ela identifica a combinação de código, dependências e locks que admite uma suíte completa de datasets. Detalhes de cada fonte estão em [Datasets](Datasets).

Os manifestos de campanhas históricas ficam em `configs/history/`:

| Arquivo | Campanha | Configurações |
| --- | --- | ---: |
| `campaign-controlled-v3.json` | `controlled-v3` | 261 |
| `campaign-versioned-v4.json` | `versioned-v4` | 289 |
| `campaign-nonlinear-v5.json` | `nonlinear-v5` | 867 execuções esperadas |

Esses manifestos são lidos pelos scripts de coleta, verificação e backfill descritos em [Tracking e Evidências](Tracking-e-Evidencias#backfill-e-verificação). Os resultados dessas campanhas não estão versionados na árvore atual.

## Versionamento de modelos e checkpoints

`ModelVersions` (`src/continual_bench/infrastructure/versioning.py`) registra cada modelo treinado durante uma execução:

- `models.bin` guarda os estados serializados em sequência;
- `model-versions.json` indexa cada versão com `version_id`, versão pai, intervalo de dados de treino (`seq_range`), ação (por exemplo `initial_fit`), deslocamento no arquivo e checksum;
- `model-metadata.json` complementa os metadados.

`load_version()` recusa uma versão cujo conteúdo ou linhagem não confere com o checksum, mesmo quando os pesos não mudaram. Um diário permite recuperar a cauda não confirmada após uma interrupção. Esses comportamentos são cobertos por `tests/test_versioning.py`, que também valida o empacotamento como modelo MLflow com o pré-processamento completo.

Os checkpoints de retomada ficam em `state/<run_key>/checkpoint.bin`, gravados por `LocalCheckpoints` (`src/continual_bench/infrastructure/storage.py`) com um prefixo `CB01` e o SHA-256 do conteúdo. O checksum detecta corrupção, não substituição maliciosa: o arquivo é um pickle e só deve ser carregado quando foi criado pelo próprio sistema.

No modo `artifact_mode: metrics_only`, nenhum estado de modelo é persistido. `verify_metrics_only_files()` falha se encontrar extensões como `.pkl`, `.bin`, `.ckpt` ou diretórios `models/` e `checkpoints/` no diretório da execução. Esse modo não aceita checkpoints de retomada, e um relatório concluído só é reutilizado quando o hash do contrato de avaliação (`evaluation_contract_sha256`) coincide.

## Onde ficam os artefatos

| Local | Conteúdo | Quem grava |
| --- | --- | --- |
| `state/<run_key>/` (padrão da CLI: `--state state`) | `events.jsonl`, `manifest.json`, `report.json`, `checkpoint.bin`, `model-versions.json`, `models.bin`, `artifacts/plots/`, `artifacts/data/`, `training-history.json`, `tracking-observations/`, `config.json`, `environment.json`, `receipt.json` | Runner e publicadores |
| MinIO, bucket `BENCH_BUCKET` (padrão `benchmarks`), chave `<run_key>/<arquivo>` | Relatório, manifesto, checkpoint e eventos; no modo `metrics_only`, todos os arquivos da execução exceto `events.jsonl` e `receipt.json` | `publish()` em `src/continual_bench/infrastructure/services.py`, com `--services` |
| MLflow | Parâmetros, métricas, tags e artefatos de cada execução | Ver [Tracking e Evidências](Tracking-e-Evidencias) |
| `s3://zenml-artifacts` no MinIO | Artefatos e logs das etapas ZenML | Stack `benchmark-minio-v2` do Docker Compose |
| S3 na AWS | Relatórios de campanhas, `final.json` por run e resultados de consultas Athena | Ver [Execução na AWS](Execucao-na-AWS) |
| `evidencias/` | Evidências científicas versionadas, sem pesos nem checkpoints | Exportações manuais auditadas |

O arquivo `receipt.json` lista as URIs enviadas ao armazenamento de objetos e os identificadores do MLflow, fechando o vínculo entre o estado local e as cópias remotas. O diretório `reports/` é ignorado pelo Git (`.gitignore`), portanto relatórios gerados ali são locais.

## Conferir checksums

Os pacotes de evidência carregam somas SHA-256. Na raiz do repositório:

```bash
(cd evidencias/semana-10/insects && shasum -a 256 -c SHA256SUMS)
(cd evidencias/semana-10/cara-upf/dados && shasum -a 256 -c SHA256SUMS)
```

O `SHA256SUMS` de INSECTS cobre dados, figuras e o PDF. O de CARA/UPF cobre apenas os arquivos de `dados/`, não as figuras.

O pacote `evidencias/aws-benchmarks` usa `manifest.json` (esquema `benchmark-tables-v1`), que lista tamanho e SHA-256 de cada arquivo e, em `tables`, as cinco tabelas obrigatórias com contagem de linhas. Uma conferência simples:

```bash
cd evidencias/aws-benchmarks
python3 -c "
import json, hashlib
m = json.load(open('manifest.json'))
bad = [f['path'] for f in m['files']
       if hashlib.sha256(open(f['path'], 'rb').read()).hexdigest() != f['sha256']]
print('divergentes:', bad)
"
```

O carregador do notebook faz uma validação mais forte: hashes semânticos das linhas das consultas, esquema, cobertura da matriz 4 × 11 × 3 e consistência interna.

## Reconstruir figuras e tabelas localmente

As figuras da Semana 10 são reconstruídas a partir dos CSVs e JSONs versionados, sem treinamento, sem nova consulta Athena e sem credenciais AWS.

### Notebook

`notebooks/figuras_benchmarks_semana_10.ipynb` (49 células) gera quatro famílias de figuras, em PNG a 300 dpi, PDF e SVG, nas variantes `notebook` (com título e nota) e `docs` (sem elementos internos, com legendas em `legendas.md` e `legendas.json`):

| Família | Entrada padrão | Figuras geradas |
| --- | --- | ---: |
| `atual` (campanha F109) | `evidencias/aws-benchmarks` | 12 |
| `insects` (histórico INSECTS) | `evidencias/semana-10/insects` | 13 |
| `paper-historico` (CARA e UPF em Electricity e Airlines) | `evidencias/semana-10/cara-upf` | 13 |
| `paper-comparativo` (valores publicados) | `evidencias/aws-benchmarks/papers` | 6 |

Requisitos e variáveis de ambiente, conforme a primeira célula do notebook:

- Python 3.12 com os extras `plots` e `services`;
- fonte Arial instalada (obrigatória; `BENCHMARK_ARIAL_FONT` aponta um arquivo de fonte licenciado);
- `BENCHMARK_FIGURES_DIR` define a saída, por padrão `reports/semana-10-redesign`, fora do versionamento;
- `BENCHMARK_ATHENA_DIR`, `BENCHMARK_INSECTS_DIR`, `BENCHMARK_CARA_UPF_DIR` e `BENCHMARK_PAPERS_DIR` substituem as entradas;
- `BENCHMARK_FIGURES_FAMILIES` escolhe entre `atual`, `historico` e `papers`;
- `BENCHMARK_SOURCE_MODE=tables` valida uma campanha nova com o contrato `benchmark-tables-v1`, sem herdar a auditoria F109.

Para executar pela linha de comando sem alterar o lock, o notebook sugere `uv run --no-sync --with nbclient --with nbformat --with ipykernel python`, o que baixa esses pacotes temporariamente. As mesmas funções também podem ser chamadas diretamente em Python:

```python
import sys
from pathlib import Path
sys.path.insert(0, "notebooks")
from benchmark_athena_data import load_athena
from benchmark_athena_figures import build_figures
from benchmark_plot_style import apply_style, export_bundle

apply_style()  # exige Arial
evidence = load_athena(Path("evidencias/aws-benchmarks"))
figures, _ = build_figures(evidence, show_title=False, show_footer=False)
export_bundle(figures, Path("/tmp/figuras/atual"), variant="docs")
```

Para uma campanha nova, `load_tables(Path(...))` aceita outro diretório com manifesto `benchmark-tables-v1`. As regras desse contrato (tabelas obrigatórias, `cohort_n`, `source_fact_present`, `availability`, diagnóstico de holdout) estão em `evidencias/aws-benchmarks/README.md`.

### Script de CARA e UPF

`scripts/plot_week10_cara_upf.py` redesenha as seis figuras de `evidencias/semana-10/cara-upf/figuras` a partir dos JSONs comprimidos. Por padrão, ele **sobrescreve** as figuras versionadas. Para gerar em outro lugar, copie o pacote e use `--root`:

```bash
cp -R evidencias/semana-10/cara-upf /tmp/cara-upf
uv run --extra plots python scripts/plot_week10_cara_upf.py --root /tmp/cara-upf
```

Na árvore atual, as seis figuras regeneradas dessa forma são idênticas byte a byte às versionadas. Com `--electricity` e `--airlines` apontando os ARFF de entrada (e o extra `large-datasets`), o script também reconstrói `dados/trajetorias.json` a partir dos dados públicos.

### Outros scripts de figuras

`scripts/plot_replication.py` e `scripts/plot_schedule.py` desenham figuras de replicação a partir de JSONs de resultado. Os exemplos em suas docstrings usam `docs/results/`, diretório que não existe na árvore atual; é preciso fornecer arquivos de resultado próprios.

## Limitações conhecidas

- Os carregadores e geradores de figuras em `notebooks/` não têm teste automatizado em `tests/`. `evidencias/aws-benchmarks/validation.json` cita um script e um relatório de regressão em `reports/semana-10-redesign/qa/`, que não estão versionados.
- O campo `git_revision` de execuções locais vale `uncommitted` sem `BENCH_GIT_REVISION`; a correspondência com o código depende do `code_sha256`.
- O manifesto de execução de CARA/UPF (`evidencias/semana-10/cara-upf/dados/manifesto-execucao.json`) registra `git_revision: uncommitted` e `python_hash_seed: random`.
- Executar a suíte de testes sem `MLFLOW_TRACKING_URI` pode criar um arquivo `mlflow.db` no diretório corrente, comportamento padrão do MLflow 3 com armazenamento local.

Veja também [Limitações e Trabalhos Pendentes](Limitacoes-e-Trabalhos-Pendentes), [Metodologia Experimental](Metodologia-Experimental) e [Métricas](Metricas).
