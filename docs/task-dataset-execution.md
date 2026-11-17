# Datasets e preparação da DAG — 4 de novembro de 2026

## Execução vigente

Stream51 saiu do catálogo, adaptador, lock e DAG em 8 de novembro de 2026. O gate de admissão exige recibos de **17 fontes**, com 100 exemplos, duas épocas e o fingerprint do runtime usado pelo worker. O plano padrão de `benchmark_task_suite` tem 518 casos (501 binários e 17 por épocas), seed 42, atraso de três e até oito tarefas Airflow ativas. A AWS usa a mesma definição de casos na [suíte `project_suite`](aws/operations.md#suíte-de-todos-os-datasets).

Para preparar a execução no Airflow, rode `scripts/validate_dataset_epochs.py` no ambiente do worker. Use `--state` com o diretório configurado em `BENCH_DATASET_SMOKES` (padrão `/state/dataset-smokes`), `--cache` com o cache e `--real-directory` com o diretório dos dados reais. Confirme `complete=true` no recibo `dataset-smoke-complete.json` e verifique `/datasets-ready` antes de disparar a DAG `benchmark_task_suite`. [Operação no Airflow](benchmark-operations.md#executar-no-airflow).

## Validação inicial de 4 de novembro

Foram implementados loaders executáveis para as 18 fontes/variantes explicitadas no projeto, na branch local `Semana-09`. Foram realizados testes iniciais com containers isolados. **A DAG completa não foi iniciada: a aquisição do Stream51 permaneceu bloqueada pela resolução DNS da origem oficial.** O seu adaptador passou pelos testes de contrato com fixtures; isso não homologou seus dados reais.

A planilha `Semana_09_Datasets_Benchmarks.xlsx` continuou indisponível neste executor: a referência Library retornou `page_reference_unsupported_format`. As 18 entradas abaixo não substituíram os 47 registros do catálogo bibliográfico, nem representaram 18 datasets independentes. Outros candidatos do catálogo não foram presumidos selecionados.

## Protocolo dos testes iniciais

Cada execução selecionou exatamente **100 exemplos originais distintos**, seed 42, preservando classes e alvos contínuos. Os exemplos de teste não participaram do treinamento. Cada uma das duas épocas percorreu somente o conjunto de treino, na ordem admitida, e foi seguida de avaliação de uma cópia congelada do modelo.

Os datasets de imagens preservaram os splits oficiais, com 80 exemplos de treino e 20 de teste no recorte. Os fluxos sem split usaram um prefixo de treino de 80% e um sufixo de teste de 20%, com purga dos rótulos ainda indisponíveis na fronteira. Nos fluxos com atraso de três unidades, ficaram 77 exemplos de treino, 20 de teste e três purgados. Portanto, duas épocas não significaram treinar duas vezes sobre o holdout nem utilizar 200 exemplos únicos.

Esse protocolo verificou carga, schema, alvos, treinamento, avaliação, artefatos e reprodutibilidade. Não representou avaliação prequential, reprodução de protocolos visuais completos, estudo de forgetting ou ranking entre bases/políticas. Os fluxos binários da campanha mantiveram o avaliador prequential existente, separado deste smoke com épocas.

| Fonte/variante | Implementação e origem fixada | Validação inicial 100 × 2 |
| --- | --- | --- |
| SEA, AGRAWAL, STAGGER | Geradores River 0.26.1, conceitos segmentados explícitos | PASS nas três fontes |
| Hyperplane, RBFDrift | Geradores River 0.26.1 com movimento obrigatório | PASS nas duas fontes |
| FriedmanDrift | River 0.26.1; regressão preservada | PASS |
| Electricity/ELEC2, Airlines | Caches fixados e verificados da Semana 08, montados somente para leitura | PASS nas duas fontes |
| INSECTS | `abrupt_balanced`, origem dos autores, 52.848 registros/33 atributos/6 classes | PASS |
| Gas Sensor Array Drift | UCI 224, dez lotes LIBSVM na ordem declarada, 128 atributos/6 classes | PASS |
| Split MNIST, Permuted MNIST | IDX oficiais, hashes fixados; splits, classes e permutações explícitos | PASS nas duas variantes |
| Split CIFAR10, Split CIFAR100 | Espelho Parquet `uoft-cs`, revisões imutáveis e leituras HTTP parciais | PASS nas duas variantes |
| CORe50 | Arquivo original 128×128, sessões oficiais, leituras parciais | PASS |
| Stream51 | ZIP/JSON/bounding boxes oficiais; suporte a arquivo local e HTTP Range | BLOCKED nos dados reais; contratos com fixture aprovados |
| CLEAR10, CLEAR100 | Release julho/2022, espelho indicado pelos autores, revisão imutável | PASS nas duas variantes |

Detalhes de origem, licença, schema, ordem, checksums e limitações: [MNIST/CIFAR nativos](image-datasets.md), [INSECTS/Gas](real-streams.md), [CORe50/Stream51/CLEAR](temporal-vision.md). O loader CIFAR usado pelo factory foi `datasets/cifar_parquet.py`: a aquisição dos arquivos completos de Toronto excedeu o timeout de aquisição, e foi escolhida explicitamente a variante Parquet versionada; não houve troca automática de origem após uma negação. Os hashes LFS integrais foram registrados como referência, sem alegar sua verificação por leitura parcial. Foram verificados os bytes selecionados e os respectivos manifestos. A representação de CORe50/CLEAR/Stream51 foi RGB 16×16 normalizado; não houve pré-treino de um extrator visual.

## Arquitetura entregue

- `datasets/catalog.py` concentrou a seleção das 18 fontes, sem converter multiclasse/regressão em booleano.
- `evaluation/epochs.py` validou IDs, ordem, tempos, atraso, features finitas, splits e orçamento. A identidade incluiu configuração, dados, manifestos, hashes dos contratos, código e versões do runtime. Cache corrompido não foi reparado silenciosamente.
- `orchestration/epoch_pipeline.py` separou avaliação e publicação em dois passos ZenML. `epoch_tracking.py` publicou relatórios e amostras no MinIO e métricas por época em um experimento MLflow separado; repetição da publicação preservou a identidade da run e das métricas.
- `orchestration/epoch_server.py` acrescentou um worker isolado com fila compartilhada de concorrência um, limites de requisições, timeout e endpoints de admissão. O gate exigiu recibos válidos de **todas as 18 fontes**, com 100 exemplos/2 épocas e fingerprint atual.
- `scripts/validate_dataset_epochs.py` consolidou sucessos e falhas, continuando as fontes independentes sem transformar ausência em skip aprovado. `scripts/export_dataset_epoch_evidence.py` exportou evidências sem alterar permissões dos arquivos produzidos pelo container.
- `benchmark_task_suite` ampliou a matriz existente para 549 casos: 531 combinações do motor binário e 18 avaliações tipadas por épocas. Manteve um seed pareado, no máximo duas tarefas Airflow ativas e um job efetivo por worker. O padrão dos casos binários foi 3.000 eventos; o recorte tipado permaneceu em 100 exemplos/2 épocas. Não foi denominada campanha exaustiva do catálogo nem comparação de arquiteturas de visão.

## Ambiente e evidências

A validação usou um snapshot do código, com caches e resultados em `state` e logs em `evidence`. O ambiente possuía 28 CPUs lógicas Xeon E5-2680 v4, aproximadamente 30 GiB de RAM e mais de 500 GiB livres no início. A stack original permaneceu em serviço. Os testes usaram containers transitórios, sem publicar portas ou parar serviços.

Foi construída somente a imagem auxiliar `continual-bench-datasets:semana09-v2`, derivada da imagem existente, acrescentando Pillow 13.0.0 e PyArrow 26.0.1 com hashes fixados. As dependências opcionais ficaram em `.[datasets]`, `uv.lock` e `requirements.lock`. Os testes unitários usaram 2 CPUs/3 GiB, os smokes 2 CPUs/4 GiB e os clientes de serviços 2 CPUs/6 GiB, com limites externos de tempo. Não foram executados testes iniciais no Mac.

| Verificação | Estado | Evidência |
| --- | --- | --- |
| Ruff, código/testes/DAGs/scripts | PASS | `evidence/reviewed-checks.txt` |
| Suíte sem serviços/benchmark | PASS, 475 testes; 32 excluídos; 38,22 s | `reports/semana09-datasets/final-checks.txt` |
| Admissão das 18 fontes, 100 exemplos/2 épocas | 17 PASS; Stream51 BLOCKED | `reports/semana09-datasets/evidence.json` e relatórios individuais em `state/dataset-smokes` |
| ZenML → MLflow/MinIO, 100 exemplos/2 épocas, e três fluxos binários com serviços reais | PASS, quatro testes; 309,34 s na rodada revisada | `reports/semana09-datasets/services.txt`, com hashes, leitura posterior, Kafka e repetição idempotente |
| Empacotamento wheel | PASS, quatro contratos JSON incluídos | `evidence/wheel-package.txt` |
| Importação no Airflow 3.3.2 e expansão da matriz | PASS, 549 casos/18 fontes | `evidence/airflow-parse-final.txt` |
| Compose do worker isolado | PASS, `config --quiet` | Configuração versionada `compose.dataset-benchmark.yaml` |
| Inicialização da DAG completa | NOT RUN, bloqueada | Nenhum dag_run foi criado por esta etapa |

O aviso preexistente de input example do MLflow permaneceu na suíte; os avisos de importação de exceção antiga vieram de DAGs existentes. As primeiras tentativas de lint encontraram a criação de cache em mount somente leitura e linhas longas no novo loader CIFAR; foram corrigidos o comando (`--no-cache`) e a formatação antes da rodada final. Fixtures, scripts de diagnóstico e tentativas incompletas de aquisição não foram apresentados como sucesso nos dados originais.

A revisão final corrigiu também a revogação de um recibo antigo antes de repetir a admissão via CLI: uma falha posterior deixou `complete=false`, com teste de regressão. Os smokes e serviços foram repetidos após essa mudança. O código Python revisado teve SHA-256 `c12041fa90c65452d7abe64e66d6f0c3bdfd2a6aafd9c79254224d63f7912d30`; o fingerprint incluiu também os contratos e as versões do runtime. A imagem usada nos testes precedeu os commits locais e registrou `git_revision=uncommitted`; a correspondência foi estabelecida pelo hash dos fontes, sem inventar uma revisão executada.

Os loaders foram registrados no commit `4c7a6ae` e o protocolo de épocas e o gate da DAG no commit `90706ad`. A consolidação da validação confirmou os 17 resultados com o mesmo fingerprint e as quatro evidências de serviços. O pico RSS foi medido no processo executor compartilhado. Os arquivos de `reports/semana09-datasets/` estão no histórico Git. Para ler a consolidação, use `git show f835ee8:reports/semana09-datasets/consolidation.json`. [Complemento do relatório no Space](https://chatgpt.com/space/page_ea19e36695b88191aeefc5cc7af299ad).

## Bloqueio registrado na primeira rodada

O endereço oficial `http://klab.cis.rit.edu/files/Stream-51.zip` não resolveu DNS durante a validação (`Name or service not known`). O MD5 publicado `5f34d542b71af7e5ecb2226d1bc7297c` ficou como referência para o arquivo integral. A falta dessa fonte impediu a aprovação do gate da época. O gate vigente usa o catálogo de 17 fontes descrito acima.

Comandos de validação, executados com o snapshot final e o estado isolado:

```bash
export TASK_BENCH_CODE=/caminho/absoluto/do/snapshot
export TASK_BENCH_STATE=/caminho/absoluto/do/estado/isolado
export TASK_BENCH_SERVICE_ENV=/caminho/privado/benchmark.env

timeout 900 docker run --rm --network none --cpus 2 --memory 4g \
  --pids-limit 192 --read-only --tmpfs /tmp:rw,size=256m \
  -v "$TASK_BENCH_CODE:/work:ro" -v "$TASK_BENCH_STATE:/state" \
  -v continual-bench_worker-data:/legacy:ro -w /work \
  -e PYTHONPATH=/work/src -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONHASHSEED=0 \
  --entrypoint python continual-bench-datasets:semana09-v2 \
  scripts/validate_dataset_epochs.py --state /state/dataset-smokes \
  --cache /state/datasets --real-directory /legacy/datasets

docker compose --env-file "$TASK_BENCH_SERVICE_ENV" \
  -f "$TASK_BENCH_CODE/compose.dataset-benchmark.yaml" config --quiet
timeout 420 docker compose --env-file "$TASK_BENCH_SERVICE_ENV" \
  -f "$TASK_BENCH_CODE/compose.dataset-benchmark.yaml" run --rm --no-deps \
  -e SERVICE_TEST_STATE=/state/reviewed-services -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONHASHSEED=0 --entrypoint python dataset-worker \
  -m pytest -q -p no:cacheprovider -m services \
  tests/test_epoch_services.py tests/test_task_services.py
```

O primeiro comando terminou com exit code 1, conforme o gate de 18 fontes na época. A repetição offline usou o cache verificado.
