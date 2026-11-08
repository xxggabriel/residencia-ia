# Datasets e preparação da DAG — 4 de novembro de 2026

Foram implementados loaders executáveis para as 18 fontes/variantes explicitadas no projeto, na branch local `Semana-09`. Foram realizados testes iniciais exclusivamente no homelab, acessado por `ssh -o BatchMode=yes -o StrictHostKeyChecking=yes gabriel@192.168.100.27`. **A DAG completa não foi iniciada: a aquisição do Stream51 permaneceu bloqueada pela resolução DNS da origem oficial.** O seu adaptador passou pelos testes de contrato com fixtures; isso não homologou seus dados reais.

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

Detalhes de origem, licença, schema, ordem, checksums e limitações: [MNIST/CIFAR nativos](image-datasets.md), [INSECTS/Gas](real-streams.md), [CORe50/Stream51/CLEAR](temporal-vision.md). O loader CIFAR usado pelo factory foi `datasets/cifar_parquet.py`: a aquisição dos arquivos completos de Toronto excedeu o timeout no homelab, e foi escolhida explicitamente a variante Parquet versionada; não houve troca automática de origem após uma negação. Os hashes LFS integrais foram registrados como referência, sem alegar sua verificação por leitura parcial. Foram verificados os bytes selecionados e os respectivos manifestos. A representação de CORe50/CLEAR/Stream51 foi RGB 16×16 normalizado; não houve pré-treino de um extrator visual.

## Arquitetura entregue

- `datasets/catalog.py` concentrou a seleção das 18 fontes, sem converter multiclasse/regressão em booleano.
- `evaluation/epochs.py` validou IDs, ordem, tempos, atraso, features finitas, splits e orçamento. A identidade incluiu configuração, dados, manifestos, hashes dos contratos, código e versões do runtime. Cache corrompido não foi reparado silenciosamente.
- `orchestration/epoch_pipeline.py` separou avaliação e publicação em dois passos ZenML. `epoch_tracking.py` publicou relatórios e amostras no MinIO e métricas por época em um experimento MLflow separado; repetição da publicação preservou a identidade da run e das métricas.
- `orchestration/epoch_server.py` acrescentou um worker isolado com fila compartilhada de concorrência um, limites de requisições, timeout e endpoints de admissão. O gate exigiu recibos válidos de **todas as 18 fontes**, com 100 exemplos/2 épocas e fingerprint atual.
- `scripts/validate_dataset_epochs.py` consolidou sucessos e falhas, continuando as fontes independentes sem transformar ausência em skip aprovado. `scripts/export_dataset_epoch_evidence.py` exportou evidências sem alterar permissões dos arquivos produzidos pelo container.
- `benchmark_task_suite` ampliou a matriz existente para 549 casos: 531 combinações do motor binário e 18 avaliações tipadas por épocas. Manteve um seed pareado, no máximo duas tarefas Airflow ativas e um job efetivo por worker. O padrão dos casos binários foi 3.000 eventos; o recorte tipado permaneceu em 100 exemplos/2 épocas. Não foi denominada campanha exaustiva do catálogo nem comparação de arquiteturas de visão.

## Ambiente e evidências

O diretório isolado foi `/home/gabriel/semana09-datasets-20261103-2341`, com snapshot final em `review-final/code`, caches e resultados em `state` e logs em `evidence`. O ambiente possuía 28 CPUs lógicas Xeon E5-2680 v4, aproximadamente 30 GiB de RAM e mais de 500 GiB livres no início. A stack original permaneceu em serviço. Os testes usaram containers transitórios, sem publicar portas ou parar serviços.

Foi construída somente a imagem auxiliar `continual-bench-datasets:semana09-v2`, derivada da imagem existente, acrescentando Pillow 13.0.0 e PyArrow 26.0.1 com hashes fixados. As dependências opcionais ficaram em `.[datasets]`, `uv.lock` e `requirements.lock`. Os testes unitários usaram 2 CPUs/3 GiB, os smokes 2 CPUs/4 GiB e os clientes de serviços 2 CPUs/6 GiB, com limites externos de tempo. Não foram executados testes iniciais no Mac.

| Verificação | Estado | Evidência |
| --- | --- | --- |
| Ruff, código/testes/DAGs/scripts | PASS | `evidence/reviewed-checks.txt` |
| Suíte sem serviços/benchmark | PASS, 475 testes; 32 excluídos; 38,22 s | [final-checks.txt](../reports/semana09-datasets/final-checks.txt) |
| Admissão das 18 fontes, 100 exemplos/2 épocas | 17 PASS; Stream51 BLOCKED | [evidence.json](../reports/semana09-datasets/evidence.json) e relatórios individuais em `state/dataset-smokes` |
| ZenML → MLflow/MinIO, 100 exemplos/2 épocas, e três fluxos binários com serviços reais | PASS, quatro testes; 309,34 s na rodada revisada | [services.txt](../reports/semana09-datasets/services.txt); hashes/readback, Kafka e repetição idempotente |
| Empacotamento wheel | PASS, quatro contratos JSON incluídos | `evidence/wheel-package.txt` |
| Importação no Airflow 3.3.2 e expansão da matriz | PASS, 549 casos/18 fontes | `evidence/airflow-parse-final.txt` |
| Compose do worker isolado | PASS, `config --quiet` | Configuração versionada `compose.dataset-benchmark.yaml` |
| Inicialização da DAG completa | NOT RUN, bloqueada | Nenhum dag_run foi criado por esta etapa |

O aviso preexistente de input example do MLflow permaneceu na suíte; os avisos de importação de exceção antiga vieram de DAGs existentes. As primeiras tentativas de lint encontraram a criação de cache em mount somente leitura e linhas longas no novo loader CIFAR; foram corrigidos o comando (`--no-cache`) e a formatação antes da rodada final. Fixtures, scripts de diagnóstico e tentativas incompletas de aquisição não foram apresentados como sucesso nos dados originais.

A revisão final corrigiu também a revogação de um recibo antigo antes de repetir a admissão via CLI: uma falha posterior deixou `complete=false`, com teste de regressão. Os smokes e serviços foram repetidos após essa mudança. O código Python revisado teve SHA-256 `c12041fa90c65452d7abe64e66d6f0c3bdfd2a6aafd9c79254224d63f7912d30`; o fingerprint incluiu também os contratos e as versões do runtime. A imagem usada nos testes precedeu os commits locais e registrou `git_revision=uncommitted`; a correspondência foi estabelecida pelo hash dos fontes, sem inventar uma revisão executada.

Os loaders foram registrados no commit `4c7a6ae` e o protocolo de épocas/gate da DAG no commit `90706ad`. A [consolidação verificada no homelab](../reports/semana09-datasets/consolidation.json) confirmou os 17 resultados no mesmo fingerprint e as quatro evidências reais de serviços. O pico RSS foi do processo executor compartilhado, não uma estimativa de memória incremental por dataset. O [complemento do relatório no Space](https://chatgpt.com/space/page_ea19e36695b88191aeefc5cc7af299ad) foi criado e vinculado ao relatório Semana 09, preservando os registros anteriores.

## Bloqueio e retomada

O endereço oficial `http://klab.cis.rit.edu/files/Stream-51.zip`, ainda indicado na implementação dos autores, não resolveu DNS no homelab (`Name or service not known`). Não houve negação de autenticação, alteração de DNS/firewall/credenciais ou contorno de controle de acesso. A obtenção do arquivo permaneceu **pendente de uma cópia oficial acessível**. O MD5 publicado `5f34d542b71af7e5ecb2226d1bc7297c` ficou como referência; uma leitura parcial não verificou o arquivo integral.

Para retomar, fornecer o caminho de uma cópia oficial de `Stream-51.zip` no homelab, ou uma URL de origem autorizada. Montar o arquivo somente para leitura no cliente isolado e definir `BENCH_STREAM51_ARCHIVE` com o caminho interno. Preparar os 100 exemplos, repetir a admissão completa e conferir `complete=true` com as 18 fontes/fingerprint atual. Não iniciar a campanha com recibo parcial.

Com a admissão aprovada, iniciar somente o worker de `compose.dataset-benchmark.yaml`, usando o arquivo de ambiente já existente, sem ler seu conteúdo. Copiar somente os dois novos arquivos (`benchmark_task_suite.py` e `task_jobs.py`) para os componentes Airflow necessários, sem substituir os arquivos antigos. Confirmar importação, saúde do worker e `/datasets-ready`; despausar e disparar `benchmark_task_suite` com run ID único. Conferir que o estado do dag_run passou a `running`; esse foi o critério solicitado para encerrar a tarefa, sem esperar a conclusão da campanha.

Comandos de validação, executados **no homelab** com o snapshot final e o estado isolado:

```bash
export TASK_BENCH_CODE=/home/gabriel/semana09-datasets-20261103-2341/review-final/code
export TASK_BENCH_STATE=/home/gabriel/semana09-datasets-20261103-2341/state
export TASK_BENCH_SERVICE_ENV=/home/gabriel/.config/continual-bench/benchmark.env

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

O primeiro comando terminou com exit code 1 enquanto faltou Stream51; isso foi o resultado esperado do gate fechado, não uma aprovação de todas as fontes. A repetição offline usou o cache verificado e não repetiu a tentativa DNS conhecida. Para uma nova aquisição explicitamente autorizada, acrescentar `--download` e o mount do arquivo local quando necessário.
