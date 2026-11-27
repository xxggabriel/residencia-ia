# Limitações e trabalhos pendentes

Esta página reúne as limitações conhecidas do código e das evidências, com o arquivo que sustenta cada item. Ela serve para delimitar o que os resultados podem e não podem afirmar.

## Escopo científico das evidências

| Limitação | Evidência |
|---|---|
| A campanha AWS registrada cobre 4 streams, 11 políticas e as seeds 42, 43 e 44 com XGBoost. Outros modelos, streams e seeds não foram executados nesse protocolo. | `evidencias/aws-benchmarks/README.md`, `cases.csv` |
| Nas 44 combinações de stream e política dessa campanha, `macro_f1` e `pr_auc` têm desvio-padrão zero entre as seeds. As três seeds não estimam a variabilidade da qualidade preditiva, apenas confirmam determinismo do pipeline nesse caso. | `evidencias/aws-benchmarks/quality-summary.csv` |
| O desvio-padrão usado nos agregados é amostral, entre três repetições computacionais. As barras dos gráficos não são intervalos de confiança. | `evidencias/aws-benchmarks/README.md` |
| A campanha INSECTS da Semana 10 reuniu 66 de 72 unidades; faltam as 6 unidades da política Online com Hoeffding Tree. | `evidencias/semana-10/insects/README.md` |
| O manifesto da comparação CARA e UPF da Semana 10 registra `git_revision: uncommitted` e `python_hash_seed: random`; o histórico CARA não preservou desvios por seed e o UPF usa desvio populacional (`ddof=0`). | `evidencias/semana-10/cara-upf/` |
| Os valores publicados por artigos ficam separados das execuções locais. Diferenças de janelas, baselines e protocolos impedem tratá-los como reprodução equivalente ou como prova de superioridade. | `evidencias/aws-benchmarks/papers/README.md`, `protocol-comparability.csv` |
| O Oracle é uma referência retrospectiva com acesso ao futuro, não uma política implantável. | [Modelos e políticas](Modelos-e-Politicas) |
| Os resultados das campanhas `controlled-v3`, `versioned-v4` e `nonlinear-v5` não estão versionados na árvore atual; restam os manifestos em `configs/history/` e o histórico Git. | `configs/history/` |
| O dossiê da campanha corretiva (60 execuções INSECTS e 24 ensaios UPF) foi removido da árvore no commit `8e1a4415e`. Os números dessa campanha só podem ser conferidos pelo histórico Git. | `git show a3aa3469a:docs/evidencias/benchmark-corrigido/README.md` |

## Métodos e protocolo

| Limitação | Evidência |
|---|---|
| Não há recalibração de probabilidades (Platt, isotônica ou similar). O código mede o erro de calibração (ECE e MCE). O termo "calibração" também nomeia a fase offline que estima parâmetros de CARA, UPF e Oracle. | `src/continual_bench/evaluation/` |
| O limiar de decisão não é otimizado em validação: é 0,5 no motor binário e `c_FP / (c_FP + c_FN)` no avaliador por blocos (0,025 em `configs/base/tracking.yaml`, 0,5 em `configs/base/tracking-dataset-suite.yaml`). | [Metodologia experimental](Metodologia-Experimental#calibração-e-limiar-de-decisão) |
| Na suíte de streams ordenados, a paciência do *early stopping* é igual ao número máximo de épocas, o que na prática desativa a parada antecipada. A paciência 10 vale apenas para o avaliador iterativo. | `scripts/run_dataset_suite.py`, `src/continual_bench/evaluation/iterative.py` |
| No avaliador por blocos, o detector não é reiniciado após um retreino e os avisos (*warning*) são ignorados. A política `drift` entrega ao ADWIN a taxa de erro do bloco, um valor contínuo, e não o erro por exemplo. | `src/continual_bench/evaluation/stream_benchmark.py` |
| EDDM, HDDM-A, HDDM-W, Page-Hinkley e KSWIN só têm testes de contrato, sem campanha registrada. | `tests/test_task_policies.py` |
| Não há detector de mudança na distribuição dos atributos, P(X); todos os detectores monitoram o erro do modelo. | [Detectores de drift](Detectores-de-Drift) |
| O avaliador de streams não mede atraso de detecção contra as fronteiras de drift publicadas do INSECTS. `validate_temporal_holdout` só é chamado em testes. | [Metodologia experimental](Metodologia-Experimental) |
| Apenas os datasets Zia recebem peso de classe; Electricity, Airlines e INSECTS não. | `src/continual_bench/datasets/streams.py` |
| A política `online` do avaliador por blocos está fora da campanha AWS registrada. | `evidencias/aws-benchmarks/cases.csv` |
| Os datasets Zia (Social, Fraud e E-commerce) ficam fora da seleção AWS atual; suas evidências são históricas. | [Datasets](Datasets) |

## Dados

| Limitação | Evidência |
|---|---|
| INSECTS balanceado e Gas Sensor Array Drift não têm SHA-256 publicado pela fonte para conferência independente. | `configs/contracts/real-streams.lock.json` |
| O hash completo dos arquivos Parquet de CIFAR não é verificado. | [Datasets](Datasets) |
| Todas as fontes preparadas usam relógio ordinal simulado, não o tempo real de coleta. | `src/continual_bench/datasets/catalog.py` |
| Permuted MNIST usa uma variante com orçamento limitado e CORe50 cobre apenas o cenário NI. | [Datasets](Datasets) |
| Covertype, CGLM, CLOC, BudgetCL e CLAD aparecem em planos, sem carregador implementado. | [Datasets](Datasets) |

## Infraestrutura, custos e operação

| Limitação | Evidência |
|---|---|
| A estimativa de custo por job Batch aplica preço On-Demand também a jobs em Fargate Spot, usa a janela de início e fim do job e um mínimo único de 60 s por job. | `src/continual_bench/application/cloud/costs.py`, `lifecycle.py` |
| Na campanha registrada, o mínimo de 60 s domina `aws_cost_total_usd`, que fica idêntico entre políticas. Esse campo não discrimina políticas. | `evidencias/aws-benchmarks/resources-summary.csv` |
| 7 de 10 preços do snapshot estão marcados como `seed_pending_price_list_refresh`; não há registro de execução de `refresh-prices`. | `configs/contracts/aws-prices.us-east-1.json` e [Custos e recursos](Custos-e-Recursos#preços-usados) |
| A reconciliação com o CUR 2.0 existe em código e testes, mas não há comando nem etapa de workflow que a execute, nem evidência registrada. Nenhum valor do repositório é faturamento observado. | `tests/test_aws_streaming_ml_costs.py` |
| 168 fatos de recursos estão ausentes (NULL) sem indicação de se são indefinidos, inaplicáveis ou não publicados. | `evidencias/aws-benchmarks/resources-by-seed.csv` |
| A campanha rodou em 8 modelos de CPU sob a mesma reserva de 2 vCPU e 4 GiB; tempos de execução são descritivos, não comparáveis como desempenho de hardware. | `evidencias/aws-benchmarks/hardware.csv` |
| Não há registro de interrupção real de Fargate Spot seguida de retomada por checkpoint; a recuperação é coberta por testes. | [Execução na AWS](Execucao-na-AWS#validações-registradas) |
| Cancelamento manual, falha induzida e nova tentativa com sufixo não foram exercitados na conta. | [Execução na AWS](Execucao-na-AWS#validações-registradas) |
| A suíte de quatro streams exige espera de fila de pelo menos 86.400 s, mas o padrão de `QueueWaitTimeoutSeconds` é 3.600 s. | `infra/cloudformation/root.yaml`, `scripts/aws/run_dataset_campaigns.py` |
| Não há evidência de ativação dos módulos AWS opcionais na conta. | [Execução na AWS](Execucao-na-AWS#módulos-opcionais) |
| No modo `metrics_only`, uma execução interrompida recomeça sem restaurar o treino; nas políticas CARA, UPF e Oracle o checkpoint só é confirmado ao final. | [Operação e diagnóstico](Operacao-e-Diagnostico) |
| A imagem `linux/amd64` não executa em máquinas arm64 sob emulação QEMU. | [Execução na AWS](Execucao-na-AWS#validações-registradas) |
| O checkpoint é serializado com pickle. O checksum detecta corrupção, não adulteração; só carregue checkpoints de origem confiável. | [Reprodutibilidade e artefatos](Reprodutibilidade-e-Artefatos) |
| Execuções locais registram `git_revision: uncommitted` quando `BENCH_GIT_REVISION` não está definido. | [Reprodutibilidade e artefatos](Reprodutibilidade-e-Artefatos) |

## Qualidade de engenharia

| Limitação | Evidência |
|---|---|
| O teste de arquitetura ignora imports relativos, não verifica `orchestration` nem `cli` e tolera ciclos entre alguns pacotes. | `tests/test_architecture.py` |
| O marcador pytest `benchmark` está declarado, mas nenhum teste o utiliza. | `pyproject.toml` |
| `requirements.lock` está defasado em relação ao `uv.lock`: uma nova exportação acrescenta `pytest-xdist` e `execnet`. | `requirements.lock`, `uv.lock` |
| Os carregadores de dados do notebook de figuras não têm teste automatizado. | `notebooks/` |
| `scripts/preflight.py` aceita apenas um endereço IP privado fixo. | `scripts/preflight.py` |
| Os plugins de `examples/` só são encontrados pela CLI quando `PYTHONPATH=.` está definido. | [Execução local](Execucao-Local) |

## Trabalhos pendentes

- Executar a política `online` do avaliador por blocos em uma rodada separada da campanha registrada.
- Atualizar o snapshot de preços com `refresh-prices` e expor a reconciliação com o CUR 2.0 por um comando ou etapa de workflow.
- Exercitar na conta a interrupção de Spot com retomada, o cancelamento manual e a nova tentativa com sufixo.
- Ampliar a validação dos detectores EDDM, HDDM, Page-Hinkley e KSWIN para além dos testes de contrato.
- Medir atraso de detecção contra as fronteiras publicadas do INSECTS no avaliador de streams.
- Reexportar `requirements.lock` a partir do `uv.lock` e cobrir os carregadores do notebook com testes.
