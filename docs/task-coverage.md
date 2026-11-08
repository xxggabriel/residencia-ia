# Matriz de cobertura da Semana 09

O escopo verificável partiu do pedido e das Pages da Semana 09, lidas em 3 de novembro de 2026. A planilha `Semana_09_Datasets_Benchmarks.xlsx` não foi materializada: o leitor autorizado retornou `page_reference_unsupported_format` para a referência fornecida. A matriz abaixo não substituiu os 47 registros, 39 associações e 64 referências da pesquisa. Não foram inventados registros nem seleções ausentes.

**Implementado** descreveu código executável; **validado inicialmente** descreveu testes pequenos no homelab; **campanha pendente** descreveu comparação experimental ainda não realizada. Fixtures não homologaram dados originais. A ampliação de 4 de novembro, incluindo os testes de 100 exemplos/2 épocas e o bloqueio de Stream51, foi registrada no [relatório de datasets](semana09-dataset-execution.md).

| Componente | Fonte e versão | Implementação | Testes e estado |
| --- | --- | --- | --- |
| SEA, AGRAWAL, STAGGER | River 0.26.1, contratos Semana 08 | Reutilizados, conceitos segmentados e recorrência explícita | Suíte anterior, integração e smokes; validação inicial |
| Hyperplane e Random RBF Drift | River 0.26.1, links no [detalhamento](dataset-catalog.md) | Novos wrappers nativos com movimento obrigatório e seeds | Replay, contrato, campanha smoke e Hyperplane com serviços; campanha comparativa pendente |
| FriedmanDrift gra, lea, gsg | River 0.26.1, Ikonomovska et al. 2011 | Adaptador preservou regressão; avaliador LinearRegression/StandardScaler | Três variantes em fixtures; smoke real do gerador com MAE/RMSE |
| Electricity Elec2 e Airlines | Fontes e SHA-256 fixados em `real-sources.lock.json` | Adaptadores existentes e factory tipado | Caches originais verificados, 100 exemplos/2 épocas aprovados nas duas fontes |
| INSECTS | Origem dos autores, `abrupt_balanced` | Loader ARFF original, 33 atributos/6 classes, cache com checksum observado | Contratos e 100 exemplos/2 épocas reais aprovados; demais variantes/campanhas pendentes |
| Gas Sensor Array Drift | UCI 224, dez lotes na ordem declarada | Loader ZIP/LIBSVM original, 128 atributos/6 classes | Contratos e 100 exemplos/2 épocas reais aprovados; recorte não cobriu todos os lotes |
| Split MNIST, CIFAR10 e CIFAR100 | MNIST IDX fixado; CIFAR Parquet `uoft-cs` em revisões imutáveis | Pixels, tarefas e splits explícitos; CIFAR nativo também implementado | 100 exemplos/2 épocas aprovados nas três variantes; aquisição nativa CIFAR excedeu timeout, variante Parquet declarada |
| Permuted MNIST | MNIST oficial, seed e permutação fixadas | Duas tarefas disjuntas no orçamento global de 100 originais; desvio do protocolo integral declarado | Contratos, cache e 100 exemplos/2 épocas reais aprovados; campanha completa pendente |
| CORe50 | Autores, arquivo original 128×128 | Leitura parcial ZIP, sessões de teste 3/7/10, RGB 16×16 diagnóstico | Contratos e 100 exemplos/2 épocas reais aprovados; NI/NC/NIC completo pendente |
| Stream51 | Implementação/arquivo dos autores, CVPRW 2020 | ZIP/JSON/bbox, arquivo local ou HTTP Range, ordem original | Fixtures aprovadas; dados reais bloqueados por DNS da origem; novelty detection não implementada |
| CLEAR10 e CLEAR100 | Release julho/2022, mirror fixado em `temporal-vision.lock.json` | ZIP parcial, metadados/buckets/splits oficiais, RGB 16×16 | Contratos e 100 exemplos/2 épocas reais aprovados nas duas variantes; direitos individuais documentados; campanha temporal completa pendente |
| Covertype e variantes | CARA e catálogo não materializado | Nenhuma cronologia natural atribuída | Reconciliação e seleção pendentes; nenhum loader fictício |
| CGLM, CLOC, BudgetCL, CLAD e demais registros | Catálogo não materializado | Mantidos como candidatos sem implementação presumida | Escopo, acesso, licença e recursos pendentes |
| Sem retreinamento, periódico, online e por alarme | Contratos Semana 08 | Preservados | Suíte anterior e smoke declarativo |
| Por desempenho | Regra operacional declarada, versão local v1 | Janela de erro dos rótulos liberados, threshold, cooldown, orçamento | Unitários, integração, retomada e serviços |
| Por custo | Heurística local v1; [CARA 2024](https://ananthmahadevan.com/publication/cost-aware-retraining/cost-aware-retraining.pdf), [UPF ICML 2025](https://proceedings.mlr.press/v267/regol25a.html) como referências | Benefício projetado versus custo de cenário; desvios explícitos | Ponto de equilíbrio e guardas testados; reprodução fiel dos papers pendente |
| ADWIN, DDM, EDDM, HDDMA, HDDMW, Page Hinkley, KSWIN | River 0.26.1 | Wrappers nativos sobre erro binário liberado; KSWIN com seed | Contratos, pickle e mudança controlada; análise de sensibilidade pendente |
| Drift em atributos P(X) | Distinção documentada | O engine monitorou erro, não atribuiu alarmes a P(X) | Detector multivariado de atributos não implementado nem presumido selecionado |
| Promoção de modelo | Gate operacional local v1 | Holdout temporal do candidato separado do treino; rejeitados consumiram orçamento | Falhas, atraso, rejeição, contabilidade e fluxo integrado testados |
| Qualidade, calibração, custo, latência, memória e recuperação | `residency-metrics-v1` | Métricas existentes reutilizadas; custo passou a incluir candidatos rejeitados | Suíte anterior e novas verificações de contabilidade |
| Métricas de detecção | Matcher existente e validação de anotações v1 | Matching um a um, censura e indisponibilidade sem ground truth | Casos calculáveis; F1 não calculado para movimento contínuo ou dados reais sem anotação |
| Retenção, forgetting, BWT/FWT | GEM arXiv 1706.08840v6; suplemento `semana09-metrics-v1` | Resumo adicional com nomes distintos dos históricos; FWT exigiu baseline inicial | Matrizes calculáveis e ausências testadas; campanha visual CL pendente |
| Near-future accuracy | Hammoud et al., arXiv 2305.09275v1 | Adaptação binária operacional com snapshot, gap, horizonte e atraso | Retomada e replay testados; executada nos smokes; sem alegação de reprodução visual ou UPF |
| Regressão e multiclasse | River 0.26.1 | Avaliador tipado com none/online/periodic e splits intercalados | Friedman e três classes em fixtures; sem colapsar classes/alvos para bool |
| Campanhas declarativas | `semana09-v1` | Casos explícitos, journal, lock, prazo e retomada verificada | Seis casos de smoke; campanha com múltiplas seeds e escalabilidade pendente |
| DAG ampliada | `benchmark_task_suite`, Airflow 3.3.2 | 549 casos: matriz binária e 18 fontes tipadas; gate 100 exemplos/2 épocas/fingerprint | Importação e plano aprovados; DAG não iniciada enquanto Stream51 permaneceu bloqueado |
| Kafka, ZenML, MLflow, MinIO | Versões da stack Semana 08 | Integração preservada, suplemento publicado com os artefatos | Verificação real de eventos, quatro passos, hashes, registro e replay; ver relatório |

Fontes, decisões e testes por área: [datasets](dataset-catalog.md), [políticas](task_evaluation-policies.md), [métricas](task_evaluation-metrics.md). Resultados e limitações observados: [relatório](semana09-report.md).
