# Matriz de cobertura da Semana 09

O escopo verificável partiu do pedido e das Pages da Semana 09, lidas em 3 de novembro de 2026. A planilha `Semana_09_Datasets_Benchmarks.xlsx` não foi materializada: o leitor autorizado retornou `page_reference_unsupported_format` para a referência fornecida. A matriz abaixo não substituiu os 47 registros, 39 associações e 64 referências da pesquisa. Não foram inventados registros nem seleções ausentes.

**Implementado** descreveu código executável; **validado inicialmente** descreveu testes pequenos no homelab; **campanha pendente** descreveu comparação experimental ainda não realizada. Fixtures não homologaram dados originais.

| Componente | Fonte e versão | Implementação | Testes e estado |
| --- | --- | --- | --- |
| SEA, AGRAWAL, STAGGER | River 0.26.1, contratos Semana 08 | Reutilizados, conceitos segmentados e recorrência explícita | Suíte anterior, integração e smokes; validação inicial |
| Hyperplane e Random RBF Drift | River 0.26.1, links no [detalhamento](week09-datasets.md) | Novos wrappers nativos com movimento obrigatório e seeds | Replay, contrato, campanha smoke e Hyperplane com serviços; campanha comparativa pendente |
| FriedmanDrift gra, lea, gsg | River 0.26.1, Ikonomovska et al. 2011 | Adaptador preservou regressão; avaliador LinearRegression/StandardScaler | Três variantes em fixtures; smoke real do gerador com MAE/RMSE |
| Electricity Elec2 | OpenML 151 v1, SHA-256 fixado em `real-sources.lock.json` | Adaptador existente preservado | Testes de admissão existentes; nenhum novo download ou benchmark real |
| INSECTS | Repositório USP dos autores, variante a selecionar | Adaptador local CSV/ARFF com manifesto obrigatório | Contratos com fixtures; aquisição, licença, variante e dados reais pendentes |
| Gas Sensor Array Drift | UCI 224, lotes na ordem declarada | Adaptador local LIBSVM multiclasse, classes preservadas | Fixture de lotes e avaliador multiclasse; dados originais e revisão de direitos pendentes |
| Split MNIST, CIFAR10 e CIFAR100 | Protocolos sobre bases oficiais; links no detalhamento | Partição por ordem de classes explícita, preservando splits | Transformações e invariantes testados; preparação de pixels/representações e campanhas pendentes |
| Permuted MNIST | Permutação artificial fixada por seed | Permutação comum a treino/teste por tarefa, tarefa zero identidade | Reprodutibilidade, IDs e preservação de splits testados; loader bruto e campanha pendentes |
| CORe50 | Autores, CoRL 2017 | Adaptador de registros preparados e transformação por sessão/grupo explícito | Contratos testados; cenário NI/NC/NIC, run e sessões oficiais pendentes |
| Stream51 | Implementação dos autores | Adaptador preparado e ordem explícita de grupos/frames | Contratos testados; extração de frames e avaliação de novidade pendentes |
| CLEAR10 e CLEAR100 | Autores, NeurIPS 2021 e releases 2022 | Adaptador preparado e buckets/splits explícitos | Contratos testados; release, imagens, licença por imagem e protocolo pendentes |
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
| Kafka, ZenML, MLflow, MinIO | Versões da stack Semana 08 | Integração preservada, suplemento publicado com os artefatos | Verificação real de eventos, quatro passos, hashes, registro e replay; ver relatório |

Fontes, decisões e testes por área: [datasets](week09-datasets.md), [políticas](week09-policies.md), [métricas](week09-metrics.md). Resultados e limitações observados: [relatório](semana09-report.md).
