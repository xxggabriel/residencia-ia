# Validação anterior à execução integral

A leitura de integridade percorreu os 201.392.014 exemplos das sete fontes preparadas. As contagens, classes, valores ausentes, ordem e hashes constam em [input-integrity.json](input-integrity.json). Valores ausentes das features foram preservados. A preparação não excluiu linhas para produzir esta validação.

## Escopo e alvos

| Stream | Exemplos integrais | Alvo do benchmark |
| --- | ---: | --- |
| Electricity | 45.312 | Projeção binária publicada no manifesto preparado |
| Airlines | 539.383 | Projeção binária publicada no manifesto preparado |
| INSECTS abrupt imbalanced | 355.275 | Classe de origem 2 contra as demais |
| INSECTS incremental abrupt reoccurring imbalanced | 452.044 | Classe de origem 2 contra as demais |
| Fraud | 50.000.000 | legitimate / fraud |
| E-commerce | 50.000.000 | other_status / returned |
| Social | 100.000.000 | follow / friend / block / mute |

As tarefas têm significados e prevalências distintos. As comparações principais entre políticas usam a mesma coorte por stream e seed, posterior à calibração. Comparações entre streams são descritivas. AP positiva binária, AP macro e os respectivos denominadores permanecem identificados. As três seeds são repetições computacionais da mesma ordem publicada, sem inferência de três populações independentes.

## Controles de recursos

Os [21 controles V6](resource-validation-v6-summary.json) ajustaram um modelo por fonte e seed, com replay real de quatro blocos e 100 épocas, seguidos por uma avaliação de 256 exemplos posteriores. Cada contêiner teve limite de 2 vCPU e 4.096 MiB, sem swap adicional, com duas threads de treino. Todos terminaram com código zero e sem OOM. O maior RSS de processo observado foi de 619.364.352 bytes. RSS não representa toda a memória do contêiner.

Esses controles medem o ajuste e seu envelope de recursos, sem representar a trajetória completa dos benchmarks. Os modelos existiram em memória para treinamento e predição. Somente métricas, históricos, hashes e tamanho lógico do modelo foram persistidos. Os relatórios completos e históricos estão comprimidos sem alterar os bytes originais; os índices registram hashes antes e depois da compressão.

A imagem V6 fixa a revisão `5a136062d564997528702b1d104d547cbd4f80f0` e o hash de código `023bfefde0046c2d3a91a5498a5838b05f4407c3da6f0c3d3458193e104aa9f7`. A versão do verificador está vinculada ao plano. Os controles V1 e V5 permanecem separados como evidências intermediárias.

## Verificação funcional e rastreamento

O smoke usa prefixos preparados e três épocas, com todas as onze políticas admitidas, sete streams e seeds 42, 43 e 44, totalizando 231 casos. Ele verifica cobertura, predições, definições das métricas e publicação. O orçamento integral de treino permanece em 100 épocas máximas.

A rodada V5 concluiu os 231 casos e preservou seus valores. Sua auditoria independente conferiu 252 pais, 1.528 fits, 867.594 pontos nativos, 3.276 objetos por SHA e 231 bundles, sem divergências nos valores ou presença de pesos. Encontrou 191 marcadores `tracking_v1_status` ainda em `publishing`, decorrentes de uma corrida entre a escrita inicial assíncrona e a conclusão síncrona. Por esse motivo, [o pacote V5](functional-smoke-v5/manifest.json) permanece incompleto como condição de admissão. Os registros originais dessa rodada não foram reparados nem substituídos.

A correção ordena de forma síncrona os marcadores de início e término de publicação. O verificador exige os três marcadores finais antes e depois da leitura dos históricos. A [rodada V6](functional-smoke-v6/manifest.json) executou novamente os 231 casos na revisão corrigida, com 77 casos por seed, e foi aprovada. O [QA científico e da API](functional-smoke-v6/validation/scientific-qa.json.gz) reconstruiu as métricas e conferiu 36.201 séries, 121.929 pontos do catálogo e os três marcadores finais antes e depois dos históricos. A [auditoria independente](functional-smoke-v6/validation/native-audit.json.gz) verificou 252 pais, 1.528 fits, 867.580 pontos nativos, 3.276 objetos por SHA e 231 bundles, sem ausências, divergências, duplicatas ou pesos persistidos. As diferenças de contagem entre as auditorias decorrem do escopo: o catálogo dos pais é parte do conjunto completo de observações, sistema e fits.

Os manifestos de seleção descrevem os payloads estruturados preservados, os membros equivalentes dos bundles e os assets de apresentação omitidos. JSON, JSONL e CSV usam compressão gzip sem perda; Parquet e arquivos gzip existentes conservam seus bytes. Valores `null`, vetores e metadados de proveniência permanecem disponíveis. Dados preparados e pesos não integram este pacote.

## Revisão V7 e publicação com memória limitada

A revisão `ef4b9966395eeaf204df20c6126b6c81e404f7d6`, com hash de código `61d4ecfb6da63610223f007f8c5af470201cd3f267361c840c4f1bca38964d25`, gera os payloads nativos dos ajustes conforme há vagas de publicação. A fila acompanha a concorrência configurada. Com um worker, um ajuste termina de ser enviado antes de gerar o próximo. Todos os pontos e a ordem dos recibos permanecem preservados. A suíte local passou 1.628 testes, incluindo os limites da fila nas três seeds e a repetição sem duplicação.

Os [21 controles V7](resource-validation-v7-summary.json) repetiram o replay de quatro blocos, as 100 épocas e a avaliação posterior. Todos terminaram com código zero, 2 vCPU, 4.096 MiB e duas threads. O maior RSS de processo observado foi de 601.985.024 bytes. Os [21 hashes de modelos em memória e os resultados das avaliações de controle](resource-v6-v7-logical-parity.json) coincidiram com V6. Tempos, RSS e amostras de sistema são observações novas; essa coincidência não estabelece igualdade de desempenho entre as rodadas.

A [rodada funcional V7](functional-smoke-v7/manifest.json) concluiu os 231 casos. O [QA científico e da API](functional-smoke-v7/validation/scientific-qa.json.gz) reconstruiu suas métricas e verificou integralmente 36.201 séries e 121.929 pontos do catálogo dos pais. Os três marcadores finais estavam corretos antes e depois da consulta dos históricos. A [auditoria SDK](functional-smoke-v7/validation/native-audit.json.gz) conferiu 252 pais, 1.528 ajustes, 867.062 pontos, 3.276 objetos por SHA e 231 bundles.

A [integração REST](rest-publication-v7/manifest.json) republicou os mesmos 231 relatórios científicos usando HTTP e armazenamento reais, com o modo Spot e retenção somente de métricas. A [prova independente dessa rota](rest-publication-v7/validation/rest-SQL-audit.json.gz) conferiu 231 pais, 1.507 ajustes e 684.604 pontos, além de 30.059 objetos por SHA. Os marcadores e metadados foram consultados pela API. O primeiro caso também teve todos os seus históricos lidos pela API e uma repetição idempotente. Os demais históricos foram confrontados por SQL offline integral, sem apresentação dessa leitura como uma auditoria API de todos os filhos.

A [prova conjunta](functional-smoke-v7/validation/broad-native-audit.json.gz) usa uma única cópia consistente do banco, com IDs disjuntos entre as rotas: 483 pais, 3.035 ajustes e 1.551.666 pontos. Não houve divergências. As diferenças de contagem entre SDK e REST refletem suas projeções, aliases e os 21 controles incluídos apenas no escopo SDK; as provas preservam o escopo e as definições de cada rota.

Três [probes suplementares de transporte](rest-publication-v7/transport-probes/proof.json.gz), um por seed, verificaram 600 contadores cada. Em cada probe, um ponto ficou deliberadamente apenas no ledger e foi recuperado. A API confirmou os 1.800 pontos finais, sem duplicação nem alteração dos timestamps nativos após repetição. O instante de observação do ledger e o instante de criação do ponto por `progress` foram preservados separadamente. São entradas controladas de teste, sem treino, excluídas dos comparativos científicos e das contagens da prova conjunta. O caminho assíncrono nativo foi exercitado; o fallback `AsyncMetricWriter` não foi acionado nesses probes.

O pacote funcional conserva 29.387 payloads estruturados e 351.926.880 bytes originais. A seleção REST mantém 4.170 payloads exclusivos, 25.659 referências por SHA a conteúdos idênticos e dez arquivos suplementares. As verificações remota e local conferiram hashes, recomposição gzip, aliases e ausência de pesos. Os controles V7 têm [índice próprio](resource-validation-v7-index.json), e o [plano integral V7](suite-full-plan-v7.json) mantém 100 épocas máximas e as três seeds.

## Admissão da execução integral

O [recibo atual de admissão](admission.json) vincula V7 às provas acima e preserva o [recibo V6](admission-v6.json). O plano integral e a implantação têm recibos próprios. A presença destes controles não declara concluída a campanha AWS. O QA e a auditoria nativa de V7 foram aprovados. Sua submissão depende do novo merge autorizado, do sucesso completo do CI/CD e da conferência das reservas reais, quota e regras de fila. O prazo de cancelamento por capacidade é alinhado às 24 horas dos pedidos.

O [recibo de atualização da configuração](runtime-override-update.json) registra a alteração condicional e a leitura de confirmação dos parâmetros. Ele prepara a implantação e não declara que a stack ou os jobs já foram executados.
