# Verificações da campanha

A revisão de avaliação foi `98a6dd5a676f46bf1df337de82b33fd0e554c085`. Os arquivos abaixo distinguem conclusão dos cálculos, consistência dos resultados, publicação das métricas e integridade do pacote. Uma verificação não substitui as demais.

| Verificação | Evidência e alcance |
| --- | --- |
| Execução integral | [Recibo final](final-campaign-completion.json): 60 casos INSECTS, oito jobs UPF com 24 ensaios e seeds 42, 43 e 44. Os processos terminaram com código zero. |
| Publicação nativa | [Resumo MLflow](mlflow/native-mlflow-verification-summary.json): 108 runs principais, 1.032 ajustes associados e 20.912.990 registros de métricas conferidos. Não foram encontradas ausências, duplicações, valores divergentes ou pontos extras segundo os critérios registrados. |
| Artefatos publicados | [Auditoria detalhada](mlflow/final98-native-mlflow-audit.json): 108 ZIPs e 10.968 membros com hashes e tamanhos conferidos. Os diretórios auditados não contêm pesos ou checkpoints; os ajustes associados não publicaram objetos de modelo. |
| Exportação integral | [QA dos dados](export-qa/qa-evidence.json): 68 payloads, 5.084 vetores e 19.337.586 linhas de ledger comparadas com as fontes, sem falhas. Foram conferidos 22.407.729 registros da tabela de métricas em sete partes e os agregados. A [QA visual](export-qa/qa-figuras-final.json) registra a inspeção individual dos 12 PNGs. |
| Tempos registrados | [Recibo de publicação](execution-times/corrective-execution-times-tracking.json): 60 pontos derivados foram conferidos por leitura de retorno. A [tabela](execution-times/corrective-execution-times.csv) tem 60 tempos por caso e oito tempos por job de três seeds, sem inferir um tempo individual por ensaio UPF. |
| Código e pós-processamento | [Checks da revisão 07aa7f5](postprocessing-checks.json): formato, lint e 1.238 testes aprovados, com 1.503 avisos. A seleção padrão do repositório exclui testes marcados como serviços e benchmark; as execuções reais têm recibos próprios. |
| Contrato real UPF | [Preflight](execution-times/real-replications-preflight.json): oito jobs, 24 ensaios e 48 relatórios de candidatos e políticas, com hashes, recibos, contratos, seeds e linhagem conferidos sem rede. |
| Consistência UPF | [Auditoria numérica](upf/prova-auditoria-upf-98a6dd5.json): 91.742 verificações de cobertura, partições, custos, agendas, oráculos e agregações, sem divergências nos critérios registrados. |
| Curvas dos candidatos | [Auditoria complementar](upf/probability-curve-consistency.json): 8.880 verificações, 360 AP e 359 áreas ROC reconstruídas. Um holdout tem classe única e AUROC indefinida. |
| Repetibilidade curta | [Comparação de 30 pares](reproducibility/reproducibility-98a6dd5.json): SEA com 1.200 exemplos, dois modelos, cinco políticas e três seeds. Os campos científicos selecionados e 27.600 pares do ledger coincidiram exatamente. Tempos e recursos não fazem parte dessa igualdade. |
| Controles integrais | [Seis controles logísticos](reproducibility/full-controls-98a6dd5.json): 1.654.130 pares do ledger e campos científicos comparados entre revisões. Não cobre repetição integral de Hoeffding Tree. |
| Operação de pontuação | [Microbenchmark](microbenchmark/batch-scores-formal-summary.json): três seeds e duas implementações com classes e perdas equivalentes no escopo medido. Os tempos são do scorer, sem alegação de aceleração do pipeline completo. |
| Ambiente | [Imagem executada](runtime-image-98a6dd5.json), [distribuições](runtime-distributions-98a6dd5.json) e [readiness da imagem posterior](portable-runtime-07aa7f5.json). O último documento verifica empacotamento e CLIs em Linux x86 sem rede; não comprova uma execução gerenciada na AWS. |
| Recursos | [Inspeção do contrato](resource-profile-policy-audit.json) e [agendamento](execution-scheduling/campaign-auxiliary-execution.json). Oito casos usaram dois workers auxiliares em fases sem sobreposição. Os perfis reais e os preços nominais não são intercambiáveis. |
| Limites históricos | [Recuperação parcial](historical-recovery.json): uma fonte e três vínculos de recibos foram recuperados. Dois CSVs e o launcher específico continuam ausentes; os comprovantes encontrados não demonstram implantação AWS nem equivalência com os resultados históricos pendentes. |

A auditoria do tracking consultou uma cópia offline e verificou os artefatos somente depois de fechar as conexões do banco. A cópia privada foi excluída após a aprovação. Nenhum banco ou credencial integra este pacote. A [metodologia da verificação](mlflow/METODOLOGIA.md) descreve esse procedimento.

As métricas primárias comparadas têm três valores válidos. O campo `n` dos demais agregados conserva a disponibilidade real: seis grupos diagnósticos `strategy_scpe` de `upf-circles` em alpha zero têm `n=1` e `missing=2`; há também células indefinidas com `n=0`. Esses valores não foram preenchidos com zero nem promovidos a três repetições válidas.

As curvas dos candidatos não permitem reconstruir o ranking agrupado das 2.712 agendas e preços UPF. AP e AUROC dessas agendas, Brier, log loss e calibração permanecem como valores registrados pelo código executado, sem uma reconstrução independente a partir de probabilidades por linha. As três seeds nos fluxos reais fixos também não constituem três populações independentes.

O [índice das provas iniciais](index.json) conserva os escopos e hashes desses controles. O [manifesto da distribuição](../package-manifest.json) e seu arquivo SHA256SUMS relacionam todos os arquivos efetivamente distribuídos, incluindo as verificações posteriores. O manifesto científico original permanece inalterado.
