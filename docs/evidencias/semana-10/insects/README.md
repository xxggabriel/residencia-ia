# Evidências do benchmark INSECTS na Semana 10

A campanha reuniu 66 de 72 execuções completas. As tabelas e figuras comparam as configurações concluídas pela média das seeds 42, 43 e 44. Para as métricas pareadas, usei a interseção dos eventos dos 33 runs de cada fluxo, com 236.587 eventos no abrupto e 301.228 no recorrente. As duas configurações Online com Hoeffding aparecem apenas no diagnóstico parcial, com janelas anteriores ao primeiro drift anotado.

## Relatórios

- [Consolidação dos benchmarks](consolidacao.md)
- [Análise da adaptação aos drifts](adaptacao-aos-drifts.md)
- [Documento de resultados e metodologia](resultados.md)
- [PDF com os sete gráficos finais](figuras-finais-insects.pdf)

## Métricas e auditoria

| Arquivo | Conteúdo |
| --- | --- |
| [runs.csv](dados/runs.csv) | Inventário e indicadores dos 66 runs concluídos |
| [metricas-mlflow.csv](dados/metricas-mlflow.csv) | 24.723 registros de métricas escalares exportadas do SageMaker MLflow |
| [medias.csv](dados/medias.csv) | Média e desvio padrão amostral no recorte original de cada configuração |
| [cobertura.csv](dados/cobertura.csv) | Unidades previstas e concluídas da matriz |
| [metricas-pareadas-por-seed.csv](dados/metricas-pareadas-por-seed.csv) | Métricas de cada um dos 66 runs no recorte em comum |
| [metricas-pareadas-medias.csv](dados/metricas-pareadas-medias.csv) | Média e desvio padrão amostral no recorte em comum |
| [regimes-por-seed.csv](dados/regimes-por-seed.csv) | F1 macro e cobertura por regime, política, modelo e seed |
| [detector-por-seed.csv](dados/detector-por-seed.csv) | Detecções, alarmes e atrasos do ADWIN para os 12 runs |
| [adaptacao-por-seed.csv](dados/adaptacao-por-seed.csv) | Acurácia, detecções e recuperação por fronteira nas três seeds |
| [janelas-f1-por-seed.csv](dados/janelas-f1-por-seed.csv) | Histórico de F1 macro por janela nos 36 runs de regressão logística |
| [drift-por-seed.json](dados/drift-por-seed.json) | Fronteiras, séries de acurácia, alarmes e recuperação dos 12 runs ADWIN |
| [janelas-parciais-online-hoeffding.csv](dados/janelas-parciais-online-hoeffding.csv) | Diagnóstico das duas configurações incompletas, com contagens por janela |
| [SHA256SUMS](SHA256SUMS) | Somas de verificação dos dados, gráficos e PDF |

## Reprodução

O [notebook da Semana 10](../../../../notebooks/figuras_benchmarks_semana_10.ipynb) lê os dados arquivados por seed, calcula as médias e gera os sete gráficos sem autenticação AWS. Também mantive a consulta individual por seed no MLflow para inspeção. Os arquivos históricos da seed 42 e o CSV de medianas continuam no diretório de dados para auditoria.

A documentação dos fluxos e das anotações está em [Souza et al., 2020](https://doi.org/10.1007/s10618-020-00698-5).
