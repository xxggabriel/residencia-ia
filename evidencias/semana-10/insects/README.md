# Evidências do benchmark INSECTS na Semana 10

A campanha reuniu 66 de 72 execuções completas. As tabelas e figuras comparam as configurações concluídas pela média das seeds 42, 43 e 44. Para as métricas pareadas, usei a interseção dos eventos dos 33 runs de cada fluxo, com 236.587 eventos no abrupto e 301.228 no recorrente. As duas configurações Online com Hoeffding aparecem apenas no diagnóstico parcial, com janelas anteriores ao primeiro drift anotado.

## Relatórios

- [Consolidação dos benchmarks](consolidacao.md)
- [Análise da adaptação aos drifts](adaptacao-aos-drifts.md)
- [Documento de resultados e metodologia](resultados.md)
- [PDF com os nove gráficos finais](figuras-finais-insects.pdf)

## Acurácia, F1 macro e recall por classe

- [Acurácia e F1 macro nos mesmos eventos](figuras/08-acuracia-f1-pareados.png) compara as 22 combinações concluídas. Cada linha liga as duas métricas e indica a diferença em pontos percentuais. A linha tracejada é a acurácia de sempre prever a classe majoritária.
- [Recall por classe com UPF](figuras/09-desempenho-por-classe.png) compara a proporção de acertos entre positivos e entre negativos para os dois modelos e fluxos. As métricas usam os mesmos eventos da figura anterior.

No Abrupto, o UPF com Hoeffding obteve 69,83% de acurácia e 42,42% de F1 macro, uma diferença de 27,41 pontos percentuais. O modelo identificou 1,53% dos positivos e 95,54% dos negativos. Se eu previsse sempre a classe negativa, a acurácia nesse recorte seria 72,65%. No Recorrente, a mesma política com Hoeffding atingiu 73,69% de acurácia, superou a referência de 72,28% e identificou 40,55% dos positivos. Para obter o recall negativo, usei a identidade que expressa a acurácia como a média dos recalls das duas classes ponderada pelas respectivas proporções. Como essas proporções são iguais entre as seeds de cada fluxo, o cálculo também vale para as médias.

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

O [notebook da Semana 10](../../../../notebooks/figuras_benchmarks_semana_10.ipynb) lê os dados arquivados por seed, calcula as médias e gera os nove gráficos sem autenticação AWS. Também mantive a consulta individual por seed no MLflow para inspeção. Os arquivos históricos da seed 42 e o CSV de medianas continuam no diretório de dados para auditoria.

A documentação dos fluxos e das anotações está em [Souza et al., 2020](https://doi.org/10.1007/s10618-020-00698-5).
