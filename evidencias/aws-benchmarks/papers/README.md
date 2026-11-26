# Dados comparativos dos artigos CARA e UPF

Os arquivos preservam números dos artigos e das execuções históricas da Semana 10. A comparação é descritiva: usar a mesma métrica permite cotejar resultados, mas não demonstra uma reprodução idêntica quando mudam fontes, partições, modelos ou versões.

| Arquivo | Conteúdo |
| --- | --- |
| `published-reference-values.csv` | 110 observações transcritas dos artigos, com tabela, página, ano, DOI, unidade e definição da dispersão disponível. |
| `paper-local-comparisons.csv` | 74 pares entre um valor publicado e uma média histórica, com caminho JSON e SHA da evidência local. |
| `comparison-parameters.csv` | Grades de α e κ, número de repetições, tamanhos dos lotes e temporização histórica. |
| `protocol-comparability.csv` | Dez dimensões que distinguem artigos, execuções históricas e a rodada F109. |
| `paper-metrics.csv` | Definições de SCPE, acurácia média, retreinamentos, custo combinado e sua AUC. |
| `limitations.json` | Ressalvas que acompanham a leitura dos dados. |

A fonte CARA é [Cost-aware retraining for machine learning (2024)](https://doi.org/10.1016/j.knosys.2024.111610), Tabelas 2, 3 e 4, página 9 do PDF publicado. Elas contêm SCPE, acurácia das consultas e número de retreinamentos. Os resumos locais usam cinco seeds e 99 preços positivos de κ, excluindo κ igual a zero nas três métricas. O arquivo antigo não contém desvios por seed, portanto esses campos permanecem vazios. O texto da seção 7.2 e a Tabela 1 apresentam uma divergência no número de lotes online; não a resolvi por suposição.

A fonte UPF é [When to retrain a machine learning model (2025)](https://proceedings.mlr.press/v267/regol25a.html), Tabelas 1 e 2, página 8 do PDF final do PMLR. O [DOI disponível](https://doi.org/10.48550/arXiv.2505.14903) identifica a versão preprint. A AUC integra o custo combinado sobre α entre 0 e 1 em Electricity e entre 0 e 0,7 em Airlines, chamado airplanes no artigo. Essa AUC não é AUROC. Os desvios locais UPF seguem a convenção populacional do harness histórico, com ddof igual a zero e dez seeds. Os valores publicados após ± foram preservados sem atribuir um tipo estatístico ou divisor não declarado na fonte consultada.

Célula numérica vazia significa informação indisponível. Não significa zero. As execuções históricas não são as 132 unidades da rodada F109. Não compare F1 ou AP atuais diretamente com acurácia, SCPE ou AUC de custo publicados. A rota histórica que adaptou CARA ao protocolo UPF também tem uso de rótulos do teste corrente na obsolescência, uma limitação que acompanha seus resultados.

Para uma figura com um representante da família, use o rótulo **CARA (CARA-T)**. Os CSVs preservam CARA-CT e CARA-P; não afirmam que seus resultados sejam equivalentes.

O notebook deve ler os valores publicados como referência fixa e recalcular as colunas locais a partir da evidência selecionada. A identificação do protocolo, dos parâmetros, das seeds e da dispersão precisa ser preservada em cada nova exportação. Um resultado de outro protocolo não se torna comparável apenas porque o dataset ou a política tem o mesmo nome.
