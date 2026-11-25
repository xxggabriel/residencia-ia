# Dados científicos dos benchmarks

Este conjunto permite reproduzir as tabelas e figuras a partir de CSVs locais, sem executar treinamento ou consultar serviços externos. A campanha contém quatro streams integrais, 11 políticas e as seeds 42, 43 e 44. As 132 unidades científicas correspondem a 4 × 11 × 3. Foram observadas 133 tentativas operacionais porque uma publicação de métricas precisou ser retomada, sem novo treinamento. A tentativa adicional não é uma quarta seed nem outro experimento.

A revisão de código dos experimentos é `f1092ee0232a6d16504b3a2211a037c41a02068a`; o protocolo é `ordered-stream-batch-prequential-metrics-v2`. O modelo observado é XGBoost. Os identificadores das consultas e os hashes estão em [provenance.json](provenance.json), e [manifest.json](manifest.json) vincula os arquivos e o contrato científico.

## Arquivos

| Arquivo | Linhas | Conteúdo |
|---|---:|---|
| `a97023a9-f737-471f-b724-d4cf5e4da0f3.csv` | 132 | Consulta 12: 34 métricas por execução, na coorte comum |
| `e931f10b-82ca-41b7-9f35-fee2b662b30e.csv` | 6.168 | Consulta 13b: 48 métricas selecionadas de recursos, com aplicabilidade própria |
| `d5acb9d0-962b-440d-8050-3f66ecb12069.csv` | 7.707 | Consulta 14b: todos os agregados por escopo/métrica, incluindo os valores das três seeds |
| `59db7383-f7d9-4030-9b68-c2ccef89ae1a.csv` | 132 | Consulta 16: 16 janelas alinhadas por execução, em `window_series_json` |
| `cases.csv` | 132 | Identidade científica, hashes das fontes e limites temporais |
| `quality-by-seed.csv` | 4.488 | 34 métricas de qualidade, em formato longo, com denominador `cohort_n` |
| `quality-summary.csv` | 1.496 | Média e desvio-padrão amostral das métricas de qualidade |
| `resources-by-seed.csv` | 6.336 | Grade de recursos com 168 ausências explícitas, preservadas como NULL |
| `resources-summary.csv` | 2.112 | Recursos agregados, com contagens de fatos presentes e ausentes |
| `metrics-by-seed.csv` | 23.121 | Todos os fatos escalares exportados, mantendo o escopo original |
| `policy-summary.csv` | 44 | Resumo por stream e política, com qualidade e recursos em colunas separadas |
| `windows-by-seed.csv` | 2.112 | Janelas em formato tabular: limites, tamanho e quatro métricas |
| `windows-summary.csv` | 2.816 | Média e desvio-padrão por janela e métrica |
| `holdout-summary.csv` | 132 | Diagnóstico separado de F1 macro, AP e tamanho do holdout |
| `training-inventory.csv` | 132 | Contagens de ajustes, rodadas registradas e verificações do modo `metrics_only` |
| `hardware.csv` | 8 | Modelos de CPU observados e quantidade de avaliações |
| `metric-dictionary.csv` | 222 | Unidade, direção, definição e limites de cada combinação de escopo e métrica |

Os quatro arquivos nomeados por identificador de consulta foram recodificados das linhas salvas da API Athena. Os valores textuais e os NULLs preservam os hashes semânticos verificados pelo carregador. Isso não afirma identidade de bytes com o CSV originalmente produzido no armazenamento remoto. Não há cópias JSON das mesmas linhas nem predições por exemplo neste diretório.

## Escopos e interpretação

Foram consumidas 1.392.014 observações das fontes. A comparação principal utiliza a mesma faixa após a calibração e antes do holdout, compartilhada pelas políticas e seeds de cada stream:

| Stream | Fonte integral | Início inclusivo | Fim exclusivo | Exemplos por execução na comparação |
|---|---:|---:|---:|---:|
| Electricity | 45.312 | 13.599 | 40.781 | 27.182 |
| Airlines | 539.383 | 161.815 | 485.445 | 323.630 |
| INSECTS abrupt imbalanced | 355.275 | 106.583 | 319.748 | 213.165 |
| INSECTS incremental abrupt reoccurring imbalanced | 452.044 | 135.615 | 406.840 | 271.225 |

O prefixo de treino inicial, a calibração e o holdout não são acrescentados ao denominador dessa comparação. As métricas originais `prequential`, `holdout` e por classe permanecem identificadas em `metric_scope`; não devem ser misturadas com `common_postcalibration`. A média das métricas de janela também não substitui uma métrica calculada sobre toda a coorte. Streams distintos não constituem uma única população de teste para produzir um ranking agregado.

Os agregados utilizam desvio-padrão amostral, `ddof=1`, entre três repetições computacionais. As barras dos gráficos não são intervalos de confiança populacionais. Um valor zero de desvio-padrão é preservado. Células vazias representam NULL, nunca imputação de zero. Em `resources-by-seed.csv`, `source_fact_present=False` identifica os 168 fatos ausentes; os arquivos originais não permitem concluir se cada ausência corresponde a uma métrica indefinida, inaplicável ou não publicada.

ROC AUC e AP utilizam histogramas uniformes de 4.096 bins. `pr_auc` representa precisão média não interpolada, e não área trapezoidal da curva precision-recall. `brier_half_sum` usa metade da soma dos erros quadráticos por classe; no caso binário equivale a `mean((y-p_1)^2)`. A integral `upf_paper_auc` é outra quantidade: integra custo combinado sobre a grade de alpha, deve ser minimizada e não é uma porcentagem de AUROC ou AP. O SCPE representa excesso percentual relativo ao Oracle do protocolo CARA, sem equivalência direta com faturamento AWS.

`train_seconds` acompanha ajustes lógicos promovidos; `actual_total_fit_seconds` acompanha o trabalho físico da unidade atual. Políticas com cache compartilham matrizes construídas por CARA, portanto o custo de uma unidade consumidora não equivale ao custo de construir uma matriz independente. A materialização já integra o total físico. `update_count` contabiliza atualizações implantadas após o modelo inicial, sem contar todos os candidatos. Custos de cenário ou de tabela de preços não são a fatura observada.

O inventário contém 2.265 registros de ajuste: 1.161 físicos e 1.104 de evidência reutilizada. As 123 materializações estão contidas nos ajustes físicos. Foram registradas 116.100 rodadas físicas de boosting, com 100 rodadas por ajuste; os 226.500 pontos incluindo reutilizações não representam treino físico adicional. O parâmetro congelado chama-se `max_epochs`, mas o eixo observado é `xgboost_boosting_round`. Não há pesos ou checkpoints nestes CSVs. Oito modelos de CPU foram observados; a mesma reserva de 2 vCPU e 4 GiB não torna o desempenho físico dos processadores idêntico.

Os campos originais `retraining_online_*` descrevem a fase de decisão após a calibração do protocolo de matriz. Seus nomes não indicam que a política de treinamento online tenha sido executada nesta campanha.

O subdiretório [papers](papers/README.md) preserva os valores publicados, locadores de tabelas e comparações com os protocolos históricos correspondentes. Ele contém 110 fatos de referência e 74 pares históricos. Esses resultados ficam separados da campanha F109: diferenças de protocolo impedem tratá-los como uma reprodução equivalente ou calcular uma superioridade causal entre os conjuntos.

## Reprodução e novos dados

O notebook [figuras_benchmarks_semana_10.ipynb](../../../notebooks/figuras_benchmarks_semana_10.ipynb) utiliza esta pasta por padrão. `load_athena` mantém a validação estrita dos quatro resultados F109, incluindo os hashes semânticos e a matriz 4 × 11 × 3. Nenhum download é necessário para usar os CSVs versionáveis daqui.

```python
from pathlib import Path
from benchmark_athena_data import load_athena

evidence = load_athena(Path("../docs/evidencias/aws-benchmarks"))
```

Para outra campanha, crie outro diretório com um manifesto `benchmark-tables-v1`. Declare datasets, políticas, seeds, modelo, protocolo, hashes das fontes, limites da coorte, métricas, número de janelas e `sd_ddof=1`. Vincule os CSVs `cases`, `primary`, `resource_values`, `windows` e `reported_metrics` em `tables`, incluindo caminho relativo, contagem de linhas e SHA-256. Preserve a grade por seed e represente valores ausentes explicitamente. A tabela primária deve declarar `cohort_n`; as janelas devem ser contíguas e alinhadas entre as políticas e seeds.

```python
from benchmark_athena_data import load_tables

evidence = load_tables(Path("../docs/evidencias/outra-campanha"))
```

`load_tables` aceita as seeds distintas declaradas no novo contrato; três é a configuração desta campanha, sem rotular outra quantidade como três. O carregador verifica hashes, esquema, identidade, cobertura e consistência interna. Sua proveniência é `contract_schema_and_hashes_only`: dados novos não herdam a auditoria F109 nem uma alegação de reprodução dos experimentos anteriores. Todos os dados das 11 políticas permanecem nesta exportação; filtros usados apenas para reduzir repetições visuais dos gráficos não removem linhas dos CSVs.

Cada linha de `reported_metrics` deve declarar `source_fact_present` e `availability`. O estado `defined` exige presença e valor numérico; `undefined` ou `source_null` exige presença e NULL; `missing` ou `missing_or_undefined` exige ausência e NULL. Contradições e estados desconhecidos são rejeitados em qualquer escopo.

O diagnóstico de holdout exige linhas para `macro_f1`, `pr_auc` e `n` em cada combinação de dataset, política e seed, mesmo quando a observação estiver ausente. Nesse caso, declare NULL explicitamente e a disponibilidade correspondente. F1 e AP definidos devem estar entre zero e um. Quando `n` estiver definido, deve corresponder a `source_rows - holdout_start`; NULL não é substituído por zero nem pelo denominador conhecido do contrato. Não existe inferência automática de um holdout parcial ou de uma campanha sem diagnóstico de holdout.
