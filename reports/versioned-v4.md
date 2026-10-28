# Integração de datasets, experimentos e versões — 27/10/2026

Implementados Electricity (OpenML 151 v1) e Airlines (OpenML 1169 v1), além de SEA,
Agrawal e STAGGER. Novas runs usam experimentos separados por política/detector, DatasetInput
nativo, metadados completos, catálogo imutável dos dados e versões recuperáveis dos modelos.

## Execução completa verificada

A DAG `benchmark_versioned`, run `versioned_v4_20261027`, concluiu **289 avaliações** e **292 tarefas**,
sem falhas nem retries. Início `2026-10-27 20:21:41.683396+00:00`; fim `2026-10-27 20:46:20.197486+00:00`;
duração 24.64 minutos, incluindo orquestração e publicação.
O monitor registrou oito `evaluate_and_publish` e oito subprocessos do worker ativos
simultaneamente em 415 de 713 observações.

| Dataset | Configurações | Sementes / seleção |
|---|---:|---|
| SEA | 85 | cinco sementes, 42–46 |
| Agrawal | 85 | cinco sementes, 42–46 |
| STAGGER | 85 | cinco sementes, 42–46 |
| Electricity | 17 | um prefixo fixo de 3000 exemplos |
| Airlines | 17 | um prefixo fixo de 3000 exemplos |

Os três protocolos foram executados: streaming com Kafka, incremental e offline com corte temporal.
A matriz mantém os orçamentos pós-warmup de 2900/5800 apresentações; online aprende cada rótulo
liberado, inclusive na drenagem final. A matriz e o SHA executados estão em
[campaign-versioned-v4.json](../configs/campaign-versioned-v4.json).

## MLflow e versionamento

As 289 runs estão FINISHED. Filtrar `tags.campaign = 'versioned-v4'` para separar a campanha
dos testes integrados. Os experimentos históricos em `continual-bench-v1` foram preservados.

| Estratégia | Experimento | ID |
|---|---|---:|
| policy-drift/detector-adwin | `continual-bench/policy-drift/detector-adwin` | 6 |
| policy-drift/detector-ddm | `continual-bench/policy-drift/detector-ddm` | 5 |
| policy-none/detector-none | `continual-bench/policy-none/detector-none` | 8 |
| policy-online/detector-none | `continual-bench/policy-online/detector-none` | 9 |
| policy-periodic/detector-none | `continual-bench/policy-periodic/detector-none` | 7 |

Foram conferidos **17 hashes de datasets**, **289 versões finais
no Model Registry**, em **55 famílias**, e **99992 registros de estado**.
Há 99272 IDs lógicos únicos e 49641 hashes de pesos únicos:
estados idênticos podem ser compartilhados entre braços, sem criar cópias artificiais de conteúdo.

Cada ajuste inicial e cada atualização posterior preserva pesos, pré-processador, versão pai,
hash dos dados, seleção de treino, disponibilidade/liberação do rótulo e SHA do código.
O Registry contém o modelo final de cada run; os estados intermediários, inclusive cada atualização
online, ficam em diário e índice imutáveis no MinIO. Os ARFF brutos, eventos JSONL, catálogo,
código de treino e seleções dos dados são recuperáveis pelos URIs registrados nos recibos.

As runs incluem fonte/licença/versão/hash do dataset, parâmetros, protocolo, semente e seu significado,
orçamento e consumo, métricas, alarmes, atualizações, configuração completa, ambiente e versões dos modelos.
Todos os pacotes finais foram comparados com os estados e com o código-fonte arquivado.
Um modelo por dataset foi recarregado pelo Registry em processo novo, usando o código embarcado,
e reproduziu a predição de referência.

## Verificação e revisão final

- 92 testes unitários passaram na imagem congelada da campanha.
- 95 testes passaram localmente e na imagem final, incluindo falha no staging MLflow e corrupção da linhagem.
- 16 testes integrados passaram: cinco datasets × três protocolos, mais a proteção de origem do painel MLflow.
- Auditoria de 289 runs: DatasetInput, experimento, Registry READY, conteúdo S3, cadeia e checksum de cada estado,
  orçamento, atualização online por rótulo e pacote/dependências de inferência.
- As 255 configurações sintéticas reproduziram dados e métricas da v3; 240 digests prequential e 15 cortes/janelas
  offline também foram reproduzidos exatamente. O versionamento preservou o comportamento preditivo.

SHA do código da campanha: `ead28c768252514ba84b0ff7dd827bb85bf8da9514cce85a40c5b123c4376499`.
A revisão final `8c9e598dd768943bd9725810ebd192fc28e160f106e67084c3eee284d1c0f629` acrescenta confirmação atômica do pacote MLflow e validação
da linhagem no carregamento de estados. Essa imagem foi implantada depois de preservar a revisão
congelada da campanha. Um pipeline completo adicional com Electricity, Kafka e 180 exemplos passou
na imagem implantada, gerando 151 estados e o modelo `models:/continual-bench-electricity-streaming-online-adwin/1`; o carregamento
em processo novo reproduziu a predição e usou o código embarcado.

As correções finais tratam recuperação/integridade e não alteram dados, modelo ou protocolo da campanha.
Os 289 pacotes produzidos pela revisão congelada foram integralmente conferidos antes da troca da imagem.

## Resultados novos e limites

No prefixo Electricity, online alcançou acurácia de 86.52% no streaming e 82.41% no incremental,
contra 68.52% e 68.48% do modelo congelado. DDM emitiu 23/24 alarmes no streaming e o teto menor
limitou seu consumo a 2900 apresentações; ADWIN emitiu sete alarmes e consumiu 1792.
Sem pontos de mudança reais anotados, esses números não medem falsos alarmes ou recall de detecção.

No Airlines, a acurácia online foi 72.93%/73.00%, próxima dos 72.86% do congelado.
O F1 online ficou em 14.21%/14.43%; DDM atingiu 14.83%/14.88%. A elevada acurácia, isoladamente,
encobre a dificuldade na classe positiva. Os prefixos completos têm 40.1% de positivos no Electricity
e 26.83% no Airlines; esses percentuais descritivos incluem o aquecimento.

Os ARFF completos foram admitidos (45312 e 539383 linhas), mas o treino desta campanha usa somente
os primeiros 3000 exemplos. A ordem publicada foi preservada; cronologia e disponibilidade de rótulos
reais não foram estabelecidas. O relógio e o atraso são simulados. Sementes não produzem repetições
independentes nos dados reais. Electricity já vem normalizado/imputado, com escopo de ajuste não documentado.
O catálogo OpenML declara licença `Public`; preservamos esse rótulo sem inventar uma licença SPDX.

Os resultados usam regressão logística River e a representação atual das features; não estabelecem
superioridade geral. Tetos de apresentações não igualam tempo de CPU ou consumo dos detectores.
Os testes integrados ocorreram durante parte da campanha; seu tempo total não é um ensaio controlado
de speedup. A prova de portabilidade cobre uma entrada por dataset no ambiente existente, sem reinstalar
dependências em um ambiente totalmente novo.

### Todas as configurações reais

Acurácia e F1 em porcentagem. Consumo é pós-warmup; offline treina 1797 exemplos e avalia
1200, com três rótulos de fronteira expurgados. Os protocolos têm janelas de avaliação distintas.

| Dataset | Protocolo | Política/detector | Teto | Consumo | Acurácia | F1 | Alarmes | Atualizações |
|---|---|---|---:|---:|---:|---:|---:|---:|
| electricity | streaming | none | 0 | 0 | 68.52 | 46.89 | 0 | 0 |
| electricity | streaming | online | 2900 | 2900 | 86.52 | 82.96 | 0 | 2900 |
| electricity | streaming | periodic | 2900 | 2900 | 74.28 | 69.30 | 0 | 12 |
| electricity | streaming | drift/adwin | 2900 | 1792 | 72.34 | 63.55 | 7 | 7 |
| electricity | streaming | drift/ddm | 2900 | 2900 | 77.00 | 72.15 | 23 | 12 |
| electricity | streaming | periodic | 5800 | 3584 | 79.38 | 72.39 | 0 | 14 |
| electricity | streaming | drift/adwin | 5800 | 1792 | 72.34 | 63.55 | 7 | 7 |
| electricity | streaming | drift/ddm | 5800 | 3328 | 77.17 | 72.28 | 24 | 13 |
| electricity | incremental | none | 0 | 0 | 68.48 | 46.80 | 0 | 0 |
| electricity | incremental | online | 2900 | 2900 | 82.41 | 77.98 | 0 | 2900 |
| electricity | incremental | periodic | 2900 | 2900 | 74.03 | 69.13 | 0 | 12 |
| electricity | incremental | drift/adwin | 2900 | 1792 | 75.07 | 69.99 | 7 | 7 |
| electricity | incremental | drift/ddm | 2900 | 2560 | 71.66 | 67.20 | 14 | 10 |
| electricity | incremental | periodic | 5800 | 3584 | 79.38 | 72.42 | 0 | 14 |
| electricity | incremental | drift/adwin | 5800 | 1792 | 75.07 | 69.99 | 7 | 7 |
| electricity | incremental | drift/ddm | 5800 | 2560 | 71.66 | 67.20 | 14 | 10 |
| electricity | offline | none | — | — | 65.83 | 28.07 | 0 | 0 |
| airlines | streaming | none | 0 | 0 | 72.86 | 3.44 | 0 | 0 |
| airlines | streaming | online | 2900 | 2900 | 72.93 | 14.21 | 0 | 2900 |
| airlines | streaming | periodic | 2900 | 2900 | 71.93 | 8.54 | 0 | 12 |
| airlines | streaming | drift/adwin | 2900 | 0 | 72.86 | 3.44 | 0 | 0 |
| airlines | streaming | drift/ddm | 2900 | 256 | 72.28 | 14.83 | 1 | 1 |
| airlines | streaming | periodic | 5800 | 3584 | 71.66 | 9.27 | 0 | 14 |
| airlines | streaming | drift/adwin | 5800 | 0 | 72.86 | 3.44 | 0 | 0 |
| airlines | streaming | drift/ddm | 5800 | 256 | 72.28 | 14.83 | 1 | 1 |
| airlines | incremental | none | 0 | 0 | 72.86 | 3.44 | 0 | 0 |
| airlines | incremental | online | 2900 | 2900 | 73.00 | 14.43 | 0 | 2900 |
| airlines | incremental | periodic | 2900 | 2900 | 71.83 | 7.68 | 0 | 12 |
| airlines | incremental | drift/adwin | 2900 | 0 | 72.86 | 3.44 | 0 | 0 |
| airlines | incremental | drift/ddm | 2900 | 256 | 72.38 | 14.88 | 1 | 1 |
| airlines | incremental | periodic | 5800 | 3584 | 71.55 | 8.44 | 0 | 14 |
| airlines | incremental | drift/adwin | 5800 | 0 | 72.86 | 3.44 | 0 | 0 |
| airlines | incremental | drift/ddm | 5800 | 256 | 72.38 | 14.88 | 1 | 1 |
| airlines | offline | none | — | — | 69.92 | 6.23 | 0 | 0 |

## Evidências e reprodução

- [Auditoria e comparação](versioned-v4-validation.json)
- [Airflow](versioned-v4-airflow.json) e [concorrência](versioned-v4-concurrency-summary.json)
- [Fontes admitidas](versioned-v4-sources.json) e [ambiente](versioned-v4-environment.json)
- [Testes unitários finais](versioned-v4-unit-tests.txt) e [integração](versioned-v4-service-tests.txt)
- [Prova após implantação](versioned-v4-final-probe.json) e [saúde final](versioned-v4-final-health.json)
- [Operação](../docs/benchmark-operations.md)

`scripts/verify_versioned_campaign.py` confere as publicações no worker;
`scripts/summarize_versioned_campaign.py` compara a evidência com a v3.
O cache completo conferido é `reports/generated/versioned-v4-inputs.json` (ignorado pelo Git).

Análise comparativa dos cinco datasets: [versioned-v4-analysis.md](versioned-v4-analysis.md).
