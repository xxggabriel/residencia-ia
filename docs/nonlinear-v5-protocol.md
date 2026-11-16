# Campanha nonlinear-v5 — protocolo histórico e recuperação

A campanha comparou regressão logística, Hoeffding Tree e kNN com os dados, protocolos e orçamentos
verificados na v4, com atenção a Agrawal e Airlines. As configurações foram fixadas antes dos testes.
O ranking posterior descreveu o desempenho nos recortes utilizados. A evidência da campanha está
no histórico Git e pode ser consultada com `git show f835ee8:reports/nonlinear-v5.md`.

## Matriz

A campanha `benchmark_nonlinear` reuniu 867 configurações, com 289 por modelo, intercaladas por caso.
O [manifesto congelado](../configs/history/campaign-nonlinear-v5.json) fixa as configurações e o SHA da implementação.

- Cinco datasets; SEA/Agrawal/STAGGER com sementes 42–46; Electricity/Airlines com um prefixo cada.
- 3000 exemplos, aquecimento100, atraso simulado3, lote incremental32, janela de replay256.
- Streaming/Kafka e incremental: congelado, online, periódico, ADWIN e DDM.
- Tetos pós-warmup2900 e5800 para periódico/ADWIN/DDM; online2900 e congelado0.
- Offline: prefixo de treino1797 com rótulos disponíveis;1200 exemplos finais para teste congelado.
- Oito `evaluate_and_publish` em paralelo e oito subprocessos no worker usado naquela campanha.
- A campanha repetiu os 289 controles logísticos e comparou com a v4 congelada.

Os três modelos são deterministas em sua construção; não usam um estimador aleatório sem semente.
`PYTHONHASHSEED=0` é registrado e protegido no contrato de retomada. Isso não implica determinismo
universal entre versões, plataformas ou ambientes diferentes.

## Modelos fixados

| Modelo | Parâmetros | Representação |
|---|---|---|
| logística | learning_rate0.05 | StandardScaler incremental para numéricos; one-hot para strings |
| Hoeffding Tree | grace_period50, max_depth8, max_size8MiB, memory_estimate_period100, leaf_prediction=nba | atributos originais; categorias nativas |
| kNN | k11, janela256, ponderação por distância, LazySearch exato | mesmo encoder incremental da logística |

Árvore: Agrawal declara elevel/car/zipcode nominais; STAGGER declara size/color/shape nominais,
apesar da codificação inteira do gerador. Nos dados reais, as strings já são nominais.
`max_size` da árvore é estimado pelo River, não uma quota rígida de memória do processo.

kNN: a janela FIFO retém no máximo256 exemplos mesmo no offline. A apresentação de um exemplo
continua contando para o orçamento, mesmo se for descartado posteriormente. Vizinhos antigos
conservam a padronização do instante de admissão; a evolução do scaler pode mudar as coordenadas.
A busca exata evita acrescentar aleatoriedade de um índice aproximado. Probabilidades podem chegar a0/1;
o log loss mantém o clipping definido pelo benchmark, sem suavização nova específica do modelo.

Referências de API: [HoeffdingTreeClassifier](https://riverml.xyz/latest/api/tree/HoeffdingTreeClassifier/),
[KNNClassifier](https://riverml.xyz/latest/api/neighbors/KNNClassifier/) e
[código LazySearch](https://github.com/online-ml/river/blob/main/river/neighbors/lazy.py).
As assinaturas e o comportamento usados foram também inspecionados na instalação fixada River0.26.1.
Esses links de consulta são móveis; a versão executada é a dependência fixada em
`requirements.lock`, registrada no ambiente de cada run, e não o conteúdo futuro de `latest/main`.

## Rastreamento e retomada

Os experimentos continuam separados por política/detector. Cada run registra `model`, `model_params`,
dataset/hash, orçamento, código e ambiente, além dos metadados anteriores. Model Registry mantém
as famílias históricas da logística; os novos nomes incluem hoeffding_tree ou knn.

Todo ajuste inicial e cada atualização posterior preservam estado e seleção dos dados no MinIO.
Os modelos finais incluem o código de inferência dos três adaptadores, assinatura e dependências.
Assinaturas numéricas usam double para não congelar um inteiro incidental da primeira amostra
(por exemplo commission=0 no Agrawal). A inferência deve normalizar essas colunas para float64;
`inference_frame` e o helper de verificação implementam essa normalização.

O campo `feature_schema` do catálogo imutável v1 descreve somente a primeira amostra e pode indicar
int em features que têm frações posteriormente. A assinatura do modelo é independente desse campo;
o catálogo antigo não é sobrescrito. Os ARFF admitidos preservam seu esquema completo de origem.

Retomadas podem mudar bytes pickle por ordem de sets ou aliases de strings imutáveis. Se isso ocorrer,
a finalização compara o estado integral usando uma representação canônica de verificação, preservando
ordem de dicts e aliases mutáveis. Pesos realmente alterados sem record continuam sendo rejeitados.
O blob e SHA originais da versão gravada permanecem os mesmos. Hash de diários grandes é calculado
em streaming; o auditor também verifica cada estado sem carregar o diário inteiro em memória.

## Comparação e limites

As comparações parearam dataset, versão, semente, protocolo, política e orçamento. O relatório
registrou acurácia/F1, Brier/log loss, consumo efetivo, alarmes, atualizações e dispersão entre
cinco sementes para as fontes sintéticas. As comparações offline e as diferenças da logística
em relação à v4 foram registradas separadamente.

O teto de apresentações não iguala CPU, tempo, tamanho do estado ou custo de persistência entre modelos.
Online continua aprendendo; periódico/drift reiniciam na janela256. Representação, capacidade e histórico
retido também diferem entre modelos. Os reais continuam com um único prefixo e relógio/atraso simulados;
Electricity conserva a incerteza sobre normalização/imputação anterior à aquisição.

Os scripts `scripts/verify_versioned_campaign.py --manifest configs/history/campaign-nonlinear-v5.json`
e `scripts/compare_nonlinear_campaign.py` verificam as execuções retidas e a comparação histórica.

## Recuperação registrada durante a execução

Após 865 casos concluídos, dois casos Airlines/árvore/streaming/DDM (tetos2900 e5800)
falharam no guard de estado final: o `NominalSplitterClassif.cond_proba` do River0.26.1
criava dicionários de classe vazios durante a predição. A correção remove somente essas novas
entradas vazias da folha visitada, mantendo os valores de probabilidade e todas as estatísticas
preexistentes. O guard de versionamento permanece ativo.

A revisão original e suas 865 runs foram preservadas. O manifesto registra um override de fonte
somente nas duas configurações recuperadas, executadas novamente desde o início; suas tentativas
anteriores ficam em `/state/failure-snapshots/nonlinear-v5/159e/` no worker. A evidência anterior
à correção está em `reports/nonlinear-v5-recovery-before.json`; a conferência posterior de métricas,
digest de predições, alarmes e ledger fica em `reports/nonlinear-v5-recovery.json`.
Os 156 testes da imagem corrigida passaram, incluindo seis regressões para essa mutação.
Não houve ajuste de hiperparâmetros, orçamento, dataset ou política na recuperação.
