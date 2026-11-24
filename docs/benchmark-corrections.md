# Correções do protocolo e campanha de avaliação

Esta revisão trata a disponibilidade temporal dos rótulos no avaliador binário por
blocos, a separação entre referência de treino e teste na variante local CARA/UPF e
a retenção exclusiva de métricas. Os resultados anteriores permanecem nos diretórios
históricos. Uma campanha posterior tem revisão, configuração, dados e identificadores
próprios; não completa retroativamente uma leitura ou execução da residência.

## Contrato temporal

No binário incremental, as predições do bloco precedem o treino. Cada candidato é
ajustado no corte do fim do bloco com rótulos cujo `available_at` não ultrapassa esse
instante. O replay seleciona as últimas N linhas liberadas na ordem dos eventos,
limitadas por `replay_window`. Um rótulo pendente continua elegível para cortes
posteriores. O relatório registra cortes, quantidades, intervalos e hashes das
sequências usadas no treino.

As perdas retrospectivas da matriz servem à pontuação. A `MatrixView` entregue às
políticas filtra por exemplo as observações já disponíveis no corte. Denominador
zero significa informação indisponível. Não se inventa erro zero para calibrar ou
decidir. O CARA sem calibração offline válida fica inativo; o UPF sem observações
suficientes não decide com um histórico fictício.

A matriz binária limita a construção retrospectiva a 24 blocos contíguos. O tamanho
efetivo é o maior entre `batch_size` e o teto do número de eventos elegíveis dividido
por 24; uma cauda muito curta é incorporada ao bloco anterior. A fase offline usa
aproximadamente o primeiro terço dos blocos. Assim, o `batch_size: 500` configurado
não significa que os candidatos dos fluxos integrais sejam ajustados a cada 500
eventos. Os intervalos efetivos constam nos relatórios. O limite reduz o custo da
matriz sem selecionar uma amostra parcial da fonte.

O cálculo das perdas retrospectivas da logística usa operações em lote apenas para
a implementação nativa, com entradas numéricas e estado compatível. Linhas próximas
da fronteira de decisão voltam ao preditor escalar, com margem conservadora para o
arredondamento. As probabilidades do ledger continuam saindo do preditor original.
Modelos nativos com predição sem mutação podem reutilizar a decodificação do bloco
corrente; tipos personalizados seguem o caminho original. Os testes comparam classes,
perdas e estado nas três seeds antes de admitir essa otimização.

O Oracle conhece perdas futuras para escolher sua agenda, mas usa candidatos sujeitos
ao mesmo corte de treino e à mesma implantação que os comparadores. Sua agenda não é
uma política prospectiva. A seleção causal dos rótulos e o limite exato do replay
mudam em conjunto nesta revisão. Uma diferença frente ao histórico não pode ser
atribuída isoladamente ao atraso de rótulos.

O protocolo UPF individual e os pares usam referência de treino e consultas sem
rótulos de teste para o sinal CARA. A variante recebe o identificador
`UPF-train-reference-v3-full-source` na campanha integral. A formulação relativa local não é a baseline absoluta de
[When to retrain a machine learning model (2025)](https://proceedings.mlr.press/v267/regol25a.html),
seção A.12. O DOI [10.48550/arXiv.2505.14903](https://doi.org/10.48550/arXiv.2505.14903)
identifica o preprint, separado da publicação final PMLR sem DOI informado.

Em `same-batch`, há holdout por lote; o CARA recebe referência de treino e atributos
das consultas correntes, enquanto o UPF decide com o histórico anterior. Isso não
representa informação idêntica entre as políticas. `next-batch` é a comparação
operacional com implantação no lote seguinte. No
[Cost-aware retraining for machine learning (2024)](https://doi.org/10.1016/j.knosys.2024.111610),
seção 7.5, accounting de custo e acurácia prequential são conceitos distintos. A
reprodução histórica CARA conserva seu identificador e pré-processamento transdutivo;
nenhuma variante causal nova é apresentada como reprodução daquele resultado.

Na campanha integral, os exemplos de Electricity e Airlines são distribuídos em 15
blocos contíguos que cobrem o arquivo inteiro, incluindo a cauda. Os sete primeiros
blocos formam a fase offline e os oito seguintes a fase online do protocolo por lotes.
O holdout interno de 20% é fixado pela seed. O tamanho efetivo de cada bloco e os
índices das partições são registrados. Essa extensão deixa de selecionar a janela
parcial histórica e tem identificação própria, sem comparação equivalente com a AUC
publicada no artigo.

## Métricas e retenção

Nos avaliadores suportados, `artifact_mode: metrics_only` mantém modelos em memória durante o cálculo, sem gravar
checkpoints ou pesos no disco, no armazenamento de objetos ou no MLflow. Séries,
relatórios, histórico de treino, cronogramas, configurações e proveniência continuam
disponíveis. Um estado de treino incompleto não é retomado; a unidade reinicia do
começo. A publicação de evidência já concluída pode ser reenviada após validação.

Cada atualização mantém uma identidade lógica de linhagem. Tamanho e digest do estado
do modelo têm cadência explícita, com medições nas fronteiras registradas e na versão
final. Uma identidade lógica não é um hash dos pesos. Valores não medidos permanecem
indisponíveis entre os cortes. O treinamento e as predições não dependem dessa cadência.

Depois que todos os episódios e suas referências terminam, o modo metrics-only
libera o prefixo de exemplos mantido para essas referências. As métricas, os modelos
em memória ainda necessários à linhagem e seus metadados permanecem disponíveis. Um
prefixo incompleto continua retido até concluir os cálculos que dependem dele.

O perfil corretivo mantém os indicadores de qualidade, detecção, adaptação, retenção,
treino, recursos e custos disponíveis no respectivo protocolo. Inclui quatro janelas
limitadas de avaliação futura com modelo congelado, horizonte 32 e gap 1. Ausência de
anotação, suporte de classes ou população elegível preserva o motivo de
indisponibilidade. As séries publicadas não recebem os limites históricos de 250
pontos por chave ou 200 amostras de sistema.

Nesse perfil, a renderização dos PNGs detalhados de cada run e ajuste fica adiada.
Os dados completos, as séries no MLflow, o HTML e a linhagem permanecem disponíveis.
Os gráficos comparativos são gerados na consolidação, reduzindo trabalho repetido
de visualização durante a campanha.

A AUROC por regime é a área da curva ROC calculada com os escores retidos e empates
agrupados. O campo escalar é preenchido quando essa curva é válida. A ausência desse
campo no coletor intermediário não significa que o valor seja indefinido. Regimes
com uma única classe mantêm a indisponibilidade explícita.

A evidência numérica e o HTML são agrupados em `metric-evidence.zip`, com um
manifesto de membros e hashes. O relatório principal permanece acessível
individualmente. As séries completas continuam como métricas nativas no MLflow;
o agrupamento reduz chamadas de upload, sem reduzir os pontos registrados. Extraia
o pacote para abrir o HTML com seus arquivos relativos.

O ledger de previsões conserva IDs, sequência, rótulo e probabilidade, sem atributos
de entrada. O exportador calcula a interseção dos eventos por dataset e modelo entre
as configurações concluídas. F1 macro sempre usa as duas classes; AP é calculada por
ranking exato com empates agrupados. As estatísticas de INSECTS usam DP amostral
(`ddof=1`); as repetições UPF preservam DP populacional (`ddof=0`). São dispersões
descritivas. Seeds no mesmo fluxo fixo e janelas sobrepostas não são amostras
independentes de generalização.

O custo objetivo das políticas, a estimativa em dólares de um cenário Fargate e o
faturamento são campos distintos. A construção dos candidatos não é confundida com
os ajustes escolhidos pela política. Tempos de processo, benchmark, instrumentação e
publicação também têm escopos separados. Métricas de um host ou cgroup compartilhado
não representam consumo exclusivo de uma política.

Em particular, `compute_seconds` do protocolo por lotes inclui a construção de seus
candidatos. Nos baselines por evento, a matriz suplementar é construída depois desse
temporizador. Esses campos não sustentam uma comparação direta de custo total entre
as políticas. `wall_seconds_this_attempt` é capturado antes das derivações finais;
os horários de início e término de cada job abrangem também finalização e publicação.
O sampler de sistema acompanha o processo durante a avaliação e a finalização, com
seu escopo compartilhado registrado no relatório.

Após a conclusão integral, `publish_corrective_execution_times.py` valida os recibos
e deriva `corrective_campaign/job_body_wall_seconds` desses horários. O intervalo
inclui materialização, avaliação, diagnósticos e publicação dentro de `execute_job`.
Ficam fora a inicialização do processo, as importações anteriores ao início registrado,
a verificação inicial do código, o hash do artefato e a escrita final do recibo.
Os 60 casos da matriz recebem um ponto cada no tracking. Nos oito jobs UPF, o tempo
corresponde às três seeds em conjunto e não é dividido entre os ensaios. O campo
legado `pipeline_elapsed_seconds` permanece com seu significado original.

## Execução portátil

No MLflow, os runs identificam dataset, modelo, política, detector, modo, protocolo
e seed por tags `cb-*` quando essas coordenadas são aplicáveis. A tag `cb-run-role`
separa os casos da matriz, o diagnóstico dos candidatos e a comparação de políticas.
Os nomes legíveis e os artefatos completos permitem filtrar e comparar os mesmos
recortes sem depender de um identificador opaco.

Nos recibos de políticas UPF, `recorded_metric_points` conta entradas do catálogo,
inclusive valores nulos. Esse campo não representa a quantidade de pontos finitos
persistidos no MLflow. A conferência nativa compara as coordenadas e os valores
finitos; as indisponibilidades continuam documentadas nos relatórios.

Use Python 3.12 com as dependências fixadas e credenciais de tracking fornecidas pelo
ambiente. O entrypoint não contém endereço de servidor nem credenciais.

As evidências versionadas ficam fora do contexto Docker. Elas não são dependências
da imagem de execução; uma comparação histórica recebe seus arquivos por montagem
ou por caminho explícito. Isso evita transferir os resultados a cada rebuild.

O paralelismo de cálculo é separado do orçamento de publicação. Em serviços de
tracking com poucas conexões, defina `BENCH_MLFLOW_PUBLISH_CONCURRENCY=1`; o cliente
mantém todos os pontos e publica os ajustes de cada processo sequencialmente.
`--parallel-cases` continua controlando os casos em execução. O plano registra os
dois limites para que a concorrência das chamadas não sobrecarregue o serviço.

```bash
uv sync --frozen --extra services --extra large-datasets --extra plots --extra dev
uv run --frozen python scripts/prepare_real_sources.py \
  --directory state/inputs --datasets electricity airlines \
  insects_abrupt_imbalanced insects_incremental_abrupt_reoccurring_imbalanced

uv run --frozen python scripts/run_corrective_benchmarks.py \
  --matrix configs/matrices/corrective-smoke.yaml --families matrix \
  --output state/corrective-smoke --data-directory state/inputs --tracking

uv run --frozen python scripts/run_corrective_benchmarks.py \
  --output state/corrective-full --data-directory state/inputs \
  --parallel-cases 2 --seeds 42 43 44 --samples 100 --tracking

uv run --frozen python scripts/export_corrective_evidence.py \
  --state state/corrective-full --output state/corrective-evidence --require-complete

uv run --frozen python scripts/publish_corrective_execution_times.py \
  --state state/corrective-full --output state/corrective-times --tracking
```

Defina `MLFLOW_TRACKING_URI` e a autenticação exigida pelo serviço antes de usar
`--tracking`. O plano registra revisão, digest do código e launcher, versões,
recursos observados, limites, paralelismo, configuração e hashes dos dados. As
entradas são verificadas contra as admissões. O estado de cada unidade só chega a
concluído depois da avaliação e publicação exigidas; uma falha não vira zero na tabela.

A matriz INSECTS desta rodada contém 60 runs: dois fluxos integrais, dois modelos,
cinco políticas e as seeds 42, 43 e 44. A política `Online`, com quatro combinações
de fluxo e modelo e 12 runs ao considerar as três seeds,
foi adiada explicitamente para a próxima rodada. Os quatro datasets do UPF individual
têm três ensaios cada; os pares Electricity/Airlines têm três ensaios por dataset em
cada timing. São 60 runs e 24 ensaios com políticas e grades internas, sem
equivalência a 84 fluxos independentes. A matriz curta também usa as três seeds e
exclui Online.

Todas as linhas dos arquivos admitidos são consumidas. Aquecimento, fase offline,
probes reservados e holdouts têm funções e denominadores distintos; consumir o
arquivo inteiro não significa pontuar exemplos usados para ajuste como se fossem
teste independente. Nas matrizes corretivas, `policy_seed_from_dataset: true`
propaga a seed do caso ao Monte Carlo UPF. Os modelos determinísticos continuam
identificados como tal, sem inventar aleatoriedade entre repetições.

Cada agenda de política e preço no UPF integral recebe os indicadores do
`Core.Metrics`, incluindo F1 macro, AP, AUROC, Brier, log loss, calibração e matriz de
confusão. O cálculo usa os holdouts efetivamente servidos e ocorre depois das
decisões. A acurácia ponderada por exemplo fica em `quality.accuracy`; ela permanece
separada da média por lote em `average_accuracy`. Predições por candidato/lote e
agendas repetidas compartilham caches em memória, sem salvar modelos ou vetores de
probabilidades como checkpoints.

Na AWS, use o mesmo arquivo `configs/matrices/insects-corrected.yaml` em uma solicitação
de matriz binária com release e fontes fixadas, conforme o [runbook](aws/operations.md).
No modo metrics-only, uma interrupção Spot reinicia a avaliação incompleta sem restaurar
pesos. Leases e tentativas permanecem rastreáveis; métricas de progresso de tentativas
distintas não se misturam. O suporte de retenção desta revisão cobre o binário;
avaliadores que não o suportam rejeitam a solicitação explicitamente. Portabilidade
de código e testes de integração não constituem prova de uma execução AWS.

## Evidência da execução

A campanha da revisão `98a6dd5a676f46bf1df337de82b33fd0e554c085` concluiu os 60 casos
INSECTS e os 24 ensaios UPF previstos. O [dossiê da campanha](evidencias/benchmark-corrigido/README.md)
reúne os resultados, a cobertura e os limites das verificações. O plano e os testes
de implementação permanecem separados das evidências de execução e de publicação.
O exportador conserva relatórios e ledgers derivados em gzip, CSVs por run, janela e
repetição, gráficos e manifesto SHA-256. As fontes históricas não foram reescritas.

`manifest.table_files` resolve cada tabela lógica para um arquivo ou para um
descritor `csv+gzip-shards`. As partes desse descritor têm ordem explícita, intervalos
de linhas e hashes; cada uma fica abaixo de 64 MiB comprimidos. Para reconstruir o
CSV, leia as partes na ordem indicada, mantendo o primeiro cabeçalho e omitindo os
cabeçalhos repetidos. O hash do CSV lógico verifica o conteúdo reconstruído.

Curvas ROC/PR e vetores densos continuam completos nos relatórios JSON comprimidos.
`vector-index.csv` localiza cada vetor pelo artefato e por um JSON Pointer RFC 6901,
com quantidade de elementos e hash canônico do array. Esses vetores não são
duplicados linha a linha no CSV de escalares, séries temporais e matrizes.
