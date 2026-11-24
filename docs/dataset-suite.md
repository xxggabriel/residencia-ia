# Suíte de streams ordenados

A suíte executa um protocolo comum de teste antes do treino por bloco em Electricity,
Airlines, duas variantes binárias de INSECTS, Fraud, E-commerce e Social. Trata-se de
uma campanha nova. Seus resultados não substituem os registros das campanhas
anteriores, que utilizavam outros modelos, atrasos de rótulos e estimadores de
ranqueamento.

## Dados e protocolo

As entradas preparadas têm manifesto, contagem de linhas e SHA-256. O escopo `full`
consome a fonte completa admitida. O escopo `smoke` utiliza o prefixo limitado
declarado pelo provider e serve para verificar o funcionamento do caminho de
avaliação e publicação. Um smoke com três épocas não constitui evidência de que o
treino com orçamento de cem épocas terminou ou terá os mesmos resultados.

Os nomes canônicos são `electricity`, `airlines`, `insects_abrupt_imbalanced`,
`insects_incremental_abrupt_reoccurring_imbalanced`, `fraud`, `ecommerce` e `social`.
As variantes INSECTS preservam a tarefa binária já definida por seu provider.
Social mantém suas quatro classes. Não se transforma esse problema em uma tarefa
binária para forçar uma comparação.

O primeiro bloco é usado para aquecimento. Os blocos seguintes são previstos antes
que seus rótulos sejam entregues ao treino e à decisão de atualização. Os rótulos
são considerados disponíveis ao final de cada bloco. Essa convenção não representa
um atraso de rotulagem medido nos dados de origem. O holdout temporal final cobre
aproximadamente 10% dos exemplos e sua fronteira preserva empates de timestamp.
Nenhum exemplo do holdout participa do treino, da calibração das políticas ou da
escolha de atualizações.

Todos os exemplos participam de alguma fase: aquecimento, avaliação prequential ou
holdout. Isso não significa que os exemplos do aquecimento também sejam exemplos
de teste. A verificação exige que a soma dessas populações corresponda à contagem
inteira do manifesto, incluindo o último bloco parcial.

Valores ausentes em features não autorizam descartar exemplos. O contrato do
modelo e a quantidade de ausências devem permanecer registrados. O XGBoost
consome os valores ausentes nativamente; as amostras geométricas de CARA usam uma
referência de preenchimento obtida apenas no aquecimento. Nenhum preenchimento
usa estatísticas futuras. Independentemente dessas ausências de entrada, todas
as probabilidades publicadas devem ser finitas e somar um dentro da tolerância
numérica verificada.

O tamanho dos blocos depende do provider e permanece no contrato de cada execução.
Os streams maiores usam blocos limitados e uma janela de replay de quatro blocos
para respeitar a memória disponível. Por isso, comparar números brutos de
atualizações ou duração entre datasets não equivale a comparar decisões em um
mesmo orçamento de exemplos. As tabelas mantêm esses denominadores.

## Políticas, seeds e métricas

Cada stream tem três repetições computacionais com seeds 42, 43 e 44. A seed entra
na identidade do caso, na configuração de treino, nos metadados de cada fit e no
gerador usado por UPF. As repetições preservam a mesma ordem de um dataset real;
não representam três populações independentes.

A matriz AWS contém `frozen`, `drift`, `drift_adwin`, `drift_ddm`, `incremental`,
`periodic`, `cara`, `cara_cumulative`, `cara_periodic`, `upf` e `oracle`. A política
`online` possui uma rodada separada. As duas variantes INSECTS são executadas
individualmente. Assim, sete streams, onze políticas e três seeds totalizam 231
casos na matriz AWS.

O perfil `configs/base/tracking-dataset-suite.yaml` utiliza custos de erro de
classificação iguais a um. A regra binária é `p_1 >= 0.5`. Os pesos de classe do
modelo, quando definidos pelo provider, são parâmetros de treinamento separados
da regra de decisão e permanecem registrados. Os preços do perfil são uma
referência declarada de Fargate On-Demand, não uma medição de faturamento Spot.

As predições individuais preservam probabilidades, classe observada, decisão,
posição na fonte, partição e versão lógica do modelo que as produziu. O avaliador
registra métricas de classificação, calibração, ranqueamento, treino, inferência,
recursos, decisões e custos. Séries, dados dos gráficos e motivos de
indisponibilidade são mantidos. Um valor indefinido continua `null`; não recebe
zero para completar tabelas.

AP e AUROC usam histogramas uniformes de 4.096 bins. As tabelas não os identificam
como áreas exatas de curvas ordenadas por score. Em problemas binários, `pr_auc`
descreve a classe de índice um. `macro_pr_auc` e `macro_roc_auc` agregam as medidas
one-vs-rest definidas das classes. A comparação exporta também `brier_half_sum`,
igual à soma dos erros quadráticos de todas as probabilidades dividida por `2*n`,
para explicitar uma escala de 0 a 1 comum. A quantidade de classes e a distribuição
de rótulos continuam diferentes entre os datasets.

As comparações principais usam a mesma faixa de posições da fonte em todas as
políticas de um stream. Seu início fica após a maior fase de calibração offline;
seu fim antecede o holdout. O restante das predições continua disponível como
evidência e diagnóstico. Médias e desvios padrão amostrais são calculados por
dataset, política e métrica, com indicação do número de seeds presentes e do
número de valores definidos. Uma única observação definida não recebe desvio
padrão inventado.

O Oracle é uma referência retrospectiva para seu objetivo formal, com acesso à
matriz futura de erros. Ele não constitui um teto universal de F1, AP ou AUROC.
Comparações entre datasets são descritivas. A avaliação do efeito de uma política
deve priorizar os contrastes pareados dentro do mesmo stream e da mesma seed.

## Smoke e conferência

As entradas devem estar previamente preparadas em um diretório com os pares
`<dataset>/<scope>.parquet` e `<dataset>/<scope>.json`. O runner não baixa dados,
altera snapshots nem escolhe silenciosamente uma amostra substituta.

```bash
uv run python scripts/run_dataset_suite.py \
  --root /inputs/prepared --output /evidence/suite-smoke \
  --scope smoke --max-epochs 3 --workers 1 --tracking

uv run python scripts/verify_dataset_suite.py \
  --output /evidence/suite-smoke --native-tracking
```

O processo que fornece `/inputs` e `/evidence` deve reservar os recursos do
container. `--workers` controla o paralelismo interno, não cria uma reserva de CPU
ou RAM. Para conferir um controle com o orçamento de treinamento final, use outro
diretório de evidências e `--max-epochs 100` no escopo `smoke`.

O plano é imutável. Cada caso é um subprocesso separado e seu recibo registra os
hashes de todos os arquivos produzidos. Uma tentativa incompleta reinicia a
avaliação; não retoma treino por checkpoint. Uma evidência concluída só é reutilizada
se sua identidade e seus hashes permanecerem iguais aos do plano.

A primeira política CARA de cada stream e seed produz um cache imutável da matriz
de métricas. As demais políticas de custo podem reutilizar esse cache depois da
validação de seu contrato científico. O cache contém métricas e proveniência, não
pesos, modelos serializados ou estado de otimizador. A reconstrução de modelos para
implantação é registrada como trabalho adicional e não é confundida com o custo
lógico da atualização simulada.

O verificador reconstrói as métricas por janela e partição a partir das predições,
confere sequência, checksums, limites das classes, soma das probabilidades e regra
de decisão. Também confere seeds por fit e ausência de arquivos de modelos. Com
`--native-tracking`, lê os históricos completos de todos os pontos finitos do
catálogo `tracking_v1` do run pai no MLflow, preservando no artefato os valores
indefinidos que não são números válidos para publicação nativa.

Esse readback tem escopo declarado. Ele não substitui a conferência independente
dos child runs de treino nem de outras famílias de observações publicadas fora do
catálogo do pai. A publicação dessas famílias conserva seus próprios recibos e
contagens.

## Pedidos AWS

O runner apenas escreve pedidos. Ele não faz deploy nem inicia campanhas:

```bash
uv run python scripts/run_dataset_suite.py \
  --root /inputs/prepared --output /evidence/suite-full-plan \
  --scope full --max-epochs 100 --tracking --plan-only \
  --write-requests /evidence/aws-requests-base --request-stage base \
  --release "$RELEASE_SHA"
```

O SHA precisa identificar uma release publicada. A primeira fase inclui as seis
políticas de referência e CARA, com três seeds por stream. A segunda fase contém
`cara_cumulative`, `cara_periodic`, `upf` e `oracle`, com referências verificadas
aos caches de métricas produzidos pela primeira fase. Os pedidos conservam 32
como limite de concorrência, fila de até 24 horas e limite explícito por tentativa.
A capacidade efetiva depende da quota, dos demais consumidores e da oferta Spot.

O conjunto de pedidos da segunda fase pode ser escrito com `--request-stage cached`
e `--metric-cache-sources <arquivo.json>`. Esse arquivo mapeia cada nome de dataset
para as três chaves de seed, cada uma contendo a URI imutável e o SHA-256 real de
seu cache. O script exige os três comprovantes e não inventa URI ou hash ausente.
