# Suíte integral de streams ordenados

A seleção AWS usa o protocolo `ordered-stream-batch-prequential-metrics-v2` para
exatamente quatro streams: `electricity`, `airlines`, `insects_abrupt_imbalanced` e
`insects_incremental_abrupt_reoccurring_imbalanced`.
Cada política é executada com as seeds 42, 43 e 44. A ordem publicada de um stream real
permanece fixa entre seeds, portanto essas repetições não representam três populações
independentes.

| Stream preparado | Exemplos na fonte integral |
| --- | ---: |
| Electricity | 45.312 |
| Airlines | 539.383 |
| INSECTS abrupt imbalanced | 355.275 |
| INSECTS incremental abrupt reoccurring imbalanced | 452.044 |
| Total das quatro fontes | 1.392.014 |

Fraud, E-commerce e Social permanecem no suporte local e nas evidências históricas,
fora desta seleção AWS. O [guia geral da suíte](../dataset-suite.md) descreve esse
suporte e o preflight local ampliado.

Todos os exemplos participam das fases definidas pelo protocolo. O treino inicial,
a avaliação prequential e o holdout temporal final têm denominadores próprios. O
protocolo pressupõe rótulos disponíveis ao final do bloco e não substitui o protocolo
binário histórico com atraso simulado. Os tamanhos de bloco, o replay, a calibração,
os modelos e as definições das métricas constam nos relatórios de cada execução.

A comparação principal usa a coorte comum posterior à calibração das políticas,
com uma única identidade de fonte por stream em todas as políticas e seeds. As
quatro tarefas são binárias. Os estimadores de ranking usam os mesmos 4.096 bins,
sem misturar seus resultados com curvas exatas dos experimentos históricos. A
interpretação de Brier considera a normalização registrada no contrato.

## Execução em duas fases

A primeira fase contém `frozen`, `drift`, `drift_adwin`, `drift_ddm`, `incremental`,
`periodic` e `cara`: 84 unidades, considerando quatro streams e três seeds. Cada
execução CARA produz uma matriz de métricas verificada. Após os três resultados
CARA de um stream serem publicados, a segunda fase pode executar `cara_cumulative`,
`cara_periodic`, `upf` e `oracle`, acrescentando 48 unidades à suíte.

O total é de 132 unidades lógicas em oito campanhas, uma base e uma cached por
stream. Cada unidade declara até 100 épocas e `artifact_mode=metrics_only`. A
matriz compartilhada contém medidas e proveniência, sem pesos ou estado de treino.
O relatório distingue a construção
original da matriz, seu reúso e os ajustes adicionais necessários para materializar
modelos em memória. Isso evita atribuir a cada política o mesmo trabalho físico de
construção. Oracle permanece uma referência retrospectiva do objetivo formal da
política, sem representar um teto universal para F1, AP ou AUROC.

`online` não integra os pedidos AWS. A reserva é de 2 vCPU e 4.096 MiB por tarefa
Fargate Spot, com um único ambiente Batch limitado a 64 vCPU. Até 32 tarefas podem
ocupar essa reserva simultaneamente. As demais aguardam na fila, e a capacidade
Spot efetivamente disponível pode reduzir o paralelismo observado.

Cada tarefa usa um worker de publicação de métricas e um worker de logging
assíncrono. A concorrência de treino e a de publicação são limites diferentes:
esse ajuste reduz a pressão simultânea sobre o rastreador sem remover pontos das
séries. Os valores efetivos acompanham os metadados do ambiente.

Os pontos de cada ajuste são gerados conforme há vagas de publicação. A fila de
payloads submetidos fica limitada à concorrência configurada, além do espaço
temporário de construção do gerador. Com um worker, o envio de um ajuste termina
antes de começar a geração do próximo. O histórico científico e as amostras de
sistema continuam integrais. Se um ajuste posterior for inválido, filhos anteriores
podem estar concluídos; a repetição reutiliza esses registros sem duplicar pontos.

O recibo de artefatos soma os arquivos e bytes enviados ao run pai e aos child
runs. As parcelas permanecem separadas, com `fit_id`, `run_id` e URI para cada
filho. A publicação confere o número de arquivos e os bytes previstos por ajuste;
um diretório ausente, um recibo incompleto ou uma falha de upload impede a
conclusão. O guard de publicação usa esse total, preservando o vínculo dos filhos
com o histórico de treino. A implementação está no
[`publisher MLflow`](../../src/continual_bench/infrastructure/aws/mlflow_tracking.py).

## Condições antes da submissão

O comando de coordenação é
[`scripts/aws/run_dataset_campaigns.py`](../../scripts/aws/run_dataset_campaigns.py).
Ele recebe pedidos novos e uma release SHA explícita. Não realiza deploy.

Antes da primeira submissão, o comando verifica:

- QA completo da matriz declarada, incluindo três seeds, métricas reconstruídas,
  integridade das predições, ausência de arquivos de modelo e conferência do catálogo
  de métricas e dos marcadores de publicação no rastreamento nativo. O QA local pode
  abranger os sete streams e seus 231 casos, desde que todos estejam completos e
  os quatro streams selecionados estejam incluídos. O comando verifica cada
  coordenada dataset/política/seed e seu recibo nativo, além dos totais;
- vínculo do hash de código do smoke ao recibo da release e sucesso atual do workflow
  `Build and deploy`, consultado pela CLI do GitHub;
- stack concluída, release publicada e imagem por digest compatíveis;
- job definition Fargate de exatamente 2 vCPU e 4.096 MiB, ambiente Spot válido com
  teto de 64 vCPU e quota regional suficiente;
- regras reais de cancelamento da fila em `RUNNABLE` com espera mínima de 86.400
  segundos para as razões de capacidade e configuração previstas no template. O
  recibo registra as regras retornadas pela AWS, e `advance` as confere novamente
  antes de iniciar a segunda fase;
- uso regional de Fargate. Se houver tarefas externas ao ambiente da suíte, o comando
  interrompe a submissão para permitir a conciliação do orçamento de recursos.

O recibo privado de deploy contém `complete`, `git_sha`, `smoke_code_sha256` e
`workflow`, com `databaseId`, `headSha`, `status` e `conclusion` obtidos da execução
real do CI/CD. `smoke_code_sha256` deve ser o hash de código efetivamente validado
no smoke, após conferir que a árvore implantada contém o mesmo código. Um commit
de merge pode ter SHA diferente da revisão avaliada, sem autorizar a substituição
do código usado no smoke.

O índice dos pedidos, gerado por `run_dataset_suite.py --write-requests`, deve ter
`stage=base`, exatamente os quatro streams integrais, três seeds e
`artifact_mode=metrics_only`. Cada pedido declara `dataset` explicitamente e
`matrix.kind=stream`. Pedidos Zia são rejeitados antes da submissão.
Os pedidos integrais mantêm 100 épocas máximas e registram a configuração completa.
Um smoke reduzido é uma validação funcional, não uma execução integral antecipada.

```bash
python scripts/run_dataset_suite.py \
  --root /inputs/prepared --output state/four-stream-full-plan \
  --datasets electricity airlines insects_abrupt_imbalanced \
    insects_incremental_abrupt_reoccurring_imbalanced \
  --scope full --max-epochs 100 --tracking --plan-only \
  --write-requests state/dataset-requests-four-streams --request-stage base \
  --release "$RELEASE_SHA"

python scripts/aws/run_dataset_campaigns.py start-base \
  --state-dir state/aws-four-stream-suite \
  --requests-dir state/dataset-requests-four-streams \
  --smoke-receipt state/dataset-smoke/suite-qa.json \
  --deployment-receipt state/deployment-receipt.json \
  --release "$RELEASE_SHA" --bucket "$DATA_BUCKET" --region us-east-1
```

O estado local registra exatamente os pedidos e as identidades retornadas. Uma
submissão com resposta incerta é conciliada pelo nome determinístico e pelo conteúdo
do pedido. O comando não cria outro nome para contornar uma identidade existente.
O contrato `contract.requests` é obrigatório, contém os quatro pedidos e precisa
coincidir com os pedidos base registrados. Estados antigos de sete streams são
rejeitados sem alteração; a seleção atual exige outro diretório de estado.

O contrato planejado contém os quatro pedidos desde o início. Os registros de
campanhas podem representar uma submissão parcial, inclusive antes da primeira
resposta bem-sucedida. Nesse caso, `status` consulta as campanhas já registradas e
seus jobs, mantendo os denominadores de 132 unidades e oito campanhas e
`complete=false`. Essa consulta não submete campanhas. Repetir `start-base` com o
mesmo contrato retoma as bases ausentes, usando os nomes determinísticos para
conciliar respostas incertas. `advance` continua limitado às fases cached de bases
registradas com três caches CARA verificados.

Uma resposta de submissão sem novo job identificado consome o orçamento global
`NoJob` da unidade. Após as quatro primeiras falhas, o workflow espera 30, 60, 120 e
240 segundos antes de conciliar novamente e tentar reenviar. A quinta falha encerra
a unidade. Esse contador não é zerado quando um job real aparece. O limite
`max_attempts` conta as tentativas com jobs reais e permanece independente desse
orçamento de submissão. A regra está no
[`workflow Step Functions`](../../infra/statemachines/benchmark.asl.json), com
[testes dos dois processadores](../../tests/test_aws_submission_retry_workflow.py).

## Acompanhamento e conferência

`status` faz uma consulta, sem iniciar campanhas. `advance` também verifica os
resultados CARA e inicia apenas as segundas fases que já tenham seus três caches
íntegros. O estado local impede que dois coordenadores o alterem simultaneamente.

```bash
python scripts/aws/run_dataset_campaigns.py status \
  --state-dir state/aws-four-stream-suite --region us-east-1

python scripts/aws/run_dataset_campaigns.py advance \
  --state-dir state/aws-four-stream-suite --region us-east-1
```

Os recibos incluem estados do Step Functions, reservas dos jobs Batch, ARNs das
tasks, tentativas e resultados verificados. A aceitação de cada unidade exige
manifesto final compatível com o plano, relatório com SHA-256 conferido, cobertura
integral, seed correspondente, rastreamento obrigatório concluído e inventário dos
objetos publicados sem pesos ou checkpoints. As listagens são paginadas por completo.

No progresso ao vivo, o ledger guarda o instante da observação, e `progress` atribui
ao ponto nativo o instante de sua criação. A conferência preserva ambos os tempos e
verifica as coordenadas e os valores. Um ponto ausente pode ser recuperado com o
timestamp do ledger; pontos já recebidos conservam o timestamp nativo, inclusive
nas repetições. A cronologia científica permanece vinculada à sequência dos
exemplos, às janelas e aos passos de cada ajuste.

A conclusão da suíte exige 132 unidades verificadas, oito campanhas concluídas com
resumos completos e nenhuma tarefa ativa da suíte. Um código de saída zero ou um
estado `SUCCEEDED` isolado não comprova essa conclusão. A conferência dos recibos de
publicação não substitui a auditoria independente das séries nativas finais.

As interrupções Spot preservam evidências por tentativa. A avaliação incompleta
reinicia sem restaurar treino; apenas a publicação de métricas já concluídas pode
ser recuperada, sob o mesmo contrato, release e imagem do plano original. As métricas
temporais e os custos das tentativas permanecem
separados. Uma estimativa de reserva computacional não é apresentada como
faturamento Spot real.

## Exportação das evidências após a campanha

O exportador
[`scripts/aws/export_dataset_evidence.py`](../../scripts/aws/export_dataset_evidence.py)
consome o estado do coordenador e exige a conclusão integral registrada, com todas
as unidades verificadas e nenhuma tarefa ativa. O comando faz somente consultas
S3 de listagem, metadados e leitura. Não inicia jobs, não altera objetos publicados
e não remove dados remotos.

```bash
python scripts/aws/export_dataset_evidence.py \
  --state-dir state/aws-four-stream-suite \
  --output docs/evidencias/aws-four-stream-suite \
  --scratch-root state/aws-evidence-scratch \
  --scratch-gib 20 --max-file-mib 64
```

Esse é um comando de pós-processamento. Sua presença no repositório não comprova
que uma campanha ou uma exportação já tenha sido executada.

Primeiro são lidos os planos e os manifestos finais, que precisam coincidir com as
identidades já verificadas pelo coordenador. Os relatórios completos são baixados
e conferidos por SHA-256. A coorte principal é determinada depois de examinar
todas as políticas e seeds de cada fonte: seu início sucede a maior fase de
calibração e seu fim antecede o holdout comum. O exportador exige os quatro streams,
as oito campanhas e a matriz completa de 132 coordenadas, com os três comprovantes
de cache CARA por stream e uma única identidade de fonte em cada coorte.

As predições são tratadas um caso de cada vez. O exportador confere seus manifestos,
fixa a versão S3 ou condiciona a leitura ao ETag, valida os bytes e o SHA-256 de
cada parte e usa o verificador da suíte para reconstruir as métricas por janela,
partição e coorte comum. O histórico de fits publicado também precisa coincidir
com o histórico integral do relatório.

O scratch tem um limite explícito de 20 GiB no comando acima. O espaço de um caso
é conferido antes dos downloads, incluindo a soma de todas as partes de predição.
Se esse orçamento não comportar o caso, a exportação falha antes de baixar suas
predições. Ao terminar a auditoria, apenas a cópia temporária local é removida.
As predições e as fontes originais continuam no S3, com suas URIs registradas.

Uma interrupção da exportação não obriga a baixar novamente os casos já auditados.
O recibo local de cada caso sela os hashes de seu QA, de sua proveniência e dos
arquivos comprimidos. A retomada exige os mesmos contratos, coortes e objetos
publicados; alterações inesperadas bloqueiam o reúso. Essa recuperação usa somente
evidências concluídas, sem restaurar estado de treino.

O pacote contém os relatórios completos comprimidos, os contratos de métricas,
históricos de treino, índices de artefatos, o catálogo `tracking_v1` derivado e os
ledgers de progresso quando presentes e compatíveis com o formato de métricas.
Os arquivos de relatório não são reduzidos a um resumo de escalares: vetores,
curvas, decisões, tipos JSON e valores `null` permanecem disponíveis. Arquivos
maiores são divididos em faixas de bytes e comprimidos sem perda. O manifesto
informa a ordem e os hashes necessários para recompor os bytes originais.

As provas de cada caso distinguem objetos efetivamente baixados e conferidos de
objetos apenas listados no inventário. Os primeiros registram URI, SHA-256,
tamanho, ETag e VersionId quando disponível. Uma listagem não é apresentada como
verificação dos bytes ou das versões dos objetos que não foram lidos. Dados brutos
e pesos não são incorporados ao pacote.

As tabelas principais mantêm a coorte pareada por stream, política e seed. As
tabelas dos demais escalares conservam os escopos prequential integral e holdout,
sem misturá-los com a comparação principal. Médias e desvios padrão amostrais
usam `ddof=1` e indicam `n_defined` e `n_undefined`. Uma métrica sem dois valores
definidos não recebe um desvio padrão fabricado. Os contratos completos preservam
a distinção entre custos lógicos dos modelos promovidos, trabalho físico,
construção da matriz, reúso de métricas e reconstrução de modelos em memória.

O manifesto e o arquivo `SHA256SUMS` descrevem o pacote exportado. Sua conclusão
comprova o escopo de reconstrução e de preservação das evidências, não uma leitura
independente das APIs do MLflow. Essa auditoria nativa deve ser vinculada como
prova separada, incluindo os child runs e as séries de observações pertinentes.
