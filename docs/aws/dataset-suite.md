# Suíte integral de streams ordenados

A suíte usa o protocolo `ordered-stream-batch-prequential-metrics-v2` para Electricity,
Airlines, as duas variantes binárias INSECTS e os streams Fraud, E-commerce e Social.
Cada política é executada com as seeds 42, 43 e 44. A ordem publicada de um stream real
permanece fixa entre seeds, portanto essas repetições não representam três populações
independentes.

| Stream preparado | Exemplos na fonte integral |
| --- | ---: |
| Electricity | 45.312 |
| Airlines | 539.383 |
| INSECTS abrupt imbalanced | 355.275 |
| INSECTS incremental abrupt reoccurring imbalanced | 452.044 |
| Fraud | 50.000.000 |
| E-commerce | 50.000.000 |
| Social | 100.000.000 |

Todos os exemplos participam das fases definidas pelo protocolo. O treino inicial,
a avaliação prequential e o holdout temporal final têm denominadores próprios. O
protocolo pressupõe rótulos disponíveis ao final do bloco e não substitui o protocolo
binário histórico com atraso simulado. Os tamanhos de bloco, o replay, a calibração,
os modelos e as definições das métricas constam nos relatórios de cada execução.

A comparação principal usa a coorte comum posterior à calibração das políticas.
Social preserva suas quatro classes. AP positiva binária e AP macro multiclasse têm
escopos distintos; a interpretação de Brier também deve considerar a normalização
registrada. Os estimadores de ranking desta suíte usam os mesmos 4.096 bins, sem
misturar seus resultados com curvas exatas dos experimentos históricos.

## Execução em duas fases

A primeira fase contém `frozen`, `drift`, `drift_adwin`, `drift_ddm`, `incremental`,
`periodic` e `cara`: 147 unidades, considerando sete streams e três seeds. Cada
execução CARA produz uma matriz de métricas verificada. Após os três resultados
CARA de um stream serem publicados, a segunda fase pode executar `cara_cumulative`,
`cara_periodic`, `upf` e `oracle`, acrescentando 84 unidades à suíte.

O total é de 231 unidades lógicas. A matriz compartilhada contém medidas e
proveniência, sem pesos ou estado de treino. O relatório distingue a construção
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

## Condições antes da submissão

O comando de coordenação é
[`scripts/aws/run_dataset_campaigns.py`](../../scripts/aws/run_dataset_campaigns.py).
Ele recebe pedidos novos e uma release SHA explícita. Não realiza deploy.

Antes da primeira submissão, o comando verifica:

- QA completo dos 231 smoke tests, incluindo três seeds, métricas reconstruídas,
  integridade das predições, ausência de arquivos de modelo e conferência do catálogo
  de métricas no rastreamento nativo;
- vínculo do hash de código do smoke ao recibo da release e sucesso atual do workflow
  `Build and deploy`, consultado pela CLI do GitHub;
- stack concluída, release publicada e imagem por digest compatíveis;
- job definition Fargate de exatamente 2 vCPU e 4.096 MiB, ambiente Spot válido com
  teto de 64 vCPU e quota regional suficiente;
- uso regional de Fargate. Se houver tarefas externas ao ambiente da suíte, o comando
  interrompe a submissão para permitir a conciliação do orçamento de recursos.

O recibo privado de deploy contém `complete`, `git_sha`, `smoke_code_sha256` e
`workflow`, com `databaseId`, `headSha`, `status` e `conclusion` obtidos da execução
real do CI/CD. `smoke_code_sha256` deve ser o hash de código efetivamente validado
no smoke, após conferir que a árvore implantada contém o mesmo código. Um commit
de merge pode ter SHA diferente da revisão avaliada, sem autorizar a substituição
do código usado no smoke.

O índice dos pedidos, gerado por `run_dataset_suite.py --write-requests`, deve ter
`stage=base`, os sete streams integrais, três seeds e `artifact_mode=metrics_only`.
Os pedidos integrais mantêm 100 épocas máximas e registram a configuração completa.
Um smoke reduzido é uma validação funcional, não uma execução integral antecipada.

```bash
python scripts/aws/run_dataset_campaigns.py start-base \
  --state-dir state/aws-dataset-suite \
  --requests-dir state/dataset-requests \
  --smoke-receipt state/dataset-smoke/suite-qa.json \
  --deployment-receipt state/deployment-receipt.json \
  --release "$RELEASE_SHA" --bucket "$DATA_BUCKET" --region us-east-1
```

O estado local registra exatamente os pedidos e as identidades retornadas. Uma
submissão com resposta incerta é conciliada pelo nome determinístico e pelo conteúdo
do pedido. O comando não cria outro nome para contornar uma identidade existente.

## Acompanhamento e conferência

`status` faz uma consulta, sem iniciar campanhas. `advance` também verifica os
resultados CARA e inicia apenas as segundas fases que já tenham seus três caches
íntegros. O estado local impede que dois coordenadores o alterem simultaneamente.

```bash
python scripts/aws/run_dataset_campaigns.py status \
  --state-dir state/aws-dataset-suite --region us-east-1

python scripts/aws/run_dataset_campaigns.py advance \
  --state-dir state/aws-dataset-suite --region us-east-1
```

Os recibos incluem estados do Step Functions, reservas dos jobs Batch, ARNs das
tasks, tentativas e resultados verificados. A aceitação de cada unidade exige
manifesto final compatível com o plano, relatório com SHA-256 conferido, cobertura
integral, seed correspondente, rastreamento obrigatório concluído e inventário dos
objetos publicados sem pesos ou checkpoints. As listagens são paginadas por completo.

A conclusão da suíte exige 231 unidades verificadas, 14 campanhas concluídas com
resumos completos e nenhuma tarefa ativa da suíte. Um código de saída zero ou um
estado `SUCCEEDED` isolado não comprova essa conclusão. A conferência dos recibos de
publicação não substitui a auditoria independente das séries nativas finais.

As interrupções Spot preservam evidências por tentativa. A avaliação incompleta
reinicia sem restaurar treino; apenas a publicação de métricas já concluídas pode
ser recuperada. As métricas temporais e os custos das tentativas permanecem
separados. Uma estimativa de reserva computacional não é apresentada como
faturamento Spot real.
