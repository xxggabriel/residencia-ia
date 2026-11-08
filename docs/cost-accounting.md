# Contabilização de custos

Os três fluxos passaram a registrar custos independentemente da política. O perfil padrão das novas execuções usa `scenario_unit`, erro de classificação com peso 1, erro absoluto de regressão com peso 1 e escala 1, apresentação de treino com custo 0.001 e atualização com custo 0.05. Os preços por segundo são zero. São estimativas de cenário, sem conversão automática para dinheiro.

No fluxo principal, configurar `tracking.costs`. No fluxo tipado (`TypedConfig` ou `run_typed`) e por épocas (`EpochConfig`), configurar `costs`. Os coeficientes aceitam valores finitos não negativos; `regression_scale` deve ser positiva. Um perfil explícito preserva os padrões anteriores de preço zero para treino e atualização quando esses campos são omitidos; declare todos os coeficientes para comparações controladas.

```yaml
costs: # usar tracking.costs no fluxo principal
  unit: scenario_unit
  false_positive: 1
  false_negative: 1
  classification_error: 1
  regression_error: 1
  regression_scale: 1
  train_sample: 0.001
  update: 0.05
  train_second: 0
  predict_second: 0
```

A classificação binária do fluxo principal usa FP × `false_positive` + FN × `false_negative`. O fluxo tipado e o fluxo por épocas usam a quantidade de erros × `classification_error`; a regressão usa a soma de |alvo − predição| / `regression_scale` × `regression_error`. A escala é fornecida pelo experimento, sem ajuste no teste. Custos entre tarefas exigem escalas e protocolos comparáveis.

O total soma erros, apresentações de treino (incluindo aquecimento e replay), atualizações posteriores ao aquecimento e tempos precificados. No fluxo por épocas, cada passagem é uma atualização e cada repetição conta como apresentação; o relatório contém custos por época e o total de todas as avaliações. O fluxo tipado precifica erros apenas nos exemplos avaliados, incluindo os splits de validação/teste, enquanto seu tempo de predição inclui também o aquecimento. O fluxo principal mantém o escopo das predições avaliadas. Essas diferenças de protocolo devem acompanhar comparações.

O relatório guarda coeficientes, versão, unidade e consumo observado em `costs` (no fluxo principal, em `tracking.costs`). Os escalares do fluxo principal seguem o namespace `tracking_v1/summary/` no MLflow. A publicação por épocas usa `costs/` por época e `costs/total/` no agregado. O fluxo tipado persiste o relatório local por seu contrato existente; não possui publisher MLflow próprio.

Tempos necessários ausentes produzem custo indisponível, não zero. Custos de aquisição, publicação e referências de avaliação ficam fora do total do aprendiz. Execuções concluídas são idempotentes; os perfis integram a identidade das novas execuções. Relatórios históricos não foram recalculados.

Os módulos antes nomeados por semana foram renomeados para `task_sources`, `evaluation_metrics` e `evaluation_reporting`. Configurações, testes, scripts e DAGs usam nomes funcionais; a configuração de avaliação futura passou a `tracking.task_evaluation`. Artefatos e evidências históricos em `reports/` foram preservados.
