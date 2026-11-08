# Contabilização de custos

Os três fluxos passaram a registrar custos independentemente da política. O perfil padrão das novas execuções usa `scenario_unit`, erro de classificação com peso 1, erro absoluto de regressão com peso 1 e escala 1, apresentação de treino com custo 0.001 e atualização com custo 0.05. Os preços por segundo são zero. Esses pesos continuam sendo estimativas de cenário. Novas execuções também registram uma estimativa independente de infraestrutura AWS em USD, descrita abaixo.

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

## Estimativa de infraestrutura AWS

`costs.aws` (ou `tracking.costs.aws`) configura a estimativa habilitada por padrão. Use `aws: null` para desabilitá-la. O resultado fica em `costs.aws`, com `settings`, `consumption`, `metrics`, versão e exclusões. `cost_total_usd` soma apenas CPU, RAM e armazenamento; não é somado a `cost_total`, que continua expresso na unidade do cenário. Erros e eventos não recebem um valor monetário implícito.

```yaml
costs: # tracking.costs no fluxo principal
  aws:
    region: us-east-1
    price_date: "2026-11-08"
    vcpus: 1
    memory_gb: 2
    vcpu_hour_usd: 0.04048
    memory_gb_hour_usd: 0.004445
    storage_gb_month_usd: 0.023
    retention_days: 30
    billing_month_days: 30
    minimum_compute_seconds: 60
```

As tarifas foram verificadas em 08/11/2026 nos catálogos públicos da AWS: [Fargate Linux/x86 On-Demand em us-east-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonECS/current/us-east-1/index.json), publicado em 19/10/2026, e [S3 Standard em us-east-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-east-1/index.json), publicado em 04/11/2026. Armazenamento usa a primeira faixa de 50 TB/mês. Os preços ficam fixados na configuração, sem consulta remota durante a execução. Outra região exige informar os três preços, fontes e data explicitamente. Atualizações de tarifa ou alocação alteram a identidade da execução.

Conforme as [regras do Fargate](https://aws.amazon.com/fargate/pricing/), CPU e RAM são recursos **reservados**, não utilização observada. A capacidade configurada é uma hipótese de execução em nuvem, não uma detecção do hardware local. Combinações de CPU/RAM precisam ser aceitas pelo Fargate. A duração estimada é a soma dos tempos medidos de treino (incluindo aquecimento/replay) e predição de cada protocolo. Aplica-se arredondamento para cima por segundo e mínimo de 60 segundos uma única vez por execução:

- `billed_seconds = max(minimum_compute_seconds, ceil(train_seconds + prediction_seconds))`
- `cost_cpu_usd = vcpus × billed_seconds / 3600 × vcpu_hour_usd`
- `cost_ram_usd = memory_gb × billed_seconds / 3600 × memory_gb_hour_usd`
- `cost_model_storage_usd = model_storage_bytes / 1024³ × retention_days / billing_month_days × storage_gb_month_usd`

O fluxo principal mede os payloads efetivamente salvos em `models.bin` (todas as versões, inclusive candidatos retidos) e `model.pkl` (cópia final). Ambas as cópias ocupam espaço e entram no consumo, sem deduplicação. O prazo de retenção é uma projeção uniforme para todos os arquivos após a execução; não representa sua idade já transcorrida. O mês de 30 dias é uma hipótese configurável. Os fluxos tipado e por épocas não salvam modelos: registram zero bytes armazenados, sem confundir o tamanho do objeto em RAM com armazenamento persistido. Por épocas, o cálculo AWS fica somente no agregado para não repetir o mínimo de cobrança.

Tempos ou tamanho de modelo desconhecidos deixam o respectivo componente e o total AWS indisponíveis (`null`), inclusive em relatórios antigos sem evidência de tamanho. Os relatórios históricos não são recalculados. A estimativa não representa uma fatura: exclui inicialização do container, ociosidade, aquisição de dados, referências de avaliação, publicação, serviços de tracking, checkpoints, metadados, requisições S3, rede, impostos, descontos e free tier. O runtime AWS real também pode ter desempenho diferente do hardware local.

Os três fluxos também geram o gráfico `costs/aws_usd` com os componentes em USD e os dados correspondentes em JSON. No MLflow, o fluxo principal publica `tracking_v1/summary/aws_cost_*_usd`; por épocas, publica `costs/aws/cost_*_usd` no agregado. O fluxo tipado conserva sua publicação local.

## Validação com serviços reais — 08/11/2026

Foi executada uma única run no homelab `192.168.100.27`, em cliente transitório com código/estado isolados, reutilizando os serviços existentes. Não foi iniciada uma campanha nem repetido o benchmark. A run processou 180 amostras SEA em streaming, com regressão logística, ADWIN e política periódica (cinco atualizações). O ZenML concluiu as quatro etapas; o MLflow finalizou a run `eba563d87b8644bfad6c3c76896c60eb` no experimento `4`.

A leitura autenticada do MLflow confirmou os quatro escalares `tracking_v1/summary/aws_cost_*_usd`, cada um com um único ponto no passo zero e valores iguais ao cálculo independente. CPU: USD 0.0006746666666666667; RAM: USD 0.00014816666666666667; modelos: USD 0.00000014829356223344801; total: USD 0.0008229816268955669. O aprendiz consumiu aproximadamente 0.008061 s; a hipótese Fargate aplicou o mínimo de 60 s. Foram contabilizados 6923 bytes de modelos, com retenção projetada de 30 dias.

O relatório, o gráfico `plots/costs/aws_usd.svg` e os dados `data/costs/aws_usd.json` foram baixados do MLflow e comparados por SHA-256 com os arquivos locais. Quatro objetos do MinIO também coincidiram por hash; a versão de modelo registrada apontou para a mesma run. IDs, valores, consumo e hashes foram preservados no arquivo local `reports/aws-costs-service-20261108/evidence.json` (`reports/` é ignorado pelo Git). [Abrir run no MLflow](http://192.168.100.27:15000/#/experiments/4/runs/eba563d87b8644bfad6c3c76896c60eb).

A primeira chamada do teste completou a pipeline e todas as verificações de custos/artefatos, mas falhou ao exigir 180 mensagens ainda retidas no Kafka: os offsets observados foram low=180/high=180. O produtor usa timestamps sintéticos antigos e o tópico configura retenção de 24 horas; o teste passou a verificar o high watermark de 180 eventos publicados e a registrar separadamente os eventos ainda retidos. A verificação foi repetida **somente em leitura**, sobre a mesma run, sem chamar `execute_pipeline` ou republicar: `1 passed in 14.74s`.

Os dois pontos de entrada ficaram em `tests/test_aws_cost_services.py`: `test_single_run_publishes_aws_costs_to_mlflow` executa uma nova run e confere a publicação; `test_existing_run_aws_cost_readback` exige `SERVICE_TEST_RUN_KEY` e somente confere uma run existente; sem essa seleção explícita, ele é ignorado pelo pytest. Não executar o módulo inteiro quando a intenção for estritamente uma run. A execução observada utilizou, no cliente isolado do homelab:

```bash
python -m pytest -q -s -p no:cacheprovider -o addopts= -m services \
  tests/test_aws_cost_services.py::test_single_run_publishes_aws_costs_to_mlflow
# Após ajustar apenas a verificação Kafka; SERVICE_TEST_RUN_KEY=c9e2cfe994de4b5f895ea493:
python -m pytest -q -s -p no:cacheprovider -o addopts= -m services \
  tests/test_aws_cost_services.py::test_existing_run_aws_cost_readback
```

O snapshot/estado e os logs originais foram mantidos em `/home/gabriel/aws-cost-single-20261108.2uaGy6/`. Essa validação conferiu a integração com os serviços reais do homelab; não envolveu consumo ou faturamento na AWS.
