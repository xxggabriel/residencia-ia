# Configurações

| Pasta | Conteúdo | Editar? |
|---|---|---|
| `base/` | Bases compartilhadas: `run.yaml` (padrões do motor binário), `smoke.yaml` (caso curto de validação), `tracking.yaml` (métricas da residência) e `tracking-fargate.yaml` (cenário de preço On-Demand) | Sim; afeta todos os arquivos que herdam delas |
| `runs/` | Execuções binárias (`continual-bench run`) | Sim |
| `typed/` | Avaliações typed prequential (`continual-bench typed`) | Sim |
| `campaigns/` | Campanhas declarativas (`continual-bench campaign`) | Sim |
| `matrices/` | Matrizes datasets × modelos × estratégias (`continual-bench matrix`); `matrices/base.yaml` guarda as sementes comuns | Sim |
| `contracts/` | Contratos de runtime: locks de datasets, definições de métricas e catálogo bibliográfico. Entram no fingerprint dos datasets e no wheel | Só com nova versão de contrato |
| `history/` | Manifestos congelados das campanhas executadas (controlled-v3, versioned-v4, nonlinear-v5), com `run_key` e hash do código | Não; são evidência |

Para enviar a matriz de smoke ao SageMaker Studio, copie a solicitação de
[campaign-binary-matrix.json](../infra/examples/campaign-binary-matrix.json), defina um
novo `campaign_id` e ajuste `request.matrix.path` para `configs/matrices/studio-smoke.yaml`.
Os exemplos de solicitação ficam em `infra/examples/`. [Runbook do Studio](../docs/aws/operations.md#campanhas-no-sagemaker-studio).

A [matriz INSECTS histórica](matrices/insects-72.yaml) planejou 72 unidades binárias
incrementais; o recorte arquivado da Semana 10 contém 66 resultados concluídos.
A [matriz piloto](matrices/insects-pilot.yaml) inclui streaming e incremental; o planejador aplica
CARA, UPF e Oracle apenas no incremental (9 casos no total). Veja [aquisição e interpretação](../docs/real-streams.md#benchmark-binário-da-semana-10).

A campanha em andamento usa a [matriz corretiva INSECTS](matrices/insects-corrected.yaml):
duas fontes integrais, dois modelos, cinco políticas e as seeds 42, 43 e 44, totalizando
60 runs. Online ficou para a próxima rodada, com quatro combinações de fonte e modelo
e 12 runs nas três seeds. A [matriz curta](matrices/corrective-smoke.yaml) também exclui
Online e usa as três seeds. Esses arquivos não substituem a matriz nem os resultados
históricos.

O [launcher corretivo](../docs/benchmark-corrections.md) organiza ainda 24 ensaios UPF:
três por dataset nos quatro conjuntos individuais e três por dataset e timing nos
pares Electricity/Airlines. Nas fontes reais, o protocolo integral distribui todo o
ARFF em 15 blocos contíguos, com sete offline, oito online e holdout interno de 20%.
Essa seleção difere das janelas e dos tamanhos de lote das comparações anteriores e
não equivale ao protocolo do artigo. O [registro da campanha corretiva](../docs/evidencias/benchmark-corrigido/README.md)
reunirá o estado observado e os artefatos; o planejamento não comprova sua conclusão.

## Herança com `extends`

Qualquer mapeamento de um arquivo YAML/JSON pode declarar `extends` com um caminho, ou uma lista
de caminhos, relativo ao arquivo que o declara. As bases são aplicadas em ordem e as chaves do
próprio mapeamento sobrescrevem as herdadas:

- mapeamentos são mesclados recursivamente, chave a chave;
- listas e valores escalares substituem o valor herdado por inteiro;
- `null` explícito sobrescreve o valor herdado com nulo.

```yaml
# runs/incremental.yaml
extends: ../base/run.yaml   # dataset, guardas da política e janelas comuns
name: agrawal-periodic
mode: incremental
dataset: {name: agrawal}    # mantém samples/seed/label_delay da base
detector: {name: ddm}
policy: {name: periodic}    # mantém every/cooldown/min_samples/min_window da base
batch_size: 32
```

`extends` também funciona dentro de seções e de itens de listas:

```yaml
tracking:
  extends: ../base/tracking.yaml
  adaptation_shadows: false
cases:
  - {extends: ../base/smoke.yaml, dataset: {name: sea}, policy: {name: none}}
```

Ciclos, arquivos ausentes e `extends` mal formado falham antes da validação. A herança só é
resolvida ao ler arquivos locais (`continual_bench.config.load_document` / `load_model`, usados por
toda a CLI). Os esquemas recusam a chave `extends`, então os payloads HTTP dos workers continuam
autocontidos.

Para ver a configuração resolvida:

```bash
python -c "import json, sys; from continual_bench.config import load_document; print(json.dumps(load_document(sys.argv[1]), indent=2))" configs/runs/streaming.yaml
```

## Identidade

A identidade de uma execução (`Config.identity()`) depende da configuração resolvida.
`tests/test_config_files.py` fixa as identidades de exemplos versionados. Alterações nos valores
efetivamente herdados mudam a identidade dos casos; uma chave sobrescrita pelo filho mantém o valor
já declarado nele. Para compartilhar as evidências de uma campanha histórica, use o manifesto
congelado correspondente.
