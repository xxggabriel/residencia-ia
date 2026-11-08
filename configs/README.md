# Configurações

| Pasta | Conteúdo | Editar? |
|---|---|---|
| `base/` | Bases compartilhadas: `run.yaml` (padrões do motor binário), `smoke.yaml` (caso curto de validação) e `tracking.yaml` (perfil de métricas da residência) | Sim; afeta todos os arquivos que herdam delas |
| `runs/` | Execuções binárias (`continual-bench run`) | Sim |
| `typed/` | Avaliações typed prequential (`continual-bench typed`) | Sim |
| `campaigns/` | Campanhas declarativas (`continual-bench campaign`) | Sim |
| `matrices/` | Matrizes datasets × modelos × estratégias (`continual-bench matrix`); `matrices/base.yaml` guarda as sementes comuns | Sim |
| `contracts/` | Contratos de runtime: locks de datasets, definições de métricas e catálogo bibliográfico. Entram no fingerprint dos datasets e no wheel | Só com nova versão de contrato |
| `history/` | Manifestos congelados das campanhas executadas (controlled-v3, versioned-v4, nonlinear-v5), com `run_key` e hash do código | Não; são evidência |

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

A identidade de uma execução (`Config.identity()`) depende só da configuração resolvida, não da
forma como ela foi escrita. A reorganização preservou as identidades de todos os arquivos
anteriores, e `tests/test_config_files.py` fixa as identidades dos exemplos versionados. Alterar
uma base muda a identidade de todos os herdeiros: esse teste falha e mostra quais arquivos foram
afetados.
