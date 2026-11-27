# Testes e CI/CD

Esta página descreve como a suíte de testes está organizada, como executá-la localmente, quais marcadores do pytest separam testes rápidos de testes que exigem serviços, e o que os workflows do GitHub Actions (`ci.yml` e `deploy.yml`) e o hook de Git `pre-push` executam.

## Organização da suíte

Os testes ficam em `tests/`, em arquivos planos `test_<assunto>.py`, sem subpastas. Os prefixos indicam a área: `test_aws_*` (perfil AWS com dublês, sem chamar a AWS), `test_zia_*`, `test_tracking_*`, `test_task_*`, `test_epoch_*`, `test_binary_*`, entre outros. A configuração do pytest está em `pyproject.toml`:

- `testpaths = ["tests"]` e `pythonpath = ["."]`, o que torna `examples/` e `tests/` importáveis durante os testes;
- `addopts = "-m 'not services and not benchmark'"`, aplicado a toda execução do pytest que não informe outro `-m`.

Na revisão atual, `uv run --frozen pytest --collect-only -q` coleta 1804 testes, dos quais 50 ficam fora da seleção padrão por serem marcados `services`. O número muda conforme o código evolui.

Alguns testes estruturais merecem destaque:

| Teste | O que garante |
|---|---|
| `tests/test_architecture.py` | Regra de dependência entre camadas e ausência de bibliotecas de terceiros no `core` (ver [Arquitetura](Arquitetura#camadas-e-regra-de-dependência)) |
| `tests/test_config_files.py` | Herança `extends`, identidades fixas das configurações versionadas e que toda configuração de entrada resolve |
| `tests/test_benchmark_dags.py` | Quantidade e unicidade de casos planejados pelas DAGs, inclusive 261 (v3), 289 (v4) e 867 (v5) configurações |
| `tests/test_application.py` | Retomada por ponto de recuperação igual à execução ininterrupta; corrupção, *lock* e manifesto divergente falham fechado |

## Marcadores

| Marcador | Significado (de `pyproject.toml`) | Situação na suíte |
|---|---|---|
| `services` | Exige Kafka, MinIO, MLflow e ZenML reais | Usado em `test_services.py`, `test_epoch_services.py`, `test_task_services.py`, `test_nonlinear_services.py`, `test_typed_cost_services.py`, `test_aws_cost_services.py`; excluído por padrão |
| `benchmark` | Experimento completo, fora dos testes rápidos | Declarado e excluído por padrão, mas nenhum teste o usa atualmente |
| `zia_extended_seeds` | Sementes adicionais (43 e 44) dos testes de integração Zia | Parâmetros em `test_zia_metrics_only.py` e `test_stream_holdout_invariance.py`; rodam localmente, excluídos no CI |
| `observation_pngs` | Renderiza os PNGs de observação | `test_training_figures.py` e um caso de `test_generic_cloud_tracking.py`; rodam no padrão e no CI |

`tests/conftest.py` tem uma *fixture* automática que desliga a renderização Matplotlib de PNGs em todos os testes sem os marcadores `observation_pngs`, `services` ou `benchmark`. Os testes rápidos continuam gerando os dados, manifestos e HTML das observações, apenas sem as imagens.

## Como executar

| Objetivo | Comando |
|---|---|
| Suíte rápida em paralelo (como no hook) | `make test` |
| Um arquivo | `uv run --frozen pytest -q tests/test_config_files.py` |
| Igual ao CI, sem paralelismo | `uv run --frozen pytest -q -m 'not services and not benchmark and not zia_extended_seeds'` |
| Somente sementes Zia adicionais | `uv run --frozen pytest -m zia_extended_seeds` |
| Integração com serviços reais | `docker compose -f compose.yaml --env-file <arquivo> exec worker-1 pytest -q -m services tests/test_services.py` |

`make test` executa `MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE=1 uv run --frozen pytest -q -n 6 --dist=loadfile`. A opção `-n 6` vem do plugin `pytest-xdist` (extra `dev`, presente em `uv.lock`), que distribui os arquivos de teste em seis processos (`--dist=loadfile` mantém cada arquivo no mesmo processo). Como herda o `addopts`, `make test` inclui as sementes `zia_extended_seeds`, ao contrário do CI.

Um `-m` informado na linha de comando substitui o filtro do `addopts`: `pytest -m services` seleciona os testes de serviços, que falham sem a pilha ativa.

### Fragmentação (*sharding*)

As opções `--shard` e `--num-shards` não vêm de um plugin: são definidas em `tests/conftest.py` (`pytest_addoption` e `pytest_collection_modifyitems`). Depois da coleta, o teste de índice `i` fica no fragmento `i % num_shards`; os demais são desmarcados. `--shard` deve estar no intervalo `[0, num_shards)`, senão o pytest encerra com erro de uso. Sem `--shard`, nada muda.

```bash
uv run --frozen pytest -q -m 'not services and not benchmark and not zia_extended_seeds' --shard=0 --num-shards=4
```

## Lint e formatação

O projeto usa [Ruff](https://docs.astral.sh/ruff/) `0.15.7` com linhas de até 110 caracteres e as regras `E` (pycodestyle), `F` (Pyflakes) e `I` (ordenação de imports), configurados em `pyproject.toml`.

| Comando | Efeito |
|---|---|
| `make lint` | `uv run ruff check src tests dags examples scripts` |
| `make format` | `ruff format` seguido de `ruff check --fix` nas mesmas pastas |

O CI roda apenas `ruff check`; a formatação é imposta localmente pelo hook `pre-push`.

## Hook de Git

`make install-hooks` aponta `core.hooksPath` para `.githooks` e torna os scripts executáveis. O único hook é `.githooks/pre-push`, que executa em sequência:

1. `make format`; se o comando alterar arquivos rastreados, o push é bloqueado até as mudanças serem commitadas;
2. `make lint`;
3. `make test`.

A variável `SKIP_PRE_PUSH=1` pula o hook inteiro.

## Workflow de CI (`.github/workflows/ci.yml`)

Disparado em *pull requests* para `main` e quando chamado por outro workflow (`workflow_call`). Pushes em outras branches não disparam o CI. Execuções antigas da mesma referência são canceladas (`concurrency` com `cancel-in-progress: true`), e o token tem permissão apenas de leitura do conteúdo.

| Job | Tempo limite | Passos |
|---|---|---|
| `lint-validate` | 10 min | Python 3.12, uv, instala `ruff==0.15.7` e `PyYAML==6.0.3`, cria um ambiente com `infra/ci-requirements.lock` (`cfn-lint`), roda `ruff check src tests dags examples scripts`, valida os templates CloudFormation (`scripts/aws/validate_templates.py`) e as máquinas de estado (`scripts/aws/validate_asl.py`) de forma offline |
| `test` (matriz `shard: [0, 1, 2, 3]`, `fail-fast: false`) | 15 min por fragmento | Python 3.12, uv, instala `requirements.lock` com *hashes* e o pacote em modo editável sem dependências, roda `python -m pytest -q -m 'not services and not benchmark and not zia_extended_seeds' --shard=<n> --num-shards=4 --durations=10` |

Os dois jobs fazem *checkout* esparso (`sparse-checkout-cone-mode: false`) com o padrão `/*` e a exclusão `!/evidencias/`. Assim, o diretório de evidências, que é grande, não é baixado. Nenhum teste lê arquivos da pasta de documentação, então a exclusão não afeta a suíte. O `.dockerignore` também exclui `evidencias` das imagens.

O job de testes usa `requirements.lock`, que não contém `pytest-xdist`; por isso o CI paraleliza por fragmentos em quatro máquinas, e não com `-n`.

## Workflow de deploy (`.github/workflows/deploy.yml`)

Disparado por push em `main` (inclusive merges) e manualmente (`workflow_dispatch`). Usa autenticação OIDC (`id-token: write`) para assumir o papel `vars.AWS_ROLE_ARN`, no ambiente protegido `production`. Deploys não se sobrepõem nem são interrompidos (`concurrency: deploy-main`, `cancel-in-progress: false`). O comentário do próprio workflow afirma que ele nunca baixa datasets nem inicia campanhas.

| Job | Depende de | O que faz |
|---|---|---|
| `verify` | nenhum | Reutiliza `ci.yml` por `workflow_call` |
| `build` | `verify` | Lê exports do CloudFormation (repositórios ECR, bucket de artefatos), valida as máquinas de estado no serviço, constrói e envia as imagens por *digest* (`scripts/aws/build_images.sh`), empacota a Lambda de controle e os módulos, empacota `infra/cloudformation/root.yaml` e gera `runtime-parameters.json`; publica `build-output` como artefato por 14 dias |
| `deploy` | `build` | Aplica um *change set* da stack de runtime com `scripts/aws/deploy_change_set.sh`, usando o papel de execução do CloudFormation do projeto |
| `release` | `deploy` | Grava o manifesto de release (`scripts/aws/write_release.py`) no bucket de dados |

Variáveis de repositório usadas: `AWS_REGION`, `PROJECT`, `RUNTIME_STACK`, `RUNTIME_CONFIG_KEY` e `AWS_ROLE_ARN`. A infraestrutura provisionada por esse fluxo está em [Execução na AWS](Execucao-na-AWS) e as estimativas de custo em [Custos e Recursos](Custos-e-Recursos).

## Workflow de publicação da Wiki (`.github/workflows/publish-wiki.yml`)

Disparado por push em `main` (inclusive merges) quando há alterações em `docs/**` ou no próprio workflow, e também manualmente (`workflow_dispatch`). Faz *checkout* esparso apenas de `docs/`, verifica a presença de `docs/Home.md` e `docs/_Sidebar.md`, clona o repositório `.wiki.git` do projeto usando o segredo `WIKI_PUSH_TOKEN` (com fallback para `github.token` via `contents: write`), sincroniza os arquivos `.md` de `docs/` e envia um commit apenas se houver diferença.

Requisito operacional no GitHub: a aba Wiki do repositório precisa estar habilitada e inicializada com pelo menos uma página antes da primeira execução automática.

## Lacunas conhecidas

| Lacuna | Evidência |
|---|---|
| Testes com serviços reais não rodam no CI | Filtro `not services` em `ci.yml` |
| Marcador `benchmark` sem testes associados | `pytest --collect-only -m benchmark` não coleta nenhum teste |
| `requirements.lock` defasado em relação a `uv.lock` | Reexportação acrescenta `pytest-xdist` e `execnet` |
| Formatação não verificada no CI | `ci.yml` roda só `ruff check` |
| Fragmentação por índice, sem balanceamento por duração | `tests/conftest.py` usa `index % num_shards` |

## Estado das capacidades

| Capacidade | Classificação | Evidência |
|---|---|---|
| Suíte rápida e lint | Validado | Executados em todo pull request para `main` pelos jobs `lint-validate` e `test` de `ci.yml` |
| CI em quatro fragmentos com checkout esparso | Implementado | `.github/workflows/ci.yml` |
| Deploy por OIDC com change set | Implementado | `.github/workflows/deploy.yml`; registros de execução na AWS em [Execução na AWS](Execucao-na-AWS) |
| Publicação automática da Wiki | Implementado | `.github/workflows/publish-wiki.yml` (push em `main` alterando `docs/**`) |
| Testes de integração com serviços | Experimental | Marcador `services`, executados apenas manualmente na pilha Compose |
