# Correção operacional — 27/10/2026

## Escopo aplicado

Projeto local `/Users/gabrieloliveira/residencia-ia`, branch `Semana-08`.
Implantação em `/home/gabriel/residencia-ia-benchmark`, servidor `192.168.100.27`.
O repositório já continha os arquivos de implementação como não rastreados; nenhum commit, push
ou PR foi criado. Volumes e execuções históricas foram preservados.

## Correções

- MLflow estava encerrado por senha de bootstrap inferior a 12 caracteres. Uma senha exclusiva
  foi configurada nos arquivos privados remoto/local, mantendo as credenciais das outras interfaces.
  O Compose agora exige `MLFLOW_PASSWORD`, e o preflight rejeita valores ausentes, curtos ou de exemplo.
  Essa exigência segue a [documentação oficial de autenticação](https://mlflow.org/docs/latest/self-hosting/security/basic-http-auth/).
- Foram observadas 252 threads/processos no contêiner MLflow, com limite de 256. OMP/BLAS/MKL/NumExpr
  passaram a usar uma thread configurada. O consumo chegou próximo de 1 GiB durante uso real;
  o limite passou a 1536 MiB, aplicado ao contêiner sem reiniciar o trabalho em execução.
- Cinco DAGs parametrizadas: `benchmark_full`, `benchmark_streaming`, `benchmark_incremental`,
  `benchmark_offline`, `benchmark_strategies`. Todas manuais e habilitadas, além de `benchmark_smoke`.
  O parser Airflow não registrou erros. A imagem contém os módulos auxiliares necessários.
- A matriz completa tem 33 configurações por semente: três datasets, dois modos prequential com
  cinco estratégias cada e três holdouts temporais. Inclui ADWIN/DDM e políticas congelada,
  periódica, online com guardas e por drift.
- Pipeline ZenML de quatro etapas, incluindo auditoria das decisões efetivas de drift/retreinamento.
  A auditoria rejeita retreinamento por drift sem alarme, atualizações no modelo congelado e
  violação do cooldown. O ciclo temporal continua no motor Python.
- Novo armazenamento ZenML S3/MinIO com segredo e identidade restrita ao bucket privado
  `zenml-artifacts`. Stack `benchmark-minio-v2`. Não foi liberado acesso local do servidor.
  Uso de MinIO é suportado pelo [armazenamento S3 do ZenML](https://docs.zenml.io/stacks/stack-components/artifact-stores/s3).
- O SDK 0.98.0 avisa explicitamente que projetos adicionais não aparecem no dashboard OSS.
  Foi criado o projeto `continual-benchmark`, mas a operação final usa `default`, com display name
  **Continual Benchmark**, descrição específica e stack dedicada. Execuções de diagnóstico no projeto
  adicional foram preservadas. Não houve alteração de licença nem contratação paga.
- MLflow passa a registrar nomes descritivos, hash do código, quantidade de amostras e métricas
  agregadas consultáveis como `summary_*`, além das janelas e artefatos já existentes.
- Worker retorna resumo e recibo de publicação; Airflow valida ambos antes de consolidar os resultados.

## Validação concluída

- Local: **69 testes aprovados**, nove testes de serviços não selecionados; comando
  `PYTHONPATH=. uv run pytest -q`. O caminho é necessário para importar os exemplos locais.
- Imagem Linux efetivamente implantada: **69 aprovados, nove não selecionados, 5.64 s**.
  Evidência: `benchmark-unit-linux-20261027.txt`.
- Integração real: **nove aprovados em 333.77 s**, com Kafka, MinIO, MLflow e ZenML.
  Comando: `docker exec continual-bench-worker-1 pytest -q -m services tests/test_services.py`.
  Confere os três datasets/modos, objetos lidos por SHA-256, publicação MLflow concluída,
  URI `mlflow-artifacts:` e reexecução sem duplicar métricas.
  Evidência: `benchmark-services-20261027.txt`.
- Ruff, validação silenciosa do Compose e preflight passaram. Verificação de 77 arquivos
  de fonte/documentação/relatório não encontrou os valores das credenciais privadas.
- LAN: health ZenML/Airflow, console MinIO e API autenticada MLflow retornaram HTTP 200.
  MLflow listou `continual-bench-v1`. Evidência: `lan-services-20261027.json`.
- Servidor ZenML leu logs de cada etapa e visualizações JSON diretamente do armazenamento S3.
  Execução verificada: `cda8cda6-433b-4883-be85-b027b31bcae7`, quatro etapas concluídas.
  Evidência: `zenml-server-artifacts-20261027.json`.
- Identidade dedicada: listagem permitida em `zenml-artifacts`, negada em `benchmarks` e `mlflow`.
  Evidência: `zenml-bucket-isolation-20261027.json`.

## Execução Airflow

Execução real `validation_20261027_full`, disparada pelo scheduler, com 3000 amostras,
semente 42 e atraso de rótulo 3. Início: 13:49:00 de 27/10/2026, America/Sao_Paulo.
Conclusão: **14:06:33 America/Sao_Paulo**, estado **success**, **36 tarefas aprovadas**:
checagem inicial, plano, 33 avaliações mapeadas e consolidação. O retorno consolidado contém 33 resultados
com métricas, contagens de alarmes/atualizações e IDs MLflow. Evidência: `airflow-full-20261027.json`.
O código local corresponde ao worker implantado: SHA-256
`6d26c045d14bd1657db53297acb74ad0702e0ad7c96c98051d09931e13961a83`.

## Correção do INTERNAL_ERROR do painel MLflow

A listagem do painel enviava `Origin: http://192.168.100.27:15000`. A API Python não envia esse
cabeçalho, portanto os testes anteriores de tracking não cobriam a falha do navegador.
Os logs mostraram `Blocked cross-origin request` e HTTP 403 em
`/ajax-api/2.0/mlflow/runs/search`, datasets e modelos. O painel apresentava `INTERNAL_ERROR`.
A reprodução autenticada com o mesmo Origin retornou 403 tanto na rota API quanto AJAX.

O Compose passou a configurar `--cors-allowed-origins http://${WEB_BIND_IP}:15000`, conforme a
[configuração oficial de origens](https://mlflow.org/docs/latest/self-hosting/security/network/).
Somente MLflow foi recriado para essa correção, após o worker terminar a publicação da matriz.

Revalidação pela LAN:

- API e AJAX de busca de runs: HTTP 200, **42 runs** (33 da matriz e nove dos testes integrados).
- AJAX de datasets e modelos: HTTP 200.
- Origem externa de controle: HTTP 403.
- Sem autenticação: HTTP 401.

Evidência: `mlflow-browser-origin-20261027.json`. Foi acrescentado um teste real de regressão
para a origem do navegador; sua evidência está em `mlflow-browser-regression-20261027.txt`.
A imagem do worker foi reconstruída com esse teste e aplicada com a fila vazia.
A regressão passou: **1 aprovado, nove não selecionados, 1.88 s**. A suíte unitária final da imagem
passou com **69 aprovados e dez não selecionados em 5.23 s** (`benchmark-unit-final-20261027.txt`).
Os nove serviços em operação estavam **healthy** (`stack-final-20261027.json`).

## Conferência dos 33 resultados

A verificação final comparou cada configuração esperada com o status do worker, relatório local,
registro MLflow FINISHED, detector/política, métricas agregadas e SHA-256 do relatório no MinIO.
Os 33 relatórios possuem a mesma revisão de fonte. Evidência: `benchmark-matrix-20261027.json`.

## Limites

Os nove testes reais são integração funcional; a matriz de uma semente com dados sintéticos
não estabelece superioridade estatística de estratégias. DDM pode legitimamente não sinalizar
drift em uma configuração. Ausência de alarme não é substituída por um gatilho artificial.
Logs e artefatos históricos de armazenamento local mantêm sua limitação original; novos artefatos
usam S3. A leitura pela API foi verificada; não se afirma inspeção visual autenticada do dashboard.

Credenciais permanecem somente em arquivos privados e no segredo ZenML. O procedimento de uso e
bootstrap está em [benchmark-operations.md](../docs/benchmark-operations.md).
