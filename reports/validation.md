> Registro histórico de 26/10/2026. A correção operacional posterior está em [validation-20261027.md](validation-20261027.md). MLflow foi iniciado e novos artefatos ZenML passaram ao MinIO; as pendências abaixo descrevem o estado anterior.

# Validação e estado do conjunto de serviços — 26/10/2026

## Estado registrado e limites

Implementação em `/Users/gabrieloliveira/residencia-ia`, ramificação `Semana-08`, base `ca10670518ec9739f08a305aa344b6f4fe044e3e`; execução em `/home/gabriel/residencia-ia-benchmark`. Sem commit, push ou PR. A rede dedicada aprovada `continual-bench_bench` usa `10.254.88.0/24`; nenhuma rede anterior foi removida e não houve alteração manual de firewall, daemon Docker, SSH ou VPN.

Airflow, ZenML, Kafka, MinIO e Postgres foram iniciados. O serviço executor responde, mas recusa tarefas enquanto o MLflow não estiver pronto. **MLflow continua parado e exige aprovação de senha específica com no mínimo 12 caracteres. Nenhuma senha alternativa foi gerada ou aplicada.** As demais credenciais foram preservadas. Kafka, Postgres, API S3 e serviço executor permanecem internos; somente as quatro interfaces web aprovadas estão vinculadas ao endereço `192.168.100.27`.

**Não há comprovação de validação positiva completa de ponta a ponta.** As nove combinações com publicação MLflow e a execução positiva de todas as avaliações comparativas pela DAG continuam pendentes. Uma rejeição correta de prontidão não equivale a uma avaliação comparativa bem-sucedida.

## Incidente relatado pelo usuário

Execução: `manual__2026-10-26T16:41:08.634869+00:00`, iniciada às 16:41:09 e encerrada às 16:42:27 UTC. As nove tarefas falharam após três tentativas cada. Os 27 erros registrados foram `URLError: [Errno -3] Temporary failure in name resolution` ao chamar o serviço executor; exemplos `sea_streaming` e `stagger_offline` apontam para a linha 41 da versão da DAG executada.

Causa imediata: `http://worker:8088/runs` não resolvia porque o contêiner do serviço executor estava em `Created`, sem processo/IP ativo. A avaliação comparativa não chegou a ser submetida. A partida do serviço executor havia sido bloqueada pela dependência MLflow, cuja inicialização recusava a senha por exigir pelo menos 12 caracteres. Havia também uma falha de implementação: a DAG liberava as nove tarefas sem verificar a prontidão do conjunto de serviços. A consulta ZenML durante a investigação mostrou somente o registro de execução anterior concluído, sem uma nova execução correspondente à tentativa Airflow.

A execução original não foi limpa, reexecutada ou alterada. Não foram apagados dados ou execuções. Fontes: `pipeline-failure-summary.json`, `zenml-runs-during-incident.json` e `readiness-deployment-audit.json`.

## Correção aplicada

- DAG com `services_ready`, sem novas tentativas, antes das nove avaliações comparativas. Ela verifica dependências e falha com mensagem útil antes da submissão.
- Serviço executor com `/health` para processo e `/ready` para prontidão autenticada do rastreamento. Submissões são recusadas com HTTP 503 quando o conjunto de serviços não está pronto. A verificação de integridade do Docker usa `/ready`.
- CLI e entrada ZenML verificam a precondição antes de materializar dados, configurar conjunto de serviços ou criar fluxo de processamento. A validação de senha não foi reduzida e a versão do MLflow não foi reduzida.
- Imagens reconstruídas; DAG e módulo auxiliar atualizados atomicamente nos três contêineres Airflow, sem reiniciar seus processos. Antes da aplicação, a consulta de execuções ativas retornou zero. Os hashes conferem entre projeto e servidor; os arquivos da DAG em uso também foram registrados.

## Revalidação real da rejeição

Execução de diagnóstico separada: `diagnostic_readiness_20261026T170318Z`, de 17:03:21 a 17:03:24 UTC. `services_ready` falhou uma vez, apontando MLflow indisponível e a exigência de senha; as nove tarefas de avaliação comparativa ficaram `upstream_failed`, todas com **zero tentativas**. O diagnóstico não executou treino nem fingiu sucesso. A API Airflow confirmou dez tarefas, com as nove dependentes da verificação inicial.

Sondagens do agendador para o serviço executor real: `/health` 200, `/ready` 503, `POST /runs` 503. CLI real retornou código 1; o diretório de artefatos solicitado não foi criado e havia zero arquivos de tarefas na fila. Evidências: `live-readiness-probe.json`, `live-cli-rejection.json`, `readiness-dag-graph.json`, `readiness-deployment-audit.json`.

## Testes executados

- Suíte Linux final: **62 testes aprovados e 9 não selecionados em 5.23s** (`linux-pytest.txt`). Os nove excluídos são testes de serviços; não são contados como integração aprovada.
- Nove regressões novas de prontidão passaram (`readiness-tests.txt`); incluem rejeição antes de acessar a rede, criar artefatos ou iniciar o fluxo de processamento e erros úteis sem divulgar credenciais.
- Regressões anteriores cobrem ordem, rótulos atrasados, avaliação antes do treino, retomada, corrupção, isolamento de transporte e identidade completa das extensões e dos módulos auxiliares declarados. A evidência anterior foi preservada em `linux-pytest-before-readiness.txt` e `review-regressions.txt`.
- 21 testes rápidos locais dos três conjuntos de dados, modos e modelos de referência concluíram antes das correções de integração/prontidão (`linux-local-smoke.jsonl`). Foram executados em contêiner sem rede e não validam serviços externos.
- Ruff e Compose config passaram nas etapas de correção. Após a correção de registros de execução ZenML, a suíte foi repetida na imagem reconstruída (5.23s). A evidência de 5.10s foi preservada em `linux-pytest-before-observability.txt`.

## Componentes reais e acesso LAN já verificados

- Pelo notebook: Airflow health/API autenticada 200; ZenML health/login 200; MinIO Console 200/login 204. URLs: `http://192.168.100.27:18081`, `http://192.168.100.27:18082`, `http://192.168.100.27:19001`. Ver `lan-services-partial.json` e `lan-auth-partial.json`. MLflow em `:15000` ainda não foi validado como disponível.
- Fluxo ZenML real de três etapas concluiu em 3.052s: execução `ba0deb5d-be2b-42cc-a57a-2703c7bb8b7f`, avaliação incremental local explicitamente identificada, 1200 eventos/1100 avaliados. Não publicou MLflow (`zenml-live-sanitized.log`).
- Kafka real processou 180 eventos SEA; MinIO confirmou envio e leitura idêntica do objeto `s3://benchmarks/independent/31dad689939a73be96b5930e/report.json` (`independent-services-sanitized.log`). Isso valida esses componentes, não a combinação completa de ponta a ponta.
- A importação da DAG Airflow retornou `[]` de erros antes do incidente; a DAG corrigida foi serializada e executou a admissão real descrita acima.

## Credenciais, versões e próximos passos

Arquivo `.env` na raiz e cópia remota em `/home/gabriel/.config/continual-bench/benchmark.env`: privados, modo 600. `.env` ignorado por Git, ausente do índice e excluído do contexto Docker. Valores não aparecem nestes relatórios. `.env.example` contém valores de exemplo. Nenhuma credencial foi alterada durante diagnóstico ou consolidação.

Serviço executor e SDK e Airflow usam ambientes separados. Pydantic foi fixado em 2.12.5 por compatibilidade ZenML. MinIO comunitário foi compilado do commit oficial `9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a`, versão `RELEASE.2025-10-15T17-29-55Z`, com limitação de manutenção e licença AGPL documentadas.

Após aprovação da senha do MLflow: atualizar somente esse valor nos arquivos privados, preservar as demais credenciais e iniciar somente MLflow e o serviço executor sem reiniciar Airflow ativo. Quando `/ready` retornar 200, executar as nove combinações de ponta a ponta reais, conferir rastreamento e objetos e uma nova execução positiva do Airflow. Preservar os registros históricos de execução. O diagnóstico posterior de registros e visualizações ZenML está registrado abaixo; sua etapa de acesso permanece pendente de aprovação específica.

## Incidente ZenML: registros de execução e visualizações

Executado diagnóstico pelo SSH e no Chrome do notebook. Na execução histórica `ba0deb5d-be2b-42cc-a57a-2703c7bb8b7f`, o grafo mostra três etapas concluídas; a aba Logs e a aba Visualization do artefato `a7169365-b6e9-4207-bef8-e454ecc1f9fa` exibem **“Files in a local artifact store cannot be accessed from the server.”** As capturas estão em `zenml-logs-local-store-error.jpg` e `zenml-artifact-local-store-error.jpg`.

Causa de leitura/renderização: armazenamento local de artefatos usado com servidor implantado. A verificação de segurança oficial do SDK 0.98.0 bloqueia esses caminhos por padrão. Todos os quatro arquivos de registro do cliente e da preparação e os três JSONs de visualização existiam no serviço executor e também no volume montado somente leitura no servidor. Não foi erro do componente de materialização, ausência de geração das visualizações ou falha do MLflow. Metadados e tamanhos em `zenml-observability-before.json`.

Causa adicional de geração: as funções das etapas não emitiam mensagens próprias; seus três registros `source=step` não possuíam arquivo. A correção em `adapters/zenml_steps.py` acrescenta mensagens operacionais de nível INFO no início e na conclusão, sem configuração completa, amostras ou segredos, e habilita registros de execução explicitamente. Não reduz a segurança do servidor.

Revalidação isolada, sem publicação de serviços: execução **`d7b5d8fc-841b-41ec-98e2-3d53741bdfed`**, SEA incremental 180 eventos, três etapas concluídas em 2.452s. Agora os três arquivos `source=step` existem: 428, 415 e 224 bytes, com 2, 2 e 1 mensagens respectivamente. Os três artefatos possuem visualização JSON persistida. Evidências: `zenml-observability-after.json`, `zenml-observability-run.log`, `zenml-observability-lint.txt`. A consulta anterior à execução encontrou zero execuções ativas ZenML e Airflow. Nenhum registro histórico de execução foi modificado ou apagado. Isso é uma execução ZenML real de avaliação local, **não** um teste completo de ponta a ponta com Kafka, MLflow e MinIO.

A nova execução também foi aberta no Chrome e seu estado `completed` e seu grafo foram confirmados; um anúncio sobrepôs o painel na reinspeção posterior. Não há comprovação de renderização positiva de registros e prévias, pois a restrição do armazenamento de artefatos continua ativa. A falha de leitura foi verificada visualmente na execução histórica e a causa está reproduzida no código oficial instalado.

A imagem `continual-bench-worker:0.1.0` foi reconstruída e aplicada somente ao serviço executor após confirmar fila vazia, sem reiniciar outros serviços. Imagem: `sha256:07c0a45ef43e1516c452145eb6c6fa514e511a07298fc903a4714bf58569d759`; hash da fonte implantada confere com o projeto. `/health` 200/fila zero e `/ready` 503 ainda apontam a senha do MLflow pendente. Ver `zenml-worker-update.json`.

**Etapa pendente de aprovação, não executada:** criar identidade/credencial MinIO exclusiva, com acesso somente ao bucket privado `zenml-artifacts`, registrar Service Connector e novo armazenamento de artefatos S3 e conjunto de componentes ZenML para execuções futuras. A política proposta, sem segredo, está em `docs/zenml-minio-policy.example.json`; o plano e as fontes oficiais estão em `docs/zenml-observability.md`. Isso requer aprovação específica de novo acesso persistente, conforme o limite de autorização registrado. Nenhuma identidade, credencial, bucket, permissão ou conector novo foi criado para esse plano. A opção de permitir arquivos locais no servidor não foi habilitada. Execuções antigas continuam com seus URIs locais; migração histórica não foi executada.
