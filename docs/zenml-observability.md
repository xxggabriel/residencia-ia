> Registro histórico de 26/10/2026. A correção operacional posterior está em [validação de 27/10](../reports/validation-20261027.md). MLflow foi iniciado e novos artefatos ZenML passaram ao MinIO; as pendências abaixo descrevem o estado anterior.

# Registros e visualizações ZenML

## Diagnóstico confirmado em 26/10/2026

No servidor ZenML 0.98.0, o armazenamento de artefatos `benchmark-local-v1` tem o tipo `local`, em `/state/zen-artifacts`. O serviço executor e o servidor montam o mesmo volume; o servidor tem montagem somente leitura. A execução `ba0deb5d-be2b-42cc-a57a-2703c7bb8b7f` concluiu as três etapas. Seu grafo aparece no Chrome do notebook, mas as abas Logs e Visualization do artefato mostram:

> Files in a local artifact store cannot be accessed from the server.

Em português: “O servidor não pode acessar arquivos em um armazenamento local de artefatos.”

O código oficial instalado, `zenml/artifact_stores/base_artifact_store.py`, aplica essa restrição quando executado no servidor. Não é um arquivo ausente nem falha de montagem: os quatro arquivos de registros do cliente e da preparação e os três JSONs de visualização existem no serviço executor e no servidor. O componente de materialização `BuiltInContainerMaterializer` registra esses JSONs como visualizações. A visualização do grafo e a prévia dos dados são funcionalidades diferentes.

As funções originais das três etapas também não emitiam mensagens próprias; portanto seus registros `source=step` não tinham arquivo. A correção acrescenta mensagens de nível INFO no início e na conclusão, com campos operacionais seguros, e habilita explicitamente os registros da execução e das etapas. Não registra configuração inteira, credenciais ou valores de amostras. Novos registros de execução não corrigem a restrição de leitura do painel.

## Etapa de acesso pendente de aprovação

Para o painel remoto, a documentação oficial recomenda armazenamento remoto de artefatos e conector de serviço (Service Connector). Proposta concreta, ainda não executada:

1. Criar bucket privado dedicado `zenml-artifacts` no MinIO existente, sem nova porta ou acesso público.
2. Criar uma identidade MinIO exclusiva, com credencial aleatória armazenada nos arquivos privados já usados pelo projeto. Política inicial: `s3:GetBucketLocation` e `s3:ListBucket` nesse bucket; `s3:GetObject` e `s3:PutObject` somente em `zenml-artifacts/*`. Não conceder administração, acesso aos buckets `benchmarks`/`mlflow` ou exclusão de dados. Exemplo sem segredo em `zenml-minio-policy.example.json`.
3. Registrar um conector de serviço e segredo restritos a esse bucket no ZenML e um novo armazenamento de artefatos S3 em `s3://zenml-artifacts`, com endpoint explícito `http://minio:9000` (`client_kwargs.endpoint_url`) e dependências S3 compatíveis nos ambientes cliente/servidor. Validar o suporte do conector ao MinIO antes de aplicar; não ampliar a política automaticamente se uma verificação exigir acesso adicional.
4. Criar um novo conjunto de componentes ZenML para execuções futuras, mantendo o conjunto de componentes e as execuções antigas. Só ativar após verificar ausência de tarefas ativas. Não presumir migração de registros e artefatos históricos: seus URIs locais permanecerão como foram criados.
5. Executar diagnóstico finito sem MLflow; conferir os registros de execução reais, JSONs persistidos, leitura autenticada pelo servidor e renderização de ambos no navegador. Depois validar as nove combinações integradas quando MLflow estiver pronto.

Isso cria acesso persistente novo para o ZenML e depende de aprovação específica. Nenhuma credencial, identidade, política ou conector dessa proposta foi criado. A opção `ZENML_SERVER_ALLOW_LOCAL_FILE_ACCESS` permanece desabilitada: habilitá-la ampliaria o acesso do servidor a arquivos locais e não faz parte da correção aplicada. Não alterar permissões do volume nem contornar a validação de caminhos.

A senha do MLflow continua sendo uma pendência separada. A falta de acesso ao armazenamento ZenML não é causada por MLflow e o diagnóstico isolado não publica experimentos nele.

## Evidências e fontes

- `reports/zenml-observability-before.json`: IDs, URIs, existência e tamanhos; nenhuma mensagem de registro ou segredo.
- `reports/zenml-logs-local-store-error.jpg` e `reports/zenml-artifact-local-store-error.jpg`: erro visto no Chrome.
- [Registros do ZenML](https://docs.zenml.io/concepts/steps_and_pipelines/logging): captura por etapa e restrição do servidor remoto.
- [Armazenamento de artefatos S3](https://docs.zenml.io/stacks/stack-components/artifact-stores/s3): armazenamento S3/MinIO, endpoint e autenticação explícita.
- [Código oficial fixado 0.98.0](https://github.com/zenml-io/zenml/blob/0.98.0/src/zenml/artifact_stores/base_artifact_store.py): validação de acesso local.

## Revalidação da geração

Execução de diagnóstico `d7b5d8fc-841b-41ec-98e2-3d53741bdfed`, SEA incremental, 180 eventos, concluiu as três etapas em 2.452s, sem publicação de serviços. Os registros de execução `source=step` passaram a existir (428/415/224 bytes) e os três JSONs foram persistidos. Ruff e 62 testes passaram; nove testes reais integrados continuam excluídos. O serviço executor foi atualizado com a imagem corrigida, após fila zero; outros serviços foram preservados. Isso valida a captura, não a leitura positiva no painel, que permanece bloqueada até a mudança de armazenamento/acesso aprovada. Veja `reports/validation.md`.
