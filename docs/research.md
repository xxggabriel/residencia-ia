# Referências e decisões de pesquisa

O levantamento de 26/10/2026 embasou a arquitetura inicial do projeto. As referências abaixo registram as escolhas daquele momento. Para a configuração atual, consulte a [arquitetura do projeto](architecture.md), o [configuração do Docker Compose](../compose.yaml) e o [guia AWS](aws/README.md).

| Escolha | Evidência e estado | Adequação e limite |
|---|---|---|
| River + ADWIN/DDM | [River, JMLR 2021, revisado por pares](https://www.jmlr.org/papers/v22/20-1380.html); [ADWIN, SIAM SDM 2007](https://epubs.siam.org/doi/10.1137/1.9781611972771.42) | Biblioteca madura, CPU, interfaces online; não prova melhor qualidade em todas as mudanças de conceito |
| Protocolos distintos | [Understanding Continual Learning Settings with Data Distribution Drift, publicação preliminar](https://arxiv.org/abs/2104.01678); [estrutura de avaliação comparativa de 2026, publicação preliminar](https://arxiv.org/abs/2606.07789) | Motiva contratos e comparação por protocolo, não transferência de resultados de um artigo para estes dados |
| Airflow 3.3.2 isolado | [Compose oficial](https://airflow.apache.org/docs/apache-airflow/3.3.2/howto/docker-compose/index.html); [Orquestrador Airflow do ZenML](https://docs.zenml.io/stacks/stack-components/orchestrators/airflow) | Airflow coordena tarefas; o serviço executor usa orquestrador local ZenML e evita dependências conflitantes |
| ZenML 0.98.0 | [Implantação com Docker](https://docs.zenml.io/deploying-zenml/deploying-zenml/deploy-with-docker), PyPI oficial e código instalado | Cliente e servidor com versões fixadas e SQLite persistente no servidor ZenML |
| Kafka 4.3.1 | [Guia oficial de início rápido](https://kafka.apache.org/quickstart/) | KRaft único para laboratório, uma partição por execução; não há alta disponibilidade |
| MLflow 3.17.2 | [Servidor de rastreamento](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/); [segurança de rede](https://mlflow.org/docs/latest/self-hosting/security/network/) | URI remota explícita, proxy de artefatos S3, autenticação real e lista de hosts permitidos; não misturar default-artifact-root S3 com proxy |
| MinIO comunitário | [repositório arquivado](https://github.com/minio/minio); [versão de segurança](https://github.com/minio/minio/releases/tag/RELEASE.2025-10-15T17-29-55Z) | Código-fonte fixado em `9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a`, AGPL; sem garantia de manutenção futura |

A pesquisa cobriu arXiv, JMLR e SIAM, além de documentação e código oficial. Os artigos orientam o desenho do experimento, sem transformar seus resultados em promessas deste sistema. Escolhas de limites/janelas e uso de geradores sintéticos são decisões de engenharia para um teste rápido, pequeno e verificável. Não foram selecionados novos algoritmos por uma suposta superioridade em relação ao estado da arte.

O resolvedor rejeitou Pydantic 2.14.1 com ZenML 0.98.0 na validação inicial. O [lock do executor](../requirements.lock) fixa Pydantic 2.12.5 e as demais dependências com hashes. O [Dockerfile do Airflow](../docker/airflow/Dockerfile) parte da imagem 3.3.2-python3.12. Construções com apt e imagem-base ainda dependem do conteúdo publicado nessas etiquetas. Na validação de uma implantação, registrar os digests observados.

Na AWS, o projeto usa SageMaker Pipelines para orquestrar campanhas Batch e jobs nativos. O MLflow gerenciado do SageMaker recebe o tracking dessas execuções, e o Unified Studio cataloga os snapshots tabulares Zia. Os contratos e procedimentos da integração estão no [guia de operações AWS](aws/operations.md).
