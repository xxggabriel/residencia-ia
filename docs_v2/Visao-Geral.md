# Visão geral

Esta página apresenta o propósito do `continual-bench`, as perguntas que ele ajuda a responder, o que está dentro e fora do escopo e o estado atual de cada grande capacidade.

## Objetivo

Modelos treinados uma única vez perdem qualidade quando os dados mudam ao longo do tempo. Essa mudança é chamada de *drift* (ou mudança de conceito). A resposta prática é atualizar o modelo, mas cada atualização consome tempo de computação e dinheiro, e atualizar cedo demais ou tarde demais tem consequências diferentes.

O repositório implementa um pipeline de benchmark para comparar, sob as mesmas condições, diferentes **políticas de treinamento e atualização** (quando e como retreinar ou ajustar um modelo) e **detectores de drift** (algoritmos que sinalizam mudanças a partir do erro do modelo). Cada execução registra:

- **desempenho preditivo**, por exemplo F1, acurácia balanceada e AUC;
- **adaptação**, isto é, como a qualidade se comporta depois de uma mudança e quantas atualizações foram feitas;
- **retenção**, quando o protocolo permite medir esquecimento de regimes anteriores;
- **esforço computacional**, como tempo de treino, CPU e memória;
- **custo**, separado em custo de cenário parametrizado, estimativa de infraestrutura e faturamento observado.

As definições estão em [Métricas](Metricas) e [Custos e recursos](Custos-e-Recursos).

## Perguntas que o benchmark ajuda a responder

- Em um fluxo com mudanças, uma política que retreina por alarme de detector obtém qualidade semelhante a uma política periódica com menos atualizações?
- Políticas sensíveis a custo, como CARA e UPF, reduzem o número de retreinos sem perda relevante de qualidade em relação ao Oracle retrospectivo?
- Quanto tempo de treino e quanta memória cada política consome no mesmo hardware reservado?

As respostas valem apenas para os datasets, modelos, protocolos e seeds de cada campanha. Ver [Metodologia experimental](Metodologia-Experimental#comparabilidade).

## Escopo

**Dentro do escopo**

- Fluxos ordenados no tempo, binários na maior parte das campanhas, com suporte a regressão e multiclasse em avaliadores específicos.
- Protocolos de avaliação explícitos: testar e depois treinar (por evento, por lote e por blocos), holdout temporal e avaliação por épocas.
- Disponibilidade atrasada de rótulos, de forma que o modelo só aprende com rótulos que já estariam disponíveis no instante da decisão.
- Execução local, com Docker Compose e no perfil gerenciado da AWS, usando o mesmo código do avaliador.
- Rastreamento das execuções e pacotes de evidência versionados com checksums.

**Fora do escopo**

- Uso em produção ou serviço de inferência com garantias operacionais.
- Alegações de estado da arte. Os testes rápidos e os *smokes* validam algoritmo, protocolo e integração, não desempenho.
- Reprodução numérica exata de artigos publicados. Onde há comparação com valores publicados, os protocolos diferem e a comparação é descritiva.

## Como o sistema está organizado

O núcleo Python em `src/continual_bench` controla todo o ciclo contínuo: ler o fluxo, prever, liberar rótulos, alimentar o detector, decidir a atualização e calcular métricas. Os serviços externos (Kafka, MinIO, MLflow, ZenML, Airflow e os serviços AWS) apenas transportam dados, guardam artefatos, registram métricas ou disparam execuções finitas. A descrição completa está em [Arquitetura](Arquitetura).

Há três caminhos de avaliação principais:

| Caminho | Uso típico | Modelos |
|---|---|---|
| Motor binário por evento | Comparações locais e matrizes de configuração | Catálogo River (regressão logística, Hoeffding Tree, kNN) |
| Avaliador por blocos | Suíte de streams ordenados e campanhas na AWS | XGBoost e regressão logística multinomial |
| Avaliadores tipado, por épocas e iterativo | Regressão, multiclasse e treino por épocas | Catálogo River |

Detalhes em [Modelos e políticas](Modelos-e-Politicas).

## Estado das capacidades

| Capacidade | Estado | Onde ver |
|---|---|---|
| Motor binário (testar e depois treinar) com políticas `none`, `online`, `periodic`, `drift` e `performance` | Validado por testes | [Modelos e políticas](Modelos-e-Politicas) |
| Avaliador por blocos com 11 políticas executadas na AWS (4 streams, 3 seeds, 132 unidades) | Validado por evidência registrada | [Tracking e evidências](Tracking-e-Evidencias) |
| Política `online` no avaliador por blocos | Implementado, fora da campanha AWS registrada | [Modelos e políticas](Modelos-e-Politicas) |
| Detectores ADWIN e DDM | Validado por testes e campanha | [Detectores de drift](Detectores-de-Drift) |
| Detectores EDDM, HDDM-A, HDDM-W, Page-Hinkley e KSWIN | Experimental (somente testes de contrato) | [Detectores de drift](Detectores-de-Drift) |
| Catálogo de 21 entradas de dados, 17 com carregador por épocas | Validado por testes | [Datasets](Datasets) |
| Perfil Docker Compose com Airflow, ZenML, MinIO, MLflow e Kafka | Implementado; integração coberta por testes marcados como `services` | [Execução local](Execucao-Local) |
| Perfil AWS com Step Functions, Batch em Fargate Spot, S3, Glue, Athena e MLflow gerenciado | Validado em implantações registradas, com lacunas | [Execução na AWS](Execucao-na-AWS) |
| Estimativa de custo de infraestrutura por job | Implementado, com simplificações conhecidas | [Custos e recursos](Custos-e-Recursos) |
| Reconciliação com faturamento observado (CUR 2.0) | Experimental: há código e testes, sem comando que a execute nem evidência registrada | [Custos e recursos](Custos-e-Recursos) |
| Recalibração de probabilidades | Não implementada; o código mede apenas erro de calibração | [Metodologia experimental](Metodologia-Experimental#calibração-e-limiar-de-decisão) |

A lista completa de lacunas está em [Limitações e trabalhos pendentes](Limitacoes-e-Trabalhos-Pendentes).

## Histórico do projeto

O repositório evoluiu em etapas semanais de uma residência em IA, iniciadas em outubro de 2026. Os resultados arquivados da Semana 10 estão em `evidencias/semana-10/` e a campanha AWS mais recente em `evidencias/aws-benchmarks/`; resultados de etapas anteriores permanecem acessíveis pelo histórico Git. Alguns protocolos antigos foram corrigidos depois, principalmente quanto à disponibilidade temporal de rótulos e à separação entre treino e teste. Por isso, números de protocolos históricos e atuais não devem ser comparados diretamente. Ver [Metodologia experimental](Metodologia-Experimental#protocolos-históricos-e-correções).
