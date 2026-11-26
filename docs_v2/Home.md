# Documentação do continual-bench

O `continual-bench` é um pipeline de benchmark para comparar políticas de treinamento e atualização de modelos e detectores de drift em fluxos de dados ordenados no tempo. Ele mede desempenho preditivo, capacidade de adaptação a mudanças, retenção de conhecimento, esforço computacional e custo, sob protocolos de avaliação explícitos e com rastreamento das execuções.

Esta Wiki descreve o estado atual do código na branch `main`. Quando uma informação depende de uma execução específica, a página indica o arquivo de evidência correspondente.

## Por onde começar

| Objetivo | Página |
|---|---|
| Entender o propósito e o escopo do projeto | [Visão geral](Visao-Geral) |
| Conhecer as camadas e o fluxo de uma execução | [Arquitetura](Arquitetura) |
| Preparar o ambiente | [Instalação e configuração](Instalacao-e-Configuracao) |
| Rodar na própria máquina ou com Docker Compose | [Execução local](Execucao-Local) |
| Rodar no perfil gerenciado da AWS | [Execução na AWS](Execucao-na-AWS) |

## Componentes do benchmark

- [Datasets](Datasets): fontes sintéticas, reais e de imagem, versões fixadas, preparação e divisões temporais.
- [Modelos e políticas](Modelos-e-Politicas): modelos disponíveis e políticas de treinamento e atualização, incluindo CARA, UPF e Oracle.
- [Detectores de drift](Detectores-de-Drift): detectores implementados, sinais monitorados e relação com as políticas.

## Avaliação e resultados

- [Metodologia experimental](Metodologia-Experimental): protocolos, disponibilidade de rótulos, calibração, coortes e comparabilidade.
- [Métricas](Metricas): o que cada família de métricas mede, unidade, direção desejável e limitações.
- [Custos e recursos](Custos-e-Recursos): escopos de custo, estimativas e medições de recursos computacionais.
- [Reprodutibilidade e artefatos](Reprodutibilidade-e-Artefatos): seeds, identidade das execuções, locks, manifestos e checksums.
- [Tracking e evidências](Tracking-e-Evidencias): MLflow, ZenML, pacotes de evidência, notebooks, figuras e interpretação.

## Operação e qualidade

- [Testes e CI/CD](Testes-e-CI-CD): suíte de testes, marcadores, integração contínua e implantação.
- [Operação e diagnóstico](Operacao-e-Diagnostico): acompanhamento, recuperação de falhas e investigação de problemas.
- [Limitações e trabalhos pendentes](Limitacoes-e-Trabalhos-Pendentes): o que não está coberto ou não foi comprovado.

## Apoio

- [Glossário](Glossario)
- [Referências](Referencias)

## Convenções desta Wiki

As capacidades são classificadas com os rótulos abaixo.

| Rótulo | Significado |
|---|---|
| Implementado | Existe código executável no repositório. |
| Validado | Há teste automatizado ou evidência de execução registrada no repositório. |
| Experimental | Implementado, mas com validação parcial ou dependente de condições específicas. |
| Planejado | Descrito em planos, sem código correspondente. |
| Pendente | Lacuna reconhecida ou trabalho não concluído. |

Caminhos de arquivos aparecem entre crases e são relativos à raiz do repositório. Quando há dúvida entre um texto e o comportamento executável (código, configurações, testes e workflows), vale o comportamento executável. Evidências de execução ficam em `evidencias/`, com manifestos e checksums.
