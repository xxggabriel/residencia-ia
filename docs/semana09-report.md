# Implementação e validação da Semana 09

Em 3 de novembro de 2026, foi ampliado o laboratório da Semana 08 na branch local `Semana-09`, criada de `origin/main` (`bd8ac1c`) do remoto confirmado `https://github.com/xxggabriel/residencia-ia.git`. O checkout inicial estava limpo. Foram implementados componentes e executadas validações funcionais no homelab; não foi concluída uma campanha comparativa, um ranking ou uma reprodução CARA/UPF.

A continuidade de 4 de novembro implementou os loaders brutos e os testes de **100 exemplos/2 épocas**. Os resultados atuais e a preparação da DAG completa ficaram no [relatório de datasets e execução](semana09-dataset-execution.md). As seções abaixo preservaram o registro da primeira etapa; seus itens então pendentes não substituíram o estado atualizado desse complemento.

## Implementação realizada

- Foram preservados os controles SEA, AGRAWAL, STAGGER, as fontes reais já fixadas e os contratos da Semana 08. Foram integrados Hyperplane e Random RBF Drift com movimento obrigatório, além de FriedmanDrift com alvos contínuos preservados.
- Foram acrescentados manifestos de admissão local para INSECTS, Gas e dados preparados de MNIST, CIFAR, CORe50, Stream51 e CLEAR. Foram exigidos origem, licença, versão, hashes, schema, ordem, variante, acesso e cache explícitos. Os formatos implementados foram CSV, ARFF denso, LIBSVM e JSONL. Não foram realizados downloads dessas bases.
- Foram implementadas transformações Split, Permuted e por grupo/sessão temporal. Elas preservaram classes, splits e metadados; não substituíram loaders brutos de imagens ou protocolos oficiais ainda não configurados.
- Foi implementado um avaliador tipado para regressão e multiclasse com River, predição antes da liberação/aprendizagem, atraso de rótulos, warm-up, orçamento e baselines none/online/periodic. Rótulos de validation/test não treinaram o modelo. Essa avaliação foi intercalada, não um holdout temporal congelado. Sua retomada reaproveitou runs completos; execuções tipadas interrompidas reiniciaram do zero.
- Foram integrados EDDM, HDDMA, HDDMW, Page Hinkley e KSWIN, mantendo ADWIN/DDM. O sinal dos detectores no motor binário foi o erro preditivo dos rótulos liberados. Alarmes não foram interpretados isoladamente como mudanças em P(X) ou P(Y|X).
- Foram implementadas políticas por desempenho e custo de cenário. `cost_heuristic` comparou uma hipótese explícita de redução de erro com custos fixo e por amostra; não implementou os algoritmos completos CARA ou UPF. Os desvios ficaram em [políticas](week09-policies.md).
- Foi separada a decisão de treinar da promoção. O gate opcional treinou no prefixo da janela e validou no sufixo histórico disponível. Candidatos rejeitados consumiram orçamento/cooldown e foram incluídos em amostras e tempo de treinamento. O holdout do candidato não integrou seu treino; o incumbente pôde ter aprendido esses exemplos anteriormente. Esse gate não constituiu evidência independente de generalização.
- Foi acrescentado o suplemento `semana09-metrics-v1`: validação de anotações de drift, resumo GEM, forgetting após aprendizado, retenção, recursos e near-future accuracy operacional com snapshots congelados. Ausência de anotações manteve F1 do detector indisponível. FWT de GEM sem baseline inicial permaneceu indisponível. As definições históricas da Semana 08 foram preservadas e distinguidas.
- Foram acrescentados campanha declarativa limitada, journal, locks, falhas visíveis e retomada verificada, além de configuração de um cliente transitório de testes com serviços. Os parâmetros vazios novos foram omitidos das identidades históricas. Não foram atualizadas dependências ou lockfiles.

A matriz com componentes, versões, fontes, testes e restrições ficou em [cobertura](semana09-coverage.md). O catálogo parcial executável ficou em `configs/week09-datasets.json`.

## Ambiente e isolamento

Todos os testes iniciais foram executados via `ssh -o BatchMode=yes -o StrictHostKeyChecking=yes gabriel@192.168.100.27`, utilizando autenticação e host conhecidos. Não foram alterados firewall, credenciais, permissões ou stacks existentes. O acesso ao ambiente gráfico do Mac não foi utilizado.

O preflight encontrou o host `gabriel`, CPU Intel Xeon E5-2680 v4, 28 CPUs lógicas, aproximadamente 30 GiB de RAM e 525 GiB livres no filesystem raiz. A carga inicial era 0,63/0,74/0,83 e havia cerca de 21 GiB disponíveis. A stack existente estava saudável. Os testes usaram snapshots próprios em `/home/gabriel/semana09-20261103-2146`, com versões finais em `revision4`.

Foi reutilizada a imagem existente `continual-bench-worker:0.1.0`, ID `sha256:75335742611805882ea6d92d0b9b5ed80a3ba8e45dc2e3caf751cf2d37364bc7`, com `PYTHONPATH=/work/src` e `PYTHONHASHSEED=0`. O ambiente registrou Python 3.12.13, River 0.26.1, pytest 9.1.1, Ruff 0.15.7 e ZenML 0.98.0; as versões completas e o hash do código ficaram nos artefatos.

A suíte sem serviços utilizou container sem rede, código somente leitura, 2 CPUs, 2 GiB de RAM, 128 processos e timeout externo de 240 segundos. Os smokes CLI usaram 1 CPU, 1 GiB e timeouts de 120/60 segundos. O cliente de serviços usou 2 CPUs, 2 GiB, 192 processos, diretório de estado e configuração ZenML próprios, nenhuma porta publicada e timeout externo de 300 segundos. Ele se conectou à rede existente somente para os testes autorizados.

O projeto/stack ZenML isolado foi `semana09-20261103-2146` / `semana09-validation-2146`. O aviso da versão OSS sobre visualizar apenas o projeto default foi esperado; a validação utilizou a API, não o dashboard. A execução real do Airflow e uma nova campanha orquestrada por ele não foram realizadas; seus testes de código permaneceram na suíte.

## Resultados observados

| Verificação | Resultado | Evidência |
| --- | --- | --- |
| Base antes das mudanças | PASS, 343 testes, 28 excluídos, 28,24 s | `evidence/baseline.txt` no homelab |
| Suíte final sem serviços/benchmark | PASS, 430 testes, 31 excluídos, 29,63 s | [unit.txt](../reports/semana09/unit.txt) |
| Ruff final | PASS | [lint.txt](../reports/semana09/lint.txt) |
| Compose do cliente transitório | PASS, `config --quiet` | Comando abaixo; nenhum serviço foi iniciado por essa validação |
| Smoke declarativo | PASS, 6 casos de 180 exemplos, seed 42, atraso 3 | [smokes.json](../reports/semana09/smokes.json), configurações versionadas |
| Smoke Friedman | PASS, 120 exemplos, 20 de warm-up, 100 avaliados, atraso 3 | [smokes.json](../reports/semana09/smokes.json) |
| Serviços reais finais | PASS, 3 testes, 279,29 s | [services.txt](../reports/semana09/services.txt), IDs e hashes em smokes.json |
| Downloads/homologação das bases reais novas | NOT RUN | Dependem de versão, licença, acesso e preparação |
| Campanha comparativa, múltiplas seeds e escalabilidade | NOT RUN | Não houve ranking nem teste de superioridade |
| Reprodução fiel CARA e UPF | NOT RUN | Heurística explicitamente distinta |

O aviso preexistente do MLflow sobre ausência de input example em um teste de backfill permaneceu na suíte final. Não foi desativado para obter sucesso.

Os seis casos do smoke combinaram fontes/políticas diferentes para exercitar componentes; não formaram uma comparação controlada de políticas. As métricas desse recorte serviram à verificação do funcionamento. O Friedman também foi um smoke de regressão, sem relação com ranking de classificação.

O hash SHA-256 do código Python validado foi `b82fa3f7699d58b73d514d5c83a73d9cba314edf523acdc81ff51f6de93d8e51`, correspondente aos fontes do commit `003a0ba`. O campo git_revision das execuções anteriores ao commit registrou `uncommitted`; a correspondência foi conferida pelo hash integral dos fontes. Ao final, os nove containers da stack original permaneceram saudáveis e não havia cliente transitório de teste em execução.

## Falhas encontradas e corrigidas

1. A primeira transferência por tar incluiu arquivos AppleDouble `._*`, que causaram falhas de leitura/lint. Foram removidos somente da área de staging desta tarefa e as transferências seguintes desabilitaram metadados do Mac.
2. O teste arquitetural detectou dependência de I/O/aplicação no novo módulo de domínio. A montagem de relatórios foi movida para `application/week09_reporting.py`; a validação não foi enfraquecida.
3. O teste de replay tipado detectou tuplas do descritor nativo versus listas lidas do JSON. A configuração foi canonicalizada antes de retornar/persistir.
4. A primeira integração Hyperplane executou o benchmark e falhou no persist por ausência de `version` no manifesto. O contrato foi corrigido; seu formato foi avançado para `river-moving-v2`, preservando os objetos imutáveis anteriores.
5. Uma repetição com código alterado tentou reutilizar run keys já publicadas, e o MLflow rejeitou os parâmetros imutáveis. Os testes passaram a derivar o namespace do hash do código. Outra tentativa encontrou o catálogo Hyperplane antigo com metadados diferentes; a versão nova da fonte produziu identidade própria. Nenhum objeto imutável foi apagado ou sobrescrito para ocultar o conflito.
6. A revisão identificou que o caminho local de cache não poderia integrar metadados compartilhados por conteúdo. Ele foi excluído, com teste de regressão.
7. A cópia de relatórios atômicos pelo usuário SSH recebeu PermissionError porque esses arquivos foram criados pelo usuário do container. A exportação foi realizada no ambiente executor que os produziu; permissões não foram alteradas. A tentativa de leitura falha não foi descrita como exportação bem-sucedida.

As rodadas falhas ficaram em `evidence/unit-first.txt`, `unit-second.txt`, `unit-third.txt`, `services-first.txt`, `services-final.txt` e `services-namespaced.txt` no diretório isolado do homelab. A rodada final ficou em `unit-v4.txt`, `lint-v4.txt`, `services-v4.txt`, `campaign-v4.txt` e `typed-v4.txt`.

## Reprodução

Os comandos abaixo foram executados no homelab. `WEEK09_SERVICE_ENV` apontou para o arquivo de ambiente já configurado, sem imprimir seu conteúdo. O script `run_week09_homelab.sh` utilizou o overlay específico da execução, preservado em `revision4/compose.yaml`.

```bash
export WEEK09_ROOT=/home/gabriel/semana09-20261103-2146/revision4
export WEEK09_SERVICE_ENV=/home/gabriel/.config/continual-bench/benchmark.env
bash "$WEEK09_ROOT/code/scripts/run_week09_homelab.sh" unit
bash "$WEEK09_ROOT/code/scripts/run_week09_homelab.sh" services
```

Para reconstruir o cliente transitório com o template versionado, em um novo diretório de estado:

```bash
export WEEK09_CODE=/caminho/absoluto/do/snapshot
export WEEK09_STATE=/caminho/absoluto/do/estado/isolado
export WEEK09_PROJECT=semana09-validacao-IDENTIFICADOR
# Manter a rede existente; não executar compose up.
docker compose --env-file "$WEEK09_SERVICE_ENV" \
  -f "$WEEK09_CODE/compose.week09-validation.yaml" config --quiet
docker compose --env-file "$WEEK09_SERVICE_ENV" \
  -f "$WEEK09_CODE/compose.week09-validation.yaml" \
  run --rm --no-deps --entrypoint python validation \
  -m pytest -q -p no:cacheprovider -m services tests/test_week09_services.py
```

No cliente limitado do homelab, os comandos CLI executados foram:

```bash
python -m continual_bench.cli campaign configs/campaign-semana09-smoke.json --state /state/smoke
python -m continual_bench.cli typed configs/semana09-friedman-smoke.json --state /state/typed
python scripts/export_week09_evidence.py --state /state
```

O script de campanha verificou o prazo entre casos e após cada retorno; o timeout externo limitou o processo inteiro. O `max_seconds` de um caso não abrangeu todas as chamadas de serviço, portanto o timeout externo continuou necessário. A execução permaneceu sequencial e limitada a 128 casos, até 100 mil eventos no motor binário e até 10 mil no avaliador tipado. Isso preparou experimentos pequenos; não demonstrou escalabilidade big data.

## Bloqueios e continuidade

A ação de ler/materializar a planilha pela referência Library fornecida retornou `page_reference_unsupported_format`, não passível de retry inalterado. Nenhuma cópia foi encontrada no projeto ou em Downloads. O catálogo completo e a seleção de todos os seus registros permaneceram **bloqueados pela ausência do arquivo legível**; o código implementou o escopo explicitamente nomeado e não afirmou cobertura dos 47 registros.

Para continuar: obter a planilha pelo fluxo suportado, reconciliar variantes e seleções; admitir dados originais com licença e hashes; fixar ordens/splits e cenários CL; executar matriz controlada com múltiplas seeds e orçamento; avaliar escalabilidade, acesso e recursos antes das cargas grandes; implementar CARA/UPF fielmente somente com seus contratos completos; registrar resultados e limitações. Os resumos por tarefa do avaliador tipado não substituíram uma matriz de testes após cada tarefa em uma campanha visual CL.

O trabalho permaneceu na branch solicitada. Não foram realizados merge, deploy público ou abertura de PR.

## Commits de implementação

- `6f50669`: adaptadores versionados, protocolos e avaliação tipada.
- `877fbd4`: políticas, detectores e promoção com contabilidade de candidatos.
- `003a0ba`: campanhas, métricas, integração e evidências.

A documentação e os recibos finais foram registrados em commit separado. A branch permaneceu local, sem push ou PR.

## Relatório no Space

O [Relatório Semana 09](https://chatgpt.com/space/page_3bc9e71db5848191a84939fa9c66e413) recebeu a síntese da implementação e validação. Foi criada a subpágina [Implementação e validação dos benchmarks da Semana 09](https://chatgpt.com/space/page_0b5f3967c7c48191a6a9b061c5a46a53), com cobertura, decisões, resultados, IDs, falhas e bloqueios. A leitura posterior confirmou a preservação dos blocos anteriores do relatório e o conteúdo novo. A renderização visual não foi inspecionada, respeitando a restrição de não operar a interface do Mac.
