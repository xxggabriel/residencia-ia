# Aquisição de INSECTS e Gas Sensor Array Drift

O [loader de fluxos reais](../src/continual_bench/datasets/real_streams.py) oferece `prepare_real_stream(name, cache, limit=100, seed=42, *, download=False)` para `insects` e `gas`, com aquisição limitada por bytes, prazo e cardinalidade. A fonte expõe `samples_iter()` de `TaskSample` e `manifest()`. A aquisição remota exige `download=True`. Na avaliação prequential a configuração usa `LocalManifestSource` para fontes locais já preparadas.

O INSECTS balanceado e o Gas nesta primeira parte pertencem à suíte por épocas. As
variantes desbalanceadas têm um [protocolo binário próprio](#benchmark-binário-da-semana-10).

## Fontes e versões

| Dataset | Fonte primária e versão selecionada | Contrato |
| --- | --- | --- |
| INSECTS | [USP Data Stream Repository](https://sites.google.com/view/uspdsrepository), Souza et al. 2020; arquivo público `abrupt_balanced.csv` indicado pelo [River 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/datasets/insects.py) | 52.848 linhas, 33 atributos numéricos e seis classes; arquivo completo esperado de 14.151.769 bytes |
| Gas Sensor Array Drift | [UCI 224](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset), DOI 10.24432/C5RP6W, batch10 corrigido em 14 de outubro de 2013 | 13.910 linhas, 128 atributos e seis classes, dez lotes na ordem 1 a 10; ZIP limitado a 16 MiB e conteúdo relevante descomprimido a 64 MiB |

Os URLs públicos, limites e requisitos ficaram em `configs/contracts/real-streams.lock.json`. O download de Gas publicado pela UCI foi anunciado como aproximadamente 9,5 MB. A aquisição armazenou somente as duas fontes selecionadas; não baixou o ZIP geral do repositório USP.

Na consulta feita para a Semana 09, o repositório USP declarou CC BY 4.0 e ressalvou direitos de fontes externas. O catálogo UCI listou CC BY 4.0 e preservou, no texto descritivo histórico, uma restrição a pesquisa. O manifesto registra ambas as referências. A validação inicial teve finalidade de pesquisa na Residência em IA.

## Integridade e cache

As fontes públicas não forneceram um SHA-256 na documentação consultada. A primeira aquisição autorizada verificou o arquivo inteiro, contagem de linhas, schema, classes e estrutura, calculou o SHA-256 integral e o fixou no recibo local `acquisition.json`. Esse recibo registrou URL original público, versão da seleção, bytes, ETag e Last-Modified quando disponíveis. O hash efetivamente observado identificou o conteúdo; não foi apresentado como checksum publicado pelos autores.

O cache foi admitido por renomeação atômica de um diretório temporário completo, somente após a validação. Falhas removeram o staging, permitindo nova tentativa explícita. A reabertura comparou URL, versão e SHA-256 e rejeitou alteração, cache incompleto e symlinks; não redownloadou conteúdo corrompido silenciosamente. A emissão das amostras voltou a conferir o arquivo original. Não foram alteradas permissões ou credenciais.

O loader Gas leu diretamente os membros conhecidos do ZIP, sem extrair arquivos arbitrários. Exigiu os dez membros únicos `batch1.dat` a `batch10.dat`, todos os 128 índices de atributos por linha, valores finitos e rótulos originais de 1 a 6. O parser INSECTS exigiu 34 colunas por linha e não inventou cabeçalho ou descartou uma linha como cabeçalho.

## Seleção e avaliação

O limite de 100 selecionou exatamente o prefixo publicado, sem embaralhamento nem transformação de rótulos para booleano. IDs derivaram do SHA-256 bruto e do offset original. A seed foi registrada como ignorada nessas fontes determinísticas. O stream retornou `split=train`; a reserva do holdout e o controle de épocas ficaram sob responsabilidade do avaliador coordenador, sem o loader modificar os dados.

Gas preservou a ordem dos dez lotes e registrou `task_id`/grupo por lote, além das cardinalidades e classes do arquivo completo. Um prefixo de 100 pôde abranger apenas parte do primeiro lote e das classes; ele não foi descrito como avaliação de todas as mudanças do dataset. INSECTS preservou a ordem do CSV `abrupt_balanced`. Nenhum loader inventou pontos de drift verdadeiros: `changepoints=None`. O relógio de replay foi ordinal simulado e não substituiu o calendário original de cada observação.

Os testes de contrato cobrem parsers, ordem de lotes, classes, integridade, corrupção, retry explícito, limites e rejeição de respostas HTML. INSECTS e Gas passaram no teste smoke com 100 exemplos reais por fonte e duas épocas. [Relatório de execução](task-dataset-execution.md)

## Benchmark binário da Semana 10

| Fonte do catálogo | CSV original | Linhas | Fronteiras abruptas anotadas |
|---|---|---:|---:|
| `insects_abrupt_imbalanced` | `abrupt_imbalanced.csv` | 355.275 | 5 |
| `insects_incremental_abrupt_reoccurring_imbalanced` | `incremental_abrupt_imbalanced.csv` | 452.044 | 2 |

O [lock de fluxos reais](../configs/contracts/real-streams.lock.json) registra SHA-256,
cardinalidade, limites dos regimes e distribuição de classes. O adaptador mantém os
33 atributos e a ordem do CSV e projeta o rótulo original `2` para positivo (`1`), com
as demais cinco classes no negativo (`0`). Os índices de mudança começam em zero e
indicam o início do novo regime. São usados para avaliação e auditoria, sem virar
atributo, task ID ou sinal prospectivo para a política.

O piloto testa streaming e incremental: Online, Frozen e ADWIN nos dois modos; CARA,
UPF e Oracle só no incremental. A [matriz completa](../configs/matrices/insects-72.yaml)
combina dois fluxos, dois modelos (regressão logística e Hoeffding Tree), seis políticas
e três seeds. O catálogo da suíte por épocas permanece com 17 entradas.

```bash
uv run --frozen continual-bench matrix configs/matrices/insects-pilot.yaml --dry-run
uv run --frozen continual-bench matrix configs/matrices/insects-72.yaml --dry-run
```

Para executar localmente, coloque os arquivos com os nomes da tabela em um diretório
e exporte `BENCH_DATASETS=/caminho/dos/csvs` antes de repetir o comando sem
`--dry-run`. Também é possível apontar essa variável para a raiz do cache com os
recibos de aquisição. O avaliador valida o SHA-256 antes de ler a fonte. Na AWS, siga o
[runbook de publicação e submissão](aws/operations.md#matriz-binária-insects).

A campanha arquivou 66 de 72 unidades. As seis pendências vieram das duas combinações
Online com Hoeffding, nas três seeds. Use as [métricas pareadas](evidencias/semana-10/insects/README.md)
quando a pergunta exigir comparar os mesmos eventos em todas as execuções; as métricas
do recorte original de cada run podem ter denominadores diferentes. Consulte F1 macro,
recall por classe e o baseline majoritário além da acurácia.
