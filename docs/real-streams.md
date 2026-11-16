# Aquisição de INSECTS e Gas Sensor Array Drift

O [loader de fluxos reais](../src/continual_bench/datasets/real_streams.py) oferece `prepare_real_stream(name, cache, limit=100, seed=42, *, download=False)` para `insects` e `gas`, com aquisição limitada por bytes, prazo e cardinalidade. A fonte expõe `samples_iter()` de `TaskSample` e `manifest()`. A aquisição remota exige `download=True`. Na avaliação prequential a configuração usa `LocalManifestSource` para fontes locais já preparadas.

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

Os testes de contrato cobrem parsers, ordem de lotes, classes, integridade, corrupção, retry explícito, limites e rejeição de respostas HTML. INSECTS e Gas passaram no smoke do homelab com 100 exemplos reais por fonte e duas épocas. [Relatório de execução](task-dataset-execution.md)
