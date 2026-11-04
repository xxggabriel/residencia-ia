# Aquisição e loaders INSECTS e Gas da Semana 09

Foram implementados loaders dos dados originais, com aquisição explícita e limitada por bytes, prazo e cardinalidade. `prepare_real_stream(name, cache, limit=100, seed=42, download=False)` retornou uma fonte com `samples_iter()` de `TaskSample` e `manifest()`. `download=False` exigiu um cache já admitido; uma ausência não disparou download implícito.

## Fontes e versões

| Dataset | Fonte primária e versão selecionada | Contrato |
| --- | --- | --- |
| INSECTS | [USP Data Stream Repository](https://sites.google.com/view/uspdsrepository), Souza et al. 2020; arquivo público `abrupt_balanced.csv` indicado pelo [River 0.26.1](https://github.com/online-ml/river/blob/0.26.1/river/datasets/insects.py) | 52.848 linhas, 33 atributos numéricos e seis classes; arquivo completo esperado de 14.151.769 bytes |
| Gas Sensor Array Drift | [UCI 224](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset), DOI 10.24432/C5RP6W, batch10 corrigido em 14 de outubro de 2013 | 13.910 linhas, 128 atributos e seis classes, dez lotes na ordem 1 a 10; ZIP limitado a 16 MiB e conteúdo relevante descomprimido a 64 MiB |

Os URLs públicos, limites e requisitos ficaram em `configs/real-streams-week09.lock.json`. O download de Gas publicado pela UCI foi anunciado como aproximadamente 9,5 MB. A aquisição armazenou somente as duas fontes selecionadas; não baixou o ZIP geral do repositório USP.

O repositório USP declarou CC BY 4.0 e ressalvou direitos de fontes externas. A licença atual no catálogo UCI também declarou CC BY 4.0, enquanto o texto descritivo histórico manteve a restrição a pesquisa. Esse contraste foi preservado no manifesto; a finalidade desta execução foi pesquisa da Residência IA. Não foi implementado aceite de termos, autenticação, confirmação de páginas HTML, cookies ou evasão de negação HTTP.

## Integridade e cache

As fontes públicas não forneceram um SHA-256 na documentação consultada. A primeira aquisição autorizada verificou o arquivo inteiro, contagem de linhas, schema, classes e estrutura, calculou o SHA-256 integral e o fixou no recibo local `acquisition.json`. Esse recibo registrou URL original público, versão da seleção, bytes, ETag e Last-Modified quando disponíveis. O hash efetivamente observado identificou o conteúdo; não foi apresentado como checksum publicado pelos autores.

O cache foi admitido por renomeação atômica de um diretório temporário completo, somente após a validação. Falhas removeram o staging, permitindo nova tentativa explícita. A reabertura comparou URL, versão e SHA-256 e rejeitou alteração, cache incompleto e symlinks; não redownloadou conteúdo corrompido silenciosamente. A emissão das amostras voltou a conferir o arquivo original. Não foram alteradas permissões ou credenciais.

O loader Gas leu diretamente os membros conhecidos do ZIP, sem extrair arquivos arbitrários. Exigiu os dez membros únicos `batch1.dat` a `batch10.dat`, todos os 128 índices de atributos por linha, valores finitos e rótulos originais de 1 a 6. O parser INSECTS exigiu 34 colunas por linha e não inventou cabeçalho ou descartou uma linha como cabeçalho.

## Seleção e avaliação inicial

O limite de 100 selecionou exatamente o prefixo publicado, sem embaralhamento nem transformação de rótulos para booleano. IDs derivaram do SHA-256 bruto e do offset original. A seed foi registrada como ignorada nessas fontes determinísticas. O stream retornou `split=train`; a reserva do holdout e o controle de épocas ficaram sob responsabilidade do avaliador coordenador, sem o loader modificar os dados.

Gas preservou a ordem dos dez lotes e registrou `task_id`/grupo por lote, além das cardinalidades e classes do arquivo completo. Um prefixo de 100 pôde abranger apenas parte do primeiro lote e das classes; ele não foi descrito como avaliação de todas as mudanças do dataset. INSECTS preservou a ordem do CSV `abrupt_balanced`. Nenhum loader inventou pontos de drift verdadeiros: `changepoints=None`. O relógio de replay foi ordinal simulado e não substituiu o calendário original de cada observação.

Os testes acrescentados cobriram os parsers com fixtures pequenas, ordem de lotes, preservação multiclasse, integridade/cache/replay, corrupção, falhas de aquisição e retry explícito, limites e rejeição de respostas HTML. Todos os testes iniciais e aquisições reais devem ser executados exclusivamente no homelab autorizado; este documento não atribuiu a presença dos testes a resultados já executados.
