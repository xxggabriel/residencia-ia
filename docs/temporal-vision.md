# Loaders de imagens temporais

O [loader temporal](../src/continual_bench/datasets/temporal_vision.py) oferece `prepare_temporal_dataset(name, cache, limit=100, seed=42, *, download=False, local_archives=None)` para `core50`, `clear10` e `clear100`. A fonte expõe `samples_iter()` e `manifest()` com imagens, alvos e metadados da publicação original.

## Contrato de aquisição

A chamada padrão lê o cache preparado e confere seu checksum. Para adquirir da origem, use `download=True`. Para preparar offline, preencha `local_archives` com todos os ZIPs necessários. CORe50 usa um único ZIP em `train`, e CLEAR usa um ZIP por split, em `train` e `test`. Se o mapa estiver parcialmente preenchido, os ZIPs faltantes são consultados na origem.

O leitor usa HTTP Range para consultar o diretório central, os metadados e as imagens selecionadas. O transporte exige HTTP 206 e Content-Range exato. Falhas transitórias 429, 5xx e de transporte recebem até três tentativas com espera limitada. No loader por amostras, o limite por arquivo é de 64 MiB transferidos e 2.048 requisições. O limite descomprimido é de 64 MiB por metadado e 8 MiB por imagem. A leitura usa blocos de 64 KiB.

O leitor confere CRC dos membros pelo `zipfile`, SHA-256 dos ranges e membros selecionados, tamanho e estabilidade de ETag/Last-Modified durante a leitura. O recorte remoto verifica apenas os bytes consultados. O MD5 integral publicado de CORe50 é uma referência. Os arquivos CLEAR são fixados na revisão Hugging Face `b518a845a98f1c913497ab98a19727cb65b74e65` indicada pelo espelho vinculado pelo projeto original.

## Recorte e representação

O loader por amostras aceita de 10 a 1.000 imagens. O padrão de **100 imagens únicas** seleciona 80 do treino e 20 do teste oficial por dataset. No smoke de duas épocas, o avaliador reapresenta as mesmas imagens de treino.

A seleção usa stride determinístico que preserva a ordem dentro de cada split. Em CLEAR, a leitura de metadados também tem limite. Os IDs derivam do dataset, da release, do caminho original e do hash da imagem. Split, tarefa, grupo, posição original e SHA-256 da imagem ficam no manifesto. O relógio de replay é ordinal e artificial.

A representação RGB passa por interpolação bilinear para 16×16 e divisão por 255, totalizando 768 features numéricas. O benchmark iterativo usa `prepare_temporal_arrays` para até 10.000 imagens, com pixels `uint8` armazenados em NPZ e normalizados conforme cada linha é usada.

## Fontes e cenários

| Dataset | Fonte original e versão | Regra implementada | Limite |
| --- | --- | --- | --- |
| CORe50 | [Autores](https://vlomonaco.github.io/core50/), arquivo 128×128 original, referência MD5 `745f3373fed08d69343f1058ee559e13` | Classe por objeto; treino em sessões 1,2,4,5,6,8,9,11 e teste nas sessões oficiais 3,7,10; ordem sessão/objeto/frame | Subconjunto NI diagnóstico; não um run NI/NC/NIC completo dos filelists |
| CLEAR10 | [Projeto original](https://clear-benchmark.github.io/), release julho de 2022, revisão do espelho fixada | Buckets e classes dos metadados originais, incluindo BACKGROUND quando presente, arquivos de treino e teste separados | A edição contém dez classes de interesse mais BACKGROUND; não foi silenciosamente reduzida a dez alvos |
| CLEAR100 | Mesma release oficial de 2022 e revisão fixada | Classes, buckets e splits originais | O smoke não sustentou avaliação de todas as classes ou comparação de retenção |

CLEAR usa os splits oficiais da release de 2022. O campo `manifest.protocol` identifica essa escolha. A avaliação streaming do artigo, com teste no bucket seguinte ao treino, exige seu próprio protocolo.

A documentação de origem consultada declarou CC BY 4.0 para CORe50 e CC BY para o benchmark CLEAR, com direitos individuais preservados para as imagens YFCC.

Implementações primárias consultadas para o formato:

- [CORe50 no Avalanche](https://github.com/ContinualAI/avalanche/blob/master/avalanche/benchmarks/datasets/core50/core50_data.py).
- [CLEAR metadata no Avalanche](https://github.com/ContinualAI/avalanche/blob/master/avalanche/benchmarks/datasets/clear/clear.py).

## Verificação inicial no homelab em novembro de 2026

Todas as execuções iniciais ocorreram em `gabriel@192.168.100.27`, via SSH com autenticação já disponível e `StrictHostKeyChecking=yes`. Não foram executados testes no Mac.

Os probes HTTP/ZIP reais confirmaram suporte a ranges em CORe50 e CLEAR. A listagem CORe50 incluiu 165428 membros e transferiu 18202869 bytes de diretório/metadados; CLEAR10 teve 33245 membros de treino com 4205433 bytes; CLEAR100 teve 112155 membros de treino com 14222299 bytes. As estruturas reais foram inspecionadas: CLEAR treino usou prefixo `train_image_only/`, enquanto teste usou `test/`.

Na validação inicial, a origem oficial do Stream51, `klab.cis.rit.edu`, falhou na resolução DNS. O adaptador da época foi exercitado com fixtures. Em 08/11/2026, Stream51 foi removido do catálogo, do adaptador, do lock e da DAG.

Sete testes do módulo passaram em 1,96 s na primeira rodada, cobrindo ranges e cache, corrupção, limites, negação sem retry, splits CORe50, JSON/bbox do antigo Stream51 e buckets/classes CLEAR10/CLEAR100. CORe50, CLEAR10 e CLEAR100 passaram também no smoke com 100 imagens reais e duas épocas. [Relatório de execução](task-dataset-execution.md)
