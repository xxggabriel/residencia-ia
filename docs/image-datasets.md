# Loaders de MNIST e CIFAR

O [registro executável de datasets](../src/continual_bench/datasets/catalog.py)
usa os arquivos IDX para Split MNIST e Permuted MNIST e os espelhos Parquet
versionados para Split CIFAR10 e Split CIFAR100. Os loaders binários de CIFAR
continuam disponíveis para seleção explícita. Os avaliadores recebem pixels
normalizados por 255, nas dimensões originais.

## Aquisição e proveniência

`prepare_image_dataset(name, cache, limit=100, seed=42, *, download=False)` retorna
uma fonte com `samples_iter()` de `TaskSample` e `manifest()`. A aquisição exige
`download=True`. Split e Permuted MNIST reutilizam os mesmos quatro arquivos IDX.

Os arquivos e tamanhos foram fixados por SHA-256 no
`configs/contracts/image-datasets.lock.json`, com base nos catálogos oficiais de checksums
do [TensorFlow Datasets](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/url_checksums).
Os links e bytes efetivamente utilizados ficaram no manifesto. A versão foi
derivada do conteúdo e do protocolo, independente da localização do cache.

| Base | Aquisição mínima desta implementação | Formato |
| --- | --- | --- |
| MNIST | 11.594.722 bytes para os quatro arquivos; compartilhados entre variantes | gzip IDX, imagens 28 por 28 e rótulos |
| CIFAR10 | 170.052.171 bytes | tar.gz binário oficial, imagens RGB 32 por 32 |
| CIFAR100 | 168.513.733 bytes | tar.gz binário oficial, rótulos coarse e fine; alvo fine preservado |

O download completo foi necessário para validar o hash oficial do arquivo; apenas
o prefixo solicitado de registros foi decodificado em features. Não foi alegado
que 100 exemplos implicaram download de apenas 100 imagens comprimidas.

A origem do MNIST foi a [página dos autores](https://yann.lecun.com/exdb/mnist/),
com os arquivos fixados no espelho CVDF. O layout binário e as divisões de CIFAR
foram conferidos na [página oficial dos autores](https://www.cs.toronto.edu/~kriz/cifar.html).
A página CIFAR solicitou citação de Krizhevsky 2009 e não declarou uma licença
SPDX; nenhuma licença do código foi atribuída aos dados, e nenhum aceite de
termos restritos foi realizado. Os direitos e referências foram registrados
literalmente no lock e no manifesto, sem inferir uma autorização SPDX.

## Protocolo do smoke de 100 exemplos originais

A seleção utilizou exatamente 100 IDs originais distintos: 80 registros do
treino oficial e 20 do teste oficial. A divisão foi preservada, sem mover imagens
de teste para treino. Dentro de cada split, os primeiros registros da publicação
foram selecionados sem embaralhamento. Essa amostra pequena pôde não cobrir todas
as classes ou tarefas e não foi descrita como benchmark visual completo.

Para Split MNIST/CIFAR10, as classes foram agrupadas numericamente em cinco
pares; para Split CIFAR100, em dez tarefas de dez classes. A sequência foi
ordenada artificialmente por tarefa, split oficial e índice de origem. Os IDs e
o grupo conservaram o dataset, split e índice original; os timestamps do replay
foram ordinais artificiais, sem alegação de cronologia natural.

Permuted MNIST utilizou duas tarefas no orçamento global: imagens originais
distintas foram distribuídas alternadamente entre uma tarefa de identidade e
outra com permutação de pixels fixada pela seed. A mesma permutação foi aplicada
a treino e teste da tarefa. Essa variante limitada foi identificada como
`disjoint_budgeted_permuted_mnist`; ela diferiu do protocolo convencional que
repete todo o conjunto-base em cada tarefa. Não houve multiplicação oculta dos
100 exemplos por quantidade de tarefas. As duas épocas de treino permaneceram
responsabilidade do avaliador, com exemplos únicos e apresentações separados.

## Contratos de falha e validação

Foram rejeitados headers IDX inconsistentes, rótulos inválidos, registros
truncados, symlinks em membros tar, tamanho e SHA-256 divergentes, redirects
inseguros e URLs com credenciais. Downloads foram gravados em arquivos temporários
e promovidos ao cache somente após a validação completa. HTTP negado e hash
inválido não acionaram retry automático. Cache corrompido não foi sobrescrito
silenciosamente. Foram aplicados lock por asset, limite de bytes e deadline.

Os testes entregues cobriram fixtures binárias reais de formato, limites,
split/ID/classe/permutação, corrupção, falhas de rede e idempotência de cache.
Os testes iniciais foram executados somente no homelab. Os resultados de
aquisição e treinamento ficaram no [relatório de execução](task-dataset-execution.md).

## Fonte CIFAR Parquet explicitamente selecionada

`prepare_cifar_parquet(name, cache, limit=100, seed=42, *, download=False)`
é o loader de CIFAR usado pelo registro executável na preparação por épocas.
A escolha foi feita no primeiro smoke, após timeout de transferência no
servidor de Toronto. Os repositórios públicos
[uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10) e
[uoft-cs/cifar100](https://huggingface.co/datasets/uoft-cs/cifar100) identificaram a
origem CIFAR, classes e splits oficiais em seus próprios dataset cards. A licença
foi registrada como `unknown`, conforme os cards, sem inferir uma licença nova.

O commit do mirror CIFAR10 foi fixado em
`0b2714987fa478483af9968de7c934580d0bb9a2`; o CIFAR100 em
`aadb3af77e9048adbea6b47c21a81e47dd092ae5`. Os arquivos train/test, tamanhos e
SHA-256 LFS publicados foram obtidos pela API pública do Hub e fixados no código.
Esses hashes completos permaneceram **referências não verificadas** na leitura
parcial; a implementação não afirmou ter validado o arquivo Parquet inteiro.

`RangeReader` buscou apenas ranges solicitados pelo leitor Parquet, com orçamento
compartilhado de 64 MiB entre treino e teste. Foram conferidos tamanho total,
validador HTTP, SHA-256 de cada range, cardinalidade declarada, schema, classes e
SHA-256 de cada imagem selecionada. URLs de CDN assinadas não foram persistidas;
a proveniência guardou somente a URL pública canônica com o commit fixado. Os
ranges não foram interpretados como ZIP nem receberam alegação de validação CRC
ZIP; o manifesto informou a integridade efetivamente verificada.

Foram decodificadas somente as primeiras 80 imagens de treino e 20 de teste do
mirror para o smoke de 100 originais, com RGB 32 por 32 e ordem de canais
planar CHW, igual ao loader binário. Não houve resize ou fitting. IDs refletiram
repositório, commit, split e índice da linha; a seleção foi declarada como
prefixo do mirror, sem alegar igualdade de bytes com o arquivo tar de Toronto.
Cache preparado completo pôde ser relido sem rede; corrupção foi rejeitada sem
reparo automático. O leitor não buscou URLs externas encontradas em campos de
imagem: os bytes precisaram estar embutidos no Parquet.

O benchmark iterativo usa uma representação compacta separada. O loader
`load_iterative` prepara 10.000 imagens por fonte em uma matriz de pixels
`uint8` e converte somente a linha em uso em features normalizadas.
