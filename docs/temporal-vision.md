# Loaders de imagens temporais da Semana 09

Foi implementado `prepare_temporal_dataset(name, cache, limit=100, seed=42, download=False)`, retornando uma fonte com `samples_iter()` e `manifest()`. Foram cobertos os nomes `core50`, `stream51`, `clear10` e `clear100`, com imagens e rótulos originais. Não foram usados exemplos sintéticos para simular essas bases.

## Contrato de aquisição

A aquisição foi explicitamente habilitada por `download=True`; a execução normal consumiu somente cache já admitido. Ausência ou corrupção do cache falhou sem download automático. Também foi aceita a opção `local_archives={"train": caminho, "test": caminho}` para arquivos oficiais já disponíveis, sem alterar credenciais ou políticas de acesso.

Foi utilizado acesso aleatório HTTP Range aos ZIPs, com diretório central, metadados e apenas as imagens selecionadas. O transporte exigiu HTTP 206 e Content-Range exato; respostas 200 não acionaram fallback de download integral. A API não desabilitou TLS, não aceitou termos novos e não tentou contornar HTTP 401/403. Falhas transitórias 429/5xx/transporte receberam até três tentativas com espera limitada. Os limites por arquivo foram 64 MiB lidos, 2048 requisições, 64 MiB descomprimidos por metadado e 8 MiB por imagem. A leitura usou blocos de 64 KiB para evitar três viagens de rede por cabeçalho/membro pequeno.

Os ZIPs não foram extraídos no filesystem nem executados via pickle. O leitor conferiu CRC dos membros pelo `zipfile`, SHA-256 dos ranges e membros selecionados, tamanho e ETag/Last-Modified estáveis durante a leitura. Os checksums completos MD5 publicados para CORe50/Stream51 foram preservados como referências, mas **não foram declarados verificados**, pois o recorte remoto não transferiu os arquivos completos. Os arquivos CLEAR ficaram fixados na revisão Hugging Face `b518a845a98f1c913497ab98a19727cb65b74e65` indicada pelo espelho vinculado pelo projeto original.

## Recorte e representação

O default selecionou exatamente **100 imagens únicas**, divididas em 80 de treino e 20 do teste oficial. O número foi um limite global por dataset, e não 100 por classe, tarefa ou época. As mesmas imagens puderam ser reapresentadas em duas épocas pelo avaliador de treinamento; o loader não duplicou registros para fingir épocas.

A seleção utilizou stride determinístico preservando a ordem dentro de cada split. Em CLEAR, foi limitado também o número de arquivos de metadados consultados, cobrindo buckets/classes sem percorrer as imagens completas. Os IDs derivaram de dataset, release, caminho original e hash da imagem. Split, tarefa, grupo, posição original, bbox quando aplicável e imagem SHA-256 ficaram no manifesto. O relógio de replay foi ordinal e explicitamente artificial, sem fingir datas reais para as features selecionadas.

A representação foi RGB redimensionado por interpolação bilinear para 16×16, normalizado por divisão por 255, sem ajuste no teste, extração aprendida ou pré-treinamento. Resultaram 768 features numéricas. Isso viabilizou testes CPU de ingestão/treino em recortes pequenos; não reproduziu as redes ou resultados publicados. Stream51 utilizou o bbox oficial sem a expansão de 10% do baseline original, com esse desvio explícito.

## Fontes e cenários

| Dataset | Fonte original e versão | Regra implementada | Limite |
| --- | --- | --- | --- |
| CORe50 | [Autores](https://vlomonaco.github.io/core50/), arquivo 128×128 original, referência MD5 `745f3373fed08d69343f1058ee559e13` | Classe por objeto; treino em sessões 1,2,4,5,6,8,9,11 e teste nas sessões oficiais 3,7,10; ordem sessão/objeto/frame | Subconjunto NI diagnóstico; não um run NI/NC/NIC completo dos filelists |
| Stream51 | [Autores e schema](https://github.com/tyler-hayes/Stream-51), CVPRW 2020, referência MD5 `5f34d542b71af7e5ecb2226d1bc7297c` | Leitura dos JSONs oficiais de treino/teste, alvos originais, sequência publicada e bbox | Não calculou novelty detection nem substituiu a ordem por um dataset distinto |
| CLEAR10 | [Projeto original](https://clear-benchmark.github.io/), release julho de 2022, revisão do espelho fixada | Buckets e classes dos metadados originais, incluindo BACKGROUND quando presente, arquivos de treino e teste separados | A edição contém dez classes de interesse mais BACKGROUND; não foi silenciosamente reduzida a dez alvos |
| CLEAR100 | Mesma release oficial de 2022 e revisão fixada | Classes, buckets e splits originais | O smoke não sustentou avaliação de todas as classes ou comparação de retenção |

CLEAR utilizou os splits oficiais da release de 2022; **não foi apresentado como o protocolo streaming de treinar no bucket atual e testar exclusivamente no próximo** do artigo. O loader registrou esse limite em `manifest.protocol`. As funções de avaliação CL mais amplas continuaram exigindo protocolo específico.

A página CORe50 declarou CC BY 4.0. O projeto CLEAR declarou CC BY e as imagens YFCC conservaram seus direitos individuais. O repositório Stream51 apresentou licença MIT para o código, o que não estabeleceu licença única dos frames derivados de GOT-10K, VID e LaSOT. As referências foram preservadas; nenhum aceite foi feito em nome do usuário.

Implementações primárias consultadas para o formato:

- [CORe50 no Avalanche](https://github.com/ContinualAI/avalanche/blob/master/avalanche/benchmarks/datasets/core50/core50_data.py).
- [CLEAR metadata no Avalanche](https://github.com/ContinualAI/avalanche/blob/master/avalanche/benchmarks/datasets/clear/clear.py).
- [Stream51 original](https://github.com/tyler-hayes/Stream-51/blob/master/StreamDataset.py).

## Verificação no homelab e bloqueio de origem

Todas as execuções iniciais ocorreram em `gabriel@192.168.100.27`, via SSH com autenticação já disponível e `StrictHostKeyChecking=yes`. Não foram executados testes no Mac.

Os probes HTTP/ZIP reais confirmaram suporte a ranges em CORe50 e CLEAR. A listagem CORe50 incluiu 165428 membros e transferiu 18202869 bytes de diretório/metadados; CLEAR10 teve 33245 membros de treino com 4205433 bytes; CLEAR100 teve 112155 membros de treino com 14222299 bytes. As estruturas reais foram inspecionadas: CLEAR treino usou prefixo `train_image_only/`, enquanto teste usou `test/`.

O domínio oficial `klab.cis.rit.edu` de Stream51 retornou falha de DNS no homelab (`Name or service not known`), e a página de projeto apontada pelos autores retornou 404 no leitor web. A pesquisa na API pública Hugging Face não encontrou repositório para Stream51/Stream-51. O adaptador de arquivo oficial foi implementado e testado por contrato, mas esse resultado **não homologou 100 imagens originais Stream51**. A execução com dados reais permanece bloqueada até haver arquivo oficial local autorizado ou endpoint oficial funcional. Não se substituiu a base por outra, por screenshots do paper ou por fixtures.

Sete testes do módulo foram aprovados no homelab em 1,96 s na primeira rodada: ranges e cache, corrupção, limites, negação sem retry, splits CORe50, JSON/bbox Stream51, buckets/classes CLEAR10/CLEAR100 e cache-only. Ruff foi aprovado após os ajustes de linhas longas. CORe50, CLEAR10 e CLEAR100 foram posteriormente aprovados com 100 imagens reais e duas épocas, conforme o [relatório de execução](semana09-dataset-execution.md); Stream51 permaneceu bloqueado.
