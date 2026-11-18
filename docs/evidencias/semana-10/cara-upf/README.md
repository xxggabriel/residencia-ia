# Auditoria dos benchmarks CARA e UPF

Registrei aqui os resultados de Electricity e Airlines usados no relatório da Semana 10. Os arquivos JSON comprimidos preservam as métricas e os registros completos produzidos pelos benchmarks em 17 de novembro de 2026, incluindo resumos, grades de custo, resultados por ensaio quando exportados, histórico de treino, candidatos e observações. Os arquivos de entrada são identificados pelos hashes e pelos nomes registrados nos resultados.

| Arquivo | Conteúdo |
| --- | --- |
| [upf.json.gz](dados/upf.json.gz) | Protocolo UPF em Electricity e Airlines, dez seeds por conjunto |
| [cara-electricity.json.gz](dados/cara-electricity.json.gz) e [cara-airlines.json.gz](dados/cara-airlines.json.gz) | Protocolo CARA, cinco seeds por conjunto |
| [same-batch.json.gz](dados/same-batch.json.gz) e [next-batch.json.gz](dados/next-batch.json.gz) | Comparação pareada local de CARA e UPF, dez seeds por conjunto e modo |
| [manifesto-execucao.json](dados/manifesto-execucao.json) | Ambiente, configuração, versões e hashes dos dados de entrada dos protocolos dos artigos |
| [trajetorias.json](dados/trajetorias.json) | Matriz de acurácia dos candidatos, tamanhos do teste e retreinamentos de Electricity e Airlines na seed 0 |
| [SHA256SUMS](dados/SHA256SUMS) | Somas de verificação dos arquivos de auditoria |

## Figuras

- [Trajetória do UPF em Electricity](figuras/trajetoria-electricity.png) e [trajetória do CARA-T em Airlines](figuras/trajetoria-airlines.png)
- [AUC do UPF local e publicado](figuras/auc-upf-publicado-local.png)
- [Métricas do protocolo UPF](figuras/metricas-protocolo-upf.png)
- [Métricas do protocolo CARA](figuras/metricas-protocolo-cara.png)
- [Excesso de custo em relação ao Oracle de erro](figuras/sensibilidade-oracle.png)

Nas trajetórias, cada linha colorida mostra a acurácia de um modelo candidato nos lotes seguintes. A linha preta mostra quais modelos serviram o fluxo sob a política indicada, com α de 0,1 e implantação no próprio lote. As bandas de Wilson de 95% correspondem aos 400 exemplos do teste de cada lote. A trajetória usa uma única seed. Comparei os hashes dos 15 modelos por dataset e recalculei a acurácia média e o custo para conferir a reconstrução.

## Reprodução

Com Python 3.12 na raiz do repositório, instale o projeto com o extra `plots` e execute o script.

```bash
uv sync --extra plots
uv run --extra plots python scripts/plot_week10_cara_upf.py
(cd docs/evidencias/semana-10/cara-upf/dados && shasum -a 256 -c SHA256SUMS)
```

Para reconstruir a matriz de acurácia a partir dos dados públicos de entrada, use os arquivos com os hashes da seção `comparison.source` em `same-batch.json.gz` e instale também o extra `large-datasets`.

```bash
uv run --extra plots --extra large-datasets python scripts/plot_week10_cara_upf.py \
  --electricity /caminho/electricity-normalized.arff \
  --airlines /caminho/airlines.arff
```

A tabela de AUC compara os números usando o protocolo UPF. A sensibilidade ao custo usa as duas variantes do protocolo pareado. Os valores de α e κ são parâmetros dos respectivos objetivos, em unidades de custo do cenário.

- [Mahadevan e Mathioudakis, CARA, 2024](https://doi.org/10.1016/j.knosys.2024.111610)
- [Regol et al., UPF, 2025](https://arxiv.org/abs/2505.14903v1)
