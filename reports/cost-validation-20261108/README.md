# Validação de custos no homelab — 08/11/2026

Foram executados 20 testes em `gabriel@192.168.100.27`; todos passaram em 643,86 segundos.
O cliente Docker usou o código montado em modo de leitura e os serviços Kafka, MinIO, MLflow e ZenML existentes. As cargas foram sintéticas e limitadas; esta execução validou a integração, sem estabelecer resultados de benchmark.

| Fluxo | Casos | Verificação |
|---|---:|---|
| Principal | 6 | Streaming, incremental e offline; cenário padrão e tempo precificado; componentes no MLflow, artefatos MinIO por SHA-256, replay sem duplicação |
| Por épocas | 2 | Duas épocas; cenário padrão e tempo precificado; componentes por época e agregado no MLflow, artefatos MinIO por SHA-256 e replay |
| Tipado | 12 | Classificação e regressão × none/online/periodic × cenário padrão/tempo precificado; fórmula de erro, apresentações de treino, tempo, relatório local e replay |

O fluxo tipado não possui publisher MLflow; seus relatórios locais foram conferidos. Os demais fluxos tiveram os componentes publicados comparados numericamente com os relatórios.

As evidências detalhadas, coeficientes e run IDs ficaram em [evidence.json](evidence.json); o resultado da suíte ficou em [cost-services.xml](cost-services.xml). O pacote enviado teve SHA-256 `94828ed264ea92b6d94be96c9733df8e3d8d28e5c249a2bbb1e251623ee7e5dc`. O estado remoto permaneceu em `/home/gabriel/cost-validation-20261108/state` e o log em `/home/gabriel/cost-validation-20261108/services.log`.

## Runs publicadas

- [main / streaming / scenario_unit](http://192.168.100.27:15000/#/experiments/11/runs/6a19377f2cec4670b7b1af155cf61937): custo total `12.464`.
- [main / incremental / custom_scenario](http://192.168.100.27:15000/#/experiments/12/runs/5bf28d84eea340d88e2b1eb258c0ac6b): custo total `42.74666306899931`.
- [main / incremental / scenario_unit](http://192.168.100.27:15000/#/experiments/12/runs/5e10870a52724e5ba812d925066ab60c): custo total `35.666`.
- [main / streaming / custom_scenario](http://192.168.100.27:15000/#/experiments/11/runs/f5cfa8c827f340639aa39d7f411c86df): custo total `17.801438723000075`.
- [main / offline / scenario_unit](http://192.168.100.27:15000/#/experiments/3/runs/369cf571af4d4f769a19140423be30c6): custo total `25.069`.
- [main / offline / custom_scenario](http://192.168.100.27:15000/#/experiments/3/runs/0a9ab95adfd24e65bfca1755ae56f003): custo total `26.385219204000126`.
- [epochs / duas épocas / scenario_unit](http://192.168.100.27:15000/#/experiments/13/runs/837e19ced1d04c87b6e62075c881ac9d): custo total `6.254`.
- [epochs / duas épocas / scenario_unit](http://192.168.100.27:15000/#/experiments/13/runs/9407bc758fc54d5b81ed2976b3da2714): custo total `9.70223135800034`.
