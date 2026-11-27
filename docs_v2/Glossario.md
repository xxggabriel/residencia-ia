# Glossário

Termos técnicos usados nesta Wiki, em ordem alfabética. Nomes entre crases correspondem a identificadores do código ou dos arquivos de evidência.

| Termo | Definição |
|---|---|
| ADWIN | Detector de drift com janela adaptativa: mantém uma janela de valores recentes e a corta quando duas subjanelas têm médias estatisticamente diferentes. Ver [Detectores de drift](Detectores-de-Drift). |
| Alocação (rateio) | Divisão de custos compartilhados entre execuções proporcionalmente aos vCPU-segundos reservados. Não é faturamento observado. |
| Aquecimento | Treino inicial do modelo em um prefixo do fluxo, fora do placar de avaliação. |
| Atraso de rótulo | Intervalo entre o momento da predição e o momento em que o rótulo correspondente fica disponível para treino. Controlado pelo campo `available_at`. |
| Backfill | Completar o registro de execuções já concluídas (métricas, figuras, histórico de treino) sem treinar novamente. |
| Bloco | Fatia contígua de linhas que o avaliador por blocos prevê por inteiro antes de liberar os rótulos. |
| CARA | Política de retreino sensível a custo que compara a obsolescência estimada do modelo com o custo de retreinar. Variantes `cara`, `cara_cumulative` e `cara_periodic`. Ver [Modelos e políticas](Modelos-e-Politicas). |
| Checkpoint | Estado salvo de uma execução que permite retomá-la após interrupção. Localmente é gravado de forma atômica; na AWS fica no S3. |
| Checkout esparso | Recurso do Git que baixa apenas parte do repositório. O CI o usa para excluir `evidencias/`. |
| Coorte comum | Mesma faixa de eventos usada por todas as políticas e seeds de um stream, para que as métricas tenham o mesmo denominador. Na campanha AWS, é a faixa `common_postcalibration`. |
| Contrato de retomada | Conjunto de campos que precisam coincidir para que um resultado existente seja retomado ou reutilizado em vez de recalculado. |
| Custo de cenário | Contabilidade parametrizada, em `scenario_unit`, que atribui pesos a predições, rótulos e retreinos. Não representa dinheiro. Ver [Custos e recursos](Custos-e-Recursos). |
| CUR 2.0 | Cost and Usage Report da AWS, a única fonte de faturamento observado prevista pelo projeto. |
| DDM | Detector de drift que monitora a taxa de erro e seu desvio-padrão, emitindo aviso e alarme quando ela sobe acima de limiares. |
| Drift | Mudança, ao longo do tempo, na distribuição dos dados ou na relação entre atributos e rótulo. Também chamado de mudança de conceito. |
| Early stopping | Interrupção do treino por épocas quando a métrica de validação deixa de melhorar por um número de épocas chamado paciência. |
| ECE | Erro esperado de calibração: diferença média entre a probabilidade prevista e a frequência observada da classe positiva, por faixas de probabilidade. |
| Estimativa de infraestrutura | Preço de tabela multiplicado pela duração e pelos recursos reservados. Não é faturamento. |
| Extra | Grupo opcional de dependências do `pyproject.toml`, instalado com `--extra <nome>`. |
| F109 | Rótulo usado nos arquivos de evidência para a campanha AWS de 4 streams, 11 políticas e 3 seeds. |
| Fargate Spot | Capacidade do AWS Fargate com desconto que a AWS pode interromper com aviso prévio (sinal SIGTERM). |
| Fronteira anotada | Índice em que a fonte declara o início de um novo regime. Usada apenas para avaliação e auditoria, nunca para treino. |
| Hash semântico | SHA-256 calculado sobre as linhas decodificadas de uma tabela, independente da codificação do arquivo CSV. |
| Holdout temporal | Parte final do fluxo reservada para avaliação, nunca usada para treino nem para decisões de política. |
| `learn_one` | Atualização incremental do modelo com um único exemplo cujo rótulo foi liberado. |
| Lockfile | Arquivo que fixa as versões exatas (e, em `requirements*.lock`, os hashes) de todas as dependências. |
| Manifesto | Arquivo JSON que lista os arquivos de um pacote de dados ou evidência com contagens e hashes. |
| Marcador (pytest) | Rótulo aplicado a testes para incluí-los ou excluí-los de uma execução, como `services`. |
| Matriz de desempenho | Tabela `err[i, j]` com o erro do modelo treinado no lote `i` avaliado no lote `j`, usada por CARA, UPF e Oracle. |
| `metrics_only` | Modo que publica métricas sem guardar pesos de modelo nem checkpoints. |
| OIDC | Protocolo usado pelo GitHub Actions para obter credenciais temporárias da AWS sem chaves de longa duração. |
| Oracle | Cronograma de retreinos ótimo calculado retrospectivamente, com acesso ao futuro. Serve como referência, não como política implantável. |
| Política de atualização | Regra que decide quando e como atualizar o modelo: nunca, periodicamente, por alarme, por desempenho, incrementalmente ou por critério de custo. |
| Replay | Janela de exemplos recentes com rótulo disponível usada para retreinar o modelo. |
| `reset_retrain` | Retreino do zero, recriando modelo e pré-processamento, sobre a janela de replay. |
| RSS | Memória residente de um processo, medida pelo sistema operacional. |
| `run_key` | Identificador de execução formado pelos primeiros 24 caracteres hexadecimais do SHA-256 da configuração. |
| Seed | Semente do gerador de números aleatórios. Repetições com seeds diferentes indicam a variabilidade computacional de um resultado. |
| Snapshot de dataset | Versão preparada de um dataset, considerada pronta quando o marcador `_SUCCESS.json` existe. |
| Testar e depois treinar | Protocolo (em inglês, *prequential*) em que cada exemplo é usado primeiro para testar o modelo e depois, quando o rótulo é liberado, para treino. |
| Unidade | Uma combinação do plano de campanha (stream, política, seed), executada como um job Batch na AWS. |
| UPF | Política de retreino baseada em previsão do desempenho futuro dos modelos candidatos e no custo relativo de retreinar (`α`). Ver [Modelos e políticas](Modelos-e-Politicas). |
