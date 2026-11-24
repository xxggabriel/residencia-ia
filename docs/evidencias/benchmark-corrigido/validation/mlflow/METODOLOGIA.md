# Verificação das métricas e evidências publicadas

A conferência abrangeu os 60 casos da matriz e os oito jobs de replicação, cada um com três seeds. Os recibos identificaram 108 execuções principais e 1.032 treinos associados. Todos estavam concluídos e foram vinculados aos respectivos relatórios, parâmetros e identificadores de treino.

Foram comparados 20.912.990 registros nativos de métricas, por chave, etapa e valor. Não foram encontrados pontos ausentes, coordenadas duplicadas ou valores divergentes. A maior série apresentou 450.852 pontos; 2.304 séries ultrapassaram 250 pontos. A comparação nativa considera valores numéricos finitos. Valores indisponíveis permanecem nos JSONs originais, cuja integridade foi conferida por hash.

Os 108 ZIPs publicados foram baixados e verificados integralmente. A conferência incluiu o hash e o tamanho do arquivo, o manifesto e o hash e o tamanho de cada um dos 10.968 membros. Os arquivos compactados somaram 4.515.195.612 bytes. Não foram encontrados membros proibidos de modelo, checkpoint ou arquivos brutos identificados pelo contrato de retenção. Também não foram encontrados arquivos de estado de modelo nos 108 diretórios de resultados, nem objetos de artefatos nos treinos filhos.

Os 60 pontos derivados de tempo foram confrontados com o resultado gravado de cada caso, sua origem, hash e instante de término. O intervalo corresponde ao corpo de execução registrado, incluindo materialização, avaliação, diagnósticos e publicação. Um caso da matriz representa uma seed; um job de replicação representa suas três seeds. A duração total de um job não foi dividida por três nem atribuída individualmente a cada trial. O campo legado de tempo não foi preenchido por esta etapa.

As consultas foram executadas apenas em uma cópia privada do banco, após a confirmação do término das execuções e da publicação dos tempos. A cópia foi criada pela API de backup, em blocos de 128 páginas, com fonte somente leitura e sem transação BEGIN explícita. Todas as conexões foram fechadas antes dos downloads. Após a aprovação integral, a cópia privada foi excluída. Não foram exportados banco de dados ou credenciais.

O arquivo `native-mlflow-verification-summary.json` contém os totais; `final98-native-mlflow-audit.json` apresenta as contagens por chave e os resultados de cada execução, treino e ZIP. O manifesto registra SHA-256 e tamanho de todos os arquivos desta evidência. O resumo de encerramentos das execuções interrompidas refere-se apenas aos nove registros próprios ainda abertos, preservando os concluídos, compartilhados ou ambíguos.
