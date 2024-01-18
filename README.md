# Projeto IA para Detecção de Defeitos em Superfícies Metálicas

Este projeto tem como objetivo a implementação de um modelo de aprendizado profundo e um modelo de aprendizado de máquina (RandomForest) para a detecção de defeitos em superfícies metálicas. O conjunto de dados utilizado é o "NEU Metal Surface Defects Data".

## Pré-requisitos

Certifique-se de rodar o requirements.txt para instalar as dependências:

```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

O projeto é composto pelos seguintes arquivos:

1. `src/TF_treinador.py`: Contém o código-fonte para treinar o modelo usando Tensorflow.
2. `src/TF_testador.py`: Contém o código-fonte para testar o modelo usando Tensorflow.
3. `src/SL_gera_hist.py`: Contém o código-fonte gerar os histogramas necessários para treinar o modelo usando Scikit-learn.
4. `src/SL_treinador.py`: Contém o código-fonte para treinar o modelo usando Scikit-learn e o método de automato celular.
5. `data`: Contém o conjunto de dados usados no modelo.

## Utilização - Tensorflow (Deep Learning)

### Treinamento do Modelo

Antes de executar o treinamento, ajuste o diretório `data_dir` no código para o caminho correto no seu sistema de arquivo.

```bash
python src/TF_treinador.py
```

Este script realiza o treinamento do modelo usando o conjunto de treinamento e valida os resultados usando o conjunto de validação. O modelo treinado é salvo como `modelo_TF.keras`.

### Avaliação do Modelo

A avaliação do modelo é realizada executando o script `TF_testador.py`. O script carrega o modelo previamente treinado (`modelo_TF.keras`) e avalia seu desempenho no conjunto de teste.

```bash
python src/testador.py
```

### Visualização de Resultados

O script também inclui a geração de gráficos para visualizar o desempenho do modelo ao longo do treinamento, como precisão (`accuracy`,`val_accuracy` )  e perda (`loss`, `val_loss`).

## Utilização - Scikit-learn (RandomForest)

### Geração dos histogramas

Antes de executar o treinamento, é necessário gerar os histogramas da imagem através do método descrito no artigo "Cellular Automaton based descriptor for pixel classification". Para isso, ajuste o diretório `data_dir` no script `SL_gera_hist.py`.

```bash
python src/SL_gera_hist.py
```

Este script gera 4 histogramas para uma imagem, em seguida tais histogramas são concatenados e salvos no diretório `data/histograms`.

### Treinamento e Avaliação do Modelo

O treinamento e a avaliação do modelo são realizados executando o script `SL_treinador.py`. O script carrega os histogramas previamente gerados e avalia seu desempenho no conjunto de teste definido pela variável `test_size`.

Lembre-se de também ajustar o diretório `data_dir` para este script.

```bash
python src/SL_treinador.py
```

O modelo treinado é salvo como `modelo_SL.joblib`.
Os resultados dos testes são mostrados no final da execução deste script.

## Resultados da Inferência

Ao final da execução, os scripts geram uma visualização dos resultados da inferência através de uma matriz de confusão e outras métricas.

Certifique-se de ajustar os diretórios `test_dir` e `model_path` no script para os caminhos corretos no seu sistema de arquivos.

### Resultados obtidos

#### Tensorflow
Matriz de Confusão:
[[27  1  0  0  2  0]
 [ 0 30  0  0  0  0]
 [ 2  0 25  2  1  0]
 [ 0  0  1 29  0  0]
 [ 0  1  0  0 29  0]
 [ 0  6  0  0  0 24]]

Relatório de Classificação:
              precision    recall  f1-score   support

   Arranhoes       0.93      0.90      0.92        30
Craqueamento       0.79      1.00      0.88        30
  Depressoes       0.96      0.83      0.89        30
    Inclusao       0.94      0.97      0.95        30
   Laminacao       0.91      0.97      0.94        30
     Manchas       1.00      0.80      0.89        30

    accuracy                           0.91       180


#### Scikit-learn
Matriz de Confusão:
[[63  0  1  4  0  0]
 [ 0 62  0  0  1  3]
 [ 0  0 58  1  0  0]
 [ 0  0  1 47  0  0]
 [ 0  1  0  0 65  0]
 [ 0  0  0  0  0 53]]

Relatório de Classificação:
              precision    recall  f1-score   support

   Arranhoes       1.00      0.93      0.96        68
Craqueamento       0.98      0.94      0.96        66
  Depressoes       0.97      0.98      0.97        59
    Inclusao       0.90      0.98      0.94        48
   Laminacao       0.98      0.98      0.98        66
     Manchas       0.95      1.00      0.97        53

    accuracy                           0.97       360


## Considerações Finais

Este README fornece uma visão geral do projeto e como utilizar os códigos fornecidos.
Esse projeto foi realizado para a disciplina de Inteligência Artificial.
