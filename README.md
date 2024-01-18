# Projeto IA para Detecção de Defeitos em Superfícies Metálicas

Este projeto tem como objetivo a implementação de um modelo de aprendizado profundo (deep learning) para a detecção de defeitos em superfícies metálicas. O conjunto de dados utilizado é o "NEU Metal Surface Defects Data".

(O projeto ainda está em fase de implementação. Atualmente estamos trabalhando em outro método que utiliza a biblioteca Scikit-learn, o qual ainda está sendo testado.)

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

## Utilização - Tensorflow

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

## Utilização - Scikit-learn

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

## Resultados da Inferência

Ao final da execução, os scripts geram uma visualização dos resultados da inferência através de uma matriz de confusão e outras métricas.

Certifique-se de ajustar os diretórios `test_dir` e `model_path` no script para os caminhos corretos no seu sistema de arquivos.

## Considerações Finais

Este README fornece uma visão geral do projeto e como utilizar os códigos fornecidos.
Esse projeto foi realizado para a disciplina de Inteligência Artificial.
