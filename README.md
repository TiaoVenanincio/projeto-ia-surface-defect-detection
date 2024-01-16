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

1. `src/treinador.py`: Contém o código-fonte para treinar o modelo.
2. `src/testador.py`: Contém o código-fonte para testar o modelo.
3. `data`: Contém o conjunto de dados usados no modelo.

## Utilização

### Treinamento do Modelo

Antes de executar o treinamento, ajuste os diretórios `train_dir`, `test_dir`, e `valid_dir` no código para os caminhos corretos no seu sistema de arquivos.

```bash
python treinador.py
```

Este script realiza o treinamento do modelo usando o conjunto de treinamento e valida os resultados usando o conjunto de validação. O modelo treinado é salvo como `modelo_TF.h5`.

### Avaliação do Modelo

A avaliação do modelo é realizada executando o script `testador.py`. O script carrega o modelo previamente treinado (`modelo_TF.h5`) e avalia seu desempenho no conjunto de teste.

```bash
python testador.py
```

### Visualização de Resultados

O script também inclui a geração de gráficos para visualizar o desempenho do modelo ao longo do treinamento, como precisão (`accuracy`,`val_accuracy` )  e perda (`loss`, `val_loss`).

## Resultados da Inferência

Ao final da execução, o script gera uma visualização dos resultados da inferência em uma amostra do conjunto de teste, destacando as previsões corretas e incorretas.

Certifique-se de ajustar os diretórios `test_dir` e `model_path` no script para os caminhos corretos no seu sistema de arquivos.

## Considerações Finais

Este README fornece uma visão geral do projeto e como utilizar os códigos fornecidos.
Esse projeto foi realizado para a disciplina de Inteligência Artificial.
