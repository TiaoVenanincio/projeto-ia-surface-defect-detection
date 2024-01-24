import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Resultados com TensorFlow
labels_tf = ['Arranhoes', 'Craqueamento', 'Depressoes', 'Inclusao', 'Laminacao', 'Manchas']
conf_matrix_tf = [[27, 1, 0, 0, 2, 0],
                  [0, 30, 0, 0, 0, 0],
                  [2, 0, 25, 2, 1, 0],
                  [0, 0, 1, 29, 0, 0],
                  [0, 1, 0, 0, 29, 0],
                  [0, 6, 0, 0, 0, 24]]

# Resultados com Scikit-learn
labels_sklearn = ['Arranhoes', 'Craqueamento', 'Depressoes', 'Inclusao', 'Laminacao', 'Manchas']
conf_matrix_sklearn = [[23, 0, 1, 2, 0, 0],
                       [0, 43, 0, 0, 0, 0],
                       [0, 0, 32, 1, 0, 0],
                       [0, 0, 0, 20, 0, 0],
                       [0, 0, 0, 0, 28, 0],
                       [0, 0, 0, 0, 0, 30]]

# Função para plotar matriz de confusão
# Função para plotar matriz de confusão e salvar como PNG
def plot_confusion_matrix(labels, conf_matrix, title, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
    else:
        plt.show()

# Especificar caminhos para salvar os arquivos PNG
save_path_tf = 'matriz_confusao_tensorflow.png'
save_path_sklearn = 'matriz_confusao_scikit-learn.png'

# Plotar matrizes de confusão e salvar como PNG
plot_confusion_matrix(labels_tf, conf_matrix_tf, 'Matriz de Confusão - TensorFlow', save_path_tf)
plot_confusion_matrix(labels_sklearn, conf_matrix_sklearn, 'Matriz de Confusão - Scikit-learn', save_path_sklearn)
