import numpy as np 
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

def exibe_9_exemplos(images, labels, model1, class_names):
    #O objetivo dessa função é exibir 9 imagens aleatórias e mostrar qual foi a predição
    # e a classe verdadeira.

    # Escolhe 9 imagens aleatórias, carregando sua classe verdadeira para compará-la com a predição.
    indices = np.random.choice(range(len(images)), size=9)
    images = images[indices]
    labels = labels[indices]
    predictions = model1.predict(images)
    

    plt.figure(figsize=(12, 12))

    # Define o espaçamento entre os subplots
    plt.subplots_adjust(wspace=0.8, hspace=0.4)

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        image = images[i]

        if image.shape[-1] == 1:
            image = np.squeeze(image)

        plt.imshow(image)

        predicted_label = np.argmax(predictions[i])

        if predicted_label == np.argmax(labels[i]):
            color = 'blue'
            result_text = "Correto"
        else:
            color = 'red'
            result_text = "Incorreto"

        label_text = "True: " + class_names[np.argmax(labels[i])] + ", Pred: " + class_names[predicted_label] + f" ({result_text})"
        plt.xlabel(label_text, color=color)

    plt.show()


# Carrega o modelo treinado e o lote de testes
model_path = r"C:\Users\Sebastiao\Desktop\Projetos\projeto-ia-surface-defect\modelo_TF.keras"
model1 = keras.models.load_model(model_path)

test_dir = r"C:\Users\Sebastiao\Desktop\Projetos\projeto-ia-surface-defect\data\NEU Metal Surface Defects Data\test"
test_datagen = ImageDataGenerator(rescale=1./255)

# Configuração do gerador de imagens para o conjunto de testes (test_datagen)
test_generator = test_datagen.flow_from_directory(test_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)

images, labels = next(test_generator)    

class_names=list(test_generator.class_indices.keys())

#Função para exibir 9 imagens como exemplos
exibe_9_exemplos(images, labels, model1, class_names)

# Obtem todas as previsões para o conjunto de testes
all_predictions = model1.predict(test_generator)

# Converte as previsões em rótulos preditos
predicted_labels = np.argmax(all_predictions, axis=1)

# Converte as labels reais em rótulos reais
true_labels = test_generator.classes

# Cria a matriz de confusão
confusion_mat = confusion_matrix(true_labels, predicted_labels)

# Exibe a matriz de confusão
print("Matriz de Confusão:")
print(confusion_mat)

# Gera um relatório de classificação
class_report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("\nRelatório de Classificação:")
print(class_report)