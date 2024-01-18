import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def plot_history(history,metric):
    #Função para plotar o histórico de treinamento de uma rede neural.
    plt.plot(history.history[metric])
    plt.plot(history.history["val_"+metric],"")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_"+metric])
    plt.show()

#Diretório dos dados
data_dir  = r"C:\Users\Sebastiao\Desktop\Projetos\projeto-ia-surface-defect\data"
train_dir = f"{data_dir}/NEU Metal Surface Defects Data/train"
test_dir = f"{data_dir}/NEU Metal Surface Defects Data/test"
valid_dir = f"{data_dir}/NEU Metal Surface Defects Data/valid"

# Configuração do gerador de imagens para o conjunto de treinamento (train_datagen)
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Reescala os valores dos pixels para o intervalo [0,1]
    rotation_range=20,           # Faixa de rotação aleatória das imagens em graus
    width_shift_range=0.1,       # Faixa de deslocamento horizontal aleatório das imagens
    height_shift_range=0.1,      # Faixa de deslocamento vertical aleatório das imagens
    horizontal_flip=True         # Habilita ou desabilita inversão horizontal aleatória das imagens
)

# Configuração do gerador de imagens para o conjunto de teste (test_datagen)
test_datagen = ImageDataGenerator(rescale=1./255)  # Reescala os valores dos pixels para o intervalo [0,1]


#Carregando e pré-processando os lotes de imagens
train_generator = train_datagen.flow_from_directory(train_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=True)
valid_generator = test_datagen.flow_from_directory(valid_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)
test_generator = test_datagen.flow_from_directory(test_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)

class_names = train_generator.class_indices
class_names = list(class_names.keys())

# Definição da arquitetura do modelo Sequential
model1 = Sequential([
    # Primeira camada convolucional com 32 filtros de tamanho (2,2), função de ativação ReLU e entrada de shape (200, 200, 3)
    Conv2D(32, (2, 2), activation='relu', input_shape=(200, 200, 3)),
    # Camada de MaxPooling para redução espacial, usando janela (2,2)
    MaxPooling2D((2, 2)),
    
    # Segunda camada convolucional com 64 filtros de tamanho (2,2) e função de ativação ReLU
    Conv2D(64, (2, 2), activation='relu'),
    # Segunda camada de MaxPooling
    MaxPooling2D((2, 2)),
    
    # Terceira camada convolucional com 128 filtros de tamanho (2,2) e função de ativação ReLU
    Conv2D(128, (2, 2), activation='relu'),
    # Terceira camada de MaxPooling
    MaxPooling2D((2, 2)),
    
    # Camada de Flatten para converter os mapas de características 2D em um vetor unidimensional
    Flatten(),
    
    # Camada densa totalmente conectada com 256 neurônios e função de ativação ReLU
    Dense(256, activation='relu'),
    
    # Camada de Dropout para evitar overfitting, desativando 20% dos neurônios aleatoriamente durante o treinamento
    Dropout(0.2),
    
    # Camada de saída com 6 neurônios (correspondentes às classes) e função de ativação softmax para classificação multiclasse
    Dense(6, activation='softmax')
])

# Compilação do modelo com otimizador "adam", função de perda "categorical_crossentropy" para classificação multiclasse,
# e métricas a serem monitoradas, no caso, "accuracy"
model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Treinamento do modelo usando os dados do gerador de treinamento (train_generator)
# por 20 épocas, com tamanho de lote (batch_size) de 32, e validação usando os dados do gerador de validação (valid_generator)
history = model1.fit(train_generator, epochs=20, batch_size=32, validation_data=valid_generator)


model1.save("modelo_TF.keras")

plot_history(history,"accuracy")
plot_history(history,"loss")

result = model1.evaluate(test_generator)
print("Test loss, Test accuracy : ", result)

