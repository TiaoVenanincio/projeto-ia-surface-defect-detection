import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

def carrega_hist(imagem_path):
    #O objetivo dessa função é buscar os histogramas associados a uma determinada imagem.

    #C: /Users /Sebastiao /Desktop /Projetos /projeto-ia-surface-defect /data\NEU Metal Surface Defects Data\test /Arranhoes /Sc_1.bmp
    imagem_path = imagem_path.split("/")
    classe, imagem = imagem_path[7], imagem_path[8]
    imagem = imagem.split(".")[0]

    histogramas_dir = "data/histograms"
    try:
        all_combined = joblib.load(f"{histogramas_dir}/{classe}/{imagem}_combined.pkl")
        return all_combined
    
    except:
        print("O histograma não foi encontrado. Verifique os diretórios e se os histogramas foram criados corretamente.")

def validacao_cruzada(modelo, X, y, n):
    scores = cross_val_score(modelo, X, y, cv=n)
    print("Acurácias em cada fold:", scores)
    print("Acurácia média: {:.2f}\n\n".format(scores.mean()))

# Define os caminhos dos diretórios
data_dir = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ia-surface-defect/data"
dataset_dir = os.path.join(data_dir, "NEU Metal Surface Defects Data")
classes_dir = os.path.join(dataset_dir, "test")
histograms_dir = os.path.join(data_dir, "histograms")      

histogramas = []
rotulos = []

for lote in os.listdir(dataset_dir):
    dir_lote = os.path.join(dataset_dir, lote)

    for classe in os.listdir(dir_lote):
        dir_classe = os.path.join(dir_lote, classe)

        for imagem in os.listdir(dir_classe):
            imagem_path = f"{dir_lote}/{classe}/{imagem}"
            
            #print(imagem_path)
            #break
            hist_cobined = carrega_hist(imagem_path)

            histogramas.append(hist_cobined)
            rotulos.append(classe)


X = np.array(histogramas)
y = np.array(rotulos)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

modelo = RandomForestClassifier(n_estimators=175, random_state=42)

modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)

precisao = accuracy_score(y_test, previsoes)
print(f"N estimators: {175}, precisao do modelo: {precisao}")

matriz_confusao = confusion_matrix(y_test, previsoes)

print("Matriz de Confusão:")
print(matriz_confusao)

class_report = classification_report(y_test, previsoes)
print("\nRelatório de Classificação:")
print(class_report)

joblib.dump(modelo, 'modelo_SL.joblib')

# Realizando a validação cruzada com n folds (8 neste caso)
validacao_cruzada(modelo, X, y, 8)

#O intuito de realizar a validacao cruzada é verificar se a acurácia do método padrao
#desvia significativamente da acuracia da validacao cruzada, caso sim, pode-se apontar
#um problema de overfitting.