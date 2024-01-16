import os
import cv2
import numpy as np
import joblib

#O objetivo desse código é gerar e salvar um histograma para cada imagem do datsaet

def obter_vizinhos(matriz_de_intensidade, linha, coluna):
    #Esse método de partição da matriz foi escolhido para manter a analise dos pixels vizinhos
    #dentro dos limites da matriz de intensidade, evitando assim valores negativos ou fora do shape da matriz.
    vizinhos = matriz_de_intensidade[max(0, linha-1):min(matriz_de_intensidade.shape[0], linha+2),
                             max(0, coluna-1):min(matriz_de_intensidade.shape[1], coluna+2)]

    #Transforma a matriz particionada em lista e remove o pixel do centro
    lista_de_vizinhos = vizinhos.flatten().tolist()
    lista_de_vizinhos.remove(matriz_de_intensidade[linha, coluna]) 

    return lista_de_vizinhos

def aplica_regras(lista_de_vizinhos, intensidade_pixel_central, estado_pixel_central):
    #Conta o número de vizinhos vivos com a mesma intensidade que o pixel central
    vizinhos_iguais = lista_de_vizinhos.count(intensidade_pixel_central)

    #Regra 1: A célula viva com dois ou três vizinhos vivos sobrevive
    if estado_pixel_central == 1 and (vizinhos_iguais == 2 or vizinhos_iguais == 3):
        return 1
    
    #Regra 2: A célula viva com menos de dois vizinhos vivos morre (subpopulação)
    elif estado_pixel_central == 1 and vizinhos_iguais < 2:
        return 0
    
    #Regra 3: A célula viva com mais de três vizinhos vivos morre (superpopulação)
    elif estado_pixel_central == 1 and vizinhos_iguais > 3:
        return 0
    
    #Regra 4: A célula morta com exatamente três vizinhos vivos se torna viva (resurreição)
    elif estado_pixel_central == 0 and vizinhos_iguais == 3:
        return 1
    
    #Pra todos os outros casos, a célula permanece no mesmo estado
    else:
        return estado_pixel_central

def percorre_imagem_aplicando_regras(matriz_de_estados, matriz_de_intensidade):
    linhas, colunas = matriz_de_intensidade.shape
    for linha in range(linhas):
        for coluna in range(colunas):
            #Obtem os vizinhos do pixel atual
            lista_de_vizinhos = obter_vizinhos(matriz_de_intensidade, linha, coluna)
            #Aplica as regras do jogo da vida no pixel atual
            matriz_de_estados[linha, coluna] = aplica_regras(lista_de_vizinhos, matriz_de_intensidade[linha, coluna], matriz_de_estados[linha, coluna])
 
    return matriz_de_estados

def gera_histogramas(imagem_cinza):
    #Transforma a imagem em uma matriz de intensidade
    matriz_de_intensidade = np.array(imagem_cinza)
    
    #Cria as matrizes de estados iniciais
    matriz_de_estados_phi = np.ones(matriz_de_intensidade.shape, dtype=int) #todos vivos
    matriz_de_estados_psi = np.zeros(matriz_de_intensidade.shape, dtype=int) #todos mortos

    #Aplica as regras do jogo da vida e atualiza as matrizes de estado inicial
    matriz_de_estados_phi = percorre_imagem_aplicando_regras(matriz_de_estados_phi, matriz_de_intensidade)
    matriz_de_estados_psi = percorre_imagem_aplicando_regras(matriz_de_estados_psi, matriz_de_intensidade)

    #As matrizes são convertidas em listas
    #Phi -> estado inicial = vivo
    phi_vivos = matriz_de_intensidade.flatten()[matriz_de_estados_phi.flatten() == 1] #se manteram vivos
    phi_mortos = matriz_de_intensidade.flatten()[matriz_de_estados_phi.flatten() == 0] #morreram

    #Psi -> estado inicial = morto
    psi_vivos = matriz_de_intensidade.flatten()[matriz_de_estados_psi.flatten() == 1] #ressuscitaram 
    psi_mortos = matriz_de_intensidade.flatten()[matriz_de_estados_psi.flatten() == 0] #se manteram mortos

    #Cria os histogramas
    hist_phi_vivos, _ = np.histogram(phi_vivos, bins=256, range=(0, 256))
    hist_phi_mortos, _ = np.histogram(phi_mortos, bins=256, range=(0, 256))
    hist_psi_vivos, _ = np.histogram(psi_vivos, bins=256, range=(0, 256))
    hist_psi_mortos, _ = np.histogram(psi_mortos, bins=256, range=(0, 256))

    return hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos

def gerador_histogramas(data_dir):
    #Funcao principal: faz a chamada das funções acima e salva os histogramas

    # Define os caminhos dos diretórios
    dataset_dir = os.path.join(data_dir, "NEU Metal Surface Defects Data")
    classes_dir = os.path.join(dataset_dir, "test")
    histograms_dir = os.path.join(data_dir, "histograms")

    # Cria as pastas para salvar os histogramas mantendo o padrão das classes do dataset
    os.makedirs(histograms_dir, exist_ok=True)

    classes = [conteudo_item for conteudo_item in os.listdir(classes_dir) if os.path.isdir(os.path.join(classes_dir, conteudo_item))]
    for classe in classes:
        dir_pastas = os.path.join(histograms_dir, classe)
        os.makedirs(dir_pastas, exist_ok=True)

    i = 0
    for lote in os.listdir(dataset_dir):
        dir_lote = os.path.join(dataset_dir, lote)

        for classe in os.listdir(dir_lote):
            dir_classe = os.path.join(dir_lote, classe)

            for imagem in os.listdir(dir_classe):
                imagem_path = os.path.join(dir_lote, classe, imagem)
                imagem_cinza = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
                hist_phi_vivos, hist_phi_mortos, hist_psi_vivos, hist_psi_mortos = gera_histogramas(imagem_cinza)

                index = imagem.split(".")[0]

                #Concatena todos os histogramas gerados em apenas um
                all_combined = np.concatenate([np.concatenate([hist_phi_vivos, hist_phi_mortos]),
                                            np.concatenate([hist_psi_vivos, hist_psi_mortos])])

                #Salva o histograma final
                file_path = os.path.join(histograms_dir, classe, f"{index}_combined.pkl")
                joblib.dump(all_combined, file_path)

                i += 1
                #1800 é o tamanho do dataset
                print(f"Progresso = {i / 1800 * 100:.1f} por cento")

data_dir = f"C:/Users/Sebastiao/Desktop/Projetos/projeto-ia-surface-defect/data"
gerador_histogramas(data_dir)