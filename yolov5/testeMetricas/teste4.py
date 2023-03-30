import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# define as dimensões da janela
window_width = 800
window_height = 600
fps = 30

# cria o objeto de escrita de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video.mp4', fourcc, fps, (window_width, window_height))

# loop pelas imagens
for i in range(80,120):
    # carrega a imagem
    texto = './imgs/imgIM0S2_'+str(i)+'.jpg'

    # carrega uma imagem)
    img = mpimg.imread(texto)

    # cria um conjunto de eixos
    fig, ax = plt.subplots(figsize=(16, 20))

    # mostra a imagem nos eixos
    ax.imshow(img)

    # adiciona um ponto e especifica a legenda
    ax.plot(0, 0, 'bo')  # b-> blue ; o -> circuli vazio
    ax.plot(0, 0, 'rs')  # r-> red ; s -> quadrado(square)
    ax.plot(0, 0, '^', color='yellow', markersize=10)

    # mostra a legenda
    ax.legend(labels=['jogador', 'bola', 'cone'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)

    # Salvar a figura
    plt.savefig('img_temporaria.png', bbox_inches='tight')

    #ler imagem salva do grafico
    entrada = cv2.imread('img_temporaria.png')
    img = cv2.resize(entrada, (window_width, window_height))

    # adiciona a imagem ao vídeo
    out.write(img)
    plt.close()

# libera o objeto de escrita de vídeo
out.release()

