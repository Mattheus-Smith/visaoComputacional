import cv2
import numpy as np

# define as configurações do vídeo
width = 640
height = 480
fps = 30

# define as dimensões da janela
window_width = 800
window_height = 600

# cria o objeto de escrita de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video.mp4', fourcc, fps, (window_width, window_height))

# cria a imagem do texto
text_image = np.zeros((100, window_width, 3), dtype=np.uint8)
cv2.putText(text_image, 'Texto', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# loop pelas imagens
for i in range(80,120):
    # carrega a imagem
    entrada = cv2.imread(f'imgIM0S2_{i}.jpg')
    img = cv2.resize(entrada, (window_width, window_height-100))

    # desenha um círculo na imagem
    cv2.circle(img, (50, 50), 30, (0, 0, 255), 2)

    # define a classe detectada
    classe = "objeto"

    # adiciona o texto na imagem
    cv2.putText(img, classe, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # cria a imagem do vídeo com o circulo
    video_image = img[0:500, 0:800]

    # adiciona o texto à imagem do vídeo
    video_text_image = cv2.hconcat([video_image, text_image])

    # adiciona a imagem ao vídeo
    out.write(video_text_image)

# libera o objeto de escrita de vídeo
out.release()
