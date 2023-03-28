import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cria uma figura com tamanho 1280x1024
fig = plt.figure(figsize=(12.8, 10.24))

# Adiciona um objeto Axes à figura
ax = fig.add_subplot(111)

# Desenha uma imagem aleatória no objeto Axes
img = cv2.imread('campoComHomografia.jpg')
ax.imshow(img)

# Ajusta o tamanho e a proporção da imagem mostrada no plot
ax.set_aspect('equal')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Salva a imagem em um arquivo JPEG
fig.savefig('imagemSemAjuste.jpg')




# import matplotlib.pyplot as plt
# import numpy as np
#
# # Cria uma figura com tamanho 800x600
# fig = plt.figure(figsize=(8, 6))
#
# # Adiciona um objeto Axes à figura
# ax = fig.add_subplot(111)
#
# # Desenha uma imagem aleatória no objeto Axes
# img = np.random.rand(600, 800, 3)
# ax.imshow(img)
#
# # Salva a imagem em um arquivo JPEG
# fig.savefig('imagem.jpg')


# # Cria uma lista de imagens do Matplotlib
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# imgs = []
# for i, ax in enumerate(axs.flat):
#     ax.plot([1, 2, 3, 4])
#     ax.set_xlabel('Eixo X')
#     ax.set_ylabel('Eixo Y')
#     ax.set_title(f'Gráfico {i+1}')
#     canvas = fig.canvas
#     canvas.draw()
#     img = np.array(canvas.renderer.buffer_rgba())
#     imgs.append(img)
#
# # Cria um objeto VideoWriter
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (imgs[0].shape[1], imgs[0].shape[0]))
#
# # Adiciona as imagens ao arquivo de vídeo
# for img in imgs:
#     out.write(img)
#
# # Libera os recursos
# out.release()
# plt.close()
