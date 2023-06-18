import cv2
import numpy as np

# carrega a imagem
image  = cv2.imread("Color_icon_blue.png")
image = cv2.resize(image, (800,600))

# conversão para o espaço de cores HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # criação da máscara para os pixels pretos
# # criação da máscara para os pixels vermelhos
lower_red = np.array([110, 50, 50])
upper_red = np.array([115, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

# aplicação de uma transformação morfológica para remover pequenos objetos e preencher buracos
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (36, 36))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
# aplicação da máscara na imagem original
result = cv2.bitwise_and(image, image, mask=mask)
#
# # salvamento da imagem resultante
# #cv2.imwrite('caminho/para/a/imagem_resultante.jpg', result)


# exibe a imagem filtrada
cv2.imshow('Imagem filtrada', result)
cv2.waitKey(0)
cv2.destroyAllWindows()