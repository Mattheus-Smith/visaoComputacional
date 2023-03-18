import cv2
import numpy as np

# Read source image.
#img_src = cv2.imread("./../data/images/imgCampoDrone.jpg")

def ordenarMaior(cones, posicao):
    for j in range(1,len(cones)):
        maior = cones[j-1][posicao]
        posicaoMaior = j-1

        for i in range(j, len(cones)):
            #print("elem: ",cones[i][posicao]," - maior: ",maior)
            if (cones[i][posicao] > maior):
                aux = cones[posicaoMaior]
                maior =cones[i][posicao]
                cones[posicaoMaior] = cones[i]
                cones[i] = aux

def ordenarMenor(cones, posicao):
    for j in range(0,len(cones)):
        menor = cones[j][posicao]
        posicaoMaior = j

        for i in range(j, len(cones)):
            if (cones[i][posicao] < menor):
                aux = menor
                menor =cones[i][posicao]
                cones[posicaoMaior][posicao] = menor
                cones[i][posicao] = aux

def getHomografiaCampo(img_src, cones_position):
    pontos = []
    for i in range(len(cones_position)):
        cone = cones_position[i]
        x1 = int(cone[0][0]);
        y1 = int(cone[0][1])
        x2 = int(cone[1][0]);
        y2 = int(cone[1][1])
        x_centro = x1 + int((x2 - x1) / 2)
        y_centro = y1 + int((y2 - y1) / 2)
        pontos.append([x_centro, y_centro])

    #print(pontos)
    ordenarMaior(pontos, 0)
    pontosEspecificos = [pontos[3], pontos[0], pontos[1], pontos[2]]
    # print(pontosEspecificos)

    # Four corners of the 3D court + mid-court circle point in source image
    # Start top-left corner and go anti-clock wise + mid-court circle point
    pts_src = np.float32(pontosEspecificos)  # C, D
    # print(pts_src)

    # encontrar a largura maxima
    width_AB = np.sqrt(((pts_src[0][0] - pts_src[1][0]) ** 2) + ((pts_src[0][1] - pts_src[1][1]) ** 2))
    width_DC = np.sqrt(((pts_src[3][0] - pts_src[2][0]) ** 2) + ((pts_src[3][1] - pts_src[2][1]) ** 2))
    maxWidth = max(int(width_AB), int(width_DC))

    # encontrar alttura maxima
    height_AD = np.sqrt(((pts_src[0][0] - pts_src[3][0]) ** 2) + ((pts_src[0][1] - pts_src[3][1]) ** 2))
    height_BC = np.sqrt(((pts_src[1][0] - pts_src[2][0]) ** 2) + ((pts_src[1][1] - pts_src[2][1]) ** 2))
    maxHeight = max(int(height_AD), int(height_BC))

    # saida pra ficar em 2D
    output_pts = np.float32([[0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0],
                             [0, 0]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(pts_src, output_pts)

    # image esult
    out = cv2.warpPerspective(img_src, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    # rotate image
    #Rotated_image = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite("./outputs/campoComHomografia.jpg", out)
    # cv2.imshow("saida", out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return out

def desenharLinhas(img, x, y):
    height, width,_ = img.shape

    width_dividido = width//x
    height_dividido = height//y

    for j in range(1,x):
        for i in range(1, y):
            if(i == 1):
                img = cv2.line(img, (j * width_dividido, 0), (j * width_dividido, height), (0, 0, 255), 2)
            img = cv2.line(img, (0, i*height_dividido), (width, i*height_dividido), (0,0,255), 2)


    return img
    #cv2.imwrite("./outputs/campoComHomografiaComLinha.png", img)
    # cv2.imshow("saida 1", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# cones = [[(1638, 829), (1670, 872)], [(1402, 241), (1421, 268)], [(447, 241), (466, 267)], [(158, 835), (188, 875)]]
#
# img_com_homografia = getHomografiaCampo(img_src, cones)
# desenharLinhas(img_com_homografia, 16, 12 )

