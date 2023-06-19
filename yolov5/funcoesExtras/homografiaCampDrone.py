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
    for j in range(1,len(cones)):
        menor = cones[j-1][posicao]
        posicaoMaior = j-1

        for i in range(j, len(cones)):
            if (cones[i][posicao] < menor):
                aux = cones[posicaoMaior]
                menor = cones[i][posicao]
                cones[posicaoMaior] = cones[i]
                cones[i] = aux

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

    listaConesOrdenados = []
    if ( len(pontos) == 4):
        # print("entrada - ",pontos)
        ordenarMaior(pontos, 1)  # acha os Y maiores
        # print("Y maiores - ", pontos)
        listaConesOrdenados.append(pontos[0]);
        listaConesOrdenados.append(pontos[1])
        pontos.pop(1);
        pontos.pop(0)
        ordenarMenor(listaConesOrdenados, 0)

        ordenarMenor(pontos, 1)  # acha os Y menores
        # print("Y menores - ", pontos)
        ordenarMaior(pontos, 0)  # acha os X maiores
        listaConesOrdenados.append(pontos[0]);
        listaConesOrdenados.append(pontos[1])

        # pontosEspecificos = [pontos[3], pontos[0], pontos[1], pontos[2]]
        #print(listaConesOrdenados)
    elif ( len(pontos) == 5):
        # print("entrada - ",pontos)
        ordenarMaior(pontos, 1)  # acha os Y maiores
        # print("Y maiores - ", pontos)
        listaConesOrdenados.append(pontos[0]);
        listaConesOrdenados.append(pontos[1])
        pontos.pop(1);
        pontos.pop(0)
        ordenarMenor(listaConesOrdenados, 0)

        ordenarMenor(pontos, 1)  # acha os Y menores
        # print("Y menores - ", pontos)
        ordenarMaior(pontos, 0)  # acha os X maiores
        listaConesOrdenados.append(pontos[0]);
        listaConesOrdenados.append(pontos[2])

        # pontosEspecificos = [pontos[3], pontos[0], pontos[1], pontos[2]]
        # print(listaConesOrdenados)

    elif( len(pontos) == 3):
        print("BO-3")
    elif( len(pontos) == 2):
        print("BO-5")
    else:
        print("menos - ", len(pontos))

    # Four corners of the 3D court + mid-court circle point in source image
    # Start top-left corner and go anti-clock wise + mid-court circle point
    pts_src = np.float32(listaConesOrdenados)  # C, D

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

    # cv2.imwrite("./outputs/campoComHomografia_15_06.jpg", out)
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

    # cv2.imwrite("./outputs/campoComHomografiaComLinha.png", img)
    # cv2.imshow("saida 1", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img

def verificar_cor(imagem, cor_min, cor_max):
    # Converter a imagem para o espaço de cores HSV
    hsv_imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Aplicar o filtro de cor na imagem HSV
    mask = cv2.inRange(hsv_imagem, cor_min, cor_max)

    # Verificar se há pixels brancos na máscara
    if cv2.countNonZero(mask) > 0:
        return True  # A cor foi encontrada

    return False  # A cor não foi encontrada

lowers =np.array([[145, 85, 225], [145, 110, 250], [145, 130, 255], [145, 165, 250],[145, 180, 250],
                    [145, 190, 255], [145, 210, 250], [145, 230, 250]])

uppers = np.array([[150, 110, 255],[155, 125, 255],[155, 160, 255],[155, 180, 255],[155, 190, 255],
                    [155, 210, 255], [155, 230, 255], [155, 240, 255]])

flags_cor = [0,0,0,0,0,0,0,0]
check_cor = []

def verificar_maior_area(image, lower_rosa, upper_rosa):
    # conversão para o espaço de cores HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # criação da máscara para os pixels rosas
    mask = cv2.inRange(hsv, lower_rosa, upper_rosa)

    # aplicação de uma transformação morfológica para remover pequenos objetos e preencher buracos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontra os contornos na máscara
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializa a área rosa
    area_rosa = 0

    # Calcula a área rosa somando os contornos encontrados
    for contorno in contornos:
        area_rosa += cv2.contourArea(contorno)
    print("essa eha area: ", area_rosa)
    return area_rosa

def verificar_duplicidade_cor(vetorA, vetorB):
    areaA = verificar_maior_area(vetorA[2], vetorA[3], vetorA[4])
    areaB = verificar_maior_area(vetorB[2], vetorB[3], vetorB[4])

    return areaA, areaB

def segmentarMatriz(img, x, y):

    altura, largura,_ = img.shape

    # Cria uma nova matriz preenchida com zeros
    matriz = np.zeros((12, 16), dtype=int)

    # Define o tamanho dos blocos para percorrer a imagem
    tamanho_bloco_x = largura // x
    tamanho_bloco_y = altura // y

    # Percorre cada bloco da imagem
    for i in range(y):
        for j in range(x):
            # Obtém as coordenadas do bloco
            x1 = j * tamanho_bloco_x
            y1 = i * tamanho_bloco_y
            x2 = x1 + tamanho_bloco_x
            y2 = y1 + tamanho_bloco_y

            # Verifica se o bloco contém algum pixel colorido
            bloco = img[y1:y2, x1:x2]
            # texto= "./outputs/bloco_{}_{}.png".format(i, j)
            # cv2.imwrite(texto, bloco)

            for lower, upper, contador in zip(lowers, uppers, range(0,len(uppers))):
                cor_encontrada = verificar_cor(bloco, lower, upper)

                if cor_encontrada and flags_cor[contador] == 0:
                    #print("A cor rosa foi encontrada na imagem.")
                    matriz[i][j] += 1
                    flags_cor[contador] += 1
                    check_cor.append([i, j, bloco,lower, upper,0])
                    texto = "{},{} | bloco_{}_{}.png | {}\n".format(lower, upper, i, j, contador)
                    # print(texto)
                elif(cor_encontrada and flags_cor[contador] != 0):
                    # texto = "{},{} | bloco_{}_{}.png | repetiu!!!\n".format(lower, upper, i, j)
                    # print(texto)
                    # check_cor.append([i, j, bloco,lower, upper])
                    # matriz[i][j] += 1
                    # flags_cor[contador] += 1
                    # check_cor.append([i, j, bloco, lower, upper])

                    for vector in check_cor:
                        if np.array_equal(vector[3], lower) and np.array_equal(vector[4], upper):
                            texto="vetor encontrado: {} {} {} {}".format(vector[0], vector[1], vector[3], vector[4])
                            # print(texto)
                            vetorB = [i, j, bloco,lower, upper, 0]
                            areaA, areaB = verificar_duplicidade_cor(vector, vetorB)

                            if areaB > areaA:
                                matriz[vector[0]][vector[1]] -= 1
                                matriz[vetorB[0]][vetorB[1]] += 1
                            else:  # areas igauis
                                print("areas igauis")
                    break

    # for linha in check_cor:
    #     print(linha[0], linha[1], linha[3], linha[4], linha[5])

    # Imprime a matriz resultante
    for linha in matriz:
        print(linha)

    return matriz, check_cor

def verificar_se_cor_bgr_pertece_a_matriz(lista, position_plys_matriz):

    listaNova = position_plys_matriz.copy()

    for vetor in lista:
        rgb_color = np.array([vetor[3][0], vetor[3][1], vetor[3][2]], dtype=np.uint8)
        bgr_color = rgb_color.reshape(1, 1, 3)  # Converter para formato BGR
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)

        # print(vetor[3], hsv_color, hsv_color.shape, lower[0], upper[0])
        for lower, upper, contador in zip(lowers, uppers, range(0, len(lowers))):
            if (lower[0] <= hsv_color[0][0][0] <= upper[0] and
                    lower[1] <= hsv_color[0][0][1] <= upper[1] and
                    lower[2] <= hsv_color[0][0][2] <= upper[2]):
                listaNova[contador][5] = lista[contador][2]
                texto = "pertence ao intervalo i={} hsv={}".format(0, hsv_color)
                # print(texto)
    return listaNova
