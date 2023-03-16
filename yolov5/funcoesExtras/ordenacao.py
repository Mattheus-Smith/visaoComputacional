
def ordenarMaior(cones, posicao):
    for j in range(0,len(cones)):
        maior = cones[j][posicao]
        posicaoMaior = j

        for i in range(j, len(cones)):
            if (cones[i][posicao] > maior):
                aux = maior
                maior =cones[i][posicao]
                cones[posicaoMaior][posicao] = maior
                cones[i][posicao] = aux

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

# cones = [[1654, 850], [1411, 254], [456, 254], [173, 855]]
# ordenarMaior(cones, 0)
# print(cones)