import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# carrega uma imagem
img = mpimg.imread('campoComHomografia.jpg')

# cria um conjunto de eixos
fig, ax = plt.subplots(figsize=(16, 20))
# 14,12 -> 1141x612
# 15x15 -> 1218x649
# 16x20 -> 1296x686

# mostra a imagem nos eixos
ax.imshow(img)

# adiciona um ponto e especifica a legenda
ax.plot(0,0, 'bo') #b-> blue ; o -> circuli vazio
ax.plot(0,0, 'rs') #r-> red ; s -> quadrado(square)
ax.plot(0,0, '^', color='yellow', markersize=10)

# mostra a legenda
ax.legend(labels = ['jogador', 'bola' ,'cone'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)

# Salvar a figura
plt.savefig('grafico_com_legenda.png', bbox_inches='tight')

fig_size = fig.get_size_inches()
print('Tamanho do gráfico:', fig_size[0]*100, 'x', fig_size[1]*100, 'pixels')
