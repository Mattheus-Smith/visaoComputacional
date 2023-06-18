# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Bibliotecas Auxiliares
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_1_image(train_images):
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def show_someone_images(train_images,class_names,train_labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

# Pasta contendo as imagens de treinamento
pasta_treinamento = './../data/todas'

# Lista para armazenar as imagens e os rótulos
imagens_treinamento = []
rotulos_treinamento = []

# Dicionário de rótulos correspondentes aos nomes das pastas
rotulos_dict = {
    '0': 0,
    '1': 1,
    '2': 2,
    '2': 3,
    '4': 4
}
class_names = ['0', '1', '2', '3', '4']

# Percorra os arquivos na pasta de treinamento
for arquivo in os.listdir(pasta_treinamento):
    # Verifique se o arquivo é uma imagem (você pode adicionar mais verificações, se necessário)
    if arquivo.endswith('.jpg') or arquivo.endswith('.png'):
        # Caminho completo para a imagem
        caminho_imagem = os.path.join(pasta_treinamento, arquivo)

        # Leitura da imagem usando o OpenCV
        imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)  # Ler a imagem em escala de cinza

        # Redimensionamento da imagem (opcional)
        imagem = cv2.resize(imagem, (15, 44))

        # achando o rotulo apartir do label da imagem
        nome = arquivo.split("_")[0]
        rotulo = rotulos_dict[nome]

        # Adicionando a imagem e o rótulo às listas
        imagens_treinamento.append(imagem)
        rotulos_treinamento.append(rotulo)

# Converter as listas em arrays NumPy
imagens = np.array(imagens_treinamento)
classes = np.array(rotulos_treinamento, dtype=np.uint8)

# Dividir os dados em conjunto de treinamento e teste
percentual_treinamento = 0.8  # 80% para treinamento, 20% para teste
tamanho_treinamento = int(len(imagens) * percentual_treinamento)

train_images = imagens[:tamanho_treinamento]
train_labels = classes[:tamanho_treinamento]
test_images = imagens[tamanho_treinamento:]
test_labels = classes[tamanho_treinamento:]

#show_1_image(train_images)

train_images = train_images / 255.0
test_images = test_images / 255.0

#show_someone_images(train_images,class_names,train_labels)


def criar_rede():

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(44, 15)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.fit(train_images, train_labels, epochs=150)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    model.save('/content/drive/MyDrive/2-PIBIC_maria/numero_camisa/weight_beta.h5')

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def usar_modelo():
    model = keras.models.load_model('./../weights/weight_v3.h5')

    # predictions = model.predict(test_images)
    # print(predictions[0])
    # print(np.argmax(predictions[0]))
    # print(test_labels[0])

    # Converter as listas em arrays NumPy
    # Leitura da imagem usando o OpenCV
    imagem = cv2.imread("frame3887.png", cv2.IMREAD_GRAYSCALE)  # Ler a imagem em escala de cinza

    # Redimensionamento da imagem (opcional)
    imagem = cv2.resize(imagem, (15, 44))
    imagens = np.array([imagem])

    predictions = model.predict(imagens)
    texto = "predicao-> {} | resultado-> {}".format(predictions, np.argmax(predictions))
    print(texto)


    # i = 34
    # plt.figure(figsize=(6, 3))
    # plt.subplot(1, 2, 1)
    # plot_image(i, predictions, test_labels, test_images)
    # plt.subplot(1, 2, 2)
    # plot_value_array(i, predictions, test_labels)
    # plt.show()

usar_modelo()