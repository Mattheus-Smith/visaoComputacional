import cv2
import numpy as np

# Carregar os vídeos
video_a = cv2.VideoCapture('C:\\Users\\Smith Fernandes\\Videos\\dados coletados\\12_05\\camera Esquerda\\jogo1Cortado.mp4')
video_b = cv2.VideoCapture('C:\\Users\\Smith Fernandes\\Videos\\dados coletados\\12_05\\camera Direita\\jogo1Cortado.mp4')

#video_a = cv2.VideoCapture('C:\\Users\\Smith Fernandes\\Videos\\dados coletados\\27_02\\10 seg\\A51Cortado.mp4')
#video_b = cv2.VideoCapture('C:\\Users\\Smith Fernandes\\Videos\\dados coletados\\27_02\\10 seg\\A72Cortado.mp4')

# Obter as dimensões dos vídeos
width = int(video_a.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
height = int(video_a.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Criar o codec de vídeo para salvar o vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_saida.mp4', fourcc, 30.0, (2 * width, height))

# Obter o número total de frames dos vídeos
total_frames_a = int(video_a.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames_b = int(video_b.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames = min(total_frames_b, total_frames_a)

cont = 0
marcas_de_progresso = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
indice_progresso = 0
while True:
    # Ler os frames dos vídeos
    ret_a, frame_a = video_a.read()
    ret_b, frame_b = video_b.read()

    if not ret_a or not ret_b:
        break

    # Redimensionar os frames para as mesmas dimensões
    frame_a = cv2.resize(frame_a, (width, height))
    frame_b = cv2.resize(frame_b, (width, height))

    # Juntar os frames lado a lado
    frame_combined = np.concatenate((frame_a, frame_b), axis=1)

    # Escrever o frame combinado no vídeo de saída
    out.write(frame_combined)

    # Mostrar procetagem do processo
    # Calcular o progresso atual
    progresso_atual = (cont + 1) / total_frames * 100
    cont+=1

    # Verificar se atingiu uma marca de progresso
    if progresso_atual >= marcas_de_progresso[indice_progresso]:
        print(f"Progresso: {marcas_de_progresso[indice_progresso]}%")
        indice_progresso += 1

    # Mostrar o frame combinado em uma janela
    #cv2.imshow('Combined Video', frame_combined)

    # Encerrar o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
video_a.release()
video_b.release()
out.release()
cv2.destroyAllWindows()