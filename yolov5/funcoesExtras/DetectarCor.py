import cv2
import numpy as np

def detectarCor(imagem, box):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    roi = imagem[p1[1]: p2[1], p1[0]: p2[0]]

    cv2.imshow("saida roi", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()