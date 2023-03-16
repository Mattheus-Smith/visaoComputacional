import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read source image.
# img_src = cv2.imread("./../data/images/imgCampoDrone.jpg")
# cones = [[(1638, 829), (1670, 872)], [(1402, 241), (1421, 268)], [(447, 241), (466, 267)], [(158, 835), (188, 875)]]
# cone1 = cones[0]
# print(cone1[0][0])
# roi = img_src[cone1[0][1]: cone1[1][1], cone1[0][0]: cone1[1][0]]

def desenhoCampDrone(img_src, cones_position):
    pontos = []
    for i in range(len(cones_position)):
        cone = cones_position[i]
        x1 = int(cone[0][0]); y1 = int(cone[0][1])
        x2 = int(cone[1][0]); y2 = int(cone[1][1])
        x_centro = x1 + int((x2 - x1) / 2); y_centro = y1 + int((y2 - y1) / 2)
        pontos.append([x_centro, y_centro])

    #print(pontos)
    pts_src = np.array(pontos)  # C, D

    # cv2.fillPoly(img_src, [pts_src], 255)
    cv2.polylines(img_src, [pts_src], isClosed=True, color=[255, 0, 0], thickness=2)
    cv2.imwrite("./outputs/campoDelimitado.jpg", img_src)

# desenhoCampDrone(img_src, cones)
# cv2.imshow("saida", img_src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()