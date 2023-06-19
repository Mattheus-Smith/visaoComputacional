import argparse
import os
import platform
import sys
from pathlib import Path

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import torch
from funcoesExtras.DesenhoCampDrone import *
from funcoesExtras.homografiaCampDrone import *

# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Bibliotecas Auxiliares
import cv2
import numpy as np
import matplotlib.pyplot as plt

def usar_modelo(frame):
    model = keras.models.load_model('./weights/weight_v3.h5')

    # Converter para níveis de cinza
    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensionamento da imagem (opcional)
    imagem = cv2.resize(imagem_cinza, (15, 44))
    imagem = np.array([imagem])

    imagem = imagem / 255.0
    imagem = np.expand_dims(imagem, axis=-1)

    predictions = model.predict(imagem)

    texto = "predicao-> {} | resultado-> {}".format(predictions, np.argmax(predictions))
    #print(texto)
    return np.argmax(predictions)

def getFrame(path, im, im0s, vid_cap, s, dt, model, increment_path, save_dir, visualize, augment, non_max_suppression, conf_thres, iou_thres,
             classes, agnostic_nms, max_det, seen, webcam,dataset, save_crop, Annotator, line_thickness, names, scale_boxes, save_txt, xyxy2xywh,
             save_conf, save_img, view_img, hide_labels, hide_conf, save_one_box, windows, vid_path, vid_writer, label_img,
             operador, out, frame_size, cont):

    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s
    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        if webcam:  # batch_size >= 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '
        else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            cones_position = []
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    labels = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    label = labels.split()

                    x1 = int(p1[0])
                    y1 = int(p1[1])
                    x2 = int(p2[0])
                    y2 = int(p2[1])
                    frame = im0s[y1:y2, x1:x2]
                    texto="./testeFrames/frame{}.png".format(cont)
                    cv2.imwrite(texto, frame)
                    print("salvo "+texto)

                    if label[0] == "cone":
                        cones_position.append([p1,p2])
                        annotator.box_label(xyxy, labels, color=(0, 255, 255), operador=operador, label_img=label_img, cont=cont, jogador=0)
                    if label[0] == "ball":
                        annotator.box_label(xyxy, labels, color=(255,0,0), operador=operador, label_img=label_img, cont=cont, jogador=0)
                    if label[0] == "person":
                        cor = (255,annotator.rosa,255)
                        annotator.players_position.append([p1,p2, usar_modelo(frame), cor])
                        annotator.box_label(xyxy, labels, color=cor, operador=operador, label_img=label_img, cont=cont, jogador=1)
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                cont +=1

            #print(players_position)

            #print(cones_position)
            #desenhoCampDrone(im0s, cones_position)
            # if(len(cones_position) > 4):
            #     cv2.imwrite("./outputs/erro_cone_"+str(cont)+"_cones_"+str(len(cones_position))+".jpg", annotator.im)
            #     # cria uma janela vazia
            #     cv2.namedWindow('Imagem', cv2.WINDOW_NORMAL)
            #
            #     # define o tamanho da janela
            #     cv2.resizeWindow('Imagem', 800, 600)
            #     for i in range(len(cones_position)):
            #         cone = cones_position[i]
            #         x1 = int(cone[0][0])
            #         y1 = int(cone[0][1])
            #         x2 = int(cone[1][0])
            #         y2 = int(cone[1][1])
            #         roi = annotator.im[y1:y2, x1:x2]
            #         # #roi = imagem[x1y1[1]: x2y2[1], x1y1[0]: x2y2[0]]
            #         cv2.imshow("Imagem", roi)
            #         cv2.waitKey(0)
            # else:
            #     homografia = getHomografiaCampo(annotator.im, cones_position)
            #     linhas = desenharLinhas(homografia, 16, 12)
            #     out.write(cv2.resize(linhas, frame_size))

            # imagem_para_homogradia = im0s.copy()

            homografia = getHomografiaCampo(annotator.homografia, cones_position)
            # linhas = desenharLinhas(homografia, 16, 12)
            matriz, position_plys_matriz = segmentarMatriz(homografia, 16,12)
            #     out.write(cv2.resize(linhas, frame_size))
            # Imprime a matriz resultante

            # for linha in annotator.players_position:
            #     print(linha)
            # print("\n")

            verificar_rgb_na_matriz(annotator.players_position, position_plys_matriz)

            # cv2.destroyAllWindows()
        # Stream results
        im0 = annotator.result()
        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

    # Print time (inference-only)
    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    return seen, dt