# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from GetFrame import getFrame
cont = 700

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source1=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        source2=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        label_img = ROOT,
        operador = ROOT
):
    global cont
    source1 = str(source1)
    save_img1 = not nosave and not source1.endswith('.txt')  # save inference images
    is_file1 = Path(source1).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url1 = source1.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam1 = source1.isnumeric() or source1.endswith('.streams') or (is_url1 and not is_file1)
    screenshot1 = source1.lower().startswith('screen')
    if is_url1 and is_file1:
        source1 = check_file(source1)  # download

    source2 = str(source2)
    save_img2 = not nosave and not source2.endswith('.txt')  # save inference images
    is_file2 = Path(source2).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url2 = source2.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam2 = source2.isnumeric() or source2.endswith('.streams') or (is_url2 and not is_file2)
    screenshot2 = source2.lower().startswith('screen')
    if is_url2 and is_file2:
        source2 = check_file(source2)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    #print("antes dos ifs")

    # Dataloader
    bs = 1  # batch_size
    if webcam1:
        #print("opcao1- 1")
        view_img = check_imshow(warn=True)
        dataset1 = LoadStreams(source1, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset1)
    elif screenshot1:
        #print("opcao2- 1")
        dataset1 = LoadScreenshots(source1, img_size=imgsz, stride=stride, auto=pt)
    else:
        #print("opcao3 - 1")
        dataset1 = LoadImages(source1, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path1, vid_writer1 = [None] * bs, [None] * bs

    if webcam2:
        #print("opcao1- 1")
        view_img = check_imshow(warn=True)
        dataset2 = LoadStreams(source2, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset2)
    elif screenshot2:
        #print("opcao2- 2")
        dataset2 = LoadScreenshots(source2, img_size=imgsz, stride=stride, auto=pt)
    else:
        #print("opcao3 - 2")
        dataset2 = LoadImages(source2, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path2, vid_writer2 = [None] * bs, [None] * bs

    #print("depois dos ifs")

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    #for path1, im1, im0s1, vid_cap1, s1 in dataset2:

    # Defina o nome do arquivo de saída
    output_video = "output_video.mp4"
    # Defina as configurações de vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    frame_size = (1280, 1024)

    # Crie o objeto VideoWriter
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # Itera sobre as duas instâncias juntas
    for data1, data2 in zip(dataset1, dataset2):

        # processa os dados de cada instância
        path1, im1, im0s1, vid_cap1, s1 = data1
        path2, im2, im0s2, vid_cap2, s2 = data2
        # print(cont)

        #cv2.imwrite("./testeFrames/frame1/imgIM0S1_"+str(cont)+".jpg", im0s1)
        #cv2.imwrite("./testeFrames/frame2/imgIM0S2_"+str(cont)+".jpg", im0s2)
        #cont += 1

        seen, dt = getFrame(path1, im1, im0s1, vid_cap1, s1, dt, model, increment_path, save_dir, visualize, augment, non_max_suppression, conf_thres,
                            iou_thres, classes, agnostic_nms, max_det, seen, webcam1,dataset1, save_crop, Annotator, line_thickness, names, scale_boxes,
                            save_txt, xyxy2xywh, save_conf, save_img1, view_img, hide_labels, hide_conf, save_one_box, windows, vid_path1, vid_writer1,
                            label_img, operador, out, frame_size)

    # Fecha o objeto "VideoWriter"
    out.release()

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img1:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source1', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--source2', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--label_img', type=int, default=0, help='if show label or not')
    parser.add_argument('--operador', type=int, default=0, help='which option use')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
