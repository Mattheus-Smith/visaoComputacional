
import torch


from tracker import Tracker
import cv2
import  os
# from deep_sort.deep_sort import DeepSort;
#coloca teu modelo aqui 
model = torch.hub.load('.', 'custom', path='weights/best.pt', source='local')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
#colo teu arquivi de entrada aqui

#out = './outFiles/PontosEntrada/'
tracker = Tracker()


cap = cv2.VideoCapture("./data/images/DroneCortado.mp4")
#arq = open(out+s[0]+'.txt','w')


# Kalman Filter
# kf = KalmanFilter()

# Object Tracking

# initialize deepsort
results_point = []

f: int = 0

while True:
    ret, frame = cap.read()

    if ret is False:
        break
    else:

        frame = cv2.resize(frame, (1080, 720))
        results = model(frame)

        box = []
        for index, row, in results.pandas().xyxy[0].iterrows():
            #print(row)
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            classe = row['name']

            box.append([x1, x2, y1, y2, classe])

        box_ids = tracker.update(box)

        box_2 = []
        for box_id in box_ids:
            x, y, w, h, classe,id = box_id

            cx = (x + y) // 2
            cy = (w + h) // 2

            results_point.append([cx, cy, classe,id])
            box_2.append([cx, cy, classe,id])

        nxa = 0
        nya = 0
        axi = 0
        ayi = 0

        for pts1 in results_point:

            x, y, classe,id = pts1

            for pts2 in box_2:
                xi, yi, classe_old,ida = pts2

                if nxa == 0: nxa = x
                if nya == 0: nya = y
                if axi == 0: xi = x
                if ayi == 0: yi = y

                # cv2.circle(frame, (x, y), 1, (255, 0, 0), 4)
                if (id == ida and classe=="person"):

                    # cv2.circle(frame, (xa, ya), 1, (255, 0, 0), 4)
                    #cv2.circle(frame, (x, y), 1, (0, 255, 0), 2)


                    if (xi == axi and yi == ayi):
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), 10)

                        # cv2.line(frame, (nxa, nya), (x, y), (255, 0, 0), 2)

                    # print(x, y, id, xi, yi, ayi, axi)

                    nxa = x
                    nya = y
                    axi = xi
                    ayi = yi

            # cv2.putText(frame, "ID: " + str(id), (x, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        cv2.imshow('FRAME', frame)

        # print(f)
        f += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
