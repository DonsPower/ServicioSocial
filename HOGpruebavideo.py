from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import imutils
# Capture video from file
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret == True:
        # Reduce la imagen para el mejor anslisis
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))

        #init HOG
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        #Detect people in the video
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Cuadro delimitador
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        posx = 0
        posy = 0

         # draw the final bounding boxes
         #Aqui empezaremos a analizar el objeto cuadro.
         #cuadro verde
        for (xA, yA, xB, yB) in pick:
            #Sacamos centro.
            posx+=(xA+xB)/2
            posy+=(yA+yB)/2
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            #cv2.circle(frame,(posx,posy), 5, (0,0,255), -1)
        cv2.imshow('frame',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
