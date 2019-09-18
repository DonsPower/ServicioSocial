from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import imutils
#person_390.bmp

image = cv2.imread("prueba2.jpg")
#resize image -scale

#image = imutils.resize(image, width=min(400, image.shape[1]))
image = imutils.resize(image, width=min(400, image.shape[1]))
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect people in the image
(rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4),
                                        padding=(8, 8), scale=1.05)

 # draw the original bounding boxes
for (x, y, w, h) in rects:
           cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

 # apply non-maxima suppression to the bounding boxes using a
 # fairly large overlap threshold to try to maintain overlapping
 # boxes that are still people
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

 # draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
             cv2.rectangle(image, (xA, yA), (xB, yB), (128, 128, 128), 2)

cv2.imshow("Output", orig)
cv2.imshow("OutputNMS", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
