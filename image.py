import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

img = cv2.imread('Resources/group.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# frame_gray = cv2.equalizeHist(frame_gray,0)
# faces = faceCascade.detectMultiScale(gray,1.1,7)
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
print(faces)
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

