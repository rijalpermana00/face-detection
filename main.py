import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

while(True):
    ret, img = capture.read()
    frame_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)


    # faces = faceCascade.detectMultiScale(gray,1.3,3)
    faces = faceCascade.detectMultiScale(frame_gray)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('img',img)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()