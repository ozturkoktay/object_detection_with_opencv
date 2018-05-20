import numpy as np
import cv2

face_file = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_file  = cv2.CascadeClassifier('haarcascade_eye.xml')
font      = cv2.FONT_HERSHEY_SIMPLEX

capture   = cv2.VideoCapture(0)

while 1:
    ret, img = capture.read()
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_file.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.putText(img,'Insan Yuzu',(0,60), font, 1,(200,255,255),1,cv2.LINE_AA)
        
        eyes = eye_file.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(img,'Insan Gozu',(0,50), font, 1,(200,255,255),1,cv2.LINE_AA)

    cv2.imshow('Nesne Tanima',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
