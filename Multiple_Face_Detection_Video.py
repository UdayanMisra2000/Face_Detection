import cv2
from random import randrange

#load some pre-trained data
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#read a chosen video
video = cv2.VideoCapture(0)

#Iterating forever over frames
while True:
    #read the current frame
    frame_check,frame = video.read()
    
    #grey-scaling the current frame
    gr_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #detecting the face coordinates
    face_coordinates = trained_face_data.detectMultiScale(gr_img)
    
    #marking the faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)
    
    #display the current frame
    cv2.imshow('faces',frame)
    key = cv2.waitKey(1)
    
    #stopping condition
    if key == 83 or key== 115:
        break

# release the VideoCapture object
video.release()
    

print('Press "s" to stop')
