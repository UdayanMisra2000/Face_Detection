import cv2
from random import randrange

#load some pre-trained data
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#read an image(chosen)
img = cv2.imread('Group_img.jpeg')

#grey-scaling the image
gr_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect the coordinates of the faces
face_coordinates = trained_face_data.detectMultiScale(gr_img)
print(face_coordinates)

# marking(here rectangle) the faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

#display the images    
cv2.imshow('face',img)
cv2.waitKey()

print('Hello')
