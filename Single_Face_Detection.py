import cv2

#load some pre-trained data
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#read a chosen image
img = cv2.imread('front_face_view.png')

#grey-scaling the image
gr_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detecting face coordinates
face_coordinates = trained_face_data.detectMultiScale(gr_img) 
print(face_coordinates)

#Marking the face
(x,y,w,h) = face_coordinates[0]
cv2.rectangle(img,(x,y),(x+w,y+h),(255,175,255),2)

#display the image
cv2.imshow('face',img)
cv2.waitKey()

print('Helo')
