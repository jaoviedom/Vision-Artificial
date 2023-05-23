import cv2
import numpy as np

img = cv2.imread('./images/suavizante/suavitel.jpeg')
bordes = cv2.Canny(img, 150, 200)

cv2.imshow("Bordes", bordes)

num_kpt = 1000
# Declaramos el objeto
orb = cv2.ORB_create(num_kpt)
# Extraemos la info de la img
keypoint, descriptor = orb.detectAndCompute(bordes, None)

print(descriptor, keypoint)

# Drawing the keypoints
kp_image = cv2.drawKeypoints(bordes, keypoint, None, color=(0, 255, 0), flags=0)
  
cv2.imshow('ORB', kp_image)

cv2.waitKey()