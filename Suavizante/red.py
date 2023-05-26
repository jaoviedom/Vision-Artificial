import cv2
import numpy as np
import pandas as pd

img1 = cv2.imread('./images/suavizante/Manzana/1.jpeg')
img2 = cv2.imread('./images/suavizante/Manzana/2.jpeg')
img3 = cv2.imread('./images/suavizante/Manzana/3.jpeg')
img4 = cv2.imread('./images/suavizante/Manzana/4.jpeg')
img5 = cv2.imread('./images/suavizante/Manzana/5.jpeg')
img6 = cv2.imread('./images/suavizante/Manzana/6.jpeg')
img7 = cv2.imread('./images/suavizante/Manzana/7.jpeg')
img8 = cv2.imread('./images/suavizante/Manzana/8.jpeg')
img9 = cv2.imread('./images/suavizante/Manzana/9.jpeg')
bordes1 = cv2.Canny(img1, 135, 150)
bordes2 = cv2.Canny(img2, 135, 150)
bordes3 = cv2.Canny(img3, 135, 150)
bordes4 = cv2.Canny(img4, 135, 150)
bordes5 = cv2.Canny(img5, 135, 150)
bordes6 = cv2.Canny(img6, 135, 150)
bordes7 = cv2.Canny(img7, 135, 150)
bordes8 = cv2.Canny(img8, 135, 150)
bordes9 = cv2.Canny(img9, 135, 150)

# cv2.imshow("Bordes", bordes)

num_kpt = 1000
# Declaramos el objeto
orb = cv2.ORB_create(num_kpt)
# Extraemos la info de la img
# keypoint1, descriptor1 = orb.detectAndCompute(img, None)
keypoint1, descriptor1 = orb.detectAndCompute(bordes1, None)
keypoint2, descriptor2 = orb.detectAndCompute(bordes2, None)
keypoint3, descriptor3 = orb.detectAndCompute(bordes3, None)
keypoint4, descriptor4 = orb.detectAndCompute(bordes4, None)
keypoint5, descriptor5 = orb.detectAndCompute(bordes5, None)
keypoint6, descriptor6 = orb.detectAndCompute(bordes6, None)
keypoint7, descriptor7 = orb.detectAndCompute(bordes7, None)
keypoint8, descriptor8 = orb.detectAndCompute(bordes8, None)
keypoint9, descriptor9 = orb.detectAndCompute(bordes9, None)

npKeyPoint1 = np.array(keypoint1)
npDescriptor1 = np.array(descriptor1)
print(npKeyPoint1.shape)
print(npDescriptor1.shape)