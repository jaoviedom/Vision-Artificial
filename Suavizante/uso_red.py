import keras
import cv2
import pandas as pd
import numpy as np
from sklearn import preprocessing

model = keras.models.load_model('Suavizante/suavizante1.h5')

# Creamos la Video Captura desde el iPhone
cap = cv2.VideoCapture(1)

while cap.isOpened():
    # Leemos los fotogramas
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bordes = cv2.Canny(gray, 135, 150)
        num_kpt = 1000
        orb = cv2.ORB_create(num_kpt)
        keypoint, descriptor = orb.detectAndCompute(bordes, None)
        # kp_image = cv2.drawKeypoints(bordes, keypoint, None, color=(0, 255, 0), flags=0)
        # cv2.imshow('ORB Bordes', kp_image)
        cv2.imshow('Cámara', frame)
        # print(keypoint, descriptor)

        # Deconstuyendo keypoint
        data = []
        xPoints = []
        yPoints = []
        angles = []
        octaves = []
        responses = []
        sizes = []
        classes = []

        for kp in keypoint:
            x,y = kp.pt
            xPoints.append(x)
            yPoints.append(y)
            angles.append(kp.angle)
            octaves.append(kp.octave)
            responses.append(kp.response)
            sizes.append(kp.size)
            classes.append(kp.class_id)
        
        print('Descriptor', descriptor.shape, type(descriptor))

        xPointsNP = np.array(xPoints)
        yPointsNP = np.array(yPoints)
        anglesNP = np.array(angles)
        octavesNP = np.array(octaves)
        responsesNP = np.array(responses)
        sizesNP = np.array(sizes)
        classesNP = np.array(classes)

        xPointsS = xPointsNP.reshape(1000, 1)
        yPointsS = yPointsNP.reshape(1000, 1)
        print('Reshape x', xPointsS.shape)
        print('Reshape y', yPointsS.shape)

        # OJO
        np.insert(xPointsS, xPointsS.shape[1], yPointsS, axis = 1)
        print('Points', xPointsS.shape)

        # Cerramos con lectura de teclado
        t = cv2.waitKey(1)
        if t == 27:
            break
    
    else:
        break

# Liberamos la VideoCaptura
cap.release()
# Cerramos la ventana
cv2.destroyAllWindows()