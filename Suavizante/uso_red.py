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
        data = xPoints = yPoints = angles = octaves = responses = sizes = classes = []

        for kp in keypoint:
            x,y = kp.pt
            xPoints.append(x)
            yPoints.append(y)
            angles.append(kp.angle)
            octaves.append(kp.octave)
            responses.append(kp.response)
            sizes.append(kp.size)
            classes.append(kp.class_id)
        
        print(descriptor.shape)
        
        c = 0
        for i in xPoints:
            c += 1
            print(c)

        xPointsNP = np.array(xPoints)
        print(xPointsNP.shape)
        # arr2 = np.column_stack((descriptor, x_np))
        # arr2 = np.append(descriptor, [x_np], axis = 1)

        # df = pd.DataFrame(descriptor)
        # print(df)
        # # df.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']
        
        # df_xPoints = pd.DataFrame(xPoints)
        # df_yPoints = pd.DataFrame(yPoints)
        # df_angles = pd.DataFrame(angles)
        # df_octaves = pd.DataFrame(octaves)
        # df_responses = pd.DataFrame(responses)
        # df_sizes = pd.DataFrame(sizes)
        # df_classes = pd.DataFrame(classes)

        # df['X'] = df_xPoints
        # df['Y'] = df_yPoints
        # df['Angles'] = df_angles
        # df['Octaves'] = df_octaves
        # df['Responses'] = df_responses
        # df['Sizes'] = df_sizes
        # df['Classes'] = df_classes

        # print(descriptor.shape)
        # x_min_max_train = preprocessing.MinMaxScaler().fit_transform(data)

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