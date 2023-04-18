# Importamos las librerias
import cv2
import numpy as np

# Modos de ejecucion
# vc = 0 --> 48  # Captura de video
# fd = 1 --> 49  # Filtro desenfoque
# fe = 2 --> 50  # Filtro detector de esquinas
# fb = 3 --> 51  # Filtro de Bordes

MODO = {'Captura': 48, 'Desenfoque': 49, 'Bordes': 50, 'Esquinas': 51}

# Parametros para detector de esquinas
esquinas_param = dict(maxCorners=500,    # Maximo numero de esquinas a detectar
                      qualityLevel=0.2,  # Umbral minimo para la deteccion de esquinas
                      minDistance=15,    # Distacia entre pixeles
                      blockSize=10)       # Area de pixeles

# Modo
mode = 48

# Creamos la Video Captura
cap = cv2.VideoCapture(1)

# Creamos un ciclo para ejecutar nuestros Frames
while cap.isOpened():
    # Leemos los fotogramas
    ret, frame = cap.read()

    if ret:
        # Decidimos el mode
        # Normal
        if mode == MODO['Captura']:
            # Mostramos los frames
            resultado = frame

        # Desenfoque
        elif mode == MODO['Desenfoque']:
            # Modificamos frames
            resultado = cv2.blur(frame, (13, 13))


        # Bordes
        elif mode == MODO['Bordes']:
            # Modificamos frames
            # Umbral inferior y superior
            # resultado = cv2.Canny(frame, 50, 150) 
            resultado = cv2.Canny(frame, 135, 150) 

        # Esquinas
        elif mode == MODO['Esquinas']:
            # Obtenemos los frames
            resultado = frame
            # Conversion a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Calculamos las caracteristicas de las esquinas
            esquinas = cv2.goodFeaturesToTrack(gray, **esquinas_param)

            # Preguntamos si detectamos esquinas con esas caracteristicas
            if esquinas is not None:
                # Iteramos
                for x, y in np.float32(esquinas).reshape(-1, 2):
                    # Convertimos en enteros
                    x, y = int(x), int(y)
                    # Dibujamos la ubicacion de las esquinas
                    cv2.circle(resultado, (x, y), 10, (255, 0, 0), 2)


        # Si presionamos otra tecla
        elif mode != 48 or mode != 49 or mode != 50 or mode != 51 or mode != -1:
            # No hacemos nada
            resultado = frame

            # Imprimimos mensaje
            print('TECLA INCORRECTA')

        # Mostramos los Frames
        cv2.imshow("VIDEO CAPTURA", resultado)

        # Cerramos con lectura de teclado
        t = cv2.waitKey(1)
        # Salimos
        if t == 27:
            break
        # Modificamos Mood
        elif t != -1:
            mode = t

    else:
        break
# Liberamos la VideoCaptura
cap.release()
# Cerramos la ventana
cv2.destroyAllWindows()
