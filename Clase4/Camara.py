# Importamos librerias
import cv2

# Creamos la Video Captura
cap = cv2.VideoCapture(1)

# Creamos un ciclo para ejecutar nuestros Frames
while cap.isOpened():
    # Leemos los fotogramas
    ret, imagen = cap.read() #En ret se almacena si la captura se hace correctamente, en imagen los fotogramas

    # print(ret)
    if ret:
        # Mostramos los Frames
        cv2.imshow("Video captura", imagen)
        # Cerramos con lectura de teclado
        t = cv2.waitKey(1) 
        if t == 27: # Si es tecla Escape
            break
    else:
        break

# Liberamos la VideoCaptura
cap.release()
# Cerramos la ventana
cv2.destroyAllWindows()