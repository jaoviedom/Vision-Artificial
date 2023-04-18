# Importar librerias
import cv2

# Creamos la Video Captura
cap = cv2.VideoCapture(1)
# Relación de aspecto
ancho = int(cap.get(16))
alto = int(cap.get(9))

print(ancho, alto)

# cv2.VideoWriter(Nombre, Codificacion, FPS, Tamaño)
out = cv2.VideoWriter('Video1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920, 1080))

# Creamos un ciclo para ejecutar nuestros Frames
while cap.isOpened():
    # Leemos los fotogramas
    ret, frame = cap.read()

    if ret:
        # Guardamos el video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out.write(frame)

        # Mostramos los Frames
        cv2.imshow("VIDEO CAPTURA", frame)

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