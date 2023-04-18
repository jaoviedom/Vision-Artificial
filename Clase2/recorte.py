# Importamos las librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Creamos imagen
# Vamos a crear nuestra matriz principal
img = 100 * np.ones((10,10,3), np.uint8)
img2 = 100 * np.ones((10,10,3), np.uint8)

# Modificamos los pixeles que vamos a extraer
# Modificamos el pixel (4,5) de la matriz R
img[4,5,0] = 255
img[4,4,0] = 0
img[5,4,0] = 255
img[5,5,0] = 0

# Modificamos el pixel (4,5) de la matriz G
img[4,5,1] = 0
img[4,4,1] = 255
img[5,4,1] = 0
img[5,5,1] = 255

# Modificamos el pixel (4,5) de la matriz B
img[4,5,2] = 255
img[4,4,2] = 255
img[5,4,2] = 0
img[5,5,2] = 0

# Realizamos recorte (filas, columnas)
# Desde fila 3 hasta 7
# Desde columna 3 hasta 7
recorte = img[4:6, 4:6]

# Mostramos nuestros canales
fig = plt.figure()
# imagen original
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img)
ax1.set_title("Original")

# recorte
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(recorte)
ax2.set_title("Recortada")

plt.show()

# Ahora con imagenes
# Leemos la img
imagen =cv2.imread('./Clase2/ironman.png')

# Truco paint
rct = imagen[0:80, 120:180]

# Mostramos el recorte
cv2.imshow('IMAGEN ORIGINAL', imagen)
cv2.imshow('RECORTE', rct)

cv2.waitKey(0)