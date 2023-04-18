#Librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Imagen color
#Creación
img = 100 * np.ones((10, 10, 3), np.uint8) # Cada pixel comienza con valor de 100

#Extraer canales
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

# Modificar el canal rojo
# R[:,:] = 0

# Modificar otros canales
R[:,:] = 255
G[:,:] = 255
B[:,:] = 0

#Mostar los valores numéricos
print(img)

#Mostrar imagen
plt.imshow(img)
plt.show()