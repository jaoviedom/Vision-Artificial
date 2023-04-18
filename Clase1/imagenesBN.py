#Librerías
import numpy as np
import matplotlib.pyplot as plt

#Imagen negra
#Creación
img = np.zeros((10, 10, 1), np.uint8)

#Cambiar algunos colores
#   F, C
img[0, 1] = 30
img[2, 3] = 50
img[4, 5] = 200
img[6, 7] = 140

#Mostar los valores numéricos
# print(img)

#Mostrar imagen
plt.imshow(img, cmap='gray')
plt.show()