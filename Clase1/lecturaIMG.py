#Librerías
import cv2
import matplotlib.pyplot as plt

# Leer imagen en gris
imgGray = cv2.imread('ironman.png', 0) # 1 canal

# Leer imagen en RGB
imgRGB = cv2.imread('ironman.png', 1) # 3 canales

# Leer imagen
img = cv2.imread('ironman.png') # 3 canales

# Extraer los atributos principales
tamanio = imgGray.shape
print('Tamaño Gray:', tamanio)
tipoRGB = imgGray.dtype
print('Tipo de dato Gray:', tipoRGB)
tamanio = imgRGB.shape
print('Tamaño RGB:', tamanio)
tipoRGB = imgRGB.dtype
print('Tipo de dato RGB:', tipoRGB)


# Mostrar imagen
cv2.imshow('Gris', imgGray)
cv2.waitKey(0) # Con el teclado se cierra la ventana de la imagen
cv2.imshow('RGB', imgRGB)
cv2.waitKey(0) # Con el teclado se cierra la ventana de la imagen


# Mostrar imagen sin corrección
plt.imshow(img)
plt.show()

# Corrección del color para pasar de BGR a RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Mostrar imagen corregida
plt.imshow(img)
plt.show()