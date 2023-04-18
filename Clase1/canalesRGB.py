#Librerías
import cv2
import matplotlib.pyplot as plt

# Leer imagen
img = cv2.imread('./Clase1/ironman.png')

# Corrección del color para pasar de BGR a RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Extraer canales
# R = img[:,:,0]
# G = img[:,:,1]
# B = img[:,:,2]
R, G, B = cv2.split(img)

print(img)

# Mostrar los canales
fig = plt.figure()

# Canal rojo
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(R, cmap='gray')
ax1.set_title('Canal rojo')

# Canal verde
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(G, cmap="gray")
ax2.set_title("Canal verde")

# Canal azul
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(B, cmap="gray")
ax3.set_title("Canal azul")

# RECONSTRUCCION
imgre = cv2.merge((R,G,B))

# Imagen original reconstruida desde los canales
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(imgre)
ax4.set_title("Original")

plt.show()