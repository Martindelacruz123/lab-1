import cv2
print(cv2.__version__)

import urllib.request

# Asigna el valor de url y filename
trumptrudeau_url = "https://upload.wikimedia.org/wikipedia/commons/e/ee/Donald_Trump_and_Justin_Trudeau_in_the_Oval_Office_-_2017.jpg"
trumptrudeau_filename = "trumptrudeau.jpg"


urllib.request.urlretrieve(trumptrudeau_url, trumptrudeau_filename) # downloads file as "trumptrudeau.jpg"

import os
os.listdir(os.curdir)
# Muestra todos los archivos en el directorio actual

# ¿Esta trump_filename en el directorio?
print(trumptrudeau_filename in os.listdir(os.curdir))

from matplotlib import pyplot as plt
%matplotlib inline

trumptrudeau = cv2.imread(trumptrudeau_filename)

plt.imshow(trumptrudeau)

img_corrected = cv2.cvtColor(trumptrudeau, cv2.COLOR_BGR2RGB)

plt.imshow(img_corrected)

plt.axis("off") # Borra las marcas de los ejes
plt.imshow(img_corrected)

from pylab import rcParams

rcParams['figure.figsize'] = 10, 12

plt.axis("off") # Borra las marcas de los ejes
plt.imshow(img_corrected)

gray_trumptrudeau = cv2.cvtColor(trumptrudeau, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_trumptrudeau, cmap = 'gray')
plt.axis("off") # Borra las marcas de los ejes
plt.title('Grayscale Image')

rcParams['figure.figsize'] = 10, 12

edges = cv2.Canny(img_corrected,
                  threshold1=100,
                  threshold2=200)

plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

rcParams['figure.figsize'] = 10, 12

edges = cv2.Canny(img_corrected,
                  threshold1=1,   ## Prueba aquí distintos valores
                  threshold2=200) ## Prueba aquí distintos valores

plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

rcParams['figure.figsize'] = 10, 12

edges = cv2.Canny(img_corrected,
                  threshold1=100,   ## Prueba aquí distintos valores
                  threshold2=500) ## Prueba aquí distintos valores

plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

import cv2
import numpy as np
from matplotlib import pyplot as plt

rcParams['figure.figsize'] = 8,4

plt.hist(gray_trumptrudeau.ravel(),256,[0,256])
plt.title('Histogram of Grayscale trumptrudeau.jpg')
plt.show()

rcParams['figure.figsize'] = 10, 12

plt.axis("off") # Borra las marcas de los ejes
plt.imshow(img_corrected)

rcParams['figure.figsize'] = 8, 4

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([trumptrudeau],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()