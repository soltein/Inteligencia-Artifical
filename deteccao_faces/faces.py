import cv2 # OpenCV
from matplotlib import pyplot as plt

imagem = cv2.imread('deteccao_faces/workplace-1245776_1920.jpg')
img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
imagem_cinza = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

detector_face = cv2.CascadeClassifier('deteccao_faces/haarcascade_frontalface_default.xml')
#deteccoes = detector_face.detectMultiScale(imagem_cinza, scaleFactor=1.3, minSize=(30,30))
deteccoes = detector_face.detectMultiScale(imagem_cinza, scaleFactor=1.3, minNeighbors=4, minSize=(60,60))
print(deteccoes)
for (x, y, l, a) in deteccoes:
  print(x, y, l, a)
  cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 2)

imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

plt.imshow(imagem)
plt.axis('off')  # Desligar os eixos
plt.show()

