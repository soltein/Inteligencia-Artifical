import cv2 # OpenCV
from matplotlib import pyplot as plt

imagem = cv2.imread('deteccao_faces/pessoas.jpg')
img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
imagem_cinza = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

detector_body = cv2.CascadeClassifier('deteccao_faces/fullbody.xml')
#deteccoes = detector_body.detectMultiScale(imagem_cinza, scaleFactor=1.3, minNeighbors=4, minSize=(60,60))
deteccoes = detector_body.detectMultiScale(imagem_cinza, scaleFactor=1.006, minSize=(50,50))
print(deteccoes)
for (x, y, l, a) in deteccoes:
  print(x, y, l, a)
  cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 2)

imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

plt.imshow(imagem)
plt.axis('off')  # Desligar os eixos
plt.show()

