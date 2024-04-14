import cv2
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
"""
#print(cv2.__version__)
caminho_train = 'deteccao_faces/yalefaces/train';
#print(os.listdir(caminho_train))


def dados_imagem():
  caminhos = [os.path.join(caminho_train, f) for f in os.listdir(caminho_train)]
  faces = []
  ids = []
  for caminho in caminhos:
    imagem = Image.open(caminho).convert('L')
    imagem_np = np.array(imagem, 'uint8')
    _id = int(os.path.split(caminho)[1].split('.')[0].replace('subject', ''))
    ids.append(_id)
    faces.append(imagem_np)
  return np.array(ids), faces

ids, faces = dados_imagem()

#print(ids)

lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')
"""
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read('classificadorLBPH.yml')

imagem_teste = 'deteccao_faces/yalefaces/test/subject11.glasses.gif'

imagem = Image.open(imagem_teste).convert('L')
imagem_np = np.array(imagem, 'uint8')
print(imagem_np)

idprevisto, _ = reconhecedor.predict(imagem_np)
idcorreto = int(os.path.split(imagem_teste)[1].split('.')[0].replace('subject', ''))
x,y = 0,0
cv2.putText(imagem_np, 'P: ' + str(idprevisto), (x,y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.putText(imagem_np, 'C: ' + str(idcorreto), (x,y + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))

imagem_np = cv2.cvtColor(imagem_np, cv2.COLOR_BGR2RGB)

plt.imshow(imagem_np)
plt.axis('off')  # Desligar os eixos
plt.show()