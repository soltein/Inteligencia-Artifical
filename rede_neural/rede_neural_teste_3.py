from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil as sh
import numpy as np

subdiretorio = "gato_cachorro"

caminho_treinamento = "rede_neural/" + subdiretorio + "/training_set/"
caminho_teste =  "rede_neural/" + subdiretorio + "/test_set/"

# Data augmentation para balanceamento de dados
gerador_treinamento = ImageDataGenerator(rescale=1./255,
                                         rotation_range=15,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True)

base_treinamento = gerador_treinamento.flow_from_directory(caminho_treinamento,
                                                           target_size=(64, 64),
                                                           batch_size=8,
                                                           class_mode='categorical')

gerador_teste = ImageDataGenerator(rescale=1./255)
base_teste = gerador_teste.flow_from_directory(caminho_teste,
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='categorical',
                                               shuffle=False)

# Arquitetura do modelo
rede_neural = Sequential()
rede_neural.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2, 2)))
rede_neural.add(Conv2D(128, (3, 3), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2, 2)))
rede_neural.add(Conv2D(256, (3, 3), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2, 2)))
rede_neural.add(Flatten())
rede_neural.add(Dense(units=512, activation='relu'))
rede_neural.add(Dropout(0.3))
rede_neural.add(Dense(units=2, activation='softmax'))

rede_neural.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
history = rede_neural.fit(base_treinamento, epochs=50, validation_data=base_teste)

# Avaliação do modelo
previsoes = rede_neural.predict(base_teste)
previsoes2 = np.argmax(previsoes, axis=1)
print(previsoes2)
print(base_teste.classes)

# Identificação e cópia das imagens incorretamente classificadas
diretorio_origem = "rede_neural/" + subdiretorio + "/test_set/"
diretorio_destino = "rede_neural/" + subdiretorio + "/imagens_incorretas/"

for i in range(len(previsoes2)):
    if previsoes2[i] != base_teste.classes[i]:
        print(f"Imagem {i + 1} - Classe real: {base_teste.classes[i]}, Previsão: {previsoes2[i]}")
        nome_arquivo = base_teste.filenames[i]
        sh.copyfile(diretorio_origem + nome_arquivo, diretorio_destino + nome_arquivo)
