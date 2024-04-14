import pandas as pd
import string
import spacy
import random
import seaborn as sns
import numpy as np
from spacy.lang.pt.stop_words import STOP_WORDS
from spacy.training import Example
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

#base_dados = pd.read_csv('processamento_linguagem_natural/base_treinamento.txt', encoding = 'utf-8')
base_dados = pd.read_csv('processamento_linguagem_natural/Twitter/Train50.csv', delimiter=';', encoding = 'utf-8')

pontuacoes = string.punctuation
#print(pontuacoes)

stop_words = STOP_WORDS
"""
#print(stop_words)
"""
pln = spacy.load("pt_core_news_sm")

def preprocessamento(texto):
  texto = texto.lower()
  documento = pln(texto)

  lista = []
  for token in documento:
    #lista.append(token.text)
    lista.append(token.lemma_)

  lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
  lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

  return lista

#teste = preprocessamento('Estou prendendo 1 10 23 processamento de linguagem natural, Curso em Curitiba')
#print(teste)

base_dados['texto'] = base_dados['texto'].apply(preprocessamento)

dic = {}
base_dados_final = []
for texto, emocao in zip(base_dados['texto'], base_dados['emocao']):
  #print(texto, emocao)
 
  if emocao == 1:
    dic = ({'POSITIVO': True, 'NEGATIVO': False})
  elif emocao == 0:
    dic = ({'POSITIVO': False, 'NEGATIVO': True})

  base_dados_final.append([texto, dic.copy()])

modelo = spacy.blank('pt')
textcat = modelo.add_pipe("textcat")
textcat.add_label("POSITIVO")
textcat.add_label("NEGATIVO")
historico = []  
"""
modelo.begin_training()
for epoca in range(3):
  random.shuffle(base_dados_final)
  losses = {}

  for batch in spacy.util.minibatch(base_dados_final, 512):
    textos = [modelo(texto) for texto, entities in batch]
    annotations = [{'cats': entities} for texto, entities in batch]
    examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(textos, annotations)]
    modelo.update(examples, losses=losses)

  if epoca % 5 == 0:
    print(losses)
    historico.append(losses)

historico_loss = []
for i in historico:
  historico_loss.append(i.get('textcat'))

historico_loss = np.array(historico_loss)
print(historico_loss)

plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Épocas')
plt.ylabel('Erro')

#modelo.to_disk("modelo")
modelo.to_disk("modelo_twitter")

"""
#modelo_carregado = spacy.load("modelo")
modelo_carregado = spacy.load("modelo_twitter")
"""

texto_positivo = 'eu adoro cor dos seus olhos'
texto_positivo = preprocessamento(texto_positivo)
#print(texto_positivo)

previsao = modelo_carregado(texto_positivo)
#print(previsao)

texto_negativo = 'estou com medo dele'
previsao = modelo_carregado(preprocessamento(texto_negativo))
print(previsao.cats)
"""

#base_dados = pd.read_csv('processamento_linguagem_natural/base_teste.txt', encoding = 'utf-8')
base_dados = pd.read_csv('processamento_linguagem_natural/Twitter/Test.csv', delimiter=';', encoding = 'utf-8')
base_dados['texto'] = base_dados['texto'].apply(preprocessamento)
previsoes = []
for texto in base_dados['texto']:
  #print(texto)
  previsao = modelo_carregado(texto)
  previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
  if previsao['POSITIVO'] > previsao['NEGATIVO']:
    previsoes_final.append(1)
  else:
    previsoes_final.append(0)

previsoes_final = np.array(previsoes_final)

respostas_reais = base_dados['emocao'].values

cm = confusion_matrix(respostas_reais, previsoes_final)

print(cm)
