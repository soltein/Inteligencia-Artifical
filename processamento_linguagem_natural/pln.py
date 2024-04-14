import bs4 as bs
import urllib.request
import nltk
import spacy
import nltk
from spacy.lang.pt.stop_words import STOP_WORDS
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
import matplotlib.pyplot as plt

pln = spacy.load('pt_core_news_sm')
#print(pln)

#documento = pln('Estou aprendendo processamento de linguagem natural, curso em Curitiba')
#print(type(documento))

#for token in documento:
#  print(token.text, token.pos_)

#for token in documento:
#  print(token.text, token.lemma_)  

#nltk.download('rslp')

#stemmer = nltk.stem.RSLPStemmer()
#print(stemmer.stem('aprender'))

#for token in documento:
#  print(token.text, token.lemma_, stemmer.stem(token.text))

#dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')
dados = urllib.request.urlopen('https://oglobo.globo.com/mundo/noticia/2024/04/13/ira-alerta-eua-para-ficar-fora-de-conflito-com-israel.ghtml')
dados = dados.read()
#print(dados)

dados_html = bs.BeautifulSoup(dados, 'lxml')
#print(dados_html)

paragrafos = dados_html.find_all('p')
#print(len(paragrafos))
#print(paragrafos[1])

conteudo = ''
for p in paragrafos:
  conteudo += p.text

conteudo = conteudo.lower()

#print(conteudo)
#print(STOP_WORDS)

doc = pln(conteudo)
lista_token = []
for token in doc:
  lista_token.append(token.text)

#print(lista_token)

sem_stop = []
for palavra in lista_token:
  if pln.vocab[palavra].is_stop == False:
    sem_stop.append(palavra)

#print(sem_stop)

color_map = ListedColormap(['orange', 'green', 'red', 'magenta'])

cloud = WordCloud(background_color = 'white', max_words = 100, colormap=color_map)
cloud = cloud.generate(' '.join(sem_stop))
plt.figure(figsize=(15,15))
plt.imshow(cloud)
plt.axis('off')
plt.show()