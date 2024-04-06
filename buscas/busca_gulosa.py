from grafo import Grafo
from vetor_ordenado import VetorOrdenadoGulosa

grafo = Grafo()

class Gulosa:
    def __init__(self, objetivo):
        self.objetivo = objetivo
        self.encontrado = False

    def buscar(self, atual):
        print("--------")
        print("Atual: {}".format(atual.rotulo))
        atual.visitado = True

        if atual == self.objetivo:
            self.encontrado = True
        else:
            vetor_ordenado = VetorOrdenadoGulosa(len(atual.adjacentes))
            for adjacente in atual.adjacentes:
                if adjacente.vertice.visitado == False:
                    adjacente.vertice.visitado = True
                    vetor_ordenado.insere(adjacente.vertice)
            
            vetor_ordenado.imprime()

            if vetor_ordenado.valores[0] != None:
                self.buscar(vetor_ordenado.valores[0])

busca_gulosa = Gulosa(grafo.bucharest)
busca_gulosa.buscar(grafo.arad) 
