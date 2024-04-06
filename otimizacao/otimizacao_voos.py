import mlrose

pessoas = [('Lisboa', 'LIS'),
           ('Madrid', 'MAD'),
           ('Paris', 'CDG'),
           ('Dublin', 'DUB'),
           ('Bruxelas', 'BRU'),
           ('Londres', 'LHR')]

destino = 'FCO'

voos = {}
for linha in open('otimizacao/flights.txt'):
  #print(linha)
  #print(linha.split(','))
  origem, destino, saida, chegada, preco = linha.split(',')
  voos.setdefault((origem, destino), [])
  voos[(origem, destino)].append((saida, chegada, int(preco)))


def imprimir_voos(agenda):
  id_voo = -1
  total_preco = 0
  for i in range(len(agenda) // 2):
    nome = pessoas[i][0]
    origem = pessoas[i][1]
    id_voo += 1
    ida = voos[(origem, destino)][agenda[id_voo]]
    total_preco += ida[2]
    id_voo += 1
    volta = voos[(destino, origem)][agenda[id_voo]]
    total_preco += volta[2]
    print('%10s%10s %5s-%5s %3s %5s-%5s %3s' % (nome, origem, ida[0], ida[1], ida[2],
                                                volta[0], volta[1], volta[2]))
  print('Preço total: ', total_preco)

agenda = [1,2, 3,2, 7,3, 6,3, 2,4, 5,3]

#Teste
#imprimir_voos(agenda)

def fitness_function(agenda):
    id_voo = -1
    total_preco = 0
    for i in range(len(agenda) // 2):
        origem = pessoas[i][1]
        id_voo += 1
        ida = voos[(origem, destino)][agenda[id_voo]]
        total_preco += ida[2]
        id_voo += 1
        volta = voos[(destino, origem)][agenda[id_voo]]
        total_preco += volta[2]

    return total_preco

fitness = mlrose.CustomFitness(fitness_function)
problema = mlrose.DiscreteOpt(length=12, fitness_fn=fitness, maximize=False, max_val=10)

#Hill climb
melhor_solucao, melhor_custo = mlrose.hill_climb(problema)
#print(melhor_solucao, melhor_custo)
imprimir_voos(melhor_solucao)

#Simulated annealing

melhor_solucao, melhor_custo = mlrose.simulated_annealing(problema)
#print(melhor_solucao, melhor_custo)
imprimir_voos(melhor_solucao)

#generic algorithm
melhor_solucao, melhor_custo = mlrose.genetic_alg(problema, pop_size=500, mutation_prob=0.2)
#print(melhor_solucao, melhor_custo)
imprimir_voos(melhor_solucao) 