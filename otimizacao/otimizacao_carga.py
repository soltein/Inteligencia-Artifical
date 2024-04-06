import mlrose

produtos = [('Refrigerador A', 0.751, 999.90),
            ('Celular', 0.0000899, 2911.12),
            ('TV 55', 0.400, 4346.99),
            ('TV 50', 0.290, 3999.90),
            ('TV 42', 0.200, 2999.00),
            ('Notebook A', 0.00350, 2499.90),
            ('Ventilador', 0.496, 199.90),
            ('Microondas A', 0.0424, 308.66),
            ('Microondas B', 0.0544, 429.90),
            ('Microondas C', 0.0319, 299.29),
            ('Refrigerador B', 0.635, 849.00),
            ('Refrigerador C', 0.870, 1199.89),
            ('Notebook B', 0.498, 1999.90),
            ('Notebook C', 0.527, 3999.00)]
espaco_disponivel = 3

def imprimir_solucao(solucao):
    total = 0.0
    espaco = 0.0
    for i in range(len(solucao)):
        if(solucao[i] == 1):
            print("%s - %s" %(produtos[i][0], produtos[i][2]))
            total += produtos[i][2]
            espaco += produtos[i][1]
    print("Total: ", round(total, 2))
    print("Espaco: ", round(espaco, 2))

#imprimir_solucao([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1])

def fitness_function(solucao):
    total = 0.0
    espaco = 0.0
    for i in range(len(solucao)):
        if(solucao[i] == 1):
            total += produtos[i][2]
            espaco += produtos[i][1]

    if(espaco > espaco_disponivel):
        total = 0

    return total

#print(fitness_function([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]))
#print(fitness_function([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]))

fitness = mlrose.CustomFitness(fitness_function)

problema = mlrose.DiscreteOpt(length = 14, fitness_fn = fitness, maximize = True, max_val = 2)
#Hill Climb
melhor_solucao, melhor_custo = mlrose.hill_climb(problema)

print(melhor_solucao, melhor_custo)
imprimir_solucao(melhor_solucao)

#Simulated annealing
melhor_solucao, melhor_custo = mlrose.simulated_annealing(problema)
print(melhor_solucao, melhor_custo)
imprimir_solucao(melhor_solucao)

#Generic Algorithm
melhor_solucao, melhor_custo = mlrose.genetic_alg(problema, pop_size=500, mutation_prob=0.2)
print(melhor_solucao, melhor_custo)
imprimir_solucao(melhor_solucao)