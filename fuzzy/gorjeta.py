import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

qualidade = ctrl.Antecedent(np.arange(0, 11, 1), 'qualidade')
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')

gorjeta = ctrl.Consequent(np.arange(0, 21, 1), 'gorjeta')

qualidade.automf(number=3, names=['ruim', 'boa', 'saborosa'])
servico.automf(number=3, names=['ruim', 'aceitavel', 'otimo'])


""" gorjeta['baixa'] = fuzz.trimf(gorjeta.universe, [0, 0, 8])
gorjeta['media'] = fuzz.trimf(gorjeta.universe, [2, 10, 18])
gorjeta['alta'] = fuzz.trimf(gorjeta.universe, [12, 20, 20]) """

gorjeta['baixa'] = fuzz.sigmf(gorjeta.universe, 5, -1)
gorjeta['media'] = fuzz.gaussmf(gorjeta.universe, 10, 3)
gorjeta['alta'] = fuzz.pimf(gorjeta.universe, 10, 20, 20, 21)

regra1 = ctrl.Rule(qualidade['ruim'] | servico['ruim'], gorjeta['baixa'])
regra2 = ctrl.Rule(qualidade['boa'] & servico['aceitavel'], gorjeta['media'])
regra3 = ctrl.Rule(qualidade['saborosa'] & servico['otimo'], gorjeta['alta'])

sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3])

sistema = ctrl.ControlSystemSimulation(sistema_controle)

sistema.input['qualidade'] = 1
sistema.input['servico'] = 1
sistema.compute()

print(sistema.output['gorjeta'])

gorjeta.view(sim = sistema)

plt.savefig('gorjeta_plot.png')
plt.show()