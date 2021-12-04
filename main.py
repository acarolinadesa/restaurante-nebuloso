import skfuzzy as fuzz #scikit-fuzzy
import numpy as np
from skfuzzy import control as ctrl

# Variáveis do problema
comida = ctrl.Antecedent(np.arange(0, 11, 1), 'comida') #avaliacao de comida de 0 a 10
atendimento = ctrl.Antecedent(np.arange(0, 11, 1), 'atendimento') #avaliacao de atendimento de 0 a 10
valorBrinde = ctrl.Consequent(np.arange(0, 50, 1), 'valorBrinde') #valorBrinde de 0 a 50


# Definição do conjunto difuso/nebuloso usando
# função de pertinencia padrão(triângulo)
comida.automf(names=['ruim', 'aceitavel', 'deliciosa'])

# Definição de funções de pertinência usando tipos variados
# trimf - função de triângulo - onde começa, ponto alto e ponto mínimo
# gaussmf - função glauciano (scikit-fuzzy) - valor médio e desvio padrão
atendimento['ruim'] = fuzz.trimf(atendimento.universe, [0, 0, 10])
atendimento['aceitavel'] = fuzz.gaussmf(atendimento.universe, 10, 2)
atendimento['otimo'] = fuzz.gaussmf(atendimento.universe, 50, 3)

valorBrinde['padrao'] = fuzz.trimf(valorBrinde.universe, [0, 0, 25])
valorBrinde['mediano'] = fuzz.trapmf(valorBrinde.universe, [0, 25, 40, 60])
valorBrinde['compensacao'] = fuzz.trimf(valorBrinde.universe, [40, 60, 60])

# Regras de decisões difusas
rule1 = ctrl.Rule(atendimento['otimo'] | comida['deliciosa'], valorBrinde['padrao'])
rule2 = ctrl.Rule(atendimento['aceitavel'], valorBrinde['mediano'])
rule3 = ctrl.Rule(atendimento['ruim'] & comida['ruim'], valorBrinde['compensacao'])

# Criando e simulando um controlador difuso
valorBrinde_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
valorBrinde_simulador = ctrl.ControlSystemSimulation(valorBrinde_ctrl)

# Entrando com valores para qualidade da comida e do atendimento
valorBrinde_simulador.input['comida'] = 1
valorBrinde_simulador.input['atendimento'] = 8

# Computando o resultado
valorBrinde_simulador.compute()
print(valorBrinde_simulador.output['valorBrinde'])

comida.view()