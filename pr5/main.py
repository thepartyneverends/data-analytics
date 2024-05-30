import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control


quality = control.Antecedent(np.arange(0, 11, 1), 'quality')
service = control.Antecedent(np.arange(0, 11, 1), 'service')
tip = control.Antecedent(np.arange(0, 26, 1), 'tip')

quality.automf(3)
service.automf(3)

tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 0, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 13, 25])

quality['average'].view()
service.view()
tip.view()

rule1 = control.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = control.Rule(service['average'], tip['medium'])
rule3 = control.Rule(service['good'] | quality['good'], tip['high'])
rule1.view()

tipping_ctrl = control.ControlSystem([rule1, rule2, rule2])
tipping = control.ControlSystemSimulation(tipping_ctrl)

tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8

tipping.compute()
print(tipping.output['tip'])
tip.view(sim=tipping)

plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()
