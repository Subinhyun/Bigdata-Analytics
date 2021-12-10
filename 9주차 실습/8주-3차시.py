import pandas as pd
from statsmodels.formula.api import ols, glm

wine = pd.read_csv("D:/BD/WINE/wine.csv", sep=',', header=0, engine='python')

wine.head()
wine.columns = wine.columns.str.replace(' ', '_')

Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + \
      residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + \
      density + pH + sulphates + alcohol'

regression_result = ols(Rformula, data = wine).fit()

sample1 = wine[wine.columns.difference(['quality', 'type'])]

sample1 = sample1[0:5][:]

sample1_predict = regression_result.predict(sample1)

sample1_predict

wine[0:5]['quality']

data = {"fixed_acidity" : [8.5, 8.1], "volatile_acidity":[0.8, 0.5], \
"citric_acid":[0.3, 0.4], "residual_sugar":[6.1, 5.8], "chlorides":[0.055, \
0.04], "free_sulfur_dioxide":[30.0, 31.0], "total_sulfur_dioxide":[98.0, \
99], "density":[0.996, 0.91], "pH":[3.25, 3.01], "sulphates":[0.4, 0.35], \
"alcohol":[9.0, 0.88]}

sample2 = pd.DataFrame(data, columns= sample1.columns)
sample2
sample2_predict = regression_result.predict(sample2)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')

red_wine_quality = wine.loc[wine['type'] == 'red', 'quality']
white_wine_quality = wine.loc[wine['type'] == 'white', 'quality']

sns.distplot(red_wine_quality, kde = True,  color = "red", label = 'red  wine')
sns.distplot(white_wine_quality, kde = True, label = 'white wine')
plt.title("Quality of Wine Type")
plt.legend()

plt.show()


import statsmodels.api as sm

fig = plt.figure(figsize = (8,13))
sm.graphics.plot_partregress_grid(regression_result, fig = fig)

others = list(set(wine.columns).difference(set(["quality", "fixed_acidity"])))


p, resids = sm.graphics.plot_partregress("quality", "fixed_acidity", others, data = wine, ret_coords = True)

sm.graphics.plot_partregress("quality", "fixed_acidity", others, data = wine, ret_coords = True)
plt.show()

































