#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols, glm

#회귀분석 => https://www.youtube.com/watch?v=X4IpcWeZHjs&list=PLFzJCjLyAoD0pDHwkN8o8fVTq9rePIe2p

# Read the data set into a pandas DataFrame
wine = pd.read_csv('winequality-both.csv', sep=',', header=0)
wine.columns = wine.columns.str.replace(' ', '_')
print(wine.head())



#회귀식 설정 
# 종속변수 ~ 독립변수1 + 독립변수2 + ...
my_formula = 'quality ~ alcohol + chlorides + citric_acid + density + fixed_acidity + free_sulfur_dioxide + pH + residual_sugar + sulphates + total_sulfur_dioxide + volatile_acidity'
#formula_all = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol'
#formula = 'quality ~ residual_sugar + alcohol'
#wine_standardized = (wine - wine.mean()) / wine.std()
#lm = ols(my_formula, data = wine_standardized).fit()


#회귀식은 wine 데이터에 적용
#ols()함수 => 최소제곱법에 의한 회귀분석 함수 

lm = ols(my_formula, data=wine).fit()

#lm = glm(my_formula, data=wine, family=sm.families.Gaussian()).fit()

print(lm.summary())
print("\nQuantities you can extract from the result:\n%s" % dir(lm))
print("\nCoefficients:\n%s" % lm.params)  # 회귀계수 
print("\nCoefficient Std Errors:\n%s" % lm.bse) # 표준오차
print("\nAdj. R-squared:\n%.2f" % lm.rsquared_adj) # R스퀘어(수정결정계수)
print("\nF-statistic: %.1f  P-value: %.2f" % (lm.fvalue, lm.f_pvalue))  # 모형적합도 
print("\nNumber of obs: %d  Number of fitted values: %s" % (lm.nobs, len(lm.fittedvalues)))

# Fit a multivariate linear model with standardized independent variables

# 데이터 표준화에 의한 회귀분석
dependent_variable = wine['quality']
independent_variables = wine[wine.columns.difference(['quality', 'type', 'in_sample'])]  #특정열 제외 
independent_variables_standardized = (independent_variables - independent_variables.mean()) / independent_variables.std()
wine_standardized = pd.concat([dependent_variable, independent_variables_standardized], axis=1)
lm_standardized = ols(my_formula, data=wine_standardized).fit()
print(lm_standardized.summary())

# 회귀식에 의한 예측 예측
# Predict quality scores for "new" observations
new_observations = wine.loc[wine.index.isin(range(10)), independent_variables.columns]
y_predicted = lm.predict(new_observations)
y_predicted2 = lm_standardized .predict(new_observations)
y_predicted_rounded = [round(score, 2) for score in y_predicted]
print(y_predicted_rounded)































