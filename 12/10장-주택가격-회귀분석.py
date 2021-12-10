# 향후 버전 업에 대한 경고 메시지 출력 안하기 
import warnings
warnings.filterwarnings(action='ignore') 

# 머신러닝 패키지 sklearn 설치
!pip install sklearn 

#데이터수집

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
boston =  load_boston()

#데이터 준비 및 탐색
print(boston.DESCR)
boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_df.head()

boston_df['PRICE'] = boston.target
boston_df.head()
print('보스톤 주택 가격 데이터셋 크기 : ', boston_df.shape)

boston_df.info()

#분석모델 구축

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# X, Y 분할하기
Y = boston_df['PRICE']
X = boston_df.drop(['PRICE'], axis=1, inplace=False)

# 훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=156)  #random_state :난수의 초기값 

# 선형회귀분석 : 모델 생성
lr = LinearRegression()
 
#선형회귀분석 : 모델 훈련
lr.fit(X_train, Y_train)

# 선형회귀분석 : 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = lr.predict(X_test)

# 결과분석 및 시각화 
#  MSE : 실제값과 예측값의 차이를 제곱하여 평균한 것
#  MSE에 루트를 씌운 것
# R² 는 분산 기반으로 예측 성능을 평가,  1에 가까울수록 예측 정확도가 높음
# R² = 예측값 Variance / 실제값 Variance

mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)

print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))


print('Y 절편 값: ', lr.intercept_)
print('회귀 계수 값: ', np.round(lr.coef_, 1))

#회귀 분석 결과를 산점도 + 선형 회귀 그래프로 시각화하기
import matplotlib.pyplot as plt
import seaborn as sns




ig, axs = plt.subplots(figsize=(16, 16), ncols=3, nrows=5)

x_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


for i, feature in enumerate(x_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x=feature, y='PRICE', data=boston_df, ax=axs[row][col])
      
      






















