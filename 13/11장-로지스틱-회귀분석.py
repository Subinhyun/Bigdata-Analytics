# 11장. 분류 분석 (1) 로지스틱 회귀 분석을 이용한 유방암 진단 프로젝트
# 1.0.1  1) 데이터 수집

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

b_cancer = load_breast_cancer()

# 1.0.2  2) 데이터 수집 및 탐색
print(b_cancer.DESCR)
b_cancer_df = pd.DataFrame(b_cancer.data, columns = b_cancer.feature_names)
#radius: 반지름, texture: 질감, perimeter:경계, concavity :오목함,....

b_cancer_df['diagnosis']= b_cancer.target
#악성:1, 양성:0
b_cancer_df.head()
print('유방암 진단 데이터셋 크기 : ', b_cancer_df.shape)
b_cancer_df.info()

# 데이터를 평균이 0, 분산이 1인 정규분포 이루도록 변환 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


#b_cancer_scaled = scaler.fit_transform(b_cancer_df)
b_cancer_scaled = scaler.fit_transform(b_cancer.data)
print(b_cancer.data[0])
print(b_cancer_scaled[0])      

# 1.0.3  3) 분석 모델 구축 : 로지스틱 회귀를 이용한 이진 분류 모델

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# # X, Y 설정하기
Y = b_cancer_df['diagnosis']
X = b_cancer_scaled 

# 훈련용 데이터와 평가용 데이터 분할하기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# 로지스틱 회귀 분석 : (1)모델 생성
lr_b_cancer = LogisticRegression()

# 로지스틱 회귀 분석 : (2)모델 훈련
lr_b_cancer.fit(X_train, Y_train)

# 로지스틱 회귀 분석 : (3)평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = lr_b_cancer.predict(X_test)

# 1.0.4  4) 결과 분석

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 오차 행렬 
cm = confusion_matrix(Y_test, Y_predict)

acccuracy = accuracy_score(Y_test, Y_predict) #정확도 
precision = precision_score(Y_test, Y_predict) # 정밀도 
recall = recall_score(Y_test, Y_predict)  #재현율 
f1 = f1_score(Y_test, Y_predict)  #   # 정밀도와 재현율의 조화평균 
roc_auc = roc_auc_score(Y_test, Y_predict) # ROC곡선 밑의 면적 => 1에 가까울수록 좋음 







 














