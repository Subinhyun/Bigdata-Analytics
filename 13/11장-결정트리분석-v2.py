# 데이터준비 및 탐색
import numpy as np
import pandas as pd

# UCI HAR Data => UCI에서 만든 데이터셋 => Human Activity Recognition 데이터 -> https://archive.ics.uci.edu
# => 30명의 실험자가 스마트폰을 허리에 차고 다양한 행동(Walking, Sitting,..)을 했을 경우, 스마트폰의 가속도 센서와 자일로스코프 센서에서 데이터 센싱 및 수집
# => 561 동작으로 구분하여 움직임 데이터 기록

# 피처 이름(561개의 동작) 파일 읽어오기 
feature_name_df = pd.read_csv('D:/BD/ch11/UCI_HAR_Dataset/UCI_HAR_Dataset/features.txt', sep='\s+',  header=None, names=['index', 'feature_name'], engine='python')
feature_name_df.head()
feature_name_df.shape
feature_name_df.info()

# index 제거하고, feature_name만 리스트로 저장
feature_name = feature_name_df.iloc[:, 1].values.tolist()


#feature_name 중에서 중복되는 이름을 조정(번호 추가)
def Duplicate_Adj(feature_name):
    size = len(feature_name)
    count = 1
    for i in range(0, size):    
        for j in range(i+1, size):
            if feature_name[i] == feature_name[j]:
                feature_name[j] = feature_name[j]+str(j) 
        print(count, feature_name[i])
        count += 1
        
Duplicate_Adj(feature_name)        

X_train = pd.read_csv('D:/BD/ch11/UCI_HAR_Dataset/UCI_HAR_Dataset/train/X_train.txt', sep='\s+', names=feature_name, engine='python')
X_test = pd.read_csv('D:/BD/ch11/UCI_HAR_Dataset/UCI_HAR_Dataset/test/X_test.txt', sep='\s+', names=feature_name, engine='python')

Y_train = pd.read_csv('D:/BD/ch11/UCI_HAR_Dataset/UCI_HAR_Dataset/train/y_train.txt', sep='\s+', header=None, names=['action'], engine='python')
Y_test = pd.read_csv('D:/BD/ch11/UCI_HAR_Dataset/UCI_HAR_Dataset/test/y_test.txt', sep='\s+', header=None, names=['action'], engine='python')


X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
X_train.info()
X_train.head()
print(Y_train['action'].value_counts())

label_name_df = pd.read_csv('D:/BD/ch11/UCI_HAR_Dataset/UCI_HAR_Dataset/activity_labels.txt', sep='\s+',  header=None, names=['index', 'label'], engine='python')
label_name = label_name_df.iloc[:, 1].values.tolist()

# 1.0.4  3) 모델 구축 : 결정트리모델
from sklearn.tree import DecisionTreeClassifier

# 결정 트리 분류 분석 : 1) 모델 생성
# dt_HAR : Decision Tree Human Activity Recognition의 약어로 변수 이름 설정 

dt_HAR = DecisionTreeClassifier(random_state=156)
# 결정 트리 분류 분석 : 2) 모델 훈련
dt_HAR.fit(X_train, Y_train)

# 결정 트리 분류 분석 : 3) 평가 데이터에 대한 예측 수행 -> 예측 결과 Y_predict 구하기
Y_predict = dt_HAR.predict(X_test)

# 1.0.5  4) 결과 분석
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Y_predict)
print('결정 트리 예측 정확도 : {0:.4f}'.format(accuracy))

#1.0.5.1  ** 성능 개선을 위해 최적 파라미터 값 찾기 => Cross Validation 
print('결정 트리의 현재 하이퍼 파라미터 : \n', dt_HAR.get_params())

from sklearn.model_selection import GridSearchCV
# 1.0.5.2  최적 파라미터 찾기 - 1

params = {
    'max_depth' : [ 6, 8, 10, 12, 16, 20, 24]
}

#결정트리모델 7개(트리깊이 6, 8, 10, 12, 16, 20,24) 생성 준비 
grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy', 
                       cv=5, return_train_score=True)

#결정트리모델 7개 생성   => 몇분 정도 시간 소요
grid_cv.fit(X_train , Y_train)  

#7개 결정트리모델의 성능 비교 => Tree depth가 16인 경우가 가장 우수  
cv_results_df = pd.DataFrame(grid_cv.cv_results_)
cv_results_df[['param_max_depth', 'mean_test_score', 'mean_train_score']]

#결정트리모델 9개(트리깊이 6, 8, 10 각각 당, 샘플 수 8, 16, 24) 생성 준비  
params = {
    'max_depth' : [ 8, 16, 20 ],
    'min_samples_split' : [ 8, 16, 24 ]
}

grid_cv = GridSearchCV(dt_HAR, param_grid=params, scoring='accuracy', 
                       cv=5, return_train_score=True)

#결정트리모델 9개 생성   => 몇분 정도 시간 소요
grid_cv.fit(X_train , Y_train)

print('최고 평균 정확도 : {0:.4f}, 최적 하이퍼 파라미터 :{1}'.format(grid_cv.best_score_ , grid_cv.best_params_))

cv_results_df = pd.DataFrame(grid_cv.cv_results_)
cv_results_df[['param_max_depth','param_min_samples_split', 'mean_test_score', 'mean_train_score']]

best_dt_HAR = grid_cv.best_estimator_
best_Y_predict = best_dt_HAR.predict(X_test)
best_accuracy = accuracy_score(Y_test, best_Y_predict)

print('best 결정 트리 예측 정확도 : {0:.4f}'.format(best_accuracy))


# ** 중요 피처 확인하기
import seaborn as sns
import matplotlib.pyplot as plt

feature_importance_values = best_dt_HAR.feature_importances_
feature_importance_values_s = pd.Series(feature_importance_values, index=X_train.columns)
feature_top10 = feature_importance_values_s.sort_values(ascending=False)[:10]

plt.figure(figsize = (10, 5))
plt.title('Feature Top 10')
sns.barplot(x=feature_top10, y=feature_top10.index)
plt.show()


# pydot를 사용한 결정트리 시각화


from sklearn.tree import export_graphviz


!pip install pydot
import pydot

export_graphviz(best_dt_HAR, out_file="DecisionTree.dot", class_names=label_name , feature_names = feature_name, impurity=True, filled=True)

(graph,) = pydot.graph_from_dot_file("DecisionTree.dot", encoding='utf8')
graph.write_png("decisionTree2.png")
























