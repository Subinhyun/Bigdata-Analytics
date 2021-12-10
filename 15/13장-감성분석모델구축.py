#*************************************************
# 13장. 텍스트마이닝_감성분석을 위한 학습모델 구축
#*************************************************
import pandas as pd
pd.show_versions()

# pyLDAvis 모듈의 정상 실행을 위하여 pandas 를 upgrade 필요(1.1.4 이상으로 )

!pip install --upgrade pandas

pd.__version__

# spyder 재실행 


# 한글 UnicoedEncodingError를 방지하기 위해 기본 인코딩을 "utf-8"로 설정
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# 경고메시지 표시 안하게 설정하기
import warnings
warnings.filterwarnings(action='ignore')

#깃허브에서 데이터 파일 다운로드 : https://github.com/e9t/nsmc
# 데이터 준비 및 탐색
# 훈련용 데이터 준비

#nsmc => Naver sentiment movie corpus 

nsmc_train_df = pd.read_csv('D:/BD/ch13/ratings_train.txt', encoding='utf-8', sep='\t', engine='python')
nsmc_train_df.head()

nsmc_train_df.info()

#결측치 제거 
# 'document'칼럼이 Null인 샘플 제거

nsmc_train_df = nsmc_train_df[nsmc_train_df['document'].notnull()]
nsmc_train_df.info()


#타겟 컬럼 label 확인 (0: 부정감성, 1: 긍정감성)
nsmc_train_df['label'].value_counts()

# 한글 이외의 문자는 공백으로 변환 (정규표현식 이용)

import re
# nsmc_train_df 확인 
nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
nsmc_train_df.head()

# nsmc_train_df 재확인 => ! ... 등이 삭제되었음을 확인  

nsmc_temp = nsmc_train_df.loc[1:100]

#  평가용 데이터 준비
nsmc_test_df = pd.read_csv('D:/BD/ch13/ratings_test.txt', encoding='utf8', sep='\t', engine='python')
nsmc_test_df.head()

nsmc_test_df.info()
nsmc_test_df = nsmc_test_df[nsmc_test_df['document'].notnull()]
print(nsmc_test_df['label'].value_counts())

# 한글 이외의 문자는 공백으로 변환 (정규표현식 이용)
nsmc_test_df['document'] = nsmc_test_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', "", x))

# *********************************************
#              분석 모델 구축
#**********************************************
# 피처 벡터화(Feature Vectorization) : TF-IDF
# 형태소를 분석하여 토큰화 : 한글 형태소 엔진으로 Okt 이용

!pip install konlpy

from konlpy.tag import Okt  #okt => Open Korea Text

okt = Okt() 

def okt_tokenizer(text):
    tokens = okt.morphs(text)  #형태소 추출 
    return tokens

# ---- TF-IDF 기반 피처 벡터 생성 : 실행시간 20분 정도 걸립니다 -----
# TF-IDF =>  https://chan-lab.tistory.com/24 참고 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfidf.fit(nsmc_train_df['document'])  # 약 8분 소요 
tfidf.vocabulary_  # # 벡터라이저가 학습한 단어사전을 출력
len(tfidf.vocabulary_)
tfidf.idf_.shape
sorted(tfidf.vocabulary_.items()) # 단어사전을 정렬
#단어들의 가중치 출력 
#tfidf.idf_
#tfidf.idf_.shape

#**************************************************
# 문서별 단어들의 가중치 계산 
# 약 10분 소요 
nsmc_train_tfidf = tfidf.transform(nsmc_train_df['document'])
nsmc_train_tfidf.shape
#temp= nsmc_train_tfidf.toarray()
#**************************************************

# 테스트 용--------------

tfidf_temp = TfidfVectorizer(tokenizer = okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfidf_temp.fit(nsmc_temp['document'])  
tfidf_temp.vocabulary_  # # 벡터라이저가 학습한 단어사전을 출력
len(tfidf_temp.vocabulary_)
tfidf_temp.idf_.shape
sorted(tfidf_temp.vocabulary_.items()) # 단어사전을 정렬
#단어들의 가중치 출력 
#tfidf_temp.idf_
#tfidf_temp.idf_.shape

nsmc_train_tfidf_temp = tfidf_temp.transform(nsmc_temp['document']) # 약 10분 소요 
nsmc_train_tfidf_temp.shape
temp= nsmc_train_tfidf_temp.toarray()
#--------------------------


# 감성 분류 모델 구축 : 로지스틱 회귀를 이용한 이진 분류
# Sentiment Analysis using Logistic Regression
# 로지스틱 회귀 기반 분석모델 생성

from sklearn.linear_model import LogisticRegression

#SA : Sentiment Analysis, lr : Logistic Regression 

SA_lr = LogisticRegression(random_state = 0)
SA_lr.fit(nsmc_train_tfidf, nsmc_train_df['label'])


#**************************************************************************
# 과적합(Overfitting) 문제 해결 => 계수들을 조정하면서 정확도 분석 
# 로지스틱 회귀의 best 하이퍼파라미터 찾기


from sklearn.model_selection import GridSearchCV

params = {'C': [1, 3, 3.5, 4, 4.5, 5]}
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params, cv=3, scoring='accuracy', verbose=1)
# C : 계수값을 조정하기 위한 기준값들 
# cv = 3: Cross Validation을 3번 수행

SA_lr_grid_cv.fit(nsmc_train_tfidf, nsmc_train_df['label'])

print(SA_lr_grid_cv.best_params_, round(SA_lr_grid_cv.best_score_, 4))

# 최적 파라미터의 best 모델 저장
SA_lr_best = SA_lr_grid_cv.best_estimator_

#*****************************************************************************

# 분석모델 평가 

# 평가용 데이터의 피처 벡터화 : 실행시간 6분 정도 걸림 
# 평가용 데이터도 단어들으 가중치 값으로 적용
nsmc_test_tfidf = tfidf.transform(nsmc_test_df['document'])

test_predict = SA_lr_best.predict(nsmc_test_tfidf)

from sklearn.metrics import accuracy_score

print('감성 분석 정확도 : ', round(accuracy_score(nsmc_test_df['label'], test_predict), 3))

#새로운 텍스트에 대한 감성 예측


st = input('감성 분석할 문장입력 >> ')

# 0) 입력 텍스트에 대한 전처리 수행
st = re.compile(r'[ㄱ-ㅣ가-힣]+').findall(st)
print(st)
st = [" ".join(st)]
print(st)

# 1) 입력 텍스트의 피처 벡터화
st_tfidf = tfidf.transform(st)

# 2) 최적 감성분석 모델에 적용하여 감성 분석 평가
st_predict = SA_lr_best.predict(st_tfidf)

# 3) 예측 값 출력하기
if(st_predict== 0):
    print(st , "->> 부정 감성")
else :
    print(st , "->> 긍정 감성")

