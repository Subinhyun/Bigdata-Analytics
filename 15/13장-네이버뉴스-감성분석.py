
#감성 분석할 데이터 수집
#4장에서 학습한 네이버 API를 이용한 크롤링 프로그램을 이용하여, 네이버 뉴스를 크롤링하여 텍스트 데이터를 수집한다

import json

file_name = '코로나_naver_news'

with open('D:/BD/ch13/'+file_name+'.json', encoding='utf8') as j_f:
    data = json.load(j_f)

print(data)

# 분석할 컬럼을 추출하여 데이터 프레임에 저장

data_title =[]
data_description = []

for item in data:
    data_title.append(item['title'])
    data_description.append(item['description'])

data_title
data_description

import pandas as pd
data_df = pd.DataFrame({'title':data_title, 'description':data_description})

#한글 이외 문자 제거

import re
data_df['title'] = data_df['title'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
data_df['description'] = data_df['description'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))

data_df.head()  #작업 확인용 출력

data_df.to_csv('D:/BD/ch13/data_df.csv', encoding='euc-kr') 


#*************************************************************
# 감성 분석 수행
#*************************************************************


# 감성분석모델 생성 => 13장-감성분석모델구축.py 실행 
#                      또는 아래 함수 실행 

from konlpy.tag import Okt  #okt => Open Korea Text
okt = Okt() 
    
def okt_tokenizer(text):
    tokens = okt.morphs(text)  #형태소 추출 
    return tokens


def 감성분석모델_생성():
    nsmc_train_df = pd.read_csv('D:/BD/ch13/ratings_train.txt', encoding='utf-8', sep='\t', engine='python') 
    nsmc_train_df = nsmc_train_df[nsmc_train_df['document'].notnull()]

    import re
# nsmc_train_df 확인 
    nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))

   
# ---- TF-IDF 기반 피처 벡터 생성 : 실행시간 20분 정도 걸립니다 -----
# TF-IDF =>  https://chan-lab.tistory.com/24 참고 
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
    tfidf.fit(nsmc_train_df['document'])  # 약 8분 소요 

# 문서별 단어들의 가중치 계산 
# 약 10분 소요 
    nsmc_train_tfidf = tfidf.transform(nsmc_train_df['document'])
    nsmc_train_tfidf.shape


# 감성 분류 모델 구축 : 로지스틱 회귀를 이용한 이진 분류
# Sentiment Analysis using Logistic Regression
# 로지스틱 회귀 기반 분석모델 생성

    from sklearn.linear_model import LogisticRegression

    SA_lr = LogisticRegression(random_state = 0)
    SA_lr.fit(nsmc_train_tfidf, nsmc_train_df['label'])

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
    return SA_lr_best, tfidf

#*****************************************************************************


SM, tfidf = 감성분석모델_생성()



# 'title'에 대한 감성 분석
# 1) 분석할 데이터의 피처 벡터화 ---<< title >> 분석

from sklearn.feature_extraction.text import TfidfVectorizer


#기존 영화평 데이터의 벡터 데이터(문서-단어별 빈도수)를 기반으로 가중치 계산해야 함
data_title_tfidf = tfidf.transform(data_df['title'])
temp = data_title_tfidf.toarray()

#  최적 파라미터 학습모델에 적용하여 감성 분석

data_title_predict = SM.predict(data_title_tfidf)

#  감성 분석 결과값을 데이터 프레임에 저장
data_df['title_label'] = data_title_predict

# 분석 결과가 추가된 데이터프레임을 CSV 파일 저장
# csv 파일로 저장 ---------------------------------------------
data_df.to_csv('D:/BD/ch13/'+file_name+'.csv', encoding='euc-kr') 

# 'description' 에 대한 감성 분석
# 분석할 데이터의 피처 벡터화 ---<< description >> 분석
data_description_tfidf = tfidf.transform(data_df['description'])

#  최적 파라미터 학습모델에 적용하여 감성 분석
data_description_predict = SM.predict(data_description_tfidf)

# 3) 감성 분석 결과값을 데이터 프레임에 저장
data_df['description_label'] = data_description_predict




# 감성 분석 결과 확인 및 시각화 - 0: 부정감성, 1: 긍정감성
# 감성 분석 결과 확인

data_df.head()

print(data_df['title_label'].value_counts())
print(data_df['description_label'].value_counts())

#결과 저장 : 긍정과 부정을 분리하여 CSV 파일 저장
columns_name = ['title','title_label','description','description_label']
NEG_data_df = pd.DataFrame(columns=columns_name)
POS_data_df = pd.DataFrame(columns=columns_name)

for i, data in data_df.iterrows(): 
    title = data["title"] 
    description = data["description"] 
    t_label = data["title_label"] 
    d_label = data["description_label"] 
    
    if d_label == 0: # 부정 감성 샘플만 추출
        NEG_data_df = NEG_data_df.append(pd.DataFrame([[title, t_label, description, d_label]],columns=columns_name),ignore_index=True)
    else : # 긍정 감성 샘플만 추출
        POS_data_df = POS_data_df.append(pd.DataFrame([[title, t_label, description, d_label]],columns=columns_name),ignore_index=True)
     
# 파일에 저장.
NEG_data_df.to_csv('D:/BD/ch13/'+file_name+'_NES.csv', encoding='euc-kr') 
POS_data_df.to_csv('D:/BD/ch13/'+file_name+'_POS.csv', encoding='euc-kr') 


# 감성 분석 결과 시각화 : 바 차트
# 긍정 감성의 데이터에서 명사만 추출하여 정리

POS_description = POS_data_df['description']
POS_description_noun_tk = []

for d in POS_description:
    POS_description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출

print(POS_description_noun_tk)  #작업 확인용 출력

POS_description_noun_join = []

for d in POS_description_noun_tk:
    d2 = [w for w in d if len(w) > 1] #길이가 1인 토큰은 제외
    POS_description_noun_join.append(" ".join(d2)) # 토큰을 연결(join)하여 리스트 구성

# 부정 감성의 데이터에서 명사만 추출하여 정리

NEG_description = NEG_data_df['description']

NEG_description_noun_tk = []
NEG_description_noun_join = []

for d in NEG_description:
    NEG_description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출
    
for d in NEG_description_noun_tk:
    d2 = [w for w in d if len(w) > 1]  #길이가 1인 토큰은 제외
    NEG_description_noun_join.append(" ".join(d2)) # 토큰을 연결(join)하여 리스트 구성

print(NEG_description_noun_join)  #작업 확인용 출력



#dtm 구성 : 단어 벡터 값을 내림차순으로 정렬

# 긍정 감성 데이터에 대한 dtm 구성, dtm을 이용하여 단어사전 구성 후 내림차순 정렬

POS_tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, min_df=2 )
POS_dtm = POS_tfidf.fit_transform(POS_description_noun_join)
POS_tfidf.get_feature_names()
len(POS_tfidf.get_feature_names())
temp =POS_dtm.toarray()
POS_tfidf.get_feature_names()[0]
POS_tfidf.get_feature_names()[1]
POS_tfidf.get_feature_names()[2]
POS_tfidf.get_feature_names()[3]
POS_dtm.getcol(0).sum()
POS_dtm.getcol(1).sum()
POS_dtm.getcol(2).sum()
POS_dtm.getcol(3).sum()
POS_vocab = dict() 

for idx, word in enumerate(POS_tfidf.get_feature_names()):
    POS_vocab[word] = POS_dtm.getcol(idx).sum()
    
POS_words = sorted(POS_vocab.items(), key=lambda x: x[1], reverse=True)



#부정 감성 데이터의 dtm 구성, dtm을 이용하여 단어사전 구성 후 내림차순 정렬

NEG_tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, min_df=2 )
NEG_dtm = NEG_tfidf.fit_transform(NEG_description_noun_join)

NEG_vocab = dict() 

for idx, word in enumerate(NEG_tfidf.get_feature_names()):
    NEG_vocab[word] = NEG_dtm.getcol(idx).sum()
    
NEG_words = sorted(NEG_vocab.items(), key=lambda x: x[1], reverse=True)



#단어사전의 상위 단어로 바 차트 그리기

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.get_fontconfig_fonts()
font_location = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

max = 15  #바 차트에 나타낼 단어의 수 

plt.bar(range(max), [i[1] for i in POS_words[:max]], color="blue")
plt.title("긍정 뉴스의 단어 상위 %d개" %max, fontsize=15)
plt.xlabel("단어", fontsize=12)
plt.ylabel("TF-IDF의 합", fontsize=12)
plt.xticks(range(max), [i[0] for i in POS_words[:max]], rotation=70)

plt.show()

plt.bar(range(max), [i[1] for i in NEG_words[:max]], color="red")
plt.title("부정 뉴스의 단어 상위 %d개" %max, fontsize=15)
plt.xlabel("단어", fontsize=12)
plt.ylabel("TF-IDF의 합", fontsize=12)
plt.xticks(range(max), [i[0] for i in NEG_words[:max]], rotation=70)

plt.show()











