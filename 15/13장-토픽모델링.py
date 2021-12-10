#*************************************************
# 13장. 토픽모델링 : LDA 기반 토픽 모델링
#*************************************************
import pandas as pd

pd.__version__

# pyLDAvis 모듈의 정상 실행을 위하여 pandas 를 upgrade 필요(1.1.4 이상으로 )
!pip install --upgrade pandas

# spyder 재실행 

#데이터 준비
data_df = pd.read_csv('D:/BD/ch13/data_df.csv', encoding="euc-kr")

#'description' 컬럼 추출
description = data_df['description']


#형태소 토큰화 : 명사만 추출
from konlpy.tag import Okt  #okt => Open Korea Text

okt = Okt() 

description_noun_tk = []
for d in description:
    description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출

description_noun_tk2 = []

for d in description_noun_tk:
    item = [i for i in d if len(i) > 1]  #토큰의 길이가 1보다 큰 것만 추출
    description_noun_tk2.append(item)

print(description_noun_tk2)


#LDA 토픽 모델 구축
# LDA 모델의 입력 벡터 생성
#gensim 라이브러리 : 자연어를 벡터데이터로 변환하는데 필요한 기능 제공
# 최초 한번만 설치
!pip install gensim   

import gensim
import gensim.corpora as corpora

#단어 사전 생성
# test-------------------------------------------------

documents = [['제주', '대학'], 
             ['애월','관광','경치'],
             ['자전거', '우도', '아이스크림'],
             ['한라산','등산', '하이킹'],
             ['바다', '함덕'],
             ['제주','박물관', '자연사']]


d = corpora.Dictionary(documents)
print(d.token2id)
d.doc2bow(['제주', '대학']) #->[(0, 1), (1, 1)]
# bow =>bag-of-words 의 약어 
# 튜플 리스트 반환 : [(단어id, 출현빈도수),..]
d.doc2bow(['애월','관광','경치'])  #=>  [(2, 1), (3, 1), (4, 1)]
#-------------------------------------------------------


dictionary = corpora.Dictionary(description_noun_tk2)
print(dictionary)
len(dictionary)
print(dictionary[1])  #작업 확인용 출력
print(dictionary.token2id)

#단어와 출현빈도(count)의 코퍼스 생성

corpus = [dictionary.doc2bow(word) for word in description_noun_tk2]
print(corpus) #작업 확인용 출력


#LDA 모델 생성 및 훈련

k = 4  #토픽의 개수 설정
lda_model = gensim.models.ldamulticore.LdaMulticore(corpus, iterations = 12, num_topics = k, id2word = dictionary, passes = 1, workers = 10)


#LDA 토픽 분석 결과 시각화

print(lda_model.print_topics(num_topics = k, num_words = 15))

#토픽 분석 결과 시각화 : pyLDAvis

#최초 한번만 설치
!pip install pyLDAvis 

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(lda_vis)

file_name = '코로나'

pyLDAvis.save_html(lda_vis, 'D:/BD/ch13/'+file_name+"_vis.html")
























