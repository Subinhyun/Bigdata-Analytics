# 13장. 텍스트마이닝_감성분석과 토픽분석

import pandas as pd


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


