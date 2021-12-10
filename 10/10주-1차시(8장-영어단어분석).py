# 아나콘다 설치 후에도 추가 설치 필요 
!pip install matplotlib
!pip install wordcloud 
#--------------------------------
import pandas as pd
import glob
import re
from functools import reduce

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

all_files = glob.glob('D:/BD/ch8/myCabinetExcelData*.xls')

all_files_data = [] #저장할 리스트 

for file in all_files:
    data_frame = pd.read_excel(file)
    all_files_data.append(data_frame)

all_files_data[0] #출력하여 내용 확인
all_files_data_concat = pd.concat(all_files_data, axis=0, ignore_index=True) #axis=0: 세로축 기준으로 병합 

all_files_data_concat #출력하여 내용 확인

all_files_data_concat.to_csv('D:/BD/ch8/riss_bigdata.csv', encoding='utf-8', index = False)

# 제목 추출
all_title = all_files_data_concat['제목']

all_title #출력하여 내용 확인
stopWords = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

words = []  

for title in all_title:
    EnWords = re.sub(r"[^a-zA-Z]+", " ", str(title))    
    EnWordsToken = word_tokenize(EnWords.lower())
    EnWordsTokenStop = [w for w in EnWordsToken if w not in stopWords]
    EnWordsTokenStopLemma = [lemma.lemmatize(w) for w in EnWordsTokenStop]
    words.append(EnWordsTokenStopLemma)
    
    
    
words2 = list(reduce(lambda x, y: x+y, words))  #2차원 리스트를 1차원 리스트로 변환 
#words1 = list(reduce(lambda x, y: x+y, words, []))

print(words2)  #작업 내용 확인    
    
    
count = Counter(words2)    
count 
    
word_count = dict()

for tag, counts in count.most_common(50):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))    
        
#검색어로 사용한 'big'과 'data' 항목 제거 하기
del word_count['big']
del word_count['data']        
    
    
plt.figure(figsize=(20,10))
plt.xlabel("word")
plt.ylabel("count")
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)

# reverse=True : 내림차순 정렬, key:정렬 기준(함수 사용 가능)
# sorted_Keys = sorted(word_count.keys(), key=word_count.get, reverse=True)


sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation='90')

plt.show()    
    
all_files_data_concat['doc_count'] = 0
summary_year = all_files_data_concat.groupby('출판일', as_index=False)['doc_count'].count()
summary_year = all_files_data_concat.groupby('출판일', as_index=True)['doc_count'].count()
summary_year  #출력하여 내용 확인    


plt.figure(figsize=(12,5))
plt.xlabel("year")
plt.ylabel("doc-count")
plt.grid(True)

plt.plot(range(len(summary_year)), summary_year['doc_count'])
plt.xticks(range(len(summary_year)), [text for text in summary_year['출판일']])

plt.show()



#stopwords=set(stopWords)
wc=WordCloud(background_color='ivory', stopwords=stopWords, width=800, height=600)
cloud=wc.generate_from_frequencies(word_count)

plt.figure(figsize=(10,10))
plt.imshow(cloud)
plt.axis('off')
plt.show()


cloud.to_file('D:/BD/ch8/riss_bigdata_wordCloud.jpg')





















    
    
    
    
    
    
    
    