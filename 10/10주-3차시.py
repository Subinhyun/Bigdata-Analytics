
import json
import re

from konlpy.tag import Okt

from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from wordcloud import WordCloud

# KoNLP 학습 ----------------------------------------
# OKT => Open Korea Text 
# 기타 : 구글링 검색 
okt = Okt()
txt = "우리는 파이썬 빅데이터 분석 기법을 공부하고 있습니다"
result = okt.morphs(txt) #형태소 단위로 분리 
noun = okt.nouns(txt) #명사 추출 
print(okt.phrases(txt)) #어절 분리 
print(okt.pos(txt)) #품사 태킹 

#----------------------------------------


inputFileName = 'D:/BD/ch8/etnews.kr_facebook_2016-01-01_2018-08-01_4차 산업혁명'
data = json.loads(open(inputFileName+'.json', 'r', encoding='utf-8').read())

data #출력하여 내용 확인


message = ''

for item in data:
    if 'message' in item.keys(): 
        message = message + re.sub(r'[^\w]', ' ', item['message']) +''
        # \w => 단어문자를 의미함. 단어문자는 영문대소문자, 숫자0-9, 언더바 _ 등을 포함
        
message #출력하여 내용 확인

nlp = Okt()
message_N = nlp.nouns(message)
message_N   

count = Counter(message_N)
count   #출력하여 내용 확인

word_count = dict()

for tag, counts in count.most_common(80):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))

#한글폰트 설정 
font_path = "c:/Windows/fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname = font_path).get_name()
matplotlib.rc('font', family=font_name)



plt.figure(figsize=(12,5))
plt.xlabel('키워드')
plt.ylabel('빈도수')
plt.grid(True)

sorted_Keys = sorted(word_count, key=word_count.get, reverse=True)
sorted_Values = sorted(word_count.values(), reverse=True)

plt.bar(range(len(word_count)), sorted_Values, align='center')
plt.xticks(range(len(word_count)), list(sorted_Keys), rotation='75')

plt.show()


wc = WordCloud(font_path, background_color='ivory', width=800, height=600)
cloud=wc.generate_from_frequencies(word_count)

plt.figure(figsize=(8,8))
plt.imshow(cloud)
plt.axis('off')
plt.show()






















