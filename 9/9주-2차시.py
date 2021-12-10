
import seaborn as sns
import pandas as pd
titanic = sns.load_dataset("titanic")
titanic.to_csv('D:/BD/titanic.csv', index=False)

titanic.isnull().sum()
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['embarked'].value_counts()
titanic['embarked'] = titanic['embarked'].fillna('S')
titanic['embark_town'].value_counts()
titanic['embark_town'] = titanic['embark_town'].fillna('Southampton')
titanic['deck'].value_counts()
titanic['deck'] = titanic['deck'].fillna('C')
titanic.isnull().sum()
titanic.info()
titanic.survived.value_counts()

import matplotlib.pyplot as plt

ratio = titanic['survived'][titanic['sex']=='male'].value_counts()
labels = ['Male', 'FeMale']
plt.pie(ratio, labels=labels, autopct='%.1f%%')
plt.show()

#pie() 매개변수
# explode => 파이조각의 돌출 정도
# autopct => 숫자의 소수자리 수
# shadow => 파이조각 그림자
# ax => 서브그래프 

f, ax = plt.subplots(1,2,figsize=(10,5))
titanic['survived'][titanic['sex']=='male'].value_counts().plot.pie(explode=[0,0.1],autopct='%.1f%%', ax=ax[0],shadow=True)
titanic['survived'][titanic['sex']=='female'].value_counts().plot.pie(explode=[0, 0.2],autopct='%.1f%%', ax=ax[1],shadow=True)
ax[0].set_title('Survived (Male)')
ax[1].set_title('Survived (FeMale)')
plt.show()

#Dec별 생존자 비율 
titanic['deck'].value_counts()
f, ax = plt.subplots(1,7,figsize=(10,5))
titanic['survived'][titanic['deck']=='C'].value_counts().plot.pie(explode=[0,0.1],autopct='%.1f%%', ax=ax[0],shadow=True)
titanic['survived'][titanic['deck']=='B'].value_counts().plot.pie(explode=[0,0.1],autopct='%.1f%%', ax=ax[1],shadow=True)
titanic['survived'][titanic['deck']=='D'].value_counts().plot.pie(explode=[0,0.1],autopct='%.1f%%', ax=ax[2],shadow=True)
titanic['survived'][titanic['deck']=='E'].value_counts().plot.pie(explode=[0,0.1],autopct='%.1f%%', ax=ax[3],shadow=True)
titanic['survived'][titanic['deck']=='A'].value_counts().plot.pie(explode=[0,0.1],autopct='%.1f%%', ax=ax[4],shadow=True)
titanic['survived'][titanic['deck']=='F'].value_counts().plot.pie(explode=[0,0.1],autopct='%.1f%%', ax=ax[5],shadow=True)
titanic['survived'][titanic['deck']=='G'].value_counts().plot.pie(explode=[0,0.1],autopct='%.1f%%', ax=ax[6],shadow=True)

ax[0].set_title('C')
ax[1].set_title('B')
ax[2].set_title('D')
ax[3].set_title('E')
ax[4].set_title('A')
ax[5].set_title('F')
ax[6].set_title('G')
plt.show()


# seaborn 시각화 
sns.countplot('pclass', hue='survived', data=titanic)
#sns.countplot('pclass', data=titanic)

plt.title('Pclass vs Survived')
plt.show()

# 상관분석을 위한 상관계수 구하고 저장하기
# 상관분석은 연속형 데이터데 대하여 적용 가능 => pclass, age

titanic_corr = titanic.corr(method='pearson')
titanic_corr
titanic_corr.to_csv('D:/BD/titanic_corr.csv', index=False)
titanic


#이산값 변수와 연속형 변수 사이의 상관분석 point-biserial correlation(이연상관계수) 사용 
from scipy import stats
surv = titanic['survived']
age = titanic['age']
stats.pointbiserialr(surv, age)

fare = titanic['fare']
age = titanic['age']
corr = stats.pearsonr(fare, age) #처음값: 상관계수, 둘째값 : p값 
corr

sns.pairplot(titanic, hue='survived')
plt.show()

sns.catplot(x='pclass', y='survived', hue='sex', data=titanic, kind='point')
plt.show()


def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7
titanic['age2'] = titanic['age'].apply(category_age) 
titanic['sex2'] = titanic['sex'].map({'male':1, 'female':0})
titanic['family'] = titanic['sibsp'] + titanic['parch']+1
titanic.to_csv('D:/BD/titanic3.csv', index=False)
heatmap_data = titanic[['survived', 'sex2', 'age2','family', 'pclass', 'fare']]
heatmap_data.info()
colormap = plt.cm.RdBu # Red Blue
#colormap = plt.cm.PuBu
#plt.figure(figsize=(10, 8))
#plt.title("Person Correlation of Features", y = 1.05, size = 15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, \
            square=True, cmap=colormap, linecolor='white', annot=True, \
            annot_kws = {"size": 10})

help(sns.heatmap)
















