#-----------------------------------------------------------------
#12장. 군집분석 : 타깃마케팅을 위한 K-평균 군집화
#-----------------------------------------------------------------



# 모델 구축 : K-평균 군집화 모델

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



customer_df = pd.read_csv('D:/BD/ch12/customer_df.csv')

# 정규 분포로 다시 스케일링하기
from sklearn.preprocessing import StandardScaler

X_features = customer_df[['Freq_log', 'SaleAmount_log', 'ElapsedDays_log']].values

temp = customer_df[['Freq_log', 'SaleAmount_log', 'ElapsedDays_log']]  
X_features = temp.values  # 배열(행렬 )
X_features_scaled = StandardScaler().fit_transform(X_features) # 표준 정규분포: 평균 0, 표준편차 1 인 분포 

# 최적의 k 찾기 (1) 엘보우 방법(elbow method)
# distortion(왜곡) => 클러스터 중심점과 클러스터내의 데이터 거리 차이의 제곱합 
# 클러스터 수를 1부터 하나씩 증가시키면서 왜곡값이 급격히 줄어드는 클러스터 갯수를 K로 선택


distortions = []

for i in range(1, 11):
    kmeans_i = KMeans(n_clusters=i, random_state=0)  # 모델 생성: randdom_state => 초기 중심점 설정 기준값
    kmeans_i.fit(X_features_scaled)   # 군집화 모델 생성 
    distortions.append(kmeans_i.inertia_) #inertia 변수값 :  오차 제곱합
    
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0) # 모델 생성

# 모델 학습과 결과 예측(클러스터 레이블 생성)
Y_labels = kmeans.fit_predict(X_features_scaled) 

#  결과 분석 및 시각화
# 최적의 k 찾기 (2) 실루엣 점수에 따른 각 클러스터의 비중 시각화 함수 정의

from matplotlib import cm

def silhouetteViz(n_cluster, X_features): 
    
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)
    
    # 각 클러스터에 속하는 데이터 각각에 대한 실루엣점수 계산     
    silhouette_values = silhouette_samples(X_features, Y_labels, metric='euclidean')

    y_ax_lower, y_ax_upper = 0, 0
    y_ticks = []

    for c in range(n_cluster): # 0, 1, 2, .., (n_cluster-1) 각각에 대하여 반복 
        c_silhouettes = silhouette_values[Y_labels == c] # c클러스터에 속한 실루엣점수 값들 
        c_silhouettes.sort() # 군집별 실루엣 점수 정렬
        y_ax_upper += len(c_silhouettes) # 실루엣점수 출력을 위한 y축값 변경
        color = cm.jet(float(c) / n_cluster) # 군집별로 색깔 설정
        # barh() => 수평 막대그래프 그림 
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouettes,
                 height=1.0, edgecolor='none', color=color)
        y_ticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouettes)
    
    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.title('Number of Cluster : '+ str(n_cluster)+'\n' \
              + 'Silhouette Score : '+ str(round(silhouette_avg,3)))
    plt.yticks(y_ticks, range(n_cluster))   
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()
    
 # 클러스터 수에 따른 클러스터 데이터 분포의 시각화 함수 정의    
def clusterScatter(n_cluster, X_features): 
    c_colors = []
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    Y_labels = kmeans.fit_predict(X_features)

    for i in range(n_cluster):
        c_color = cm.jet(float(i) / n_cluster) #클러스터의 색상 설정
        c_colors.append(c_color)
        #클러스터의 데이터 분포를 동그라미로 시각화
        plt.scatter(X_features[Y_labels == i,0], X_features[Y_labels == i,1],
                     marker='o', color=c_color, edgecolor='black', s=50, 
                     label='cluster '+ str(i))       
    
    #각 클러스터의 중심점을 삼각형으로 표시
    for i in range(n_cluster):
        plt.scatter(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1], 
                    marker='^', color=c_colors[i], edgecolor='w', s=200)
        
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

silhouetteViz(3, X_features_scaled) #클러스터 3개인 경우의 실루엣 score 및 각 클러스터 비중 시각화
silhouetteViz(4, X_features_scaled) #클러스터 4개인 경우의 실루엣 score 및 각 클러스터 비중 시각화
silhouetteViz(5, X_features_scaled) #클러스터 5개인 경우의 실루엣 score 및 각 클러스터 비중 시각화
silhouetteViz(6, X_features_scaled) #클러스터 6개인 경우의 실루엣 score 및 각 클러스터 비중 시각화


clusterScatter(3, X_features_scaled) #클러스터 3개인 경우의 클러스터 데이터 분포 시각
clusterScatter(4, X_features_scaled)  #클러스터 4개인 경우의 클러스터 데이터 분포 시각화
clusterScatter(5, X_features_scaled)  #클러스터 5개인 경우의 클러스터 데이터 분포 시각화
clusterScatter(6, X_features_scaled)  #클러스터 6개인 경우의 클러스터 데이터 분포 시각화


# 결정된 k를 적용하여 최적의 K-mans 모델 완성

best_cluster = 4

kmeans = KMeans(n_clusters=best_cluster, random_state=0)
Y_labels = kmeans.fit_predict(X_features_scaled)

customer_df['ClusterLabel'] = Y_labels

customer_df.head()


# ClusterLabel이 추가된 데이터를 파일로 저장

customer_df.to_csv('D:/BD/ch12/Online_Retail_Customer_Cluster.csv')

# 클러스터 분석하기


# 각 클러스터의 고객수

customer_df.groupby('ClusterLabel')['CustomerID'].count()

# 각 클러스터의 특징

customer_cluster_df = customer_df.drop(['Freq_log', 'SaleAmount_log', 'ElapsedDays_log'],axis=1, inplace=False)

# 주문 1회당 평균 구매금액 : SaleAmountAvg
customer_cluster_df['SaleAmountAvg'] = customer_cluster_df['SaleAmount']/customer_cluster_df['Freq']

customer_cluster_df.head()


# 클러스터별 분석
customer_cluster_df.drop(['CustomerID'],axis=1, inplace=False).groupby('ClusterLabel').mean()




















