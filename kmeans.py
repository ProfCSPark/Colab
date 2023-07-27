#!/usr/bin/env python
# coding: utf-8

# In[1]:

# KMeans - 부원 데이터
# https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
# 2021.12.12. cspark.
# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans   # KMeans( ~ ) . 이 방법 추천.
# import sklearn.cluster as sk        # sk.KMeans( ~ )

# reading the data and looking at the first five rows of the data
# data = read.csv(file.choose(), sep = ",", header = T, quote = "")

#df = pd.read_csv("outlier_data.csv")
df = pd.read_csv("Buwon_Labelled_2_Class_ShotNo.csv") # result of isolation_forest for 2 classes.

df2 = df.loc[df['anomaly'] == -1] # -1:anomaly,  1:normal



print('\n ********* Before Slicing: *************** \n')
print(df2.head())


# statistics of the data
# df.describe()

# data2 = df.copy() # 원본데이터 복사.. Anomaly Col을 살려두고 뒤에서 Cluster 붙임.

# data = df2[['V1', 'V2', 'V3',.......,'V15']] # select multiple col's with col names
# data = df2.loc[:, "V1":"V15"] # OK

data = df2.iloc[:, 1:-1]  # OK .. exclude first and last columns.. shotNo and Anomaly cols.
# data = only anomaly data set without ShotNo and Anomaly Cols



print('\n ****** After Slicing: ************** \n')
print(data.head())


#data2 = data.copy()

#Here, we see that there is a lot of variation in the magnitude of the data. Variables like Channel and Region have low magnitude whereas variables like Fresh, Milk, Grocery, etc. have a higher magnitude.

#Since K-Means is a distance-based algorithm, this difference of magnitude can create a problem. So let’s first bring all the variables to the same magnitude:

# standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()

# ****************************************************
Num_Given_Cluster = 4  # 4 defective types 존재 가정함.
#Num_Given_Cluster = 5
# ****************************************************


# defining the kmeans function with initialization as k-means++
#kmeans = KMeans(n_clusters=2, init='k-means++')
#kmeans = KMeans(n_clusters=5, init='k-means++')
# kmeans = KMeans(n_clusters=5, init='k-means++')
kmeans = KMeans(n_clusters=Num_Given_Cluster)

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)

# inertia on the fitted data
# inertia_ : Sum of squared distances of samples to their closest cluster center
# print(' *** inertia_ : **** ', kmeans.inertia_ )

# 참고) n_jobs : scikit-learn의 기본적인 병렬처리로 내부적으로 멀티프로세스를 사용하는 것. 
# 만약 CPU 코어의 수가 충분하다면 n_jobs를 늘릴수록 속도가 증가
# n_jobs 는 향후 사라질 예정이므로 사용하지 말것...csp.

# *************************************************************************************
#  아래와 같이 1~20정도의 클러스터에 대해서 돌린후 SSE를 만들어서  Cluster vs. Inertia  그림을 그려서 
# 기울기가 변하지 않는 지점의 Cluster을 클러스터 갯수로  선정함.
# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    # kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++') # n_jobs : will be deprecated. No more use it!!!
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_) # inertia_ : Sum of squared distances of samples to their closest cluster center,
# converting the results into a dataframe and plotting them
# 확 꺽이는 지점 (클러스터 갯수) 를 찾기 위해서 그림.
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE}) # DataFrame 생성. SSE에는 20개 클러스터에 대한 SSE 값이 있음.
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
# *************************************************************************************


# k means using 5 clusters and k-means++ initialization
# kmeans = KMeans(n_clusters = Num_Given_Cluster, init='k-means++')
kmeans = KMeans(n_clusters = Num_Given_Cluster) # default  init = 'random'  


kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

data['cluster'] = pred # to use with original data set
data.to_csv('Clustering_Result_wo_shotNo_Anomaly.csv', index=False) # Cluster 붙임

df2['cluster'] = pred
df2.to_csv('Clustering_Result_shotNo_Anomaly.csv', index=False) # Cluster 붙임

df3 = df2.copy() ####
df3['cluster'] = df3['cluster'] + 1  # 0,1,2,3 => 1,2,3,4 로 변경하기 위해.

df3.to_csv('Clustering_Result_shotNo_Anomaly_cluster_pp.csv', index=False) # Cluster 붙임


df['cluster'] = 0
df.to_csv('Buwon_5_class.csv', index=False) # Cluster 붙임

#####################  중요 #####################################
# 이후는 시간 관계상 엑셀 이용해서 클러스터 칼럼을 작업함  .
# 위에서 Buwon_5_class.csv 파일과 _pp.csv 파일을  anomaly 와 shotNo 로  정렬후  _pp.csv 의 cluster 컬럼을
# Buwon_5_class.csv 의 cluster col 에  복사. 
# 즉, 0 으로 cluster 칼럼을 세팅한 후 불량 레코드만 찾아서 cluster cell 값을  위에서 찾은
# 클러스터 값으로 update ^^
# 2022.1. cspark.
################################################################






'''



df_temp = df.copy()
# df2 의 shotNo의 cluster 를 읽어서 df 의 동일한 ShotNo  cluster 갱신.

# grades 값이 90 이라면 result 는 'A'
df.loc[df['grades'] == 90, 'result'] = 'A'

# grades 값이 80 이라면 result 는 'B'
df.loc[df['grades'] == 80, 'result'] = 'B'

# result 값이 'A' 나 'B' 가 아니라면 result 는 'F'
df.loc[df['result'] != ('A' or 'B') , 'result'] = 'F'
# https://wooono.tistory.com/293

'''

# 아래 내용은 위의 엑셀 수작업을  자동으로 처리해 볼려 했으나.... 실패.
# 아래 내용은 나중에 시간 있을 때  완성해 볼것. 2022.1. csp.
#df_temp[df_temp[df_temp['shot_No'] == df2['shot_No']]].cluster= df2['cluster']
# df_temp[df_temp.loc[df_temp['shot_No']== df2['shot_No']].cluster= df2.loc['shot_No'].cluster
# 참고: https://note.espriter.net/1326
'''
df_temp.to_csv('Clustering_Result_Ori_Final.csv', index=False) # Cluster 붙임


###############################################################################333

#Finally, let’s look at the value count of points in each of the above-formed clusters:
frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred

# insert cluster no to original data file
df['cluster'] = pred
df['Label'] = kmeans.labels_  # pred 와 같은 생성된 라벨 값
df.to_csv('Clustering_Result_scaled.csv', index=False)

print('****** Cluster prediction: ********** \n', frame.head())

val_counts = frame['cluster'].value_counts()  # val_counts : Series
#val_counts.sort
print('val_counts type=', type(val_counts))
print('\n ****** Number of each Clusters: *********\n', val_counts)


lst_val = [] # create empty list
lst_key = [] # create empty list
dict_ = {} # create empty dictionary

for i in range(0, Num_Given_Cluster):
    pr = val_counts[i]/len(df)    
    lst_val.append(pr)
    lst_key.append(i)

# 두 개의 리스트를 이용하여 key와 value를 구성
# 키는 키끼리 값은 값끼리 묶어서 zip함수에 전달
# zip 결과를 dict 에 전달하여  Dictionary 생성.
d = dict(zip(lst_key, lst_val))    
print(' ********* 비율: ************:\n', d)
'''




