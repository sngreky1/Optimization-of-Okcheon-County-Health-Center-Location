import pandas as pd
import folium
import random
from sklearn import cluster

pd10 = pd.read_csv('충청북도 옥천군_경로당현황_좌표추가.csv')
pd20 = pd.read_csv('충청북도 옥천군_음식점현황_좌표추가.csv')

pd10

pd10 = pd10.rename(columns={"경로당명": "경로당명/음식점명"})
pd10 = pd10.rename(columns={"경로당 주소": "도로명 주소"})
pd10

pd10 = pd10.drop("연번", axis=1)
pd10 = pd10.drop("읍면", axis=1)
pd10 = pd10.drop("데이터기준일", axis=1)
pd10 = pd10.dropna()
pd10

pd20 = pd20.rename(columns={"업소명": "경로당명/음식점명"})
pd20 = pd20.rename(columns={"소재지(도로명)": "도로명 주소"})
pd20 = pd20[1:]
pd20

pd20 = pd20.drop("연번", axis=1)
pd20 = pd20.drop("업종명", axis=1)
pd20 = pd20.drop("군분", axis=1)
pd20 = pd20.drop("소재지(지번)", axis=1)
pd20 = pd20.drop("소재지전화", axis=1)
pd20 = pd20.drop("데이터기준일", axis=1)
pd20 = pd20[:-1].dropna()
pd20


def merge_dataframes(pd10, pd20):
    # 두 데이터프레임을 합침
    merged_df = pd.concat([pd10, pd20], ignore_index=True)
    return merged_df


merged_df = merge_dataframes(pd10, pd20)
merged_df

# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "청성면"이 포함된 행을 뽑음
merged_df_청성면 = merged_df[merged_df["도로명 주소"].str.contains("청성면")]

# 필터링된 데이터프레임 출력
print(merged_df_청성면)

merged_df_청성면.shape[0]

# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "청산면"이 포함된 행을 뽑음
merged_df_청산면 = merged_df[merged_df["도로명 주소"].str.contains("청산면")]

# 필터링된 데이터프레임 출력
print(merged_df_청산면)

merged_df_청산면.shape[0]

# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "동이면"이 포함된 행을 뽑음
merged_df_동이면 = merged_df[merged_df["도로명 주소"].str.contains("동이면")]

# 필터링된 데이터프레임 출력
print(merged_df_동이면)

merged_df_동이면.shape[0]

# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군북면"이 포함된 행을 뽑음
merged_df_군북면 = merged_df[merged_df["도로명 주소"].str.contains("군북면")]

# 필터링된 데이터프레임 출력
print(merged_df_군북면)

merged_df_군북면.shape[0]

# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군서면"이 포함된 행을 뽑음
merged_df_군서면 = merged_df[merged_df["도로명 주소"].str.contains("군서면")]

# 필터링된 데이터프레임 출력
print(merged_df_군서면)

merged_df_군서면.shape[0]

# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "안남면"이 포함된 행을 뽑음
merged_df_안남면 = merged_df[merged_df["도로명 주소"].str.contains("안남면")]

# 필터링된 데이터프레임 출력
print(merged_df_안남면)

merged_df_안남면.shape[0]

# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "안내면"이 포함된 행을 뽑음
merged_df_안내면 = merged_df[merged_df["도로명 주소"].str.contains("안내면")]

# 필터링된 데이터프레임 출력
print(merged_df_안내면)

merged_df_안내면.shape[0]

# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "이원면"이 포함된 행을 뽑음
merged_df_이원면 = merged_df[merged_df["도로명 주소"].str.contains("이원면")]

# 필터링된 데이터프레임 출력
print(merged_df_이원면)

merged_df_이원면.shape[0]

# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "옥천읍"이 포함된 행을 뽑음
merged_df_옥천읍 = merged_df[merged_df["도로명 주소"].str.contains("옥천읍")]

# 필터링된 데이터프레임 출력
print(merged_df_옥천읍)

merged_df_옥천읍.shape[0]

import random
import pandas as pd

random.seed(123)

# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values = merged_df_옥천읍["Latitude"].apply(pd.to_numeric, errors='coerce')
longitude_values = merged_df_옥천읍["Longitude"].apply(pd.to_numeric, errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
min_latitude = latitude_values.dropna().min()
max_latitude = latitude_values.dropna().max()
min_longitude = longitude_values.dropna().min()
max_longitude = longitude_values.dropna().max()

x1_list = []
y1_list = []

for i in range(5330):
    x1_list.append(random.uniform(min_latitude, max_latitude))
    y1_list.append(random.uniform(min_longitude, max_longitude))

target1 = pd.DataFrame({'Latitude': x1_list, 'Longitude': y1_list})
target1

import random
import pandas as pd

# merged_df_이원면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values = merged_df_이원면["Latitude"].apply(pd.to_numeric, errors='coerce')
longitude_values = merged_df_이원면["Longitude"].apply(pd.to_numeric, errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
min_latitude = latitude_values.dropna().min()
max_latitude = latitude_values.dropna().max()
min_longitude = longitude_values.dropna().min()
max_longitude = longitude_values.dropna().max()

x2_list = []
y2_list = []

for i in range(1613):
    x2_list.append(random.uniform(min_latitude, max_latitude))
    y2_list.append(random.uniform(min_longitude, max_longitude))

target2 = pd.DataFrame({'Latitude': x2_list, 'Longitude': y2_list})
target2

import random
import pandas as pd

# merged_df_안내면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values = merged_df_안내면["Latitude"].apply(pd.to_numeric, errors='coerce')
longitude_values = merged_df_안내면["Longitude"].apply(pd.to_numeric, errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
min_latitude = latitude_values.dropna().min()
max_latitude = latitude_values.dropna().max()
min_longitude = longitude_values.dropna().min()
max_longitude = longitude_values.dropna().max()

x3_list = []
y3_list = []

for i in range(852):
    x3_list.append(random.uniform(min_latitude, max_latitude))
    y3_list.append(random.uniform(min_longitude, max_longitude))

target3 = pd.DataFrame({'Latitude': x3_list, 'Longitude': y3_list})
target3

import random
import pandas as pd

# merged_df_안남면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values = merged_df_안남면["Latitude"].apply(pd.to_numeric, errors='coerce')
longitude_values = merged_df_안남면["Longitude"].apply(pd.to_numeric, errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
min_latitude = latitude_values.dropna().min()
max_latitude = latitude_values.dropna().max()
min_longitude = longitude_values.dropna().min()
max_longitude = longitude_values.dropna().max()

x4_list = []
y4_list = []

for i in range(573):
    x4_list.append(random.uniform(min_latitude, max_latitude))
    y4_list.append(random.uniform(min_longitude, max_longitude))

target4 = pd.DataFrame({'Latitude': x4_list, 'Longitude': y4_list})
target4

import random
import pandas as pd

# merged_df_군서면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values = merged_df_군서면["Latitude"].apply(pd.to_numeric, errors='coerce')
longitude_values = merged_df_군서면["Longitude"].apply(pd.to_numeric, errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
min_latitude = latitude_values.dropna().min()
max_latitude = latitude_values.dropna().max()
min_longitude = longitude_values.dropna().min()
max_longitude = longitude_values.dropna().max()

x5_list = []
y5_list = []

for i in range(895):
    x5_list.append(random.uniform(min_latitude, max_latitude))
    y5_list.append(random.uniform(min_longitude, max_longitude))

target5 = pd.DataFrame({'Latitude': x5_list, 'Longitude': y5_list})
target5

import random
import pandas as pd

# merged_df_군북면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values = merged_df_군북면["Latitude"].apply(pd.to_numeric, errors='coerce')
longitude_values = merged_df_군북면["Longitude"].apply(pd.to_numeric, errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
min_latitude = latitude_values.dropna().min()
max_latitude = latitude_values.dropna().max()
min_longitude = longitude_values.dropna().min()
max_longitude = longitude_values.dropna().max()

x6_list = []
y6_list = []

for i in range(1029):
    x6_list.append(random.uniform(min_latitude, max_latitude))
    y6_list.append(random.uniform(min_longitude, max_longitude))

target6 = pd.DataFrame({'Latitude': x6_list, 'Longitude': y6_list})
target6

import random
import pandas as pd

# merged_df_동이면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values = merged_df_동이면["Latitude"].apply(pd.to_numeric, errors='coerce')
longitude_values = merged_df_동이면["Longitude"].apply(pd.to_numeric, errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
min_latitude = latitude_values.dropna().min()
max_latitude = latitude_values.dropna().max()
min_longitude = longitude_values.dropna().min()
max_longitude = longitude_values.dropna().max()

x7_list = []
y7_list = []

for i in range(1122):
    x7_list.append(random.uniform(min_latitude, max_latitude))
    y7_list.append(random.uniform(min_longitude, max_longitude))

target7 = pd.DataFrame({'Latitude': x7_list, 'Longitude': y7_list})
target7

import random
import pandas as pd

# merged_df_청산면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values = merged_df_청산면["Latitude"].apply(pd.to_numeric, errors='coerce')
longitude_values = merged_df_청산면["Longitude"].apply(pd.to_numeric, errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
min_latitude = latitude_values.dropna().min()
max_latitude = latitude_values.dropna().max()
min_longitude = longitude_values.dropna().min()
max_longitude = longitude_values.dropna().max()

x8_list = []
y8_list = []

for i in range(1325):
    x8_list.append(random.uniform(min_latitude, max_latitude))
    y8_list.append(random.uniform(min_longitude, max_longitude))

target8 = pd.DataFrame({'Latitude': x8_list, 'Longitude': y8_list})
target8

import random
import pandas as pd

# merged_df_청성면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values = merged_df_청성면["Latitude"].apply(pd.to_numeric, errors='coerce')
longitude_values = merged_df_청성면["Longitude"].apply(pd.to_numeric, errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
min_latitude = latitude_values.dropna().min()
max_latitude = latitude_values.dropna().max()
min_longitude = longitude_values.dropna().min()
max_longitude = longitude_values.dropna().max()

x9_list = []
y9_list = []

for i in range(1110):
    x9_list.append(random.uniform(min_latitude, max_latitude))
    y9_list.append(random.uniform(min_longitude, max_longitude))

target9 = pd.DataFrame({'Latitude': x9_list, 'Longitude': y9_list})
target9


# 읍면 데이터 병합
ndf = pd.concat([target1, target2, target3, target4, target5, target6, target7, target8, target9])

ndf.describe()

## 기존데이터(경로당+음식점) + 난수 데이터 프레임 병합

merged_df_옥천읍_target1 = pd.concat([target1, merged_df_옥천읍], ignore_index=True)

# 'Latitude'와 'Longitude' 열의 데이터 유형을 숫자(float)로 변환
merged_df_옥천읍_target1["Latitude"] = merged_df_옥천읍_target1["Latitude"].astype(float)
merged_df_옥천읍_target1["Longitude"] = merged_df_옥천읍_target1["Longitude"].astype(float)

merged_df_옥천읍_target1 = merged_df_옥천읍_target1.drop(['경로당명/음식점명', '도로명 주소'], axis=1)
merged_df_옥천읍_target1

merged_df_이원면_target2 = pd.concat([target2, merged_df_이원면], ignore_index=True)

# 'Latitude'와 'Longitude' 열의 데이터 유형을 숫자(float)로 변환
merged_df_이원면_target2["Latitude"] = merged_df_이원면_target2["Latitude"].astype(float)
merged_df_이원면_target2["Longitude"] = merged_df_이원면_target2["Longitude"].astype(float)

merged_df_이원면_target2 = merged_df_이원면_target2.drop(['경로당명/음식점명', '도로명 주소'], axis=1)
merged_df_이원면_target2

merged_df_안내면_target3 = pd.concat([target3, merged_df_안내면], ignore_index=True)

# 'Latitude'와 'Longitude' 열의 데이터 유형을 숫자(float)로 변환
merged_df_안내면_target3["Latitude"] = merged_df_안내면_target3["Latitude"].astype(float)
merged_df_안내면_target3["Longitude"] = merged_df_안내면_target3["Longitude"].astype(float)

merged_df_안내면_target3 = merged_df_안내면_target3.drop(['경로당명/음식점명', '도로명 주소'], axis=1)
merged_df_안내면_target3

merged_df_안남면_target4 = pd.concat([target4, merged_df_안남면], ignore_index=True)

# 'Latitude'와 'Longitude' 열의 데이터 유형을 숫자(float)로 변환
merged_df_안남면_target4["Latitude"] = merged_df_안남면_target4["Latitude"].astype(float)
merged_df_안남면_target4["Longitude"] = merged_df_안남면_target4["Longitude"].astype(float)

merged_df_안남면_target4 = merged_df_안남면_target4.drop(['경로당명/음식점명', '도로명 주소'], axis=1)
merged_df_안남면_target4

merged_df_군서면_target5 = pd.concat([target5, merged_df_군서면], ignore_index=True)

# 'Latitude'와 'Longitude' 열의 데이터 유형을 숫자(float)로 변환
merged_df_군서면_target5["Latitude"] = merged_df_군서면_target5["Latitude"].astype(float)
merged_df_군서면_target5["Longitude"] = merged_df_군서면_target5["Longitude"].astype(float)

merged_df_군서면_target5 = merged_df_군서면_target5.drop(['경로당명/음식점명', '도로명 주소'], axis=1)
merged_df_군서면_target5

merged_df_군북면_target6 = pd.concat([target6, merged_df_군북면], ignore_index=True)

# 'Latitude'와 'Longitude' 열의 데이터 유형을 숫자(float)로 변환
merged_df_군북면_target6["Latitude"] = merged_df_군북면_target6["Latitude"].astype(float)
merged_df_군북면_target6["Longitude"] = merged_df_군북면_target6["Longitude"].astype(float)

merged_df_군북면_target6 = merged_df_군북면_target6.drop(['경로당명/음식점명', '도로명 주소'], axis=1)
merged_df_군북면_target6

merged_df_동이면_target7 = pd.concat([target7, merged_df_동이면], ignore_index=True)

# 'Latitude'와 'Longitude' 열의 데이터 유형을 숫자(float)로 변환
merged_df_동이면_target7["Latitude"] = merged_df_동이면_target7["Latitude"].astype(float)
merged_df_동이면_target7["Longitude"] = merged_df_동이면_target7["Longitude"].astype(float)

merged_df_동이면_target7 = merged_df_동이면_target7.drop(['경로당명/음식점명', '도로명 주소'], axis=1)
merged_df_동이면_target7

merged_df_청산면_target8 = pd.concat([target8, merged_df_청산면], ignore_index=True)

# 'Latitude'와 'Longitude' 열의 데이터 유형을 숫자(float)로 변환
merged_df_청산면_target8["Latitude"] = merged_df_청산면_target8["Latitude"].astype(float)
merged_df_청산면_target8["Longitude"] = merged_df_청산면_target8["Longitude"].astype(float)

merged_df_청산면_target8 = merged_df_청산면_target8.drop(['경로당명/음식점명', '도로명 주소'], axis=1)
merged_df_청산면_target8

merged_df_청성면_target9 = pd.concat([target9, merged_df_청성면], ignore_index=True)

# 'Latitude'와 'Longitude' 열의 데이터 유형을 숫자(float)로 변환
merged_df_청성면_target9["Latitude"] = merged_df_청성면_target9["Latitude"].astype(float)
merged_df_청성면_target9["Longitude"] = merged_df_청성면_target9["Longitude"].astype(float)

merged_df_청성면_target9 = merged_df_청성면_target9.drop(['경로당명/음식점명', '도로명 주소'], axis=1)
merged_df_청성면_target9

# 전체 DBSCAN

# 기본 라이브러리 불러오기

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import folium

plt.scatter(ndf.Latitude, ndf.Longitude)
plt.show

from sklearn import cluster

# DBSCAN 모형 객체 생성
dbscan0 = cluster.DBSCAN(eps=0.009, min_samples=15)

# dbscan0 = cluster.DBSCAN(eps = 0.007, min_samples = 30)일때 : 20개
# dbscan0 = cluster.DBSCAN(eps = 0.005, min_samples = 30) : 12개
# dbscan0 = cluster.DBSCAN(eps = 0.009, min_samples = 30) : 23개
# dbscan0 = cluster.DBSCAN(eps = 0.009, min_samples = 15) : 27개

# 모형학습
dbscan0.fit(ndf)

# 예측
cluster_label0 = dbscan0.labels_
print(cluster_label0)
print('\n')

# 예측 결과를 데이터 프레임에 추가
ndf['Cluster_num'] = cluster_label0
print(ndf.Cluster_num)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(cluster_label0)) - (1 if -1 in cluster_label0 else 0)
n_noise_ = list(cluster_label0).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 클러스터링 결과를 바탕으로 동적으로 색을 할당
unique_clusters = ndf['Cluster_num'].unique()
num_clusters = len(unique_clusters)
colors_dynamic = list(mcolors.CSS4_COLORS.values())  # CSS4 색상 목록 사용

# 색상 리스트에서 필요한 수만큼 색상 추출
colors = colors_dynamic[:num_clusters]

# 그래프로 표현 - 시각화
cluster_map_oc = folium.Map(location=[np.mean(ndf.Latitude), np.mean(ndf.Longitude)], zoom_start=11)

for lat, lng, clus in zip(ndf.Latitude, ndf.Longitude, ndf.Cluster_num):
    color_index = np.where(unique_clusters == clus)[0][0]
    color = colors[color_index]

    folium.CircleMarker([lat, lng],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7
                        ).add_to(cluster_map_oc)

# 지도 html 파일로 저장
cluster_map_oc.save('./dbscan_난수데이터.html')

## 읍면별 DBSCAN

# from sklearn.cluster import DBSCAN
from sklearn import cluster
import folium
import pandas as pd

### target1 옥천읍

# 중앙 좌표 기준으로 지도 생성
map_center1 = [merged_df_옥천읍_target1["Latitude"].mean(), merged_df_옥천읍_target1["Longitude"].mean()]  # 중앙 좌표 설정
map = folium.Map(location=map_center1, zoom_start=10)

# DBSCAN 모형 객체 생성
dbscan1 = cluster.DBSCAN(eps=0.001, min_samples=15)

# 모형 학습
dbscan1.fit(merged_df_옥천읍_target1[['Latitude', 'Longitude']])

# 예측
cluster_label1 = dbscan1.labels_
print(cluster_label1)
print('\n')

# 예측 결과를 데이터 프레임에 추가
merged_df_옥천읍_target1['Cluster_num'] = cluster_label1
print(merged_df_옥천읍_target1.Cluster_num)

# 클러스터링 결과를 바탕으로 동적으로 색을 할당
unique_clusters = merged_df_옥천읍_target1['Cluster_num'].unique()
num_clusters = len(unique_clusters)
colors_dynamic = list(mcolors.CSS4_COLORS.values())  # CSS4 색상 목록 사용

# 색상 리스트에서 필요한 수만큼 색상 추출
colors = colors_dynamic[:num_clusters]

# 지도에 마커 추가
for lat, lng, clus in zip(merged_df_옥천읍_target1.Latitude, merged_df_옥천읍_target1.Longitude,
                          merged_df_옥천읍_target1.Cluster_num):
    color_index = np.where(unique_clusters == clus)[0][0]
    color = colors[color_index]

    folium.CircleMarker([lat, lng],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5
                        ).add_to(map)

# 지도를 html 파일로 저장
map.save('./dbscan_target1_옥천읍.html')

# 클러스터 개수, 노이즈 개수
n_clusters_ = len(set(cluster_label1)) - (1 if -1 in cluster_label1 else 0)
n_noise_ = list(cluster_label1).count(-1)

print("clusters 수: %d" % n_clusters_)
print("noise points 수: %d" % n_noise_)

### target2 이원면

# 중앙 좌표 기준으로 지도 생성
map_center2 = [merged_df_이원면_target2["Latitude"].mean(), merged_df_이원면_target2["Longitude"].mean()]  # 중앙 좌표 설정
map = folium.Map(location=map_center2, zoom_start=10)

# DBSCAN 모형 객체 생성
dbscan2 = cluster.DBSCAN(eps=0.004, min_samples=20)

# 모형학습
dbscan2.fit(merged_df_이원면_target2[['Latitude', 'Longitude']])

# 예측
cluster_label2 = dbscan2.labels_
print(cluster_label2)
print('\n')

# 예측 결과를 데이터 프레임에 추가
merged_df_이원면_target2['Cluster_num'] = cluster_label2
print(merged_df_이원면_target2.Cluster_num)

# 클러스터링 결과를 바탕으로 동적으로 색을 할당
unique_clusters = merged_df_이원면_target2['Cluster_num'].unique()
num_clusters = len(unique_clusters)
colors_dynamic = list(mcolors.CSS4_COLORS.values())  # CSS4 색상 목록 사용

# 색상 리스트에서 필요한 수만큼 색상 추출
colors = colors_dynamic[:num_clusters]

# 지도에 마커 추가
for lat, lng, clus in zip(merged_df_이원면_target2.Latitude, merged_df_이원면_target2.Longitude,
                          merged_df_이원면_target2.Cluster_num):
    color_index = np.where(unique_clusters == clus)[0][0]
    color = colors[color_index]

    folium.CircleMarker([lat, lng],
                        radius=3,
                        color=colors[clus],
                        fill=True,
                        fill_color=colors[clus],
                        fill_opacity=0.5
                        ).add_to(map)

# 지도를 html 파일로 저장
map.save('./dbscan_target2_이원면.html')

# 클러스터 개수, 노이즈 개수
n_clusters_ = len(set(cluster_label2)) - (1 if -1 in cluster_label2 else 0)
n_noise_ = list(cluster_label2).count(-1)

print("clusters 수: %d" % n_clusters_)
print("noise points 수: %d" % n_noise_)

### target3 안내면

# 중앙 좌표 기준으로 지도 생성
map_center3 = [merged_df_안내면_target3["Latitude"].mean(), merged_df_안내면_target3["Longitude"].mean()]  # 중앙 좌표 설정
map = folium.Map(location=map_center3, zoom_start=10)

# DBSCAN 모형 객체 생성
dbscan3 = cluster.DBSCAN(eps=0.007, min_samples=20)

# 모형학습
dbscan3.fit(merged_df_안내면_target3[['Latitude', 'Longitude']])

# 예측
cluster_label3 = dbscan3.labels_
print(cluster_label3)
print('\n')

# 예측 결과를 데이터 프레임에 추가
merged_df_안내면_target3['Cluster_num'] = cluster_label3
print(merged_df_안내면_target3.Cluster_num)

# 클러스터링 결과를 바탕으로 동적으로 색을 할당
unique_clusters = merged_df_안내면_target3['Cluster_num'].unique()
num_clusters = len(unique_clusters)
colors_dynamic = list(mcolors.CSS4_COLORS.values())  # CSS4 색상 목록 사용

# 색상 리스트에서 필요한 수만큼 색상 추출
colors = colors_dynamic[:num_clusters]

# 지도에 마커 추가
for lat, lng, clus in zip(merged_df_안내면_target3.Latitude, merged_df_안내면_target3.Longitude,
                          merged_df_안내면_target3.Cluster_num):
    color_index = np.where(unique_clusters == clus)[0][0]
    color = colors[color_index]

    folium.CircleMarker([lat, lng],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5
                        ).add_to(map)

# 지도를 html 파일로 저장
map.save('./dbscan_target3_안내면.html')

# 클러스터 개수, 노이즈 개수
n_clusters_ = len(set(cluster_label3)) - (1 if -1 in cluster_label3 else 0)
n_noise_ = list(cluster_label3).count(-1)

print("clusters 수: %d" % n_clusters_)
print("noise points 수: %d" % n_noise_)

### target4 안남면

# 중앙 좌표 기준으로 지도 생성
map_center4 = [merged_df_안남면_target4["Latitude"].mean(), merged_df_안남면_target4["Longitude"].mean()]  # 중앙 좌표 설정
map = folium.Map(location=map_center4, zoom_start=10)

# DBSCAN 모형 객체 생성
dbscan4 = cluster.DBSCAN(eps=0.0045, min_samples=20)

# 모형학습
dbscan4.fit(merged_df_안남면_target4[['Latitude', 'Longitude']])

# 예측
cluster_label4 = dbscan4.labels_
print(cluster_label4)
print('\n')

# 예측 결과를 데이터 프레임에 추가
merged_df_안남면_target4['Cluster_num'] = cluster_label4
print(merged_df_안남면_target4.Cluster_num)

# 클러스터링 결과를 바탕으로 동적으로 색을 할당
unique_clusters = merged_df_안남면_target4['Cluster_num'].unique()
num_clusters = len(unique_clusters)
colors_dynamic = list(mcolors.CSS4_COLORS.values())  # CSS4 색상 목록 사용

# 색상 리스트에서 필요한 수만큼 색상 추출
colors = colors_dynamic[:num_clusters]

# 지도에 마커 추가
for lat, lng, clus in zip(merged_df_안남면_target4.Latitude, merged_df_안남면_target4.Longitude,
                          merged_df_안남면_target4.Cluster_num):
    color_index = np.where(unique_clusters == clus)[0][0]
    color = colors[color_index]

    folium.CircleMarker([lat, lng],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5
                        ).add_to(map)

# 지도를 html 파일로 저장
map.save('./dbscan_target4_안남면.html')

# 클러스터 개수, 노이즈 개수
n_clusters_ = len(set(cluster_label4)) - (1 if -1 in cluster_label4 else 0)
n_noise_ = list(cluster_label4).count(-1)

print("clusters 수: %d" % n_clusters_)
print("noise points 수: %d" % n_noise_)

### target5 군서면

# 중앙 좌표 기준으로 지도 생성
map_center5 = [merged_df_군서면_target5["Latitude"].mean(), merged_df_군서면_target5["Longitude"].mean()]  # 중앙 좌표 설정
map = folium.Map(location=map_center5, zoom_start=10)

# DBSCAN 모형 객체 생성
dbscan5 = cluster.DBSCAN(eps=0.0045, min_samples=20)

# 모형학습
dbscan5.fit(merged_df_군서면_target5[['Latitude', 'Longitude']])

# 예측
cluster_label5 = dbscan5.labels_
print(cluster_label5)
print('\n')

# 예측 결과를 데이터 프레임에 추가
merged_df_군서면_target5['Cluster_num'] = cluster_label5
print(merged_df_군서면_target5.Cluster_num)

# 클러스터링 결과를 바탕으로 동적으로 색을 할당
unique_clusters = merged_df_군서면_target5['Cluster_num'].unique()
num_clusters = len(unique_clusters)
colors_dynamic = list(mcolors.CSS4_COLORS.values())  # CSS4 색상 목록 사용

# 색상 리스트에서 필요한 수만큼 색상 추출
colors = colors_dynamic[:num_clusters]

# 지도에 마커 추가
for lat, lng, clus in zip(merged_df_군서면_target5.Latitude, merged_df_군서면_target5.Longitude,
                          merged_df_군서면_target5.Cluster_num):
    color_index = np.where(unique_clusters == clus)[0][0]
    color = colors[color_index]

    folium.CircleMarker([lat, lng],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5
                        ).add_to(map)

# 지도를 html 파일로 저장
map.save('./dbscan_target5_군서면.html')

# 클러스터 개수, 노이즈 개수
n_clusters_ = len(set(cluster_label5)) - (1 if -1 in cluster_label5 else 0)
n_noise_ = list(cluster_label5).count(-1)

print("clusters 수: %d" % n_clusters_)
print("noise points 수: %d" % n_noise_)

### target6 군북면

# 중앙 좌표 기준으로 지도 생성
map_center6 = [merged_df_군북면_target6["Latitude"].mean(), merged_df_군북면_target6["Longitude"].mean()]  # 중앙 좌표 설정
map = folium.Map(location=map_center6, zoom_start=10)

# DBSCAN 모형 객체 생성
dbscan6 = cluster.DBSCAN(eps=0.0055, min_samples=20)

# 모형학습
dbscan6.fit(merged_df_군북면_target6[['Latitude', 'Longitude']])

# 예측
cluster_label6 = dbscan6.labels_
print(cluster_label6)
print('\n')

# 예측 결과를 데이터 프레임에 추가
merged_df_군북면_target6['Cluster_num'] = cluster_label6
print(merged_df_군북면_target6.Cluster_num)

# 클러스터링 결과를 바탕으로 동적으로 색을 할당
unique_clusters = merged_df_군북면_target6['Cluster_num'].unique()
num_clusters = len(unique_clusters)
colors_dynamic = list(mcolors.CSS4_COLORS.values())  # CSS4 색상 목록 사용

# 색상 리스트에서 필요한 수만큼 색상 추출
colors = colors_dynamic[:num_clusters]

# 지도에 마커 추가
for lat, lng, clus in zip(merged_df_군북면_target6.Latitude, merged_df_군북면_target6.Longitude,
                          merged_df_군북면_target6.Cluster_num):
    color_index = np.where(unique_clusters == clus)[0][0]
    color = colors[color_index]

    folium.CircleMarker([lat, lng],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5
                        ).add_to(map)

# 지도를 html 파일로 저장
map.save('./dbscan_target6_군북면.html')

# 클러스터 개수, 노이즈 개수
n_clusters_ = len(set(cluster_label6)) - (1 if -1 in cluster_label6 else 0)
n_noise_ = list(cluster_label6).count(-1)

print("clusters 수: %d" % n_clusters_)
print("noise points 수: %d" % n_noise_)

### target7 동이면

# 중앙 좌표 기준으로 지도 생성
map_center7 = [merged_df_동이면_target7["Latitude"].mean(), merged_df_동이면_target7["Longitude"].mean()]  # 중앙 좌표 설정
map = folium.Map(location=map_center7, zoom_start=10)

# DBSCAN 모형 객체 생성
dbscan7 = cluster.DBSCAN(eps=0.0045, min_samples=20)

# 모형학습
dbscan7.fit(merged_df_동이면_target7[['Latitude', 'Longitude']])

# 예측
cluster_label7 = dbscan7.labels_
print(cluster_label7)
print('\n')

# 예측 결과를 데이터 프레임에 추가
merged_df_동이면_target7['Cluster_num'] = cluster_label7
print(merged_df_동이면_target7.Cluster_num)

# 클러스터링 결과를 바탕으로 동적으로 색을 할당
unique_clusters = merged_df_동이면_target7['Cluster_num'].unique()
num_clusters = len(unique_clusters)
colors_dynamic = list(mcolors.CSS4_COLORS.values())  # CSS4 색상 목록 사용

# 색상 리스트에서 필요한 수만큼 색상 추출
colors = colors_dynamic[:num_clusters]

# 지도에 마커 추가
for lat, lng, clus in zip(merged_df_동이면_target7.Latitude, merged_df_동이면_target7.Longitude,
                          merged_df_동이면_target7.Cluster_num):
    color_index = np.where(unique_clusters == clus)[0][0]
    color = colors[color_index]

    folium.CircleMarker([lat, lng],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5
                        ).add_to(map)

# 지도를 html 파일로 저장
map.save('./dbscan_target7_동이면.html')

# 클러스터 개수, 노이즈 개수
n_clusters_ = len(set(cluster_label7)) - (1 if -1 in cluster_label7 else 0)
n_noise_ = list(cluster_label7).count(-1)

print("clusters 수: %d" % n_clusters_)
print("noise points 수: %d" % n_noise_)

### target8 청산면

# 중앙 좌표 기준으로 지도 생성
map_center8 = [merged_df_청산면_target8["Latitude"].mean(), merged_df_청산면_target8["Longitude"].mean()]  # 중앙 좌표 설정
map = folium.Map(location=map_center8, zoom_start=10)

# DBSCAN 모형 객체 생성
dbscan8 = cluster.DBSCAN(eps=0.0045, min_samples=20)

# 모형학습
dbscan8.fit(merged_df_청산면_target8[['Latitude', 'Longitude']])

# 예측
cluster_label8 = dbscan8.labels_
print(cluster_label8)
print('\n')

# 예측 결과를 데이터 프레임에 추가
merged_df_청산면_target8['Cluster_num'] = cluster_label8
print(merged_df_청산면_target8.Cluster_num)

# 클러스터링 결과를 바탕으로 동적으로 색을 할당
unique_clusters = merged_df_청산면_target8['Cluster_num'].unique()
num_clusters = len(unique_clusters)
colors_dynamic = list(mcolors.CSS4_COLORS.values())  # CSS4 색상 목록 사용

# 색상 리스트에서 필요한 수만큼 색상 추출
colors = colors_dynamic[:num_clusters]

# 지도에 마커 추가
for lat, lng, clus in zip(merged_df_청산면_target8.Latitude, merged_df_청산면_target8.Longitude,
                          merged_df_청산면_target8.Cluster_num):
    color_index = np.where(unique_clusters == clus)[0][0]
    color = colors[color_index]

    folium.CircleMarker([lat, lng],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5
                        ).add_to(map)

# 지도를 html 파일로 저장
map.save('./dbscan_target8_청산면.html')

# 클러스터 개수, 노이즈 개수
n_clusters_ = len(set(cluster_label8)) - (1 if -1 in cluster_label8 else 0)
n_noise_ = list(cluster_label8).count(-1)

print("clusters 수: %d" % n_clusters_)
print("noise points 수: %d" % n_noise_)

### target9 청성면

# 중앙 좌표 기준으로 지도 생성
map_center9 = [merged_df_청성면_target9["Latitude"].mean(), merged_df_청성면_target9["Longitude"].mean()]  # 중앙 좌표 설정
map = folium.Map(location=map_center9, zoom_start=10)

# DBSCAN 모형 객체 생성
dbscan9 = cluster.DBSCAN(eps=0.0065, min_samples=20)

# 모형학습
dbscan9.fit(merged_df_청성면_target9[['Latitude', 'Longitude']])

# 예측
cluster_label9 = dbscan9.labels_
print(cluster_label9)
print('\n')

# 예측 결과를 데이터 프레임에 추가
merged_df_청성면_target9['Cluster_num'] = cluster_label9
print(merged_df_청성면_target9.Cluster_num)

# 클러스터링 결과를 바탕으로 동적으로 색을 할당
unique_clusters = merged_df_청성면_target9['Cluster_num'].unique()
num_clusters = len(unique_clusters)
colors_dynamic = list(mcolors.CSS4_COLORS.values())  # CSS4 색상 목록 사용

# 색상 리스트에서 필요한 수만큼 색상 추출
colors = colors_dynamic[:num_clusters]

# 지도에 마커 추가
for lat, lng, clus in zip(merged_df_청성면_target9.Latitude, merged_df_청성면_target9.Longitude,
                          merged_df_청성면_target9.Cluster_num):
    color_index = np.where(unique_clusters == clus)[0][0]
    color = colors[color_index]

    folium.CircleMarker([lat, lng],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5
                        ).add_to(map)

# 지도를 html 파일로 저장
map.save('./dbscan_target9_청성면.html')

# 클러스터 개수, 노이즈 개수
n_clusters_ = len(set(cluster_label9)) - (1 if -1 in cluster_label9 else 0)
n_noise_ = list(cluster_label9).count(-1)

print("clusters 수: %d" % n_clusters_)
print("noise points 수: %d" % n_noise_)

# 실루엣 계수

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score

### 1. 옥천읍

merged_df_옥천읍_target1['dbscan_silhouette'] = silhouette_samples(X=merged_df_옥천읍_target1[['Latitude', 'Longitude']],
                                                                labels=merged_df_옥천읍_target1['Cluster_num'])

dbscan옥천읍_silhouette = silhouette_score(X=merged_df_옥천읍_target1[['Latitude', 'Longitude']],
                                        labels=merged_df_옥천읍_target1['Cluster_num'])

print('dbscan')
print('\n옥천읍_실루엣 계수:')
print(merged_df_옥천읍_target1.dbscan_silhouette)
print('\n옥천읍 data의 실루엣 계수(평균):')
print(dbscan옥천읍_silhouette)

### 2. 이원면

merged_df_이원면_target2['dbscan_silhouette'] = silhouette_samples(X=merged_df_이원면_target2[['Latitude', 'Longitude']],
                                                                labels=merged_df_이원면_target2['Cluster_num'])

dbscan이원면_silhouette = silhouette_score(X=merged_df_이원면_target2[['Latitude', 'Longitude']],
                                        labels=merged_df_이원면_target2['Cluster_num'])

print('dbscan')
print('\n이원면_실루엣 계수:')
print(merged_df_이원면_target2.dbscan_silhouette)
print('\n이원면 data의 실루엣 계수(평균):')
print(dbscan이원면_silhouette)

### 3. 안내면

merged_df_안내면_target3['dbscan_silhouette'] = silhouette_samples(X=merged_df_안내면_target3[['Latitude', 'Longitude']],
                                                                labels=merged_df_안내면_target3['Cluster_num'])

dbscan안내면_silhouette = silhouette_score(X=merged_df_안내면_target3[['Latitude', 'Longitude']],
                                        labels=merged_df_안내면_target3['Cluster_num'])

print('dbscan')
print('\n안내면_실루엣 계수:')
print(merged_df_안내면_target3.dbscan_silhouette)
print('\n안내면 data의 실루엣 계수(평균):')
print(dbscan안내면_silhouette)

### 4. 안남면

merged_df_안남면_target4['dbscan_silhouette'] = silhouette_samples(X=merged_df_안남면_target4[['Latitude', 'Longitude']],
                                                                labels=merged_df_안남면_target4['Cluster_num'])

dbscan안남면_silhouette = silhouette_score(X=merged_df_안남면_target4[['Latitude', 'Longitude']],
                                        labels=merged_df_안남면_target4['Cluster_num'])

print('dbscan')
print('\n안남면_실루엣 계수:')
print(merged_df_안남면_target4.dbscan_silhouette)
print('\n안남면 data의 실루엣 계수(평균):')
print(dbscan안남면_silhouette)

### 5. 군서면

merged_df_군서면_target5['dbscan_silhouette'] = silhouette_samples(X=merged_df_군서면_target5[['Latitude', 'Longitude']],
                                                                labels=merged_df_군서면_target5['Cluster_num'])

dbscan군서면_silhouette = silhouette_score(X=merged_df_군서면_target5[['Latitude', 'Longitude']],
                                        labels=merged_df_군서면_target5['Cluster_num'])

print('dbscan')
print('\n군서면_실루엣 계수:')
print(merged_df_군서면_target5.dbscan_silhouette)
print('\n군서면 data의 실루엣 계수(평균):')
print(dbscan군서면_silhouette)

### 6. 군북면

merged_df_군북면_target6['dbscan_silhouette'] = silhouette_samples(X=merged_df_군북면_target6[['Latitude', 'Longitude']],
                                                                labels=merged_df_군북면_target6['Cluster_num'])

dbscan군북면_silhouette = silhouette_score(X=merged_df_군북면_target6[['Latitude', 'Longitude']],
                                        labels=merged_df_군북면_target6['Cluster_num'])

print('dbscan')
print('\n군북면_실루엣 계수:')
print(merged_df_군북면_target6.dbscan_silhouette)
print('\군북면 data의 실루엣 계수(평균):')
print(dbscan군북면_silhouette)

### 7. 동이면

merged_df_동이면_target7['dbscan_silhouette'] = silhouette_samples(X=merged_df_동이면_target7[['Latitude', 'Longitude']],
                                                                labels=merged_df_동이면_target7['Cluster_num'])

dbscan동이면_silhouette = silhouette_score(X=merged_df_동이면_target7[['Latitude', 'Longitude']],
                                        labels=merged_df_동이면_target7['Cluster_num'])

print('dbscan')
print('\n동이면_실루엣 계수:')
print(merged_df_동이면_target7.dbscan_silhouette)
print('\n동이면 data의 실루엣 계수(평균):')
print(dbscan동이면_silhouette)

### 8. 청산면

merged_df_청산면_target8['dbscan_silhouette'] = silhouette_samples(X=merged_df_청산면_target8[['Latitude', 'Longitude']],
                                                                labels=merged_df_청산면_target8['Cluster_num'])

dbscan청산면_silhouette = silhouette_score(X=merged_df_청산면_target8[['Latitude', 'Longitude']],
                                        labels=merged_df_청산면_target8['Cluster_num'])

print('dbscan')
print('\n청산면_실루엣 계수:')
print(merged_df_청산면_target8.dbscan_silhouette)
print('\n청산면 data의 실루엣 계수(평균):')
print(dbscan청산면_silhouette)

### 9. 청성면

merged_df_청성면_target9['dbscan_silhouette'] = silhouette_samples(X=merged_df_청성면_target9[['Latitude', 'Longitude']],
                                                                labels=merged_df_청성면_target9['Cluster_num'])

dbscan청성면_silhouette = silhouette_score(X=merged_df_청성면_target9[['Latitude', 'Longitude']],
                                        labels=merged_df_청성면_target9['Cluster_num'])

print('dbscan')
print('\n청성면_실루엣 계수:')
print(merged_df_청성면_target9.dbscan_silhouette)
print('\n청성면 data의 실루엣 계수(평균):')
print(dbscan청성면_silhouette)

# 실루엣계수 시각화

dbscan_silhouette_sample = merged_df_옥천읍_target1['dbscan_silhouette']
labels = dbscan1.labels_

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(dbscan_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(dbscan_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('옥천읍 dbscan 분석 결과')

plt.rc('font', size=20)

plt.show()

dbscan_silhouette_sample = merged_df_이원면_target2['dbscan_silhouette']
labels = dbscan2.labels_

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(dbscan_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(dbscan_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('이원면 dbscan 분석 결과')

plt.rc('font', size=20)

plt.show()

dbscan_silhouette_sample = merged_df_안내면_target3['dbscan_silhouette']
labels = dbscan3.labels_

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(dbscan_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(dbscan_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('안내면 dbscan 분석 결과')

plt.rc('font', size=20)

plt.show()

dbscan_silhouette_sample = merged_df_안남면_target4['dbscan_silhouette']
labels = dbscan4.labels_

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(dbscan_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(dbscan_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('안남면 dbscan 분석 결과')

plt.rc('font', size=20)

plt.show()

dbscan_silhouette_sample = merged_df_군서면_target5['dbscan_silhouette']
labels = dbscan5.labels_

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(dbscan_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(dbscan_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('군서면 dbscan 분석 결과')

plt.rc('font', size=20)

plt.show()

dbscan_silhouette_sample = merged_df_군북면_target6['dbscan_silhouette']
labels = dbscan6.labels_

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(dbscan_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(dbscan_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('군북면 dbscan 분석 결과')

plt.rc('font', size=20)

plt.show()

dbscan_silhouette_sample = merged_df_동이면_target7['dbscan_silhouette']
labels = dbscan7.labels_

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(dbscan_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(dbscan_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('동이면 dbscan 분석 결과')

plt.rc('font', size=20)

plt.show()

dbscan_silhouette_sample = merged_df_청산면_target8['dbscan_silhouette']
labels = dbscan8.labels_

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(dbscan_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(dbscan_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('청산면 dbscan 분석 결과')

plt.rc('font', size=20)

plt.show()

dbscan_silhouette_sample = merged_df_청성면_target9['dbscan_silhouette']
labels = dbscan9.labels_

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(dbscan_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(dbscan_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('청성면 dbscan 분석 결과')

plt.rc('font', size=20)

plt.show()