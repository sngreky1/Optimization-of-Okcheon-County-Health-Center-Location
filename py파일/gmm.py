
import pandas as pd
import folium
import random

pd10 = pd.read_csv('충청북도 옥천군_경로당현황_좌표추가.csv')
pd20 = pd.read_csv('충청북도 옥천군_음식점현황_좌표추가.csv')

pd10

pd10 = pd10.rename(columns={"경로당명": "경로당명/음식점명"})
pd10 = pd10.rename(columns={"경로당 주소": "도로명 주소"})
pd10

pd10 = pd10.drop("연번", axis=1)
pd10 = pd10.drop("읍면", axis=1)
pd10 = pd10.drop("데이터기준일", axis=1)
pd10

pd20 = pd20.rename(columns={"업소명": "경로당명/음식점명"})
pd20 = pd20.rename(columns={"소재지(도로명)": "도로명 주소"})
pd20 = pd20[1:].dropna()
pd20

pd20 = pd20.drop("연번", axis=1)
pd20 = pd20.drop("업종명", axis=1)
pd20 = pd20.drop("군분", axis=1)
pd20 = pd20.drop("소재지(지번)", axis=1)
pd20 = pd20.drop("소재지전화", axis=1)
pd20 = pd20.drop("데이터기준일", axis=1)
pd20 = pd20[:-1]
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

# target1~9

map_center1 = [target1["Latitude"].mean(), target1["Longitude"].mean()]  # 중앙 좌표 설정

map = folium.Map(location=map_center1, zoom_start=10)  # 중앙 좌표 기준으로 지도 생성

# 랜덤 좌표들을 지도에 마커로 추가
# target1
for lat, lng in zip(target1.Latitude, target1.Longitude):
    folium.CircleMarker([lat, lng],
                        radius=2,
                        color='blue',
                        fill=True,
                        fill_color='skyblue',
                        fill_opacity=0.5
                        ).add_to(map)
# target2
for lat, lng in zip(target2.Latitude, target2.Longitude):
    folium.CircleMarker([lat, lng],
                        radius=2,
                        color='red',
                        fill=True,
                        fill_color='pink',
                        fill_opacity=0.5
                        ).add_to(map)
# target3
for lat, lng in zip(target3.Latitude, target3.Longitude):
    folium.CircleMarker([lat, lng],
                        radius=3,
                        color='green',
                        fill=True,
                        fill_color='yellow',
                        fill_opacity=0.5
                        ).add_to(map)
# target4
for lat, lng in zip(target4.Latitude, target4.Longitude):
    folium.CircleMarker([lat, lng],
                        radius=3,
                        color='white',
                        fill=True,
                        fill_color='yellow',
                        fill_opacity=0.2
                        ).add_to(map)

# target5
for lat, lng in zip(target5.Latitude, target5.Longitude):
    folium.CircleMarker([lat, lng],
                        radius=2,
                        color='black',
                        fill=True,
                        fill_color='skyblue',
                        fill_opacity=0.2
                        ).add_to(map)
# target6
for lat, lng in zip(target6.Latitude, target6.Longitude):
    folium.CircleMarker([lat, lng],
                        radius=2,
                        color='yellow',
                        fill=True,
                        fill_color='pink',
                        fill_opacity=0.5
                        ).add_to(map)
# target7
for lat, lng in zip(target7.Latitude, target7.Longitude):
    folium.CircleMarker([lat, lng],
                        radius=3,
                        color='skyblue',
                        fill=True,
                        fill_color='yellow',
                        fill_opacity=0.5
                        ).add_to(map)

# target8
for lat, lng in zip(target8.Latitude, target8.Longitude):
    folium.CircleMarker([lat, lng],
                        radius=3,
                        color='violet',
                        fill=True,
                        fill_color='yellow',
                        fill_opacity=0.5
                        ).add_to(map)

# target9
for lat, lng in zip(target9.Latitude, target9.Longitude):
    folium.CircleMarker([lat, lng],
                        radius=3,
                        color='coral',
                        fill=True,
                        fill_color='yellow',
                        fill_opacity=0.5
                        ).add_to(map)

# 지도를 html 파일로 저장
# map.save('./oc_randnum1_9.html')

# 읍면 난수 데이터 병합
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

# GNN 옥천군 전체

# 기본 라이브러리 불러오기

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import folium

import matplotlib as mpl
import seaborn as sns

import warnings

mpl.rc('font', family='NanumGothic')  # 폰트 설정
mpl.rc('axes', unicode_minus=False)  # 유니코드에서 음수 부호 설정

# 차트 스타일 설정
sns.set(font="NanumGothic", rc={"axes.unicode_minus": False}, style='darkgrid')
plt.rc("figure", figsize=(10, 8))

warnings.filterwarnings("ignore")


def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter:
        centers = clusterobj.cluster_centers_

    unique_labels = np.unique(dataframe[label_name].values)

    # markers 리스트 길이 조정
    markers = ['o', 's', '^', 'x', '*'] * ((len(unique_labels) // 5) + 1)
    isNoise = False

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name] == label]

        if label == -1:
            cluster_legend = 'Noise'
            isNoise = True
        else:
            cluster_legend = 'Cluster ' + str(label)

        # 각 군집 시각화
        plt.scatter(x=label_cluster['Longitude'], y=label_cluster['Latitude'], s=70,
                    edgecolor='k', marker=markers[label], label=cluster_legend)

        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                        alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k',
                        edgecolor='k', marker='$%d$' % label)

    if isNoise:
        legend_loc = 'upper center'
    else:
        legend_loc = 'upper right'

    plt.legend(loc=legend_loc)
    plt.show()


from sklearn.mixture import GaussianMixture

# 군집의 수 35개로 설정
gmm = GaussianMixture(n_components=35, random_state=0)
gmm_cluster_labels = gmm.fit_predict(ndf)

ndf["gmm_cluster"] = gmm_cluster_labels

visualize_cluster_plot(gmm, ndf, 'gmm_cluster', iscenter=False)

print('\n Gaussian Mixture Clustering _옥천군 전체')
print(ndf['gmm_cluster'].value_counts().sort_index())

## 읍/면별로 GNN

### target1 옥천읍

from sklearn.mixture import GaussianMixture

# 군집의 수 7개로 설정
gmm = GaussianMixture(n_components=7, random_state=0)
gmm_cluster_labels1 = gmm.fit_predict(merged_df_옥천읍_target1)

merged_df_옥천읍_target1["gmm_cluster"] = gmm_cluster_labels1

visualize_cluster_plot(gmm, merged_df_옥천읍_target1, 'gmm_cluster', iscenter=False)

print('\n Gaussian Mixture Clustering_target1 옥천읍')
print(merged_df_옥천읍_target1['gmm_cluster'].value_counts().sort_index())

### target2 이원면

# 군집의 수 7개로 설정
gmm = GaussianMixture(n_components=7, random_state=0)
gmm_cluster_labels2 = gmm.fit_predict(merged_df_이원면_target2)

merged_df_이원면_target2["gmm_cluster"] = gmm_cluster_labels2

visualize_cluster_plot(gmm, merged_df_이원면_target2, 'gmm_cluster', iscenter=False)

print('\n Gaussian Mixture Clustering_target2 이원면')
print(merged_df_이원면_target2['gmm_cluster'].value_counts().sort_index())

### target3 안내면

# 군집의 수 7개로 설정
gmm = GaussianMixture(n_components=7, random_state=0)
gmm_cluster_labels3 = gmm.fit_predict(merged_df_안내면_target3)

merged_df_안내면_target3["gmm_cluster"] = gmm_cluster_labels3

visualize_cluster_plot(gmm, merged_df_안내면_target3, 'gmm_cluster', iscenter=False)

print('\n Gaussian Mixture Clustering_target3 안내면')
print(merged_df_안내면_target3['gmm_cluster'].value_counts().sort_index())

### target4 안남면

# 군집의 수 7개로 설정
gmm = GaussianMixture(n_components=7, random_state=0)
gmm_cluster_labels4 = gmm.fit_predict(merged_df_안남면_target4)

merged_df_안남면_target4["gmm_cluster"] = gmm_cluster_labels4

visualize_cluster_plot(gmm, merged_df_안남면_target4, 'gmm_cluster', iscenter=False)

print('\n Gaussian Mixture Clustering_target4 안남면')
print(merged_df_안남면_target4['gmm_cluster'].value_counts().sort_index())

### target5 군서면

# 군집의 수 10개로 설정
gmm = GaussianMixture(n_components=7, random_state=0)
gmm_cluster_labels5 = gmm.fit_predict(merged_df_군서면_target5)

merged_df_군서면_target5["gmm_cluster"] = gmm_cluster_labels5

visualize_cluster_plot(gmm, merged_df_군서면_target5, 'gmm_cluster', iscenter=False)

print('\n Gaussian Mixture Clustering_target5 군서면')
print(merged_df_군서면_target5['gmm_cluster'].value_counts().sort_index())

### target6 군북면

# 군집의 수 10개로 설정
gmm = GaussianMixture(n_components=7, random_state=0)
gmm_cluster_labels6 = gmm.fit_predict(merged_df_군북면_target6)

merged_df_군북면_target6["gmm_cluster"] = gmm_cluster_labels6

visualize_cluster_plot(gmm, merged_df_군북면_target6, 'gmm_cluster', iscenter=False)

print('\n Gaussian Mixture Clustering_target6 군북면')
print(merged_df_군북면_target6['gmm_cluster'].value_counts().sort_index())

### target7 동이면

# 군집의 수 7개로 설정
gmm = GaussianMixture(n_components=7, random_state=0)
gmm_cluster_labels7 = gmm.fit_predict(merged_df_동이면_target7)

merged_df_동이면_target7["gmm_cluster"] = gmm_cluster_labels7

visualize_cluster_plot(gmm, merged_df_동이면_target7, 'gmm_cluster', iscenter=False)

print('\n Gaussian Mixture Clustering_target7 동이면')
print(merged_df_동이면_target7['gmm_cluster'].value_counts().sort_index())

### target8 청산면

# 군집의 수 7개로 설정
gmm = GaussianMixture(n_components=7, random_state=0)
gmm_cluster_labels8 = gmm.fit_predict(merged_df_청산면_target8)

merged_df_청산면_target8["gmm_cluster"] = gmm_cluster_labels8

visualize_cluster_plot(gmm, merged_df_청산면_target8, 'gmm_cluster', iscenter=False)

print('\n Gaussian Mixture Clustering_target8 청산면')
print(merged_df_청산면_target8['gmm_cluster'].value_counts().sort_index())

### target9 청성면

# 군집의 수 7개로 설정
gmm = GaussianMixture(n_components=7, random_state=0)
gmm_cluster_labels9 = gmm.fit_predict(merged_df_청성면_target9)

merged_df_청성면_target9["gmm_cluster"] = gmm_cluster_labels9

visualize_cluster_plot(gmm, merged_df_청성면_target9, 'gmm_cluster', iscenter=False)

print('\n Gaussian Mixture Clustering_target9 청성면')
print(merged_df_청성면_target9['gmm_cluster'].value_counts().sort_index())

# 실루엣 계수

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score

### 1. 옥천읍

merged_df_옥천읍_target1['gmm_silhouette'] = silhouette_samples(X=merged_df_옥천읍_target1[['Latitude', 'Longitude']],
                                                             labels=merged_df_옥천읍_target1['gmm_cluster'])

gmm옥천읍_silhouette = silhouette_score(X=merged_df_옥천읍_target1[['Latitude', 'Longitude']],
                                     labels=merged_df_옥천읍_target1['gmm_cluster'])

print('gnn')
print('\n옥천읍_실루엣 계수:')
print(merged_df_옥천읍_target1.gmm_silhouette)
print('\n옥천읍 data의 실루엣 계수(평균):')
print(gmm옥천읍_silhouette)

### 2. 이원면

merged_df_이원면_target2['gmm_silhouette'] = silhouette_samples(X=merged_df_이원면_target2[['Latitude', 'Longitude']],
                                                             labels=merged_df_이원면_target2['gmm_cluster'])

gmm이원면_silhouette = silhouette_score(X=merged_df_이원면_target2[['Latitude', 'Longitude']],
                                     labels=merged_df_이원면_target2['gmm_cluster'])

print('gnn')
print('\n이원면_실루엣 계수:')
print(merged_df_이원면_target2.gmm_silhouette)
print('\n이원면 data의 실루엣 계수(평균):')
print(gmm이원면_silhouette)

### 3. 안내면

merged_df_안내면_target3['gmm_silhouette'] = silhouette_samples(X=merged_df_안내면_target3[['Latitude', 'Longitude']],
                                                             labels=merged_df_안내면_target3['gmm_cluster'])

gmm안내면_silhouette = silhouette_score(X=merged_df_안내면_target3[['Latitude', 'Longitude']],
                                     labels=merged_df_안내면_target3['gmm_cluster'])

print('dbscan')
print('\n안내면_실루엣 계수:')
print(merged_df_안내면_target3.gmm_silhouette)
print('\n안내면 data의 실루엣 계수(평균):')
print(gmm안내면_silhouette)

### 4. 안남면

merged_df_안남면_target4['gmm_silhouette'] = silhouette_samples(X=merged_df_안남면_target4[['Latitude', 'Longitude']],
                                                             labels=merged_df_안남면_target4['gmm_cluster'])

gmm안남면_silhouette = silhouette_score(X=merged_df_안남면_target4[['Latitude', 'Longitude']],
                                     labels=merged_df_안남면_target4['gmm_cluster'])

print('dbscan')
print('\n안남면_실루엣 계수:')
print(merged_df_안남면_target4.gmm_silhouette)
print('\n안남면 data의 실루엣 계수(평균):')
print(gmm안남면_silhouette)

### 5. 군서면

merged_df_군서면_target5['gmm_silhouette'] = silhouette_samples(X=merged_df_군서면_target5[['Latitude', 'Longitude']],
                                                             labels=merged_df_군서면_target5['gmm_cluster'])

gmm군서면_silhouette = silhouette_score(X=merged_df_군서면_target5[['Latitude', 'Longitude']],
                                     labels=merged_df_군서면_target5['gmm_cluster'])

print('gnn')
print('\n군서면_실루엣 계수:')
print(merged_df_군서면_target5.gmm_silhouette)
print('\n군서면 data의 실루엣 계수(평균):')
print(gmm군서면_silhouette)

### 6. 군북면

merged_df_군북면_target6['gmm_silhouette'] = silhouette_samples(X=merged_df_군북면_target6[['Latitude', 'Longitude']],
                                                             labels=merged_df_군북면_target6['gmm_cluster'])

gmm군북면_silhouette = silhouette_score(X=merged_df_군북면_target6[['Latitude', 'Longitude']],
                                     labels=merged_df_군북면_target6['gmm_cluster'])

print('gnn')
print('\n군북면_실루엣 계수:')
print(merged_df_군북면_target6.gmm_silhouette)
print('\n군북면 data의 실루엣 계수(평균):')
print(gmm군북면_silhouette)

### 7. 동이면

merged_df_동이면_target7['gmm_silhouette'] = silhouette_samples(X=merged_df_동이면_target7[['Latitude', 'Longitude']],
                                                             labels=merged_df_동이면_target7['gmm_cluster'])

gmm동이면_silhouette = silhouette_score(X=merged_df_동이면_target7[['Latitude', 'Longitude']],
                                     labels=merged_df_동이면_target7['gmm_cluster'])

print('dbscan')
print('\n동이면_실루엣 계수:')
print(merged_df_동이면_target7.gmm_silhouette)
print('\n동이면 data의 실루엣 계수(평균):')
print(gmm동이면_silhouette)

### 8. 청산면

merged_df_청산면_target8['gmm_silhouette'] = silhouette_samples(X=merged_df_청산면_target8[['Latitude', 'Longitude']],
                                                             labels=merged_df_청산면_target8['gmm_cluster'])

gmm청산면_silhouette = silhouette_score(X=merged_df_청산면_target8[['Latitude', 'Longitude']],
                                     labels=merged_df_청산면_target8['gmm_cluster'])

print('dbscan')
print('\n청산면_실루엣 계수:')
print(merged_df_청산면_target8.gmm_silhouette)
print('\n청산면 data의 실루엣 계수(평균):')
print(gmm청산면_silhouette)

### 9. 청성면

merged_df_청성면_target9['gmm_silhouette'] = silhouette_samples(X=merged_df_청성면_target9[['Latitude', 'Longitude']],
                                                             labels=merged_df_청성면_target9['gmm_cluster'])

gmm청성면_silhouette = silhouette_score(X=merged_df_청성면_target9[['Latitude', 'Longitude']],
                                     labels=merged_df_청성면_target9['gmm_cluster'])

print('dbscan')
print('\n청성면_실루엣 계수:')
print(merged_df_청성면_target9.gmm_silhouette)
print('\n청성면 data의 실루엣 계수(평균):')
print(gmm청성면_silhouette)

# 실루엣계수 시각화


gnn_silhouette_sample = merged_df_옥천읍_target1['gmm_silhouette']
labels = gmm_cluster_labels1

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(gnn_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(gnn_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('옥천읍 gmm 분석 결과')

plt.rc('font', size=20)

plt.show()

gnn_silhouette_sample = merged_df_이원면_target2['gmm_silhouette']
labels = gmm_cluster_labels2

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(gnn_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(gnn_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('이원면 gmm 분석 결과')

plt.rc('font', size=20)

plt.show()

gnn_silhouette_sample = merged_df_안내면_target3['gmm_silhouette']
labels = gmm_cluster_labels3

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(gnn_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(gnn_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('안내면 gmm 분석 결과')

plt.rc('font', size=20)

plt.show()

gnn_silhouette_sample = merged_df_안남면_target4['gmm_silhouette']
labels = gmm_cluster_labels4

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(gnn_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(gnn_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('안남면 gmm 분석 결과')

plt.rc('font', size=20)

plt.show()

gnn_silhouette_sample = merged_df_군서면_target5['gmm_silhouette']
labels = gmm_cluster_labels5

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(gnn_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(gnn_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('군서면 gmm 분석 결과')

plt.rc('font', size=20)

plt.show()

gnn_silhouette_sample = merged_df_군북면_target6['gmm_silhouette']
labels = gmm_cluster_labels6

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(gnn_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(gnn_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('군북면 gmm 분석 결과')

plt.rc('font', size=20)

plt.show()

gnn_silhouette_sample = merged_df_동이면_target7['gmm_silhouette']
labels = gmm_cluster_labels7

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(gnn_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(gnn_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('동이면 gmm 분석 결과')

plt.rc('font', size=20)

plt.show()

gnn_silhouette_sample = merged_df_청산면_target8['gmm_silhouette']
labels = gmm_cluster_labels8

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(gnn_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(gnn_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('청산면 gmm 분석 결과')

plt.rc('font', size=20)

plt.show()

gnn_silhouette_sample = merged_df_청성면_target9['gmm_silhouette']
labels = gmm_cluster_labels9

# 각 data의 실루엣 계수를 막대 그래프로 시각화합
y_ticks = []
y_lower, y_upper = 0, 0
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_values = np.array(gnn_silhouette_sample[labels == cluster])
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    color = plt.cm.get_cmap("Spectral")(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
             edgecolor='none', color=color)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_values)

# 각 data의 실루엣 계수의 평균값을 수직선으로 그림
silhouette_avg = np.mean(gnn_silhouette_sample)
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

# 축과 레이블 설정
plt.yticks(y_ticks, np.unique(labels))
plt.rc('font', family='NanumGothic')
plt.ylabel('클러스터')
plt.xlabel('실루엣 계수')
plt.title('청성면 gmm 분석 결과')

plt.rc('font', size=20)

plt.show()