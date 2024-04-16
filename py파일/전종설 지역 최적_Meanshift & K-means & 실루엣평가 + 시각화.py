#!/usr/bin/env python
# coding: utf-8

# In[14]:


get_ipython().system('pip install folium')


# # 경로당 + 음식점 좌표 합침

# In[15]:


import pandas as pd
import folium
import random
pd10 = pd.read_csv("충청북도 옥천군_경로당현황_좌표추가.csv")
pd20 = pd.read_csv("충청북도 옥천군_음식점현황_좌표추가.csv")


# In[16]:


pd10 = pd10.rename(columns={"경로당명": "경로당명/음식점명"})
pd10 = pd10.rename(columns={"경로당 주소": "도로명 주소"})
pd10


# In[17]:


pd20 = pd20.drop("군분", axis=1)
pd20 = pd20.drop("소재지(지번)",axis=1)
pd20 = pd20.drop("소재지전화", axis=1)
pd20 = pd20.rename(columns={"업소명": "경로당명/음식점명"})
pd20 = pd20.rename(columns={"소재지(도로명)": "도로명 주소"})
pd20


# In[18]:


def merge_dataframes(pd10, pd20):
    # 두 데이터프레임을 합침
    merged_df = pd.concat([pd10, pd20], ignore_index=True)
    return merged_df

merged_df = merge_dataframes(pd10, pd20)
merged_df


# In[19]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "옥천읍"이 포함된 행을 뽑음
merged_df_옥천읍 = merged_df[merged_df["도로명 주소"].str.contains("옥천읍")]

# 필터링된 데이터프레임 출력
print(merged_df_옥천읍)

merged_df_옥천읍.shape[0]


# In[20]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "이원면"이 포함된 행을 뽑음
merged_df_이원면 = merged_df[merged_df["도로명 주소"].str.contains("이원면")]

# 필터링된 데이터프레임 출력
print(merged_df_이원면)

merged_df_이원면.shape[0]


# In[21]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "안내면"이 포함된 행을 뽑음
merged_df_안내면 = merged_df[merged_df["도로명 주소"].str.contains("안내면")]

# 필터링된 데이터프레임 출력
print(merged_df_안내면)

merged_df_안내면.shape[0]


# In[22]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "안남면"이 포함된 행을 뽑음
merged_df_안남면 = merged_df[merged_df["도로명 주소"].str.contains("안남면")]

# 필터링된 데이터프레임 출력
print(merged_df_안남면)

merged_df_안남면.shape[0]


# In[23]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군서면"이 포함된 행을 뽑음
merged_df_군서면 = merged_df[merged_df["도로명 주소"].str.contains("군서면")]

# 필터링된 데이터프레임 출력
print(merged_df_군서면)

merged_df_군서면.shape[0]


# In[24]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군북면"이 포함된 행을 뽑음
merged_df_군북면 = merged_df[merged_df["도로명 주소"].str.contains("군북면")]

# 필터링된 데이터프레임 출력
print(merged_df_군북면)

merged_df_군북면.shape[0]


# In[25]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "동이면"이 포함된 행을 뽑음
merged_df_동이면 = merged_df[merged_df["도로명 주소"].str.contains("동이면")]

# 필터링된 데이터프레임 출력
print(merged_df_동이면)

merged_df_동이면.shape[0]


# In[26]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "청산면"이 포함된 행을 뽑음
merged_df_청산면 = merged_df[merged_df["도로명 주소"].str.contains("청산면")]

# 필터링된 데이터프레임 출력
print(merged_df_청산면)

merged_df_청산면.shape[0]


# In[27]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "청성면"이 포함된 행을 뽑음
merged_df_청성면 = merged_df[merged_df["도로명 주소"].str.contains("청성면")]

# 필터링된 데이터프레임 출력
print(merged_df_청성면)

merged_df_청성면.shape[0]


# ### 옥천읍 난수발생 + 경로당&음식점

# In[36]:


import random
import pandas as pd

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

merged_df_옥천읍_1 = merged_df_옥천읍[['Latitude','Longitude']]
merged_df_옥천읍_1.dropna()

#난수발생 target1와 옥천읍에 해당하는 경로당 + 음식점의 좌표 합치기
merged_df_옥천읍_target1 = pd.concat([target1, merged_df_옥천읍_1], ignore_index=True)
merged_df_옥천읍_target1.dropna()
merged_df_옥천읍_target1


# ### 이원면 난수발생 + 경로당&음식점

# In[37]:


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

merged_df_이원면_1 = merged_df_이원면[['Latitude','Longitude']]
merged_df_이원면_1.dropna()

#난수발생 target2와 이원면에 해당하는 경로당 + 음식점의 좌표 합치기
merged_df_이원면_target2 = pd.concat([target2, merged_df_이원면_1], ignore_index=True)
merged_df_이원면_target2


# ### 안내면 난수발생 + 경로당&음식점

# In[46]:


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

merged_df_안내면_1 = merged_df_안내면[['Latitude','Longitude']]
merged_df_안내면_1.dropna()

#난수발생 target3와 안내면에 해당하는 경로당 + 음식점의 좌표 합치기
merged_df_안내면_target3 = pd.concat([target3, merged_df_안내면_1], ignore_index=True)
merged_df_안내면_target3


# ### 안남면 난수발생 + 경로당&음식점

# In[47]:


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

merged_df_안남면_1 = merged_df_안남면[['Latitude','Longitude']]
merged_df_안남면_1.dropna()

#난수발생 target4와 안남면에 해당하는 경로당 + 음식점의 좌표 합치기
merged_df_안남면_target4 = pd.concat([target4, merged_df_안남면_1], ignore_index=True)
merged_df_안남면_target4


# ### 군서면 난수발생 + 경로당&음식점

# In[48]:


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

merged_df_군서면_1 = merged_df_군서면[['Latitude','Longitude']]
merged_df_군서면_1.dropna()

#난수발생 target5와 군서면에 해당하는 경로당 + 음식점의 좌표 합치기
merged_df_군서면_target5 = pd.concat([target5, merged_df_군서면_1], ignore_index=True)
merged_df_군서면_target5


# ### 군북면 난수발생 + 경로당&음식점

# In[49]:


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


merged_df_군북면_1 = merged_df_군북면[['Latitude','Longitude']]
merged_df_군북면_1.dropna()

#난수발생 target6와 군북면에 해당하는 경로당 + 음식점의 좌표 합치기
merged_df_군북면_target6 = pd.concat([target6, merged_df_군북면_1], ignore_index=True)
merged_df_군북면_target6


# ### 동이면 난수발생 + 경로당&음식점

# In[50]:


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


merged_df_동이면_1 = merged_df_동이면[['Latitude','Longitude']]
merged_df_동이면_1.dropna()

#난수발생 target7와 동이면에 해당하는 경로당 + 음식점의 좌표 합치기
merged_df_동이면_target7 = pd.concat([target7, merged_df_동이면_1], ignore_index=True)
merged_df_동이면_target7


# ### 청산면 난수발생 + 경로당&음식점

# In[51]:


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


merged_df_청산면_1 = merged_df_청산면[['Latitude','Longitude']]
merged_df_청산면_1.dropna()

#난수발생 target8와 청산면에 해당하는 경로당 + 음식점의 좌표 합치기
merged_df_청산면_target8 = pd.concat([target8, merged_df_청산면_1], ignore_index=True)
merged_df_청산면_target8


# ### 청성면 난수발생 + 경로당&음식점

# In[52]:


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


merged_df_청성면_1 = merged_df_청성면[['Latitude','Longitude']]
merged_df_청성면_1.dropna()

#난수발생 target9와 청성면에 해당하는 경로당 + 음식점의 좌표 합치기
merged_df_청성면_target9 = pd.concat([target9, merged_df_청성면], ignore_index=True)
merged_df_청성면_target9


# ## Mean-shift 옥천읍 모델링 & 라벨링

# In[109]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_1 = pd.to_numeric(merged_df_옥천읍_target1["Latitude"], errors='coerce')
longitude_values_1 = pd.to_numeric(merged_df_옥천읍_target1["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_1 = latitude_values_1.dropna()
valid_longitude_values_1 = longitude_values_1.dropna()

X = np.column_stack((valid_latitude_values_1, valid_longitude_values_1))

# bandwidth 추정
# quantile = 0.2에서 0.1로 줄일 시 추정된 클러스터의 개수가 2개 -> 7개로 증가함.
# quantile 값을 증가시키면 대역폭이 감소하고 클러스터의 개수가 증가할 수 있다. 
# 반대로 quantile 값을 감소시키면 대역폭이 증가하여 클러스터의 크기가 커지고 개수가 감소할 수 있다.
bandwidth_1 = estimate_bandwidth(X, quantile=0.1, n_samples=500)

# Mean Shift 클러스터링 모델 생성
ms_1 = MeanShift(bandwidth=bandwidth_1, bin_seeding=True)

# 모델 피팅
ms_1.fit(X)

# 클러스터 중심 추출
cluster_centers_1 = ms_1.cluster_centers_

# 클러스터 개수와 중심 출력
labels_1 = ms_1.labels_
n_clusters_1 = len(np.unique(labels_1))
print("추정된 클러스터 개수:", n_clusters_1)
print("클러스터 중심:")
print(cluster_centers_1)

# 클러스터링 결과(라벨)를 데이터프레임에 추가
merged_df_옥천읍_target1['Mean-shift'] = labels_1

# 결과 확인
print(merged_df_옥천읍_target1.head())


# ## Mean-shift 이원면 모델링 & 라벨링

# In[110]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_이원면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_2 = pd.to_numeric(merged_df_이원면_target2["Latitude"], errors='coerce')
longitude_values_2 = pd.to_numeric(merged_df_이원면_target2["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_2 = latitude_values_2.dropna()
valid_longitude_values_2 = longitude_values_2.dropna()

X2 = np.column_stack((valid_latitude_values_2, valid_longitude_values_2))

# bandwidth 추정
# quantile = 0.2에서 0.1로 줄일 시 추정된 클러스터의 개수가 2개 -> 7개로 증가함.
# quantile 값을 증가시키면 대역폭이 감소하고 클러스터의 개수가 증가할 수 있다. 
# 반대로 quantile 값을 감소시키면 대역폭이 증가하여 클러스터의 크기가 커지고 개수가 감소할 수 있다.
bandwidth_2 = estimate_bandwidth(X2, quantile=0.1, n_samples=500)

# Mean Shift 클러스터링 모델 생성
ms_2 = MeanShift(bandwidth=bandwidth_2, bin_seeding=True)

# 모델 피팅
ms_2.fit(X2)

# 클러스터 중심 추출
cluster_centers_2 = ms_2.cluster_centers_

# 클러스터 개수와 중심 출력
labels_2 = ms_2.labels_
n_clusters_2 = len(np.unique(labels_2))
print("추정된 클러스터 개수:", n_clusters_2)
print("클러스터 중심:")
print(cluster_centers_2)

# 클러스터링 결과(라벨)를 데이터프레임에 추가
merged_df_이원면_target2['Mean-shift'] = labels_2

# 결과 확인
print(merged_df_이원면_target2.head())


# ## Mean-shift 안내면 모델링 & 라벨링

# In[111]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_안내면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_3 = pd.to_numeric(merged_df_안내면_target3["Latitude"], errors='coerce')
longitude_values_3 = pd.to_numeric(merged_df_안내면_target3["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_3 = latitude_values_3.dropna()
valid_longitude_values_3 = longitude_values_3.dropna()

X3 = np.column_stack((valid_latitude_values_3, valid_longitude_values_3))

# bandwidth 추정
# quantile = 0.2에서 0.1로 줄일 시 추정된 클러스터의 개수가 3개 -> 8개로 증가함.
# quantile 값을 증가시키면 대역폭이 감소하고 클러스터의 개수가 증가할 수 있다. 
# 반대로 quantile 값을 감소시키면 대역폭이 증가하여 클러스터의 크기가 커지고 개수가 감소할 수 있다.
bandwidth_3 = estimate_bandwidth(X3, quantile=0.1, n_samples=500)

# Mean Shift 클러스터링 모델 생성
ms_3 = MeanShift(bandwidth=bandwidth_3, bin_seeding=True)

# 모델 피팅
ms_3.fit(X3)

# 클러스터 중심 추출
cluster_centers_3 = ms_3.cluster_centers_

# 클러스터 개수와 중심 출력
labels_3 = ms_3.labels_
n_clusters_3 = len(np.unique(labels_3))
print("추정된 클러스터 개수:", n_clusters_3)
print("클러스터 중심:")
print(cluster_centers_3)

# 클러스터링 결과(라벨)를 데이터프레임에 추가
merged_df_안내면_target3['Mean-shift'] = labels_3

# 결과 확인
print(merged_df_안내면_target3.head())


# ## Mean-shift 안남면 모델링 & 라벨링

# In[112]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_안남면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_4 = pd.to_numeric(merged_df_안남면_target4["Latitude"], errors='coerce')
longitude_values_4 = pd.to_numeric(merged_df_안남면_target4["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_4 = latitude_values_4.dropna()
valid_longitude_values_4 = longitude_values_4.dropna()

X4 = np.column_stack((valid_latitude_values_4, valid_longitude_values_4))

# bandwidth 추정
# quantile = 0.2에서 0.1로 줄일 시 추정된 클러스터의 개수가 7개 -> 12개로 증가함.
# quantile 값을 증가시키면 대역폭이 감소하고 클러스터의 개수가 증가할 수 있다. 
# 반대로 quantile 값을 감소시키면 대역폭이 증가하여 클러스터의 크기가 커지고 개수가 감소할 수 있다.
bandwidth_4 = estimate_bandwidth(X4, quantile=0.1, n_samples=500)

# Mean Shift 클러스터링 모델 생성
ms_4 = MeanShift(bandwidth=bandwidth_4, bin_seeding=True)

# 모델 피팅
ms_4.fit(X)

# 클러스터 중심 추출
cluster_centers_4 = ms_4.cluster_centers_

# 클러스터 개수와 중심 출력
labels_4 = ms_4.labels_
n_clusters_4 = len(np.unique(labels_4))
print("추정된 클러스터 개수:", n_clusters_4)
print("클러스터 중심:")
print(cluster_centers_4)

# 클러스터링 결과(라벨)를 데이터프레임에 추가
if len(valid_latitude_values_4) == len(labels_4):
    merged_df_안남면_target4['Mean-shift'] = labels_4
    print(merged_df_안남면_target4.head())
else:
    # 데이터프레임의 인덱스를 재설정하여 길이를 일치시킴
    merged_df_안남면_target4.reset_index(drop=True, inplace=True)
    merged_df_안남면_target4['Mean-shift'] = labels_4[:len(valid_latitude_values_4)]  # 일치하는 길이만큼만 사용
    print(merged_df_안남면_target4.head())

# 클러스터링 결과(라벨)를 데이터프레임에 추가
# merged_df_안남면_target4['Mean-shift'] = labels_4

# 결과 확인
# print(merged_df_안남면_target4.head())


# ## Mean-shift 군서면 모델링 & 라벨링

# In[113]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_군서면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_5 = pd.to_numeric(merged_df_군서면_target5["Latitude"], errors='coerce')
longitude_values_5 = pd.to_numeric(merged_df_군서면_target5["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_5 = latitude_values_5.dropna()
valid_longitude_values_5 = longitude_values_5.dropna()

X5 = np.column_stack((valid_latitude_values_5, valid_longitude_values_5))

# bandwidth 추정
# quantile = 0.2에서 0.1로 줄일 시 추정된 클러스터의 개수가 3개 -> 8개로 증가함.
# quantile 값을 증가시키면 대역폭이 감소하고 클러스터의 개수가 증가할 수 있다. 
# 반대로 quantile 값을 감소시키면 대역폭이 증가하여 클러스터의 크기가 커지고 개수가 감소할 수 있다.
bandwidth_5 = estimate_bandwidth(X5, quantile=0.1, n_samples=500)

# Mean Shift 클러스터링 모델 생성
ms_5 = MeanShift(bandwidth=bandwidth_5, bin_seeding=True)

# 모델 피팅
ms_5.fit(X)

# 클러스터 중심 추출
cluster_centers_5 = ms_5.cluster_centers_

# 클러스터 개수와 중심 출력
labels_5 = ms_5.labels_
n_clusters_5 = len(np.unique(labels_5))
print("추정된 클러스터 개수:", n_clusters_5)
print("클러스터 중심:")
print(cluster_centers_5)

# 클러스터링 결과(라벨)를 데이터프레임에 추가
if len(valid_latitude_values_5) == len(labels_5):
    merged_df_군서면_target5['Mean-shift'] = labels_5
    print(merged_df_군서면_target5.head())
else:
    # 데이터프레임의 인덱스를 재설정하여 길이를 일치시킴
    merged_df_군서면_target5.reset_index(drop=True, inplace=True)
    merged_df_군서면_target5['Mean-shift'] = labels_5[:len(valid_latitude_values_5)]  # 일치하는 길이만큼만 사용
    print(merged_df_군서면_target5.head())


# ## Mean-shift 군북면 모델링 & 라벨링

# In[114]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_군북면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_6 = pd.to_numeric(merged_df_군북면_target6["Latitude"], errors='coerce')
longitude_values_6 = pd.to_numeric(merged_df_군북면_target6["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_6 = latitude_values_6.dropna()
valid_longitude_values_6 = longitude_values_6.dropna()

X6 = np.column_stack((valid_latitude_values_6, valid_longitude_values_6))

# bandwidth 추정
# quantile = 0.2에서 0.1로 줄일 시 추정된 클러스터의 개수가 1개 -> 5개로 증가함.
# quantile 값을 증가시키면 대역폭이 감소하고 클러스터의 개수가 증가할 수 있다. 
# 반대로 quantile 값을 감소시키면 대역폭이 증가하여 클러스터의 크기가 커지고 개수가 감소할 수 있다.
bandwidth_6 = estimate_bandwidth(X6, quantile=0.1, n_samples=500)

# Mean Shift 클러스터링 모델 생성
ms_6 = MeanShift(bandwidth=bandwidth_6, bin_seeding=True)

# 모델 피팅
ms_6.fit(X)

# 클러스터 중심 추출
cluster_centers_6 = ms_6.cluster_centers_

# 클러스터 개수와 중심 출력
labels_6 = ms_6.labels_
n_clusters_6 = len(np.unique(labels_6))
print("추정된 클러스터 개수:", n_clusters_6)
print("클러스터 중심:")
print(cluster_centers_6)

# 클러스터링 결과(라벨)를 데이터프레임에 추가
if len(valid_latitude_values_6) == len(labels_6):
    merged_df_군북면_target6['Mean-shift'] = labels_6
    print(merged_df_군북면_target6.head())
else:
    # 데이터프레임의 인덱스를 재설정하여 길이를 일치시킴
    merged_df_군북면_target6.reset_index(drop=True, inplace=True)
    merged_df_군북면_target6['Mean-shift'] = labels_6[:len(valid_latitude_values_6)]  # 일치하는 길이만큼만 사용
    print(merged_df_군북면_target6.head())


# ## Mean-shift 동이면 모델링 & 라벨링

# In[115]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_동이면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_7 = pd.to_numeric(merged_df_동이면_target7["Latitude"], errors='coerce')
longitude_values_7 = pd.to_numeric(merged_df_동이면_target7["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_7 = latitude_values_7.dropna()
valid_longitude_values_7 = longitude_values_7.dropna()

X7 = np.column_stack((valid_latitude_values_7, valid_longitude_values_7))

# bandwidth 추정
# quantile = 0.2에서 0.1로 줄일 시 추정된 클러스터의 개수가 2개 -> 7개로 증가함.
# quantile 값을 증가시키면 대역폭이 감소하고 클러스터의 개수가 증가할 수 있다. 
# 반대로 quantile 값을 감소시키면 대역폭이 증가하여 클러스터의 크기가 커지고 개수가 감소할 수 있다.
bandwidth_7 = estimate_bandwidth(X7, quantile=0.1, n_samples=500)

# Mean Shift 클러스터링 모델 생성
ms_7 = MeanShift(bandwidth=bandwidth_7, bin_seeding=True)

# 모델 피팅
ms_7.fit(X)

# 클러스터 중심 추출
cluster_centers_7 = ms_7.cluster_centers_

# 클러스터 개수와 중심 출력
labels_7 = ms_7.labels_
n_clusters_7 = len(np.unique(labels_7))
print("추정된 클러스터 개수:", n_clusters_7)
print("클러스터 중심:")
print(cluster_centers_7)

# 클러스터링 결과(라벨)를 데이터프레임에 추가
if len(valid_latitude_values_7) == len(labels_7):
    merged_df_동이면_target7['Mean-shift'] = labels_7
    print(merged_df_동이면_target7.head())
else:
    # 데이터프레임의 인덱스를 재설정하여 길이를 일치시킴
    merged_df_동이면_target7.reset_index(drop=True, inplace=True)
    merged_df_동이면_target7['Mean-shift'] = labels_7[:len(valid_latitude_values_7)]  # 일치하는 길이만큼만 사용
    print(merged_df_동이면_target7.head())


# ## Mean-shift 청산면 모델링 & 라벨링

# In[116]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_청산면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_8 = pd.to_numeric(merged_df_청산면_target8["Latitude"], errors='coerce')
longitude_values_8 = pd.to_numeric(merged_df_청산면_target8["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_8 = latitude_values_8.dropna()
valid_longitude_values_8 = longitude_values_8.dropna()

X8 = np.column_stack((valid_latitude_values_8, valid_longitude_values_8))

# bandwidth 추정
# quantile = 0.2에서 0.1로 줄일 시 추정된 클러스터의 개수가 3개 -> 6개로 증가함.
# quantile 값을 증가시키면 대역폭이 감소하고 클러스터의 개수가 증가할 수 있다. 
# 반대로 quantile 값을 감소시키면 대역폭이 증가하여 클러스터의 크기가 커지고 개수가 감소할 수 있다.
bandwidth_8 = estimate_bandwidth(X8, quantile=0.1, n_samples=500)

# Mean Shift 클러스터링 모델 생성
ms_8 = MeanShift(bandwidth=bandwidth_8, bin_seeding=True)

# 모델 피팅
ms_8.fit(X8)

# 클러스터 중심 추출
cluster_centers_8 = ms_8.cluster_centers_

# 클러스터 개수와 중심 출력
labels_8 = ms_8.labels_
n_clusters_8 = len(np.unique(labels_8))
print("추정된 클러스터 개수:", n_clusters_8)
print("클러스터 중심:")
print(cluster_centers_8)

# 클러스터링 결과(라벨)를 데이터프레임에 추가
if len(valid_latitude_values_8) == len(labels_8):
    merged_df_청산면_target8['Mean-shift'] = labels_8
    print(merged_df_청산면_target8.head())
else:
    # 데이터프레임의 인덱스를 재설정하여 길이를 일치시킴
    merged_df_청산면_target8.reset_index(drop=True, inplace=True)
    merged_df_청산면_target8['Mean-shift'] = labels_8[:len(valid_latitude_values_8)]  # 일치하는 길이만큼만 사용
    print(merged_df_청산면_target8.head())


# ## Mean-shift 청성면 모델링 & 라벨링

# In[119]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_청성면 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_9 = pd.to_numeric(merged_df_청성면_target9["Latitude"], errors='coerce')
longitude_values_9 = pd.to_numeric(merged_df_청성면_target9["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_9 = latitude_values_9.dropna()
valid_longitude_values_9 = longitude_values_9.dropna()

X9 = np.column_stack((valid_latitude_values_9, valid_longitude_values_9))

# bandwidth 추정
# quantile = 0.2에서 0.1로 줄일 시 추정된 클러스터의 개수가 2개 -> 8개로 증가함.
# quantile 값을 증가시키면 대역폭이 감소하고 클러스터의 개수가 증가할 수 있다. 
# 반대로 quantile 값을 감소시키면 대역폭이 증가하여 클러스터의 크기가 커지고 개수가 감소할 수 있다.
bandwidth_9 = estimate_bandwidth(X9, quantile=0.1, n_samples=500)

# Mean Shift 클러스터링 모델 생성
ms_9 = MeanShift(bandwidth=bandwidth_9, bin_seeding=True)

# 모델 피팅
ms_9.fit(X9)

# 클러스터 중심 추출
cluster_centers_9 = ms_9.cluster_centers_

# 클러스터 개수와 중심 출력
labels_9 = ms_9.labels_
n_clusters_9 = len(np.unique(labels_9))
print("추정된 클러스터 개수:", n_clusters_9)
print("클러스터 중심:")
print(cluster_centers_9)


# 클러스터링 결과(라벨)를 데이터프레임에 추가
if len(valid_latitude_values_9) == len(labels_9):
    merged_df_청성면_target9['Mean-shift'] = labels_9
    print(merged_df_청성면_target9.head())
else:
    # 데이터프레임의 인덱스를 재설정하여 길이를 일치시킴
    merged_df_청성면_target9.reset_index(drop=True, inplace=True)
    merged_df_청성면_target9['Mean-shift'] = labels_9[:len(valid_latitude_values_9)]  # 일치하는 길이만큼만 사용
    print(merged_df_청성면_target9.head())


# # 실루엣 계수 평가 - 옥천읍

# In[120]:


from sklearn.metrics import silhouette_score

# 실루엣 계수 평가
silhouette_avg_1 = silhouette_score(X, labels_1)
print("Mean Shift 클러스터링의 실루엣 계수:", silhouette_avg_1)


# In[128]:


from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
# 실루엣 분석 metric 값을 구하기 위한 API 추가
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[129]:


# iris 의 모든 개별 데이터에 실루엣 계수값을 구함
score_samples1 = silhouette_samples(X, merged_df_옥천읍_target1['Mean-shift'])
print('silhouette_samples( ) return 값의 shape' , score_samples1.shape)

# irisDF에 실루엣 계수 컬럼 추가
merged_df_옥천읍_target1['silhouette_coeff'] = score_samples1


# In[130]:


print(np.mean(score_samples1))
print(silhouette_score(X, merged_df_옥천읍_target1['Mean-shift']))


# # 실루엣 계수 평가 - 이원면

# In[121]:


from sklearn.metrics import silhouette_score

# 실루엣 계수 평가
silhouette_avg_2 = silhouette_score(X2, labels_2)
print("Mean Shift 클러스터링의 실루엣 계수:", silhouette_avg_2)


# # 실루엣 계수 평가 - 안내면

# In[122]:


from sklearn.metrics import silhouette_score

# 실루엣 계수 평가
silhouette_avg_3 = silhouette_score(X3, labels_3)
print("Mean Shift 클러스터링의 실루엣 계수:", silhouette_avg_3)


# # 실루엣 계수 평가 - 안남면

# In[124]:


from sklearn.metrics import silhouette_score

# labels_4와 X4의 샘플 수 일치시키기
labels_4 = labels_4[:X4.shape[0]]

# 실루엣 계수 평가
silhouette_avg_4 = silhouette_score(X4, labels_4)
print("Mean Shift 클러스터링의 실루엣 계수:", silhouette_avg_4)


# # 실루엣 계수 평가 - 군서면

# In[126]:


from sklearn.metrics import silhouette_score

# labels_5와 X5의 샘플 수 일치시키기
labels_5 = labels_5[:X5.shape[0]]

# 실루엣 계수 평가
silhouette_avg_5 = silhouette_score(X5, labels_5)
print("Mean Shift 클러스터링의 실루엣 계수:", silhouette_avg_5)


# # 실루엣 계수 평가 - 군북면

# In[133]:


from sklearn.metrics import silhouette_score

# labels_6와 X6의 샘플 수 일치시키기
n_samples = min(X6.shape[0], len(labels_6))
X6_subset = X6[:n_samples]
labels_6_subset = labels_6[:n_samples]

# 실루엣 계수 평가
silhouette_avg_6 = silhouette_score(X6_subset, labels_6_subset)
print("Mean Shift 클러스터링의 실루엣 계수:", silhouette_avg_6)


# # 실루엣 계수 평가 - 동이면

# In[137]:


from sklearn.metrics import silhouette_score

# labels_7와 X7의 샘플 수 일치시키기
labels_7 = labels_7[:X7.shape[0]]

# 실루엣 계수 평가
silhouette_avg_7 = silhouette_score(X7, labels_7)
print("Mean Shift 클러스터링의 실루엣 계수:", silhouette_avg_7)


# # 실루엣 계수 평가 - 청산면
# 

# In[138]:


from sklearn.metrics import silhouette_score

# labels_8와 X8의 샘플 수 일치시키기
labels_8 = labels_8[:X8.shape[0]]

# 실루엣 계수 평가
silhouette_avg_8 = silhouette_score(X8, labels_8)
print("Mean Shift 클러스터링의 실루엣 계수:", silhouette_avg_8)


# # 실루엣 계수 평가 - 청성면

# In[139]:


from sklearn.metrics import silhouette_score

# labels_9와 X9의 샘플 수 일치시키기
labels_9 = labels_9[:X9.shape[0]]

# 실루엣 계수 평가
silhouette_avg_9 = silhouette_score(X9, labels_9)
print("Mean Shift 클러스터링의 실루엣 계수:", silhouette_avg_9)


# # K-means

# ## K-Means 클러스터링 - 옥천읍

# In[200]:


import numpy as np
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points_옥천읍 = merged_df_옥천읍.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_옥천읍 = int(np.sqrt(n_data_points_옥천읍 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_옥천읍}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_옥천읍 = KMeans(n_clusters=rule_of_thumb_k_옥천읍, init='k-means++', random_state=0)

# K-means 모델 피팅
kmeans_옥천읍.fit(X)
y_kmeans_옥천읍 = kmeans_옥천읍.predict(X)

# 초기 중심점 출력
print("초기 중심점:")
print(kmeans_옥천읍.cluster_centers_)

# 클러스터링 결과 가져오기
labels_옥천읍 = kmeans_옥천읍.labels_

#클러스터링 결과를 데이터 프레임에 추가
merged_df_옥천읍_target1['KCluster'] = kmeans_옥천읍.labels_
print(merged_df_옥천읍_target1.head())

from sklearn.metrics import silhouette_score

# 실루엣 계수 평가
silhouette_avg_옥천읍 = silhouette_score(X, labels_옥천읍)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_옥천읍)


# In[201]:


# 클러스터링 결과 시각화
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans_옥천읍,cmap='viridis',s=10)
plt.scatter(kmeans_옥천읍.cluster_centers_[:, 0], kmeans_옥천읍.cluster_centers_[:, 1], marker='*', color='red',s=100,alpha=0.5, label='Centroids')

plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.title('Warehouse k-means clustering result')
plt.legend()
plt.show()


# ## K-Means 클러스터링 - 이원면

# In[183]:


import numpy as np
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points_이원면 = merged_df_이원면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_이원면 = int(np.sqrt(n_data_points_이원면 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_이원면}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_이원면 = KMeans(n_clusters=rule_of_thumb_k_이원면, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
# 예를 들어, latitude_values와 longitude_values는 숫자로 변환 가능한 경우에 선택
# latitude_values = pd.to_numeric(merged_df_옥천읍["Latitude"], errors='coerce')
# longitude_values = pd.to_numeric(merged_df_옥천읍["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
# valid_latitude_values = latitude_values.dropna()
# valid_longitude_values = longitude_values.dropna()

# X = np.column_stack((valid_latitude_values, valid_longitude_values))

# 예시로 사용할 데이터 X 생성 (실제 데이터셋에 맞게 적절히 조정 필요)
# X_옥천읍 = np.random.rand(n_data_points_옥천읍, 2)  # 랜덤 데이터 생성 예시

# K-means 모델 피팅
kmeans_이원면.fit(X2)
y_kmeans_이원면 = kmeans_이원면.predict(X2)

# 초기 중심점 출력
print("초기 중심점:")
print(kmeans_이원면.cluster_centers_)

# 클러스터링 결과 가져오기
labels_이원면 = kmeans_이원면.labels_

#클러스터링 결과를 데이터 프레임에 추가
merged_df_이원면_target2['KCluster'] = kmeans_이원면.labels_
print(merged_df_이원면_target2.head())

from sklearn.metrics import silhouette_score

# 실루엣 계수 평가
silhouette_avg_이원면 = silhouette_score(X2, labels_이원면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_이원면)


# In[202]:


import matplotlib.pyplot as plt

# 클러스터링 결과 시각화
plt.figure(figsize=(8, 8))
plt.scatter(X2[:, 0], X2[:, 1], c=y_kmeans_이원면, cmap='viridis', s=10)
plt.scatter(kmeans_이원면.cluster_centers_[:, 0], kmeans_이원면.cluster_centers_[:, 1], marker='*', color='red', s=100, alpha=0.5, label='Centroids')

plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.title('Warehouse k-means clustering result')
plt.legend()
plt.show()


# ## K-Means 클러스터링 - 안내면

# In[184]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points_안내면 = merged_df_안내면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_안내면 = int(np.sqrt(n_data_points_안내면 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_안내면}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_안내면 = KMeans(n_clusters=rule_of_thumb_k_안내면, init='k-means++', random_state=0)

# K-means 모델 피팅
kmeans_안내면.fit(X3)
y_kmeans_안내면 = kmeans_안내면.predict(X3)

# 초기 중심점 출력
print("초기 중심점:")
print(kmeans_안내면.cluster_centers_)

# 클러스터링 결과 가져오기
labels_안내면 = kmeans_안내면.labels_

# 클러스터링 결과를 데이터 프레임에 추가
merged_df_안내면['KCluster'] = kmeans_안내면.labels_
print(merged_df_안내면.head())

# 실루엣 계수 평가
silhouette_avg_안내면 = silhouette_score(X3, labels_안내면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_안내면)


# In[203]:


import matplotlib.pyplot as plt

# 클러스터링 결과 시각화
plt.figure(figsize=(8, 8))
plt.scatter(X3[:, 0], X3[:, 1], c=y_kmeans_안내면, cmap='viridis', s=10)
plt.scatter(kmeans_안내면.cluster_centers_[:, 0], kmeans_안내면.cluster_centers_[:, 1], marker='*', color='red', s=100, alpha=0.5, label='Centroids')

plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.title('Warehouse k-means clustering result')
plt.legend()
plt.show()


# ## K-means 클러스터링 - 안남면

# In[187]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points_안남면 = merged_df_안남면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_안남면 = int(np.sqrt(n_data_points_안남면 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_안남면}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_안남면 = KMeans(n_clusters=rule_of_thumb_k_안남면, init='k-means++', random_state=0)

# K-means 모델 피팅
kmeans_안남면.fit(X4)
y_kmeans_안남면 = kmeans_안남면.predict(X4)

# 초기 중심점 출력
print("초기 중심점:")
print(kmeans_안남면.cluster_centers_)

# 클러스터링 결과 가져오기
labels_안남면 = kmeans_안남면.labels_

# 클러스터링 결과를 데이터 프레임에 추가
merged_df_안남면['KCluster'] = labels_안남면
print(merged_df_안남면.head())

# 실루엣 계수 평가
silhouette_avg_안남면 = silhouette_score(X4, labels_안남면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_안남면)


# In[204]:


import matplotlib.pyplot as plt

# 클러스터링 결과 시각화
plt.figure(figsize=(8, 8))
plt.scatter(X4[:, 0], X4[:, 1], c=y_kmeans_안남면, cmap='viridis', s=10)
plt.scatter(kmeans_안남면.cluster_centers_[:, 0], kmeans_안남면.cluster_centers_[:, 1], marker='*', color='red', s=100, alpha=0.5, label='Centroids')

plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.title('Warehouse k-means clustering result')
plt.legend()
plt.show()


# ## K-means 클러스터링 - 군서면

# In[214]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points_군서면 = merged_df_군서면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_군서면 = int(np.sqrt(n_data_points_군서면 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_군서면}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_군서면 = KMeans(n_clusters=rule_of_thumb_k_군서면, init='k-means++', random_state=0)

# K-means 모델 피팅
kmeans_군서면.fit(X5)
y_kmeans_군서면 = kmeans_군서면.predict(X5)

# 클러스터링 결과 가져오기
labels_군서면 = kmeans_군서면.labels_

# 실루엣 계수 평가
silhouette_avg_군서면 = silhouette_score(X5, labels_군서면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_군서면)

# 데이터프레임에 클러스터링 결과 추가
# labels_군서면의 길이를 데이터프레임의 인덱스에 맞게 조정
labels_군서면 = labels_군서면[:n_data_points_군서면]

# 데이터프레임에 클러스터링 결과 열 추가
merged_df_군서면['KCluster'] = labels_군서면

# 결과 출력
print(merged_df_군서면.head())


# In[215]:


import matplotlib.pyplot as plt

# 클러스터링 결과 시각화
plt.figure(figsize=(8, 8))
plt.scatter(X5[:, 0], X5[:, 1], c=y_kmeans_군서면, cmap='viridis', s=10)
plt.scatter(kmeans_군서면.cluster_centers_[:, 0], kmeans_군서면.cluster_centers_[:, 1], marker='*', color='red', s=100, alpha=0.5, label='Centroids')

plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.title('Warehouse k-means clustering result')
plt.legend()
plt.show()


# ## K-means 클러스터링 - 군북면

# In[216]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points_군북면 = merged_df_군북면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_군북면 = int(np.sqrt(n_data_points_군북면 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_군북면}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_군북면 = KMeans(n_clusters=rule_of_thumb_k_군북면, init='k-means++', random_state=0)

# K-means 모델 피팅
kmeans_군북면.fit(X6)
y_kmeans_군북면 = kmeans_군북면.predict(X6)

# 클러스터링 결과 가져오기
labels_군북면 = kmeans_군북면.labels_

# 실루엣 계수 평가
silhouette_avg_군북면 = silhouette_score(X6, labels_군북면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_군북면)

# 데이터프레임에 클러스터링 결과 추가
# labels_군북면의 길이를 데이터프레임의 인덱스에 맞게 조정
labels_군북면 = labels_군북면[:n_data_points_군북면]

# 데이터프레임에 클러스터링 결과 열 추가
merged_df_군북면['KCluster'] = labels_군북면

# 결과 출력
print(merged_df_군북면.head())


# In[217]:


import matplotlib.pyplot as plt

# 클러스터링 결과 시각화
plt.figure(figsize=(8, 8))
plt.scatter(X6[:, 0], X6[:, 1], c=y_kmeans_군북면, cmap='viridis', s=10)
plt.scatter(kmeans_군북면.cluster_centers_[:, 0], kmeans_군북면.cluster_centers_[:, 1], marker='*', color='red', s=100, alpha=0.5, label='Centroids')

plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.title('Warehouse k-means clustering result')
plt.legend()
plt.show()


# ## K-means 클러스터링 - 동이면

# In[218]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points_동이면 = merged_df_동이면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_동이면 = int(np.sqrt(n_data_points_동이면 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_동이면}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_동이면 = KMeans(n_clusters=rule_of_thumb_k_동이면, init='k-means++', random_state=0)

# K-means 모델 피팅
kmeans_동이면.fit(X7)
y_kmeans_동이면 = kmeans_동이면.predict(X7)

# 클러스터링 결과 가져오기
labels_동이면 = kmeans_동이면.labels_

# 실루엣 계수 평가
silhouette_avg_동이면 = silhouette_score(X7, labels_동이면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_동이면)

# 데이터프레임에 클러스터링 결과 추가
# labels_동이면의 길이를 데이터프레임의 인덱스에 맞게 조정
labels_동이면 = labels_동이면[:n_data_points_동이면]

# 데이터프레임에 클러스터링 결과 열 추가
merged_df_동이면['KCluster'] = labels_동이면

# 결과 출력
print(merged_df_동이면.head())


# In[219]:


import matplotlib.pyplot as plt

# 클러스터링 결과 시각화
plt.figure(figsize=(8, 8))
plt.scatter(X7[:, 0], X7[:, 1], c=y_kmeans_동이면, cmap='viridis', s=10)
plt.scatter(kmeans_동이면.cluster_centers_[:, 0], kmeans_동이면.cluster_centers_[:, 1], marker='*', color='red', s=100, alpha=0.5, label='Centroids')

plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.title('Warehouse k-means clustering result')
plt.legend()
plt.show()


# ## K-means 클러스터링 - 청산면

# In[220]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points_청산면 = merged_df_청산면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_청산면 = int(np.sqrt(n_data_points_청산면 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_청산면}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_청산면 = KMeans(n_clusters=rule_of_thumb_k_청산면, init='k-means++', random_state=0)

# K-means 모델 피팅
kmeans_청산면.fit(X8)
y_kmeans_청산면 = kmeans_청산면.predict(X8)

# 클러스터링 결과 가져오기
labels_청산면 = kmeans_청산면.labels_

# 실루엣 계수 평가
silhouette_avg_청산면 = silhouette_score(X8, labels_청산면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_청산면)

# 데이터프레임에 클러스터링 결과 추가
# labels_청산면의 길이를 데이터프레임의 인덱스에 맞게 조정
labels_청산면 = labels_청산면[:n_data_points_청산면]

# 데이터프레임에 클러스터링 결과 열 추가
merged_df_청산면['KCluster'] = labels_청산면

# 결과 출력
print(merged_df_청산면.head())


# In[221]:


import matplotlib.pyplot as plt

# 클러스터링 결과 시각화
plt.figure(figsize=(8, 8))
plt.scatter(X8[:, 0], X8[:, 1], c=y_kmeans_청산면, cmap='viridis', s=10)
plt.scatter(kmeans_청산면.cluster_centers_[:, 0], kmeans_청산면.cluster_centers_[:, 1], marker='*', color='red', s=100, alpha=0.5, label='Centroids')

plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.title('Warehouse k-means clustering result')
plt.legend()
plt.show()


# In[222]:


## K-means 클러스터링 - 청성면


# In[224]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points_청성면 = merged_df_청성면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_청성면 = int(np.sqrt(n_data_points_청성면 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_청성면}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_청성면 = KMeans(n_clusters=rule_of_thumb_k_청성면, init='k-means++', random_state=0)

# K-means 모델 피팅
kmeans_청성면.fit(X9)
y_kmeans_청성면 = kmeans_청성면.predict(X9)

# 클러스터링 결과 가져오기
labels_청성면 = kmeans_청성면.labels_

# 실루엣 계수 평가
silhouette_avg_청성면 = silhouette_score(X9, labels_청성면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_청성면)

# 데이터프레임에 클러스터링 결과 추가
# labels_청성면의 길이를 데이터프레임의 인덱스에 맞게 조정
labels_청성면 = labels_청성면[:n_data_points_청성면]

# 데이터프레임에 클러스터링 결과 열 추가
merged_df_청성면['KCluster'] = labels_청성면

# 결과 출력
print(merged_df_청성면.head())


# In[225]:


import matplotlib.pyplot as plt

# 클러스터링 결과 시각화
plt.figure(figsize=(8, 8))
plt.scatter(X9[:, 0], X9[:, 1], c=y_kmeans_청성면, cmap='viridis', s=10)
plt.scatter(kmeans_청성면.cluster_centers_[:, 0], kmeans_청성면.cluster_centers_[:, 1], marker='*', color='red', s=100, alpha=0.5, label='Centroids')

plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.title('Warehouse k-means clustering result')
plt.legend()
plt.show()


# In[197]:


# 실루엣 계수 평가
silhouette_avg_옥천읍 = silhouette_score(X, labels_옥천읍)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_옥천읍)

# 실루엣 계수 평가
silhouette_avg_이원면 = silhouette_score(X2, labels_이원면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_이원면)

# 실루엣 계수 평가
silhouette_avg_안내면 = silhouette_score(X3, labels_안내면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_안내면)

# 실루엣 계수 평가
silhouette_avg_안남면 = silhouette_score(X4, labels_안남면)
print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_안남면)

# 실루엣 계수 평가
# silhouette_avg_군서면 = silhouette_score(X5, labels_군서면)
# print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_군서면)

# 실루엣 계수 평가
# silhouette_avg_군북면 = silhouette_score(X6, labels_군북면)
# print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_군북면)

# 실루엣 계수 평가
# silhouette_avg_동이면 = silhouette_score(X7, labels_동이면)
# print("K-means 클러스터링의 실루엣 계수:", silhouette_avg_동이면)


# In[ ]:




