#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install folium')


# In[2]:


import pandas as pd
import folium
import random
pd10 = pd.read_csv("충청북도 옥천군_경로당현황_좌표추가.csv")
pd20 = pd.read_csv("충청북도 옥천군_음식점현황_좌표추가.csv")


# In[3]:


pd10


# In[4]:


pd10 = pd10.rename(columns={"경로당명": "경로당명/음식점명"})


# In[5]:


pd10


# In[6]:


pd20


# In[7]:


pd20 = pd20.drop("군분", axis=1)
pd20 = pd20.drop("소재지(지번)",axis=1)
pd20 = pd20.drop("소재지전화", axis=1)


# In[8]:


pd10
pd10 = pd10.rename(columns={"경로당 주소": "도로명 주소"})
pd10


# In[9]:


pd20
pd20 = pd20.rename(columns={"업소명": "경로당명/음식점명"})
pd20 = pd20.rename(columns={"소재지(도로명)": "도로명 주소"})
pd20


# In[10]:


def merge_dataframes(pd10, pd20):
    # 두 데이터프레임을 합침
    merged_df = pd.concat([pd10, pd20], ignore_index=True)
    return merged_df

merged_df = merge_dataframes(pd10, pd20)
merged_df


# In[11]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군서면"이 포함된 행을 뽑음
merged_df_청성면 = merged_df[merged_df["도로명 주소"].str.contains("청성면")]

# 필터링된 데이터프레임 출력
print(merged_df_청성면)

merged_df_청성면.shape[0]


# In[12]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군서면"이 포함된 행을 뽑음
merged_df_청산면 = merged_df[merged_df["도로명 주소"].str.contains("청산면")]

# 필터링된 데이터프레임 출력
print(merged_df_청산면)

merged_df_청산면.shape[0]


# In[13]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군서면"이 포함된 행을 뽑음
merged_df_동이면 = merged_df[merged_df["도로명 주소"].str.contains("동이면")]

# 필터링된 데이터프레임 출력
print(merged_df_동이면)

merged_df_동이면.shape[0]


# In[14]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군서면"이 포함된 행을 뽑음
merged_df_군북면 = merged_df[merged_df["도로명 주소"].str.contains("군북면")]

# 필터링된 데이터프레임 출력
print(merged_df_군북면)

merged_df_군북면.shape[0]


# In[15]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군서면"이 포함된 행을 뽑음
merged_df_군서면 = merged_df[merged_df["도로명 주소"].str.contains("군서면")]

# 필터링된 데이터프레임 출력
print(merged_df_군서면)

merged_df_군서면.shape[0]


# In[16]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군서면"이 포함된 행을 뽑음
merged_df_안남면 = merged_df[merged_df["도로명 주소"].str.contains("안남면")]

# 필터링된 데이터프레임 출력
print(merged_df_안남면)

merged_df_안남면.shape[0]


# In[17]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군서면"이 포함된 행을 뽑음
merged_df_안내면 = merged_df[merged_df["도로명 주소"].str.contains("안내면")]

# 필터링된 데이터프레임 출력
print(merged_df_안내면)

merged_df_안내면.shape[0]


# In[18]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "군서면"이 포함된 행을 뽑음
merged_df_이원면 = merged_df[merged_df["도로명 주소"].str.contains("이원면")]

# 필터링된 데이터프레임 출력
print(merged_df_이원면)

merged_df_이원면.shape[0]


# In[19]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)

# "소재지(도로명)" 칼럼에서 "옥천읍"이 포함된 행을 뽑음
merged_df_옥천읍 = merged_df[merged_df["도로명 주소"].str.contains("옥천읍")]

# 필터링된 데이터프레임 출력
print(merged_df_옥천읍)

merged_df_옥천읍.shape[0]


# In[ ]:





# In[53]:


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
target1


# In[ ]:


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
target1


# In[59]:


merged_df_옥천읍_1 = merged_df_옥천읍[['Latitude','Longitude']]
merged_df_옥천읍_1


# In[61]:


merged_df_옥천읍_target1 = pd.concat([target1, merged_df_옥천읍], ignore_index=True)
pd20.dropna()
merged_df_옥천읍_target1


# In[21]:


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


# In[63]:


merged_df_이원면_target2 = pd.concat([target2, merged_df_이원면], ignore_index=True)
merged_df_이원면_target2


# In[22]:


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


# In[64]:


merged_df_안내면_target3 = pd.concat([target3, merged_df_안내면], ignore_index=True)
merged_df_안내면_target3


# In[23]:


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


# In[65]:


merged_df_안남면_target4 = pd.concat([target4, merged_df_안남면], ignore_index=True)
merged_df_안남면_target4


# In[24]:


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


# In[66]:


merged_df_군서면_target5 = pd.concat([target5, merged_df_군서면], ignore_index=True)
merged_df_군서면_target5


# In[25]:


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


# In[ ]:


merged_df_군북면_target6 = pd.concat([target6, merged_df_군북면], ignore_index=True)
merged_df_군북면_target6


# In[26]:


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


# In[ ]:


merged_df_동이면_target7 = pd.concat([target7, merged_df_동이면], ignore_index=True)
merged_df_동이면_target7


# In[27]:


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


# In[ ]:


merged_df_청산면_target8 = pd.concat([target8, merged_df_청산면], ignore_index=True)
merged_df_청산면_target8


# In[28]:


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


# In[ ]:


merged_df_청성면_target9 = pd.concat([target9, merged_df_청성면], ignore_index=True)
merged_df_청성면_target9


# In[94]:


map_center = [target9["Latitude"].mean(), target9["Longitude"].mean()]  # 중앙 좌표 설정
map = folium.Map(location=map_center, zoom_start=10)  # 중앙 좌표 기준으로 지도 생성

# 랜덤 좌표들을 지도에 마커로 추가
for index, row in target9.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']]).add_to(map)


# In[95]:


map


# In[29]:


# NaN 값을 빈 문자열로 대체
merged_df["도로명 주소"].fillna("", inplace=True)


# In[30]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_1 = pd.to_numeric(merged_df_옥천읍["Latitude"], errors='coerce')
longitude_values_1 = pd.to_numeric(merged_df_옥천읍["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_1 = latitude_values_1.dropna()
valid_longitude_values_1 = longitude_values_1.dropna()

X = np.column_stack((valid_latitude_values_1, valid_longitude_values_1))

# bandwidth 추정
bandwidth_1 = estimate_bandwidth(X, quantile=0.2, n_samples=500)

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


# In[31]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_2 = pd.to_numeric(merged_df_이원면["Latitude"], errors='coerce')
longitude_values_2 = pd.to_numeric(merged_df_이원면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_2 = latitude_values_2.dropna()
valid_longitude_values_2 = longitude_values_2.dropna()

X2 = np.column_stack((valid_latitude_values_2, valid_longitude_values_2))

# bandwidth 추정
bandwidth_2 = estimate_bandwidth(X2, quantile=0.2, n_samples=500)

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


# In[32]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_3 = pd.to_numeric(merged_df_안내면["Latitude"], errors='coerce')
longitude_values_3 = pd.to_numeric(merged_df_안내면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_3 = latitude_values_3.dropna()
valid_longitude_values_3 = longitude_values_3.dropna()

X3 = np.column_stack((valid_latitude_values_3, valid_longitude_values_3))

# bandwidth 추정
bandwidth_3 = estimate_bandwidth(X3, quantile=0.2, n_samples=500)

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


# In[33]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_4 = pd.to_numeric(merged_df_안남면["Latitude"], errors='coerce')
longitude_values_4 = pd.to_numeric(merged_df_안남면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_4 = latitude_values_4.dropna()
valid_longitude_values_4 = longitude_values_4.dropna()

X4 = np.column_stack((valid_latitude_values_4, valid_longitude_values_4))

# bandwidth 추정
bandwidth_4 = estimate_bandwidth(X4, quantile=0.2, n_samples=500)

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


# In[34]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_5 = pd.to_numeric(merged_df_군서면["Latitude"], errors='coerce')
longitude_values_5 = pd.to_numeric(merged_df_군서면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_5 = latitude_values_5.dropna()
valid_longitude_values_5 = longitude_values_5.dropna()

X5 = np.column_stack((valid_latitude_values_5, valid_longitude_values_5))

# bandwidth 추정
bandwidth_5 = estimate_bandwidth(X5, quantile=0.2, n_samples=500)

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


# In[35]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_6 = pd.to_numeric(merged_df_군북면["Latitude"], errors='coerce')
longitude_values_6 = pd.to_numeric(merged_df_군북면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_6 = latitude_values_6.dropna()
valid_longitude_values_6 = longitude_values_6.dropna()

X6 = np.column_stack((valid_latitude_values_6, valid_longitude_values_6))

# bandwidth 추정
bandwidth_6 = estimate_bandwidth(X6, quantile=0.2, n_samples=500)

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


# In[36]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_7 = pd.to_numeric(merged_df_동이면["Latitude"], errors='coerce')
longitude_values_7 = pd.to_numeric(merged_df_동이면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_7 = latitude_values_7.dropna()
valid_longitude_values_7 = longitude_values_7.dropna()

X7 = np.column_stack((valid_latitude_values_7, valid_longitude_values_7))

# bandwidth 추정
bandwidth_7 = estimate_bandwidth(X7, quantile=0.2, n_samples=500)

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


# In[37]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_8 = pd.to_numeric(merged_df_청산면["Latitude"], errors='coerce')
longitude_values_8 = pd.to_numeric(merged_df_청산면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_8 = latitude_values_8.dropna()
valid_longitude_values_8 = longitude_values_8.dropna()

X8 = np.column_stack((valid_latitude_values_8, valid_longitude_values_8))

# bandwidth 추정
bandwidth_8 = estimate_bandwidth(X8, quantile=0.2, n_samples=500)

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


# In[38]:


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# 데이터셋 준비
# merged_df_옥천읍 데이터프레임에서 숫자로 변환할 수 있는 값만 선택
latitude_values_9 = pd.to_numeric(merged_df_청성면["Latitude"], errors='coerce')
longitude_values_9 = pd.to_numeric(merged_df_청성면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_9 = latitude_values_9.dropna()
valid_longitude_values_9 = longitude_values_9.dropna()

X9 = np.column_stack((valid_latitude_values_9, valid_longitude_values_9))

# bandwidth 추정
bandwidth_9 = estimate_bandwidth(X9, quantile=0.2, n_samples=500)

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


# In[39]:


import numpy as np

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points = merged_df_옥천읍.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k = int(np.sqrt(n_data_points / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k}")


# In[40]:


import numpy as np
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points = merged_df_옥천읍.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k = int(np.sqrt(n_data_points / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans = KMeans(n_clusters=rule_of_thumb_k, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
# 예를 들어, latitude_values와 longitude_values는 숫자로 변환 가능한 경우에 선택
# latitude_values = pd.to_numeric(merged_df_옥천읍["Latitude"], errors='coerce')
# longitude_values = pd.to_numeric(merged_df_옥천읍["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
# valid_latitude_values = latitude_values.dropna()
# valid_longitude_values = longitude_values.dropna()

# X = np.column_stack((valid_latitude_values, valid_longitude_values))

# 예시로 사용할 데이터 X 생성 (실제 데이터셋에 맞게 적절히 조정 필요)
X = np.random.rand(n_data_points, 2)  # 랜덤 데이터 생성 예시

# K-means 모델 피팅
kmeans.fit(X)

# 초기 중심점 출력
print("초기 중심점:")
print(kmeans.cluster_centers_)


# In[41]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points = merged_df_옥천읍.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k = int(np.sqrt(n_data_points / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans = KMeans(n_clusters=rule_of_thumb_k, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
# 예를 들어, latitude_values와 longitude_values는 숫자로 변환 가능한 경우에 선택
# latitude_values = pd.to_numeric(merged_df_옥천읍["Latitude"], errors='coerce')
# longitude_values = pd.to_numeric(merged_df_옥천읍["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
# valid_latitude_values = latitude_values.dropna()
# valid_longitude_values = longitude_values.dropna()

# X = np.column_stack((valid_latitude_values, valid_longitude_values))

# 예시로 사용할 데이터 X 생성 (실제 데이터셋에 맞게 적절히 조정 필요)
X = np.random.rand(n_data_points, 2)  # 랜덤 데이터 생성 예시

# K-means 모델 피팅
kmeans.fit(X)

# 각 데이터포인트를 군집에 할당(배정)
labels = kmeans.labels_

# 할당된 군집(클러스터) 확인
print("데이터 포인트의 군집 할당 결과:")
print(labels)


# In[42]:


# 각 군집의 중심점 재설정(갱신)
new_centers = []
for i in range(rule_of_thumb_k):
    cluster_points = X[labels == i]  # 현재 군집에 속하는 데이터 포인트들 선택
    new_center = np.mean(cluster_points, axis=0)  # 군집 내 데이터 포인트들의 평균 계산
    new_centers.append(new_center)

new_centers = np.array(new_centers)

# 갱신된 중심점 출력
print("갱신된 중심점:")
print(new_centers)


# In[43]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산
n_data_points = merged_df_옥천읍.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k = int(np.sqrt(n_data_points / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans = KMeans(n_clusters=rule_of_thumb_k, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
# 예를 들어, latitude_values와 longitude_values는 숫자로 변환 가능한 경우에 선택
# latitude_values = pd.to_numeric(merged_df_옥천읍["Latitude"], errors='coerce')
# longitude_values = pd.to_numeric(merged_df_옥천읍["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
# valid_latitude_values = latitude_values.dropna()
# valid_longitude_values = longitude_values.dropna()

# X = np.column_stack((valid_latitude_values, valid_longitude_values))

# 예시로 사용할 데이터 X 생성 (실제 데이터셋에 맞게 적절히 조정 필요)
X = np.random.rand(n_data_points, 2)  # 랜덤 데이터 생성 예시

# K-means 클러스터링 반복
max_iterations = 100
for i in range(max_iterations):
    # K-means 모델 피팅
    kmeans.fit(X)
    
    # 현재 중심점 출력
    print(f"Iteration {i + 1} - 현재 중심점:")
    print(kmeans.cluster_centers_)
    
    # 각 데이터포인트를 군집에 할당(배정)
    labels = kmeans.labels_
    
    # 각 군집의 중심점 재설정(갱신)
    new_centers = []
    for j in range(rule_of_thumb_k):
        cluster_points = X[labels == j]  # 현재 군집에 속하는 데이터 포인트들 선택
        new_center = np.mean(cluster_points, axis=0)  # 군집 내 데이터 포인트들의 평균 계산
        new_centers.append(new_center)
    
    new_centers = np.array(new_centers)
    
    # 중심점 변화가 없으면 반복 중지
    if np.allclose(kmeans.cluster_centers_, new_centers):
        print(f"수렴 조건 충족 - Iteration {i + 1}")
        break
    
    # 중심점 업데이트
    kmeans.cluster_centers_ = new_centers

# 최종 군집 할당 결과
final_labels = kmeans.labels_
print("최종 데이터 포인트의 군집 할당 결과:")
print(final_labels)


# In[44]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산 (이원면 데이터셋)
n_data_points_2 = merged_df_이원면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_2 = int(np.sqrt(n_data_points_2 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_2}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_2 = KMeans(n_clusters=rule_of_thumb_k_2, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
# 예를 들어, latitude_values와 longitude_values는 숫자로 변환 가능한 경우에 선택
# latitude_values_2 = pd.to_numeric(merged_df_이원면["Latitude"], errors='coerce')
# longitude_values_2 = pd.to_numeric(merged_df_이원면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
# valid_latitude_values_2 = latitude_values_2.dropna()
# valid_longitude_values_2 = longitude_values_2.dropna()

# X_2 = np.column_stack((valid_latitude_values_2, valid_longitude_values_2))

# 예시로 사용할 데이터 X 생성 (실제 데이터셋에 맞게 적절히 조정 필요)
X_2 = np.random.rand(n_data_points_2, 2)  # 랜덤 데이터 생성 예시

# K-means 클러스터링 반복
max_iterations_2 = 100
for i in range(max_iterations_2):
    # K-means 모델 피팅
    kmeans_2.fit(X_2)
    
    # 현재 중심점 출력
    print(f"Iteration {i + 1} - 현재 중심점:")
    print(kmeans_2.cluster_centers_)
    
    # 각 데이터포인트를 군집에 할당(배정)
    labels_2 = kmeans_2.labels_
    
    # 각 군집의 중심점 재설정(갱신)
    new_centers_2 = []
    for j in range(rule_of_thumb_k_2):
        cluster_points_2 = X_2[labels_2 == j]  # 현재 군집에 속하는 데이터 포인트들 선택
        new_center_2 = np.mean(cluster_points_2, axis=0)  # 군집 내 데이터 포인트들의 평균 계산
        new_centers_2.append(new_center_2)
    
    new_centers_2 = np.array(new_centers_2)
    
    # 중심점 변화가 없으면 반복 중지
    if np.allclose(kmeans_2.cluster_centers_, new_centers_2):
        print(f"수렴 조건 충족 - Iteration {i + 1}")
        break
    
    # 중심점 업데이트
    kmeans_2.cluster_centers_ = new_centers_2

# 최종 군집 할당 결과
final_labels_2 = kmeans_2.labels_
print("최종 데이터 포인트의 군집 할당 결과:")
print(final_labels_2)


# In[45]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산 (안내면 데이터셋)
n_data_points_3 = merged_df_안내면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_3 = int(np.sqrt(n_data_points_3 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_3}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_3 = KMeans(n_clusters=rule_of_thumb_k_3, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
# 예를 들어, latitude_values와 longitude_values는 숫자로 변환 가능한 경우에 선택
# latitude_values_3 = pd.to_numeric(merged_df_안내면["Latitude"], errors='coerce')
# longitude_values_3 = pd.to_numeric(merged_df_안내면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
# valid_latitude_values_3 = latitude_values_3.dropna()
# valid_longitude_values_3 = longitude_values_3.dropna()

# X_3 = np.column_stack((valid_latitude_values_3, valid_longitude_values_3))

# 예시로 사용할 데이터 X 생성 (실제 데이터셋에 맞게 적절히 조정 필요)
X_3 = np.random.rand(n_data_points_3, 2)  # 랜덤 데이터 생성 예시

# K-means 클러스터링 반복
max_iterations_3 = 100
for i in range(max_iterations_3):
    # K-means 모델 피팅
    kmeans_3.fit(X_3)
    
    # 현재 중심점 출력
    print(f"Iteration {i + 1} - 현재 중심점:")
    print(kmeans_3.cluster_centers_)
    
    # 각 데이터포인트를 군집에 할당(배정)
    labels_3 = kmeans_3.labels_
    
    # 각 군집의 중심점 재설정(갱신)
    new_centers_3 = []
    for j in range(rule_of_thumb_k_3):
        cluster_points_3 = X_3[labels_3 == j]  # 현재 군집에 속하는 데이터 포인트들 선택
        new_center_3 = np.mean(cluster_points_3, axis=0)  # 군집 내 데이터 포인트들의 평균 계산
        new_centers_3.append(new_center_3)
    
    new_centers_3 = np.array(new_centers_3)
    
    # 중심점 변화가 없으면 반복 중지
    if np.allclose(kmeans_3.cluster_centers_, new_centers_3):
        print(f"수렴 조건 충족 - Iteration {i + 1}")
        break
    
    # 중심점 업데이트
    kmeans_3.cluster_centers_ = new_centers_3

# 최종 군집 할당 결과
final_labels_3 = kmeans_3.labels_
print("최종 데이터 포인트의 군집 할당 결과:")
print(final_labels_3)


# In[46]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산 (안남면 데이터셋)
n_data_points_4 = merged_df_안남면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_4 = int(np.sqrt(n_data_points_4 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_4}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_4 = KMeans(n_clusters=rule_of_thumb_k_4, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
# 예를 들어, latitude_values와 longitude_values는 숫자로 변환 가능한 경우에 선택
# latitude_values_4 = pd.to_numeric(merged_df_안남면["Latitude"], errors='coerce')
# longitude_values_4 = pd.to_numeric(merged_df_안남면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
# valid_latitude_values_4 = latitude_values_4.dropna()
# valid_longitude_values_4 = longitude_values_4.dropna()

# X_4 = np.column_stack((valid_latitude_values_4, valid_longitude_values_4))

# 예시로 사용할 데이터 X 생성 (실제 데이터셋에 맞게 적절히 조정 필요)
X_4 = np.random.rand(n_data_points_4, 2)  # 랜덤 데이터 생성 예시

# K-means 클러스터링 반복
max_iterations_4 = 100
for i in range(max_iterations_4):
    # K-means 모델 피팅
    kmeans_4.fit(X_4)
    
    # 현재 중심점 출력
    print(f"Iteration {i + 1} - 현재 중심점:")
    print(kmeans_4.cluster_centers_)
    
    # 각 데이터포인트를 군집에 할당(배정)
    labels_4 = kmeans_4.labels_
    
    # 각 군집의 중심점 재설정(갱신)
    new_centers_4 = []
    for j in range(rule_of_thumb_k_4):
        cluster_points_4 = X_4[labels_4 == j]  # 현재 군집에 속하는 데이터 포인트들 선택
        new_center_4 = np.mean(cluster_points_4, axis=0)  # 군집 내 데이터 포인트들의 평균 계산
        new_centers_4.append(new_center_4)
    
    new_centers_4 = np.array(new_centers_4)
    
    # 중심점 변화가 없으면 반복 중지
    if np.allclose(kmeans_4.cluster_centers_, new_centers_4):
        print(f"수렴 조건 충족 - Iteration {i + 1}")
        break
    
    # 중심점 업데이트
    kmeans_4.cluster_centers_ = new_centers_4

# 최종 군집 할당 결과
final_labels_4 = kmeans_4.labels_
print("최종 데이터 포인트의 군집 할당 결과:")
print(final_labels_4)


# In[47]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산 (군서면 데이터셋)
n_data_points_5 = merged_df_군서면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_5 = int(np.sqrt(n_data_points_5 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_5}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_5 = KMeans(n_clusters=rule_of_thumb_k_5, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
# 예를 들어, latitude_values와 longitude_values는 숫자로 변환 가능한 경우에 선택
# latitude_values_5 = pd.to_numeric(merged_df_군서면["Latitude"], errors='coerce')
# longitude_values_5 = pd.to_numeric(merged_df_군서면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
# valid_latitude_values_5 = latitude_values_5.dropna()
# valid_longitude_values_5 = longitude_values_5.dropna()

# X_5 = np.column_stack((valid_latitude_values_5, valid_longitude_values_5))

# 예시로 사용할 데이터 X 생성 (실제 데이터셋에 맞게 적절히 조정 필요)
X_5 = np.random.rand(n_data_points_5, 2)  # 랜덤 데이터 생성 예시

# K-means 클러스터링 반복
max_iterations_5 = 100
for i in range(max_iterations_5):
    # K-means 모델 피팅
    kmeans_5.fit(X_5)
    
    # 현재 중심점 출력
    print(f"Iteration {i + 1} - 현재 중심점:")
    print(kmeans_5.cluster_centers_)
    
    # 각 데이터포인트를 군집에 할당(배정)
    labels_5 = kmeans_5.labels_
    
    # 각 군집의 중심점 재설정(갱신)
    new_centers_5 = []
    for j in range(rule_of_thumb_k_5):
        cluster_points_5 = X_5[labels_5 == j]  # 현재 군집에 속하는 데이터 포인트들 선택
        new_center_5 = np.mean(cluster_points_5, axis=0)  # 군집 내 데이터 포인트들의 평균 계산
        new_centers_5.append(new_center_5)
    
    new_centers_5 = np.array(new_centers_5)
    
    # 중심점 변화가 없으면 반복 중지
    if np.allclose(kmeans_5.cluster_centers_, new_centers_5):
        print(f"수렴 조건 충족 - Iteration {i + 1}")
        break
    
    # 중심점 업데이트
    kmeans_5.cluster_centers_ = new_centers_5

# 최종 군집 할당 결과
final_labels_5 = kmeans_5.labels_
print("최종 데이터 포인트의 군집 할당 결과:")
print(final_labels_5)


# In[48]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산 (군북면 데이터셋)
n_data_points_6 = merged_df_군북면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_6 = int(np.sqrt(n_data_points_6 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_6}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_6 = KMeans(n_clusters=rule_of_thumb_k_6, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
# 예를 들어, latitude_values와 longitude_values는 숫자로 변환 가능한 경우에 선택
# latitude_values_6 = pd.to_numeric(merged_df_군북면["Latitude"], errors='coerce')
# longitude_values_6 = pd.to_numeric(merged_df_군북면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
# valid_latitude_values_6 = latitude_values_6.dropna()
# valid_longitude_values_6 = longitude_values_6.dropna()

# X_6 = np.column_stack((valid_latitude_values_6, valid_longitude_values_6))

# 예시로 사용할 데이터 X 생성 (실제 데이터셋에 맞게 적절히 조정 필요)
X_6 = np.random.rand(n_data_points_6, 2)  # 랜덤 데이터 생성 예시

# K-means 클러스터링 반복
max_iterations_6 = 100
for i in range(max_iterations_6):
    # K-means 모델 피팅
    kmeans_6.fit(X_6)
    
    # 현재 중심점 출력
    print(f"Iteration {i + 1} - 현재 중심점:")
    print(kmeans_6.cluster_centers_)
    
    # 각 데이터포인트를 군집에 할당(배정)
    labels_6 = kmeans_6.labels_
    
    # 각 군집의 중심점 재설정(갱신)
    new_centers_6 = []
    for j in range(rule_of_thumb_k_6):
        cluster_points_6 = X_6[labels_6 == j]  # 현재 군집에 속하는 데이터 포인트들 선택
        new_center_6 = np.mean(cluster_points_6, axis=0)  # 군집 내 데이터 포인트들의 평균 계산
        new_centers_6.append(new_center_6)
    
    new_centers_6 = np.array(new_centers_6)
    
    # 중심점 변화가 없으면 반복 중지
    if np.allclose(kmeans_6.cluster_centers_, new_centers_6):
        print(f"수렴 조건 충족 - Iteration {i + 1}")
        break
    
    # 중심점 업데이트
    kmeans_6.cluster_centers_ = new_centers_6

# 최종 군집 할당 결과
final_labels_6 = kmeans_6.labels_
print("최종 데이터 포인트의 군집 할당 결과:")
print(final_labels_6)


# In[49]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산 (동이면 데이터셋)
n_data_points_7 = merged_df_동이면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_7 = int(np.sqrt(n_data_points_7 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_7}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_7 = KMeans(n_clusters=rule_of_thumb_k_7, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
# 예를 들어, latitude_values와 longitude_values는 숫자로 변환 가능한 경우에 선택
# latitude_values_7 = pd.to_numeric(merged_df_동이면["Latitude"], errors='coerce')
# longitude_values_7 = pd.to_numeric(merged_df_동이면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
# valid_latitude_values_7 = latitude_values_7.dropna()
# valid_longitude_values_7 = longitude_values_7.dropna()

# X_7 = np.column_stack((valid_latitude_values_7, valid_longitude_values_7))

# 예시로 사용할 데이터 X 생성 (실제 데이터셋에 맞게 적절히 조정 필요)
X_7 = np.random.rand(n_data_points_7, 2)  # 랜덤 데이터 생성 예시

# K-means 클러스터링 반복
max_iterations_7 = 100
for i in range(max_iterations_7):
    # K-means 모델 피팅
    kmeans_7.fit(X_7)
    
    # 현재 중심점 출력
    print(f"Iteration {i + 1} - 현재 중심점:")
    print(kmeans_7.cluster_centers_)
    
    # 각 데이터포인트를 군집에 할당(배정)
    labels_7 = kmeans_7.labels_
    
    # 각 군집의 중심점 재설정(갱신)
    new_centers_7 = []
    for j in range(rule_of_thumb_k_7):
        cluster_points_7 = X_7[labels_7 == j]  # 현재 군집에 속하는 데이터 포인트들 선택
        new_center_7 = np.mean(cluster_points_7, axis=0)  # 군집 내 데이터 포인트들의 평균 계산
        new_centers_7.append(new_center_7)
    
    new_centers_7 = np.array(new_centers_7)
    
    # 중심점 변화가 없으면 반복 중지
    if np.allclose(kmeans_7.cluster_centers_, new_centers_7):
        print(f"수렴 조건 충족 - Iteration {i + 1}")
        break
    
    # 중심점 업데이트
    kmeans_7.cluster_centers_ = new_centers_7

# 최종 군집 할당 결과
final_labels_7 = kmeans_7.labels_
print("최종 데이터 포인트의 군집 할당 결과:")
print(final_labels_7)


# In[50]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산 (청산면 데이터셋)
n_data_points_8 = merged_df_청산면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_8 = int(np.sqrt(n_data_points_8 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_8}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_8 = KMeans(n_clusters=rule_of_thumb_k_8, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
latitude_values_8 = pd.to_numeric(merged_df_청산면["Latitude"], errors='coerce')
longitude_values_8 = pd.to_numeric(merged_df_청산면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_8 = latitude_values_8.dropna()
valid_longitude_values_8 = longitude_values_8.dropna()

X_8 = np.column_stack((valid_latitude_values_8, valid_longitude_values_8))

# K-means 클러스터링 반복
max_iterations_8 = 100
for i in range(max_iterations_8):
    # K-means 모델 피팅
    kmeans_8.fit(X_8)
    
    # 현재 중심점 출력
    print(f"Iteration {i + 1} - 현재 중심점:")
    print(kmeans_8.cluster_centers_)
    
    # 각 데이터포인트를 군집에 할당(배정)
    labels_8 = kmeans_8.labels_
    
    # 각 군집의 중심점 재설정(갱신)
    new_centers_8 = []
    for j in range(rule_of_thumb_k_8):
        cluster_points_8 = X_8[labels_8 == j]  # 현재 군집에 속하는 데이터 포인트들 선택
        new_center_8 = np.mean(cluster_points_8, axis=0)  # 군집 내 데이터 포인트들의 평균 계산
        new_centers_8.append(new_center_8)
    
    new_centers_8 = np.array(new_centers_8)
    
    # 중심점 변화가 없으면 반복 중지
    if np.allclose(kmeans_8.cluster_centers_, new_centers_8):
        print(f"수렴 조건 충족 - Iteration {i + 1}")
        break
    
    # 중심점 업데이트
    kmeans_8.cluster_centers_ = new_centers_8

# 최종 군집 할당 결과
final_labels_8 = kmeans_8.labels_
print("최종 데이터 포인트의 군집 할당 결과:")
print(final_labels_8)


# In[51]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 데이터프레임에서 데이터포인트의 수 계산 (청성면 데이터셋)
n_data_points_9 = merged_df_청성면.shape[0]

# Rule of Thumb을 적용하여 K 값 추정
rule_of_thumb_k_9 = int(np.sqrt(n_data_points_9 / 2))

print(f"Rule of Thumb을 통한 추정된 K 값: {rule_of_thumb_k_9}")

# K-means++ 방법을 사용하여 K-means 클러스터링 모델 생성
kmeans_9 = KMeans(n_clusters=rule_of_thumb_k_9, init='k-means++', random_state=0)

# 데이터프레임에서 필요한 숫자 형태로 변환 가능한 데이터 선택
latitude_values_9 = pd.to_numeric(merged_df_청성면["Latitude"], errors='coerce')
longitude_values_9 = pd.to_numeric(merged_df_청성면["Longitude"], errors='coerce')

# NaN 값을 제거한 후 최소값과 최대값 계산
valid_latitude_values_9 = latitude_values_9.dropna()
valid_longitude_values_9 = longitude_values_9.dropna()

X_9 = np.column_stack((valid_latitude_values_9, valid_longitude_values_9))

# K-means 클러스터링 반복
max_iterations_9 = 100
for i in range(max_iterations_9):
    # K-means 모델 피팅
    kmeans_9.fit(X_9)
    
    # 현재 중심점 출력
    print(f"Iteration {i + 1} - 현재 중심점:")
    print(kmeans_9.cluster_centers_)
    
    # 각 데이터포인트를 군집에 할당(배정)
    labels_9 = kmeans_9.labels_
    
    # 각 군집의 중심점 재설정(갱신)
    new_centers_9 = []
    for j in range(rule_of_thumb_k_9):
        cluster_points_9 = X_9[labels_9 == j]  # 현재 군집에 속하는 데이터 포인트들 선택
        new_center_9 = np.mean(cluster_points_9, axis=0)  # 군집 내 데이터 포인트들의 평균 계산
        new_centers_9.append(new_center_9)
    
    new_centers_9 = np.array(new_centers_9)
    
    # 중심점 변화가 없으면 반복 중지
    if np.allclose(kmeans_9.cluster_centers_, new_centers_9):
        print(f"수렴 조건 충족 - Iteration {i + 1}")
        break
    
    # 중심점 업데이트
    kmeans_9.cluster_centers_ = new_centers_9

# 최종 군집 할당 결과
final_labels_9 = kmeans_9.labels_
print("최종 데이터 포인트의 군집 할당 결과:")
print(final_labels_9)


# In[52]:


# 최종 군집 할당 결과에서 유니크한 레이블 개수 계산
n_clusters_final = len(np.unique(final_labels))
print("최종 클러스터 개수:", n_clusters_final)


# In[ ]:




