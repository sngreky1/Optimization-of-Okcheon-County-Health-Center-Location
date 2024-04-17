## 원핫인코딩 코드
# pd10 원핫인코딩
# 읍면 정보 col 추가
# loc = ['옥천읍','이원면','안내면','안남면','군서면','군북면','동이면','청산면','청성면']

# 도로명주소에 '옥천읍'이 포함된 경우 col로 '은평구'를 추가
# pd10['읍면'] = pd10.apply(lambda x: ' '.join([i for i in loc if i in x['도로명 주소']]), axis=1)
# pd20['읍면'] = pd20.apply(lambda x: ' '.join([i for i in loc if i in x['도로명 주소']]), axis=1)

# 원핫인코딩
# from sklearn import preprocessing
#
# label_encoder = preprocessing.LabelEncoder()
# onehot_encoder = preprocessing.OneHotEncoder()
#
# onehot_읍면10 = label_encoder.fit_transform(pd10['읍면'])
# onehot_읍면20 = label_encoder.fit_transform(pd20['읍면'])
#
# pd10['loc'] = onehot_읍면10
# pd20['loc'] = onehot_읍면20
#
# print(pd10.head())
# print(pd20.head())

# dfw.to_excel('파일명.xlsx', encoding='utf-8', index=False)

# merged_df랑 target 합칠 때 target에 'loc' column 추가해서 읍면 원핫인코딩 번호 추가 필요
#target1 = pd.DataFrame({'Latitude': x1_list, 'Longitude': y1_list, 'loc':5})
