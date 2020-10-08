# 모델 개선

# 분류 모델의 성능을 더욱 끌어 올리기 위해서...
# 1) 좋은 분류 기법을 사용해야 한다
# 2) 더 많은 데이터를 사용한다
# 3) 피처 엔지니어링 Feature Enineering
#    피처 엔지니어링은 모델에 사용 할 피처를 가공하는 분석 작업을 말한다.

import pandas as pd

df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv('titanic_test.csv')

import numpy as np
import matplotlib.pyplot as pyplot
import seaborn as sns

# 전처리

# age의 결측값을 평균값으로 대체하자
replace_mean = df_train[df_train['age']>0]['age'].mean()
df_train['age'] = df_train['age'].fillna(replace_mean)
df_test['age'] = df_test['age'].fillna(replace_mean)

# embark의 결측값을 최빈값 대체하자
embarked_mode = df_train['embarked'].value_counts().index[0]
df_train['embarked'] = df_train['embarked'].fillna(embarked_mode)
df_test['embarked'] = df_test['embarked'].fillna(embarked_mode)

# one-hot encoding / 통합 데이터 프레임(whole_df) 생성
whole_df = df_train.append(df_test)
train_idx_num = len(df_train)

# 피처 엔지니어링 $ 전처리

# cabin 피처 활용

# 결측 데이터를 'X' 대체
whole_df['cabin'] = whole_df['cabin'].fillna('X')

# cabin 피처의 첫번째 알파벳을 추출한다
whole_df['cabin'] = whole_df['cabin'].apply(lambda x : x[0])

# 추출한 알파벳 중 G T 수가 너무 작기 때문에 X 대체
whole_df['cabin'] = whole_df['cabin'].replace({"G" : "X", "T" : "x"})
ax = sns.countplot(x='cabin', hue = 'survived', data = whole_df)
plt.show()

# name 피처 성 호칭 이름
whole_df.head()

name_grade = whole_df['name'].apply(lambda x : x.split(", ",1)[1].split(".")[0])
name_grade = name_grade.unique().tolist()
print(name_grade)


# 호칭에 따라서 사회적 지위를 정의 (1910' 기준)
def give_grade(x):
    grade = x.split(", ", 1)[1].split(".")[0]
    for key, value in grade_dict.items():
        for title in value:
            if grade == title:
                return key
    return 'G'

whole_df['name'] = whole_df['name'].apply(lambda x : give_grade(x))
print(whole_df['name'].value_counts())


whole_df_encoded = pd.get_dummies(whole_df)
df_train = whole_df_encoded[:train_idx_num]
df_test = whole_df_encoded[:train_idx_num]
df_train.head(10)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


x_train, y_train = df_train.loc[:, df_train.columns != 'survived'].values, df_train['survived'].values
x_test, y_test = df_test.loc[:, df_test.columns != 'survived'].values, df_train['survived'].values


lr = LogisticRegression(random_state=0)
lr.fit(x_train , y_train)

y_pred = lr.predict(x_test)
print(y_pred)

y_pred_p = lr.predict_proba(x_test)[:,1]
print(y_pred_p)




