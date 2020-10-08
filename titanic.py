import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv('titanic_test.csv')
df_train = df_train.drop(['name','ticket','body','cabin','home.dest'], axis =1)
df_test = df_test.drop(['name','ticket','body','cabin','home.dest'], axis =1)

from scipy import stats

# 두 집단의 피처를 비교해주며 탐색작업을 자동화하는 함수를 정의합니다.
def valid_features(df, col_name, distribution_check=True):
    
    # 두 집단 (survived=1, survived=0)의 분포 그래프를 출력합니다.
    g = sns.FacetGrid(df, col='survived')
    g.map(plt.hist, col_name, bins=30)

    # 두 집단 (survived=1, survived=0)의 표준편차를 각각 출력합니다.
    titanic_survived = df[df['survived']==1]
    titanic_survived_static = np.array(titanic_survived[col_name])
    print("data std is", '%.2f' % np.std(titanic_survived_static))
    titanic_n_survived = df[df['survived']==0]
    titanic_n_survived_static = np.array(titanic_n_survived[col_name])
    print("data std is", '%.2f' % np.std(titanic_n_survived_static))
    
     # T-test로 두 집단의 평균 차이를 검정합니다.
    tTestResult = stats.ttest_ind(titanic_survived[col_name], titanic_n_survived[col_name])
    tTestResultDiffVar = stats.ttest_ind(titanic_survived[col_name], titanic_n_survived[col_name], equal_var=False)
    print("The t-statistic and p-value assuming equal variances is %.3f and %.3f." % tTestResult)
    print("The t-statistic and p-value not assuming equal variances is %.3f and %.3f" % tTestResultDiffVar)
    
    if distribution_check:
        # Shapiro-Wilk 검정 : 분포의 정규성 정도를 검증합니다.
        print("The w-statistic and p-value in Survived %.3f and %.3f" % stats.shapiro(titanic_survived[col_name]))
        print("The w-statistic and p-value in Non-Survived %.3f and %.3f" % stats.shapiro(titanic_n_survived[col_name]))

# 함수 실행 age, sibsp
valid_features(df_train[df_train['age']>0],'age',distribution_check=True)
plt.show()

# sibsp 동승한 형제 또는 배우자 수
valid_features(df_train,'sibsp',distribution_check=False)
plt.show()

# sex : 탑승자 성별
valid_features(df_train,'sex',distribution_check=True)
plt.show()

# parch : 동승한 부모 또는 자녀의 수
valid_features(df_train,'parch',distribution_check=True)
plt.show()

# pclass : 탑승자 등급
valid_features(df_train,'pclass',distribution_check=True)
plt.show()

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
test_idx_num = len(df_test)

# pandas 패키지를 이용해서 one-hor encoding 수행
whole_df_encoded = pd.get_dummies(whole_df)
df_train = whole_df_encoded[:train_idx_num]
df_test = whole_df_encoded[:test_idx_num]

df_train.head()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 데이터를 학습 데이터와 테스트 데이터로 분리

X_train, y_train = df_train.loc[:, df_train.columns != 'survived'].values, df_train['survived'].values
X_test, y_test = df_test.loc[:, df_test.columns != 'survived'].values, df_train['survived'].values

# 로지스틱 회귀 모델 학습
lr = LogisticRegression(random_state=0)
lr.fit(X_train , y_train)

y_pred = lr.predict(X_test)
print(y_pred)

y_pred_p = lr.predict_proba(X_test)[:,1]
print(y_pred_p)


# 테스트 데이터에 대한 정확도, 정밀도, 특이도, 평가 지표

print("정확도 : %.2f" % accuracy_score(y_test, y_pred))
print("정밀도 : %.2f" % precision_score(y_test, y_pred))
print("특이도 : %.2f" % recall_score(y_test, y_pred))
print("평가지표 : %.2f" % f1_score(y_test, y_pred))

# 의사결정나무 모델은 피처 단위로 조건을 분리하여 정답의 집합을 좁혀나가는 방법
# 마치 스무고개 놀이에서 정답을 찾아 나가는 과정과 유사하다

from sklearn.tree import DecisionTreeClassifier

# 의사 결정 나무를 학습하고, 학습한 모델로 테시트 데이터셋에 대한 예측값을 반환한다.

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

print(y_pred)

y_pred_p = dtc.predict_proba(X_test)[:,1]
print(y_pred_p)

print("expect : %.2f" % accuracy_score(y_test, y_pred))