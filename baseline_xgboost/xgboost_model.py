from xgboost import XGBClassifier
import lightgbm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

file_path = '../criteo_data/criteo_small.txt'
data = pd.read_csv(file_path)
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )

le = LabelEncoder()
data[sparse_features] = data[sparse_features].apply(le.fit_transform)
# data = pd.get_dummies(data,columns=sparse_features)

print(data.head())

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# xgb = XGBClassifier(learning_rate=0.01, n_estimators=20)
# xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=['error'], verbose=True)

# 不使用one-hot编码 正确率在0.766767左右
# 使用one-hot编码 内存溢出

lgb = lightgbm.LGBMClassifier(learning_rate=0.01, n_estimators=2000)
lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='error', categorical_feature=sparse_features,
        verbose=True)

# lightgbm 加入categorical_feature  正确率为 0.773067左右
