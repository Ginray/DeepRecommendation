import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_data():
    file_path = '../criteo_data/criteo_small.txt'
    data = pd.read_csv(file_path)
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    le = LabelEncoder()
    data[sparse_features] = data[sparse_features].apply(le.fit_transform)

    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test