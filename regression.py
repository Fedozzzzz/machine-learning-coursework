import pandas as pd
from sklearn.linear_model import Lasso
from sklearn import preprocessing

data = pd.read_csv('C:/Users/FedozZz/PycharmProjects/machine-learning-coursework/data/data.csv')

data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

features_mean = list(data.columns[1:11])

diag_map = {'M': 1, 'B': 0}
data['diagnosis'] = data['diagnosis'].map(diag_map)

X = data.loc[:, features_mean]
y = data.loc[:, 'diagnosis']


def get_regression_coeff(x, y):
    x = x.to_numpy()
    y = y.to_numpy()
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    log = Lasso(alpha=0.00001)
    log.fit(x, y)
    return log.coef_


print("X:{}".format(get_regression_coeff(X, y)))
