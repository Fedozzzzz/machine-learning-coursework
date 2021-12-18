import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

data = pd.read_csv('C:/Users/FedozZz/PycharmProjects/machine-learning-coursework/data/data.csv')\

data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

features_mean = list(data.columns[1:11])

diag_map = {'M': 1, 'B': 0}
data['diagnosis'] = data['diagnosis'].map(diag_map)

X = data.loc[:, features_mean]
y = data.loc[:, 'diagnosis']

X_radius = data.loc[:, 'radius_mean']
X_texture = data.loc[:, 'texture_mean']
X_perimeter = data.loc[:, 'perimeter_mean']
X_area = data.loc[:, 'area_mean']
X_smoothness = data.loc[:, 'smoothness_mean']
X_compactness = data.loc[:, 'compactness_mean']
X_concavity = data.loc[:, 'concavity_mean']
X_concave_points = data.loc[:, 'concave points_mean']
X_symmetry = data.loc[:, 'symmetry_mean']
X_fractal_dimension = data.loc[:, 'fractal_dimension_mean']


def get_regression_coeff(x, y):
    x = x.to_numpy()
    y = y.to_numpy()

    x = x.reshape(-1, 1)

    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    log = LogisticRegression(penalty='l1', solver='liblinear')
    log.fit(X_train, y_train)

    return log.coef_[0][0]


print("radius:{}".format(get_regression_coeff(X_radius, y)))
print("texture:{}".format(get_regression_coeff(X_texture, y)))
print("perimeter:{}".format(get_regression_coeff(X_perimeter, y)))
print("area:{}".format(get_regression_coeff(X_area, y)))
print("smoothness:{}".format(get_regression_coeff(X_smoothness, y)))
print("compactness:{}".format(get_regression_coeff(X_compactness, y)))
print("concavity:{}".format(get_regression_coeff(X_concavity, y)))
print("concave_points:{}".format(get_regression_coeff(X_concave_points, y)))
print("symmetry:{}".format(get_regression_coeff(X_symmetry, y)))
print("fractal_dimension:{}".format(get_regression_coeff(X_fractal_dimension, y)))