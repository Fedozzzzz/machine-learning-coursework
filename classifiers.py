import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
import time


def test_classifier(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    start = time.time()

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    scores = cross_val_score(clf, X, y, cv=5)

    end = time.time()

    print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
    # print("MSE: {0:.2%}".format(mean_squared_error(y_test, prediction, squared=False)))
    print("MSE: {0:.2%}".format(mean_squared_error(y_test, prediction)))
    print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores) * 2))
    print("Execution time: {0:.5} seconds \n".format(end - start))


def get_best_parameters_clf(clf, X, y, parameters):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start = time.time()

    clf = GridSearchCV(clf, parameters, scoring='average_precision', n_jobs=-1)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    scores = cross_val_score(clf, X, y, cv=5)

    end = time.time()

    print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
    print("MSE: {0:.2%}".format(mean_squared_error(y_test, prediction)))
    print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores) * 2))
    print("Execution time: {0:.5} seconds \n".format(end - start))

    print("Best parameters: {0}".format(clf.best_params_))

    return clf.best_params_


data = pd.read_csv('C:/Users/FedozZz/PycharmProjects/machine-learning-coursework/data/data.csv')

print("\n \t The data frame has {0[0]} rows and {0[1]} columns. \n".format(data.shape))
data.info()

data.head(3)

data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

data.info()

features_mean = list(data.columns[1:11])

plt.figure(figsize=(10, 10))
sns.heatmap(data[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()

color_dic = {'M': 'red', 'B': 'blue'}
colors = data['diagnosis'].map(lambda x: color_dic.get(x))

sm = scatter_matrix(data[features_mean], c=colors, alpha=0.4, figsize=((15, 15)))

# plt.show()

diag_map = {'M': 1, 'B': 0}
data['diagnosis'] = data['diagnosis'].map(diag_map)

X = data.loc[:, features_mean]
y = data.loc[:, 'diagnosis']

X = X.to_numpy()
y = y.to_numpy()

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

print('-----------------TESTING CLASSIFIERS-----------------')
print('KNeighborsClassifier:')
test_classifier(KNeighborsClassifier(), X, y)
print('----------------------------------------------------')
print('GaussianNB:')
# test_classifier(GaussianNB(priors=[0.2, 0.8]), X, y)
test_classifier(GaussianNB(), X, y)
print('----------------------------------------------------')
print('GradientBoostingClassifier:')
test_classifier(GradientBoostingClassifier(), X, y)
print('----------------------------------------------------')

print('------------CALCULATING THE BEST PARAMETERS FOR CLASSIFIERS-------------')
print('KNeighborsClassifier:')
parameters = {'n_neighbors': list(range(1, 5)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2, 3]}
bp_k_neib = get_best_parameters_clf(KNeighborsClassifier(), X, y, parameters)
print('----------------------------------------------------')
print('GaussianNB:')
parameters = {
    'priors': [None, [0.01, 0.99], [0.1, 0.9], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7], [0.35, 0.65], [0.4, 0.6]]}
bp_gauss = get_best_parameters_clf(GaussianNB(), X, y, parameters)
print('----------------------------------------------------')
print('GradientBoostingClassifier:')
parameters = {'max_depth': list(range(3, 10)), 'criterion': ['friedman_mse', 'squared_error'],
              'loss': ['deviance', 'exponential']}
bp_boost = get_best_parameters_clf(GradientBoostingClassifier(), X, y, parameters)
print('----------------------------------------------------')

print('------------TESTING WITH CALCULATED PARAMETERS-------------')
print('KNeighborsClassifier:')
test_classifier(KNeighborsClassifier(algorithm='auto', n_neighbors=4, p=2), X, y)
print('----------------------------------------------------')
print('GaussianNB:')
test_classifier(GaussianNB(priors=None), X, y)
print('----------------------------------------------------')
print('GradientBoostingClassifier:')
test_classifier(GradientBoostingClassifier(criterion='friedman_mse', max_depth=3), X, y)
print('----------------------------------------------------')
