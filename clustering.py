import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram

data = pd.read_csv('C:/Users/FedozZz/PycharmProjects/machine-learning-coursework/data/data.csv')

data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

features_mean = list(data.columns[1:11])

diag_map = {'M': 1, 'B': 0}
data['diagnosis'] = data['diagnosis'].map(diag_map)

X = data.loc[:, features_mean]
y = data.loc[:, 'diagnosis']

X = X.to_numpy()
y = y.to_numpy()

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('KMeans:')
kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(X_train)
prediction = kmeans.predict(X_test)

print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print('----------------------------------------------------')


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)


model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)

plt.figure(figsize=(15, 15))
plt.grid(True)
plt.title('Dendrogram')
plot_dendrogram(model, truncate_mode='level', p=15)
plt.show()

print('AgglomerativeClustering (tree): ')
model = AgglomerativeClustering(n_clusters=2)
prediction = model.fit_predict(X)
print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y)))
print('----------------------------------------------------')

print('KMedoids: ')
kmedoids = KMedoids(n_clusters=2, random_state=0)
kmedoids.fit(X_train)
prediction = kmedoids.predict(X_test)
print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print('----------------------------------------------------')
