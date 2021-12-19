from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from keras import regularizers
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_TNSE(X, y):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)

    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['y'] = y

    plt.figure(figsize=(16, 10))

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_subset,
        legend="full",
        alpha=0.8
    )
    plt.show()


def get_KNeighborsClf_accuracy(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    return accuracy_score(prediction, y_test), mean_squared_error(y_test, prediction)



data = pd.read_csv('C:/Users/FedozZz/PycharmProjects/machine-learning-coursework/data/data.csv')

data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

features_mean = list(data.columns[1:11])

diag_map = {'M': 1, 'B': 0}
data['diagnosis'] = data['diagnosis'].map(diag_map)

X = data.loc[:, features_mean]
y = data.loc[:, 'diagnosis']

X = X.to_numpy()
y = y.to_numpy().astype(float)

plot_TNSE(X, y)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AUTOENCODER FOR DIMENSION REDUCTION
model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=120, verbose=0)

X_train_coded = model.predict(x_train)
X_test_coded = model.predict(x_test)

acc, mse = get_KNeighborsClf_accuracy(X_train_coded, X_test_coded, y_train, y_test)

print("Dimension reduction Accuracy: {0:.2%}".format(acc))
print("MSE: {0:.2%}".format(mse))

X_coded = model.predict(X)

plot_TNSE(X_coded, y)

# SPARSE AUTOENCODER
model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense(8, activation='relu', activity_regularizer=regularizers.l1(10e-5)),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=120, verbose=0)

X_train_coded = model.predict(x_train)
X_test_coded = model.predict(x_test)

acc, mse = get_KNeighborsClf_accuracy(X_train_coded, X_test_coded, y_train, y_test)

print("Sparse Accuracy: {0:.2%}".format(acc))
print("MSE: {0:.2%}".format(mse))

X_coded = model.predict(X)
plot_TNSE(X_coded, y)

# SPARSE AUTOENCODER
model = tf.keras.Sequential([
    tf.keras.layers.GaussianNoise(0.2),
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=120, verbose=0)

X_train_coded = model.predict(x_train)
X_test_coded = model.predict(x_test)

acc, mse = get_KNeighborsClf_accuracy(X_train_coded, X_test_coded, y_train, y_test)
print("Denoising Accuracy: {0:.2%}".format(acc))
print("MSE: {0:.2%}".format(mse))

X_coded = model.predict(X)
plot_TNSE(X_coded, y)

# RESULTS:
# Dimension reduction Accuracy: 93.86%
# Sparse Accuracy: 92.98%
# Denoising Accuracy: 96.49%
