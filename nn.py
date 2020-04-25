import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

dataset = pd.read_csv('~/Downloads/Iris.csv')

# ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
dataset['Species'].replace(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], [0, 1, 2], inplace=True)
ohe = OneHotEncoder(categories='auto')
y = ohe.fit_transform(dataset['Species'].values.reshape(-1, 1)).toarray()
X = MinMaxScaler().fit_transform(dataset.drop(['Species', 'Id'], axis=1).values)

"""
Architecture
------------
layer_0 [150 x 4]
layer_1 [4 x 5]
layer_2 [5 x 3]

"""
#Normalize array
def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


W0 = 2 * np.random.random((4, 5)) - 1
W1 = 2 * np.random.random((5, 3)) - 1

epochs = 10000
errors = []
n = 0.1

for i in range(epochs):
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, W0))
    layer_2 = sigmoid(np.dot(layer_1, W1))

    error = y - layer_2
    error_term = error * sigmoid_deriv(layer_2)

    hidden_error = np.dot(error_term, W1.T)
    hidden_error_term = hidden_error * sigmoid_deriv(layer_1)

    W1 += np.dot(layer_1.T, error_term) * n
    W0 += np.dot(layer_0.T, hidden_error_term) * n

    errors.append(np.mean(np.abs(error)))

plt.plot(errors)
plt.xlabel('Training')
plt.ylabel('Error')
plt.show()