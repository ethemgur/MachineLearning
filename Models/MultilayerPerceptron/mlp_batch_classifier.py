import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data_x = np.genfromtxt('images.csv', delimiter=',')
data_y = np.genfromtxt('labels.csv', delimiter=',').astype(int)

data_x = np.c_[np.ones(len(data_x)), data_x]
data_y = np.eye(np.max(data_y))[(data_y-1).tolist()]

train_size = 500
train_x = data_x[:train_size]
train_y = data_y[:train_size]


safelog = lambda a: np.log(a + 1e-100)
sigmoid = lambda a: 1 / (1 + np.exp(-a))
softmax = lambda mz, mv: np.exp(mz.dot(mv)) / np.sum(np.exp(mz.dot(mv)), axis=1)[:, None]

n = train_x.shape[0]
eta = 5e-4
epsilon = 1e-3
max_iteration = 500

v = np.genfromtxt('initial_V.csv', delimiter=',')
w = np.genfromtxt('initial_W.csv', delimiter=',')

z = sigmoid(train_x.dot(w))
y_pred = softmax(np.c_[np.ones(n), z], v)
objective_values = [-np.sum(train_y * safelog(y_pred))]

for iteration in range(max_iteration):
  delta_v = eta * (np.c_[np.ones(n), z]).T.dot(train_y - y_pred)
  delta_w = eta * train_x.T.dot((train_y - y_pred).dot(v[1:].T) * z * (1 - z))

  v += delta_v
  w += delta_w

  z = sigmoid(train_x.dot(w))
  y_pred = softmax(np.c_[np.ones(n), z], v)

  objective_values.append(-np.sum(train_y * safelog(y_pred)))
  if np.abs(objective_values[-1] - objective_values[-2]) < epsilon:
    print('Iteration stopped at: ', iteration)
    break


print('\n\n===== Train Accuracy =====\n')
print(confusion_matrix(np.argmax(train_y, axis=1), np.argmax(y_pred, axis=1)))

test_x = data_x[train_size:]
test_y = data_y[train_size:]

z = sigmoid(test_x.dot(w))
test_y_pred = sigmoid(np.c_[np.ones(n), z].dot(v))

print('\n\n===== Test Accuracy =====\n')
print(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(test_y_pred, axis=1)))

plt.plot(objective_values)
plt.show()
