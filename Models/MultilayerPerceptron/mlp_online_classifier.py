import numpy as np
from sklearn.metrics import confusion_matrix

data_x = np.genfromtxt('images.csv', delimiter=',')[:500]
data_y = np.genfromtxt('labels.csv', delimiter=',')[:500].astype(int)
data_y = np.eye(np.max(data_y))[(data_y-1).tolist()]

n, d = data_x.shape

safelog = lambda a: np.log(a + 1e-100)
sigmoid = lambda a: 1 / (1 + np.exp(-a))

eta = 5e-4
epsilon = 1e-3
h = 20
max_iteration = 500

v = np.genfromtxt('initial_V.csv', delimiter=',')
w = np.genfromtxt('initial_W.csv', delimiter=',')

z = sigmoid(np.c_[np.ones(n), data_x].dot(w))
y_pred = sigmoid(np.c_[np.ones(n), z].dot(v))
objective_values = [-0.5*np.sum(data_y * safelog(y_pred) + (1 - data_y) * safelog(1 - y_pred))]

iteration = 0
while(iteration < max_iteration):
  if iteration % 50 == 0:
    print('Iteration: ', iteration)
  for i in np.random.choice(n, n, replace=False):
    z[i, :] = sigmoid(np.append(1, data_x[i, :]).dot(w))
    y_pred[i] = sigmoid(np.append(1, z[i, :]).dot(v))

    delta_v = eta * (data_y[i:i+1] - y_pred[i:i+1]).T.dot(np.c_[np.ones(1), z[i:i+1, :]]).T
    a = data_y[i] - y_pred[i]
    b = np.append(1, data_x[i, :])
    c = v[1:].T * z[i, :] * (1 - z[i, :])
    delta_w = eta * np.outer(a,b).T.dot(c)

    v += delta_v
    w += delta_w

  z = sigmoid(np.c_[np.ones(n), data_x].dot(w))
  y_pred = sigmoid(np.c_[np.ones(n), z].dot(v))
  objective_values.append(-0.5*np.sum(data_y * safelog(y_pred) + (1 - data_y) * safelog(1 - y_pred)))

  if np.abs(objective_values[-1] - objective_values[-2]) < epsilon:
    pass

  iteration += 1

print('Iteration: ', iteration)
print(confusion_matrix(np.argmax(data_y, axis=1), np.argmax(y_pred, axis=1)))
