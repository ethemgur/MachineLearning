import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

max_iteration = 500
eta = 1e-4
epsilon = 1e-3

safelog = lambda num: np.log(num + 1e-100)

x_data = np.asmatrix(np.genfromtxt('images.csv', delimiter=','))
y_data = np.asmatrix(np.genfromtxt('labels.csv', delimiter=',').astype(int)).T

x_train = x_data[:500, :]
y_train = y_data[:500, :]

def one_hot_encoding(y):
  tmp = np.zeros((y.shape[0], 5))
  tmp[np.arange(y.shape[0]), (y.T - 1).tolist()[0]] = 1
  return np.asmatrix(tmp)

sigmoid = lambda x, mw, mw0: 1 / (1 + np.exp(-(x * mw.T + mw0.T)))
gradient_w = lambda x, y, yp: -np.sum(np.multiply(np.broadcast_to(y - yp, x.shape), x), axis=0)
gradient_w0 = lambda y, yp: -np.sum(y - yp)

w = np.asmatrix(np.genfromtxt('initial_W.csv', delimiter=',')).T
w0 = np.asmatrix(np.genfromtxt('initial_w0.csv', delimiter=',')).T

y_train = one_hot_encoding(y_train)

iteration = 0
objective_values = []

while(iteration < max_iteration):
  iteration += 1

  y_pred = sigmoid(x_train, w, w0)

  objective_values.append(-np.sum(y_train.T * safelog(y_pred) + (1 - y_train).T * safelog(1 - y_pred)))

  w_old = w
  w0_old = w0

  gradients_w = [gradient_w(x_train, y_train[:, i], y_pred[:, i]) for i in range(y_train.shape[1])]
  gradients_w = np.concatenate(gradients_w)

  w = w - eta * gradients_w
  w0 = w0 - eta * gradient_w0(y_train, y_pred)

  if (np.sqrt(np.square(w0 - w0_old) + np.sum(np.square(w - w_old))) < epsilon).all():
    break


print('Iteration no: ', iteration)

print(3*'=', 'Train Accuracy', 3*'=')
print(confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=1)))

x_test = np.asmatrix(np.genfromtxt('images.csv', delimiter=','))[500:, :]
y_test = np.asmatrix(np.genfromtxt('labels.csv', delimiter=',').astype(int)).T[500:, :]
y_pred_test = sigmoid(x_test, w, w0)

print(3*'=', 'Test Accuracy', 3*'=')
print(confusion_matrix(y_test-1, y_pred_test.argmax(axis=1)))
plt.plot(objective_values)
