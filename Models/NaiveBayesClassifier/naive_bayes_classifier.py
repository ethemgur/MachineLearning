from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from math import pi

data_x = pd.read_csv('images.csv', header=None).values
data_y = np.loadtxt(open('labels.csv', 'rb'), delimiter=',').astype(int)

train_size = 200

x = data_x[:train_size]
y = data_y[:train_size]
x_test = data_x[train_size:]
y_test = data_y[train_size:]

K = range(max(y))

means = [np.mean(x[y==i+1], axis=0) for i in K]
deviations = [np.std(x[y==i+1], axis=0) for i in K]
priors = [np.mean(y==i+1) for i in K]

score = lambda c, m: np.sum(np.log(np.exp(-1*(m-means[c])**2/2/deviations[c]**2)/np.sqrt(2*pi*deviations[c]**2)))

def get_accuracy(mx, my):
  score_values = np.array([[score(c, m) for c in K] for m in mx])
  s = np.argmax(score_values, axis=1) + 1
  print(confusion_matrix(my, s))

print(5*'=', 'Train Accuracy', 5*'=')
get_accuracy(x,y)
print(5*'=', 'Test Accuracy', 5*'=')
get_accuracy(x_test, y_test)
