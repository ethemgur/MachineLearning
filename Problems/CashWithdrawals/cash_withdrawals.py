import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from sklearn.neural_network import MLPRegressor


def get_rank(d, c):
  c_sort = d.groupby(c)['TRX_COUNT'].mean().sort_values().index.values
  return d[c].apply(lambda x: np.where(c_sort == x)[0][0])


def get_pre_special_date(x):
  special_dates2 = [(8, 11, 21), (6, 5, 15)]
  years = [2018, 2019]
  for j in range(len(years)):
    for i in special_dates2:
      if x['MONTH'] == i[0] and x['DAY'] > i[1] - j*10 and x['DAY'] < i[2] - j*10 and x['YEAR'] == years[j]:
        return 1
  return 0


def get_special_date(x):
  special_dates = [(7, 15), (5, 19), (8, 30), (1, 1), (10, 29)]
  special_dates2 = [(8, 20, 25), (6, 14, 18)]
  years = [2018, 2019]
  for i in special_dates:
    if x['MONTH'] == i[0] and x['DAY'] == i[1]:
      return 1
  for j in range(len(years)):
    for i in special_dates2:
      if x['MONTH'] == i[0] and x['DAY'] > i[1] - j*10 and x['DAY'] < i[2] - j*10 and x['YEAR'] == years[j]:
        return 1
  return 0


def try_model(regressor, X_train, X_test, y_train, y_test):
  regressor.fit(X_train, y_train)
  y_predicted = regressor.predict(X_test)
  y_model = regressor.predict(X_train)
  print('Train RMSE:', np.sqrt(np.mean(np.square(y_train - y_model))), end=' - ')
  print('Test RMSE:', np.sqrt(np.mean(np.square(y_test - y_predicted))))
  print()
  return np.sqrt(np.mean(np.square(y_test - y_predicted)))


def test(d, depth, r=1):
  errors = []
  for i in range(r):
    X_train, X_test, y_train, y_test = train_test_split(d.drop('TRX_COUNT', axis=1), d['TRX_COUNT'], test_size=0.025, random_state=i)
    if r > 1:
      print('\nATTEMPT {}\n------------\n\n'.format(i))
    try_model(RandomForestRegressor(n_estimators = 500, random_state = 42+i, max_depth=depth), X_train, X_test, y_train, y_test)


df = pd.read_csv('training_data.csv')
df['weekday'] = df.apply(lambda x: datetime(x['YEAR'], x['MONTH'], x['DAY']).weekday(), axis=1)
df['pre_special_date'] = df.apply(get_pre_special_date, axis=1)
df['special_date'] = df.apply(get_special_date, axis=1)
test(df, 19, 10)
