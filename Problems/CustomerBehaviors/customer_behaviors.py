import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def test_models(x, y):
  models = [
    ('KNN', KNeighborsClassifier(n_neighbors=3)),
    ('RF', RandomForestClassifier(random_state = 42)),
    ('LR', LogisticRegression()),
    ('MLP', MLPClassifier(hidden_layer_sizes=(10,))),
    ('NB', GaussianNB())
  ]
  results = []
  names = []
  scoring = make_scorer(roc_auc_score)
  for name, model in models:
    kfold = model_selection.StratifiedKFold(n_splits=5, random_state=42)
    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


def try_model(regressor, X_train, X_test, y_train, y_test):
  regressor.fit(X_train, y_train)
  y_predicted = regressor.predict(X_test)
  y_model = regressor.predict(X_train)
  print('Train Error:', roc_auc_score(y_train.values.flatten(), y_model), end=' - ')
  print('Test Error:', roc_auc_score(y_test.values.flatten(), y_predicted))
  print(confusion_matrix(y_train, y_model))
  print(confusion_matrix(y_test, y_predicted))
  print()
  return y_model, y_predicted


def test(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    models = [
      KNeighborsClassifier(n_neighbors=3),
      RandomForestClassifier(n_estimators = 100, random_state = 42)
    ]
    try_model(RandomForestClassifier(random_state=42), X_train, X_test, y_train, y_test)

    for i in models:# [svc, rf, nb]:
      print(type(i).__name__)
      y_model, y_pred = try_model(i, X_train, X_test, y_train, y_test)
      X_train = np.concatenate((y_model[:, None], X_train), axis=1)
      X_test = np.concatenate((y_pred[:, None], X_test), axis=1)

#     model = VotingClassifier(estimators=[('svc', svc), ('rf', rf), ('nb', nb)], voting='hard')
#     try_model(model, X_train, X_test, y_train, y_test)


data_x = pd.read_csv('training_data.csv', index_col=0)
data_y = pd.read_csv('training_label.csv', index_col=0)
test_data_x = pd.read_csv('test_data.csv')

def data_handle(df1, df2):
  df1['Train'] = 1
  df2['Train'] = 0
  df = pd.concat([df1, df2])
  df = df.drop(df.columns[df.isna().sum()>len(df)*0.3], axis=1)
  object_cols = df.dtypes[(df.dtypes != int) & (df.dtypes != float)].index
  df = pd.get_dummies(df, object_cols)
  df = df.fillna(df.median())
  return df[df['Train']==1], df[df['Train']==0]

data_x, test_data_x = data_handle(data_x, test_data_x)

y_pred = pd.DataFrame()
y_pred['ID'] = test_data_x.index
for i in range(6):
  print('\n{}\nTarget {}\n{}\n'.format('-'*25, i+1, '-'*25))
  y = data_y.iloc[:, i]
  valid_rows = y.notnull()
  y = y[valid_rows]
  x = data_x[valid_rows]
  print(y.value_counts())
  x['Y'] = y
  x = pd.concat([x[x['Y']==0], resample(x[x['Y']==1], replace=True, n_samples=len(x[x['Y']==0]), random_state=42)])
  y = x['Y']
  x = x.drop('Y', axis=1)
