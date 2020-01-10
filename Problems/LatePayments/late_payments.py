import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample
from sklearn.decomposition import PCA
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE

def try_model(regressor, X_train, X_test, y_train, y_test):
  regressor.fit(X_train, y_train)
  y_predicted = regressor.predict(X_test)
  y_model = regressor.predict(X_train)
  print('Train AUROC:', roc_auc_score(y_train.values.flatten(), y_model), end=' - ')
  print('Test AUROC:', roc_auc_score(y_test.values.flatten(), y_predicted))
  print()
  return y_predicted


def test(d, y):
  X_train, X_test, y_train, y_test = train_test_split(d, y, test_size=0.2, random_state=42, shuffle=True)
  rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
  svc = LinearSVC()
  nb = GaussianNB()
  algs = ['Random Forest', 'SVC', 'Naive Bayes']
  for i, j in enumerate([rf, svc, nb]):
    print(algs[i])
    try_model(j, X_train, X_test, y_train, y_test)

  print('Majority Voting')
  model = VotingClassifier(estimators=[('svc', svc), ('rf', rf), ('nb', nb)], voting='hard')
  try_model(model, X_train, X_test, y_train, y_test)



for dataset in range(1, 4):
  print()
  print('TARGET', dataset)
  data = pd.read_csv('target{}_training_data.csv'.format(dataset))
  y = pd.read_csv('target{}_training_label.csv'.format(dataset), index_col=0)

  test_data_x = pd.read_csv('target{}_test_data.csv'.format(dataset))

  def data_handle(df1, df2):
    df1['Train'] = 1
    df2['Train'] = 0
    df = pd.concat([df1, df2])
    df = df.reset_index()
    df = df.drop(df.columns[df.isna().sum()>len(df)*0.2], axis=1)
    df = df.drop('ID', axis=1)
    df = df.fillna(df.mean())
    df = df.fillna(df.mode())
    object_cols = df.dtypes[(df.dtypes != int) & (df.dtypes != float)].index
    df = pd.get_dummies(df, object_cols)
    return df

  df = data_handle(data, test_data_x)
  df = df[df['Train']==1]
  df['Y'] = y
  df = pd.concat([df[df['Y']==0], resample(df[df['Y']==1], replace=True, n_samples=len(df[df['Y']==0]), random_state=42)])
  y = df['Y']
  df = df.drop('Y', axis=1)

  pca = PCA(n_components=20)
  pca.fit(df)
  X_pca = pca.transform(df)

  test(X_pca, y)

#   rf = RandomForestClassifier(n_estimators = 200, random_state = 42)
#   rf.fit(df[df['Train']==1], y)
#   y_pred = pd.DataFrame()
#   y_pred['ID'] = test_data_x.ID
#   y_pred['TARGET'] = rf.predict(df[df['Train']==0])
#   print(dataset)
#   y_pred.to_csv('prediction{}.csv'.format(dataset), index=False)
