import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

path = "/home/mukai/DASeminar2023/mukai/minpaku/dataset/"

X_train = pd.read_csv(path + 'data_train.csv')
X_valid = pd.read_csv(path + 'data_valid.csv')
X_test = pd.read_csv(path + 'data_test.csv')
y_train = pd.read_csv(path + 'label_train.csv')
y_valid = pd.read_csv(path + 'label_valid.csv')

from optuna.integration import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import re

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

train_data = lgb.Dataset(X_train_scaled, label=y_train)
valid_data = lgb.Dataset(X_valid_scaled, label=y_valid)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1,
    'learning_rate': 0.01,
    "num_iterations":1000,
}
verbose_eval = 100

#optuna
from optuna.integration import lightgbm as lgb
model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=1000000000,callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True),lgb.log_evaluation(verbose_eval)])
y_test_pred = model.predict(X_test_scaled)

result_df = pd.DataFrame({'id': range(0, 18528), 'y': y_test_pred})
result_df.to_csv('submission.csv', index=False, header=False)


result_df = pd.DataFrame({'id': range(0, 18528), 'y': y_test_pred})
result_df.to_csv(path + 'submission6.csv', index=False, header=False)
