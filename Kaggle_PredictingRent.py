# -*- coding: utf-8 -*-

import pandas as pd

analysisDataPath = '/content/drive/MyDrive/Airbnb/analysisData.csv'
scoringDataPath = '/content/drive/MyDrive/Airbnb/scoringData.csv'

# Keep only numeric columns
analysisData = pd.read_csv(analysisDataPath)
analysisData.drop(columns=['id'], inplace=True)

#def to_bool(s):
 #   return 1 if analysisData['s'] == 't' else 0
#to_bool(analysisData['host_is_superhost'])

# a=nrow(analysisData)
# b=ncol(analysisData)

# aaa=analysisData['host_is_superhost']
# bbb=[1 if i =='t' else 0 for i in aaa]

#np.shape(csv_data)[1]

"""## Pre-Processing Analysis Data Utilities"""

analysisData.columns

from sklearn.preprocessing import LabelEncoder

tf_columns_list = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'is_location_exact', 'has_availability', 'requires_license', 'instant_bookable', 'is_business_travel_ready', 'require_guest_profile_picture', 'require_guest_phone_verification']
perc_columns_list = ['host_response_rate', 'host_acceptance_rate']
string_columns_list = ['square_feet', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee']
encoding_columns_list = ['host_response_time', 'neighbourhood_group_cleansed', 'smart_location', 'country_code', 'property_type', 'room_type', 'bed_type', 'cancellation_policy']
le = LabelEncoder()

def bool_to_int(input_df):
  for column in tf_columns_list:
    input_df[column] = input_df[column].apply(lambda x: 1 if x == 't' else 0)

def label_encoder(input_df):
  input_df[encoding_columns_list] = input_df[encoding_columns_list].apply(le.fit_transform)

def string_to_int(input_df):
  input_df[string_columns_list] = input_df[string_columns_list].astype(float)
  
def perc_to_float(input_df):
  for column in perc_columns_list:
    input_df[column] = input_df[column].apply(lambda x: float(x.strip('%'))/100 if not pd.isna(x) else x)

def pre_process(input_df):
  bool_to_int(input_df)
  label_encoder(input_df)
  string_to_int(input_df)
  perc_to_float(input_df)
  input_df = input_df['zipcode'].apply(lambda x: pd.to_numeric(x,errors='coerce'))

pre_process(analysisData)
analysisData = analysisData.select_dtypes(include='number')
analysisData.fillna(0, inplace=True)

analysisData.head()

from sklearn.model_selection import train_test_split
# analysisData = pd.read_csv('/content/data2.csv')

X = analysisData.loc[:, analysisData.columns != 'price']
y = analysisData['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt 

# Fitting the model
xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(X_train, y_train)
training_preds_xgb_reg = xgb_reg.predict(X_train)
val_preds_xgb_reg = xgb_reg.predict(X_test)

# Printing the results
# print(f"Time taken to run: {round((xgb_reg_end - xgb_reg_start)/60,1)} minutes")
print("\nTraining RMSE:", round(mean_squared_error(y_train, training_preds_xgb_reg, squared=False),4))
print("Validation RMSE:", round(mean_squared_error(y_test, val_preds_xgb_reg, squared=False),4))
print("\nTraining r2:", round(r2_score(y_train, training_preds_xgb_reg),4))
print("Validation r2:", round(r2_score(y_test, val_preds_xgb_reg),4))

# Producing a dataframe of feature importances
ft_weights_xgb_reg = pd.DataFrame(xgb_reg.feature_importances_, columns=['weight'], index=X_train.columns)
ft_weights_xgb_reg.sort_values('weight', inplace=True)

# Plotting feature importances
plt.figure(figsize=(8,20))
plt.barh(ft_weights_xgb_reg.index, ft_weights_xgb_reg.weight, align='center') 
plt.title("Feature importances in the XGBoost model", fontsize=14)
plt.xlabel("Feature importance")
plt.margins(y=0.01)
plt.show()

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import numpy as np

features = ft_weights_xgb_reg.sort_values(by=['weight'], ascending=False).index[:20].tolist()
print(features)

from xgboost.sklearn import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

other_params = {'eta': 0.3, 'n_estimators': 500, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 1,
                'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
                'seed': 33}
cv_params = {'n_estimators': np.linspace(150, 250, 11, dtype=int)}
regress_model = xgb.XGBRegressor(**other_params)  # 注意这里的两个 * 号！
gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=5, n_jobs=-1)
gs.fit(X_train[features], y_train)  # X为训练数据的特征值，y为训练数据的label
# 性能测评
print("参数的最佳取值：:", gs.best_params_)
print("最佳模型得分:", gs.best_score_)

param = {
  # 'max_depth':range(9,15,2),
  # 'min_child_weight':range(5,7,1)
  'gamma':[i/10.0 for i in range(0,5)],
  #'subsample':[i/10.0 for i in range(6,10)],
  #'colsample_bytree':[i/10.0 for i in range(6,10)],
  #'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

xgb = XGBRegressor(n_estimators=150, max_depth=9, min_child_weight=6)
gsearch = GridSearchCV(estimator = xgb, param_grid = param, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=1)
gsearch.fit(X, y)

gsearch.best_params_, gsearch.best_score_

# Evalaute Fine-Tuned XGBoostRegressor on Analysis Data

from xgboost.sklearn import XGBRegressor
xgb = XGBRegressor(n_estimators=150, max_depth=9, min_child_weight=6, gamma=0, colsample_bytree=0.8, subsample=0.9, reg_alpha=0.1)

xgb.fit(X_train, y_train)
training_preds_xgb = xgb.predict(X_train)
val_preds_xgb = xgb.predict(X_test)

print("Validation RMSE:", round(mean_squared_error(y_test, val_preds_xgb, squared=False),4))
print("Validation r2:", round(r2_score(y_test, val_preds_xgb),4))

##SVR 另一种方式
pipe = Pipeline(steps=[("SC", StandardScaler()), ("svr", SVR())])

param_grid = {
    "svr__kernel": ('linear', 'poly', 'rbf', 'sigmoid'),
    "svr__C": [1,5,10],
    "svr__degree": [3,8],
    "svr__coef0": [0.01,0.5,10],
    "svr__gamma": ('auto', 'scale')
}

regr = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=5, verbose=True)
regr.fit(X[features], y)

from sklearn.metrics import mean_squared_error

regr = make_pipeline(StandardScaler(), SVR(C=10))
regr.fit(X[features], y)

y_pred = regr.predict(X_test[features])
mean_squared_error(y_test, y_pred, squared=False)

"""# Generate Submission File"""

# Read Scoring Data
scoringData = pd.read_csv('/content/score3.csv')

# Pre-process Data
scoringData = scoringData.select_dtypes(['number'])
scoringData.fillna(scoringData.mean(), inplace=True)

# Using SVR
scoringData['price'] = regr.predict(scoringData[features])

# Using XGBoost
#xgb = XGBRegressor(n_estimators=1000, max_depth=9, min_child_weight=1, gamma=0, colsample_bytree=0.8, subsample=0.8, reg_alpha=1e-2)
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

scoringData = pd.read_csv(scoringDataPath)
pre_process(scoringData)
scoringData = scoringData.select_dtypes(include='number')
scoringData.fillna(0, inplace=True)

other_params = {'eta': 0.36, 'n_estimators': 150, 'gamma': 0, 'max_depth': 9, 'min_child_weight': 6,
                'colsample_bytree': 0.6, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 38, 'reg_alpha': 0,
                'seed': 33}
xgb = xgb.XGBRegressor(**other_params)
# xgb = XGBRegressor(n_estimators=150, max_depth=9, min_child_weight=6, gamma=0, colsample_bytree=0.8, subsample=0.9, reg_alpha=0.01)
X = analysisData.loc[:, analysisData.columns != 'price']
y = analysisData['price']

xgb.fit(X, y)
scoringData['price'] = xgb.predict(scoringData.loc[:, scoringData.columns != 'id'])

scoringData = scoringData[['id', 'price']]
scoringData.columns = ['id', 'price']
scoringData.head()

scoringData.to_csv('/content/submission2.1.csv', index=False)

# copy
#Read Scoring Data
scoringData = pd.read_csv('/content/score3.csv')

# Pre-process Data
scoringData = scoringData.select_dtypes(['number'])
scoringData.fillna(scoringData.mean(), inplace=True)

# Using XGBoost
xgb = XGBRegressor(eta=0.36,n_estimators=89, max_depth=5, min_child_weight=9, gamma=0, colsample_bytree=0.6, colsample_bylevel=1, subsample=1,reg_lambda=38,reg_alpha=0,
                seed=33)
X = analysisData.loc[:, analysisData.columns != 'price']
y = analysisData['price']

xgb.fit(X, y)
scoringData['price'] = xgb.predict(scoringData.loc[:, scoringData.columns != 'id'])

scoringData = scoringData[['id', 'price']]
scoringData.columns = ['id', 'price']
scoringData.head()

scoringData.to_csv('/content/submission5.2.csv', index=False)
