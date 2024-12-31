import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.model_selection
pd.options.display.max_columns = None
pd.options.display.max_rows = 200
from datetime import datetime
from datetime import timedelta
import sklearn
import xgboost as xgb
import os
os.chdir('E:/Data_Science_Study/' + \
         'Hands-On Gradient Boosting with XGBoost and scikit-learn')


# ----------------------------------------------------------------------------
# read the raw dataset 1
# ----------------------------------------------------------------------------
df_bikes = pd.read_csv('dataset_1_bike_rentals/bike_rentals.csv')
df_bikes.dtypes

# convert dteday to datetime
df_bikes['dteday'] = pd.to_datetime(df_bikes['dteday'])

# checking columns with missing values
df_bikes.isnull().sum()
df_bikes.loc[df_bikes.isnull().any(axis=1)]

# impute windspeed with median
df_bikes['windspeed'] = (df_bikes['windspeed']
                         .fillna(df_bikes['windspeed'].median()))

# impute hum with median group by season
df_bikes['hum'] = (df_bikes['hum']
                   .fillna(df_bikes
                           .groupby('season')['hum']
                           .transform('median')))

# impute temp and atemp with 
# the average values of the day before and the day after
df_bikes = df_bikes.sort_values('dteday')
df_bikes.loc[df_bikes['temp'].isnull()]
df_bikes.loc[701, 'temp'] = (
    (df_bikes.loc[700, 'temp'] + df_bikes.loc[702, 'temp']) / 2
)

df_bikes.loc[df_bikes['atemp'].isnull()]
df_bikes.loc[701, 'atemp'] = (
    (df_bikes.loc[700, 'atemp'] + df_bikes.loc[702, 'atemp']) / 2
)

# redefine mth and yr
df_bikes['mnth'] = df_bikes['dteday'].dt.month

# Change row 730, column 'yr' to 1.0
df_bikes.loc[730, 'yr'] = 1.0

# drop non-numerical columns
df_bikes = df_bikes.drop(['dteday'], axis=1)

# drop'casual' and 'registered'
df_bikes = df_bikes.drop(['casual', 'registered'], axis=1)


# ----------------------------------------------------------------------------
# saving the cleaned data
# ----------------------------------------------------------------------------
df_bikes.to_csv('dataset_1_bike_rentals/bike_rentals_cleaned.csv')


# ----------------------------------------------------------------------------
# prepare for the models
# ----------------------------------------------------------------------------
# define X and y
X = df_bikes.drop(['cnt'], axis=1)
y = df_bikes['cnt']

# train_test_split
X_train, X_test = sklearn.model_selection.train_test_split(X, random_state=2)
y_train, y_test = sklearn.model_selection.train_test_split(y, random_state=2)

# silence warning
import warnings
warnings.filterwarnings('ignore')

# define a dictionary to keep the models' metrics
model_metrics = {}


# ----------------------------------------------------------------------------
# build a Linear Regression model
# ----------------------------------------------------------------------------
model_metrics['lin_reg'] = {}

# build the model
lin_reg = sklearn.linear_model.LinearRegression()

# fit the model
lin_reg.fit(X_train, y_train)

# predict with the model
y_pred = lin_reg.predict(X_test)

# meausre the model performance
rmse = sklearn.metrics.root_mean_squared_error(y_test, y_pred)
print('The root mean squared error of the model is {:.2f}'.format(rmse))

model_metrics['lin_reg']['rmse'] = rmse


# ----------------------------------------------------------------------------
# build an XGBoost model
# ----------------------------------------------------------------------------
model_metrics['xg_reg'] = {}

# build the model
xg_reg = xgb.XGBRegressor()

# fit the model
xg_reg.fit(X_train, y_train)

# predict with the model
y_pred = xg_reg.predict(X_test)

# meausre the model performance
rmse = sklearn.metrics.root_mean_squared_error(y_test, y_pred)
print('The root mean squared error of the model is {:.2f}'.format(rmse))

model_metrics['xg_reg']['rmse'] = rmse


# ----------------------------------------------------------------------------
# cross validation with linear regression
# ----------------------------------------------------------------------------
model_metrics['lin_reg_cv'] = {}

# build the model
model = sklearn.linear_model.LinearRegression()

# cross-validation
scores = sklearn.model_selection.cross_val_score(
    model, X, y, scoring='neg_mean_squared_error', cv=20)

rmse = np.power(-scores, 1/2)

avg_rmse = rmse.mean()
print('The root mean squared error of the model is {:.2f}'.format(avg_rmse))

model_metrics['lin_reg_cv']['rmse'] = avg_rmse


# ----------------------------------------------------------------------------
# cross validation with XGBoost regression
# ----------------------------------------------------------------------------
model_metrics['xg_reg_cv'] = {}

# build the model
model = xgb.XGBRegressor(objective="reg:squarederror")

# cross-validation
scores = sklearn.model_selection.cross_val_score(
    model, X, y, scoring='neg_mean_squared_error', cv=20)

rmse = np.power(-scores, 1/2)

avg_rmse = rmse.mean()
print('The root mean squared error of the model is {:.2f}'.format(avg_rmse))

model_metrics['xg_reg_cv']['rmse'] = avg_rmse










# ----------------------------------------------------------------------------

# now we are going to see whether XGBoost Classifier works
# for regression case

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# read the raw dataset 2
# ----------------------------------------------------------------------------
df_census = pd.read_csv('dataset_2_census/adult.data',
                        header=None)
df_census.columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    'marital-status', 'occupation', 'relationship', 'race', 'sex', 
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 
    'income'
]

df_census.dtypes


# deal with null
df_census.isnull().sum()


# drop education since education-num is available
df_census = df_census.drop(['education'], axis=1)


# creat dummy variables with one-hot encoder
df_census = pd.get_dummies(df_census, 
                        #    drop_first=True
                           )


# define X and y
X = df_census.drop(['income_ <=50K', 'income_ >50K'], axis=1)
y = df_census['income_ >50K']


# ----------------------------------------------------------------------------
# Logistic Regression vs XGB Classifier
# ----------------------------------------------------------------------------
def perf_metrics(model, cv):
    model = model
    scores = sklearn.model_selection.cross_val_score(model, X, y, cv=cv)
    print('Accuracy:', np.round(scores, 2))
    print('Accuracy mean is {:.2f}'.format(scores.mean()))

perf_metrics(sklearn.linear_model.LogisticRegression(), 10)

perf_metrics(xgb.XGBClassifier(), 10)


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------























# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------























# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------























# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------























# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------























# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------























# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------


























