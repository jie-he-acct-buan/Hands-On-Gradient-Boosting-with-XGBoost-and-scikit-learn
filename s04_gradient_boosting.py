import numpy as np
import pandas as pd
pd.options.display.max_rows = 300
pd.options.display.max_columns = None

import os
os.chdir('E:/Data_Science_Study/' +
         'Hands-On Gradient Boosting with XGBoost and scikit-learn/')

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error


# ----------------------------------------------------------------------------
# read the raw data
# ----------------------------------------------------------------------------
df_bike = pd.read_csv('dataset_1_bike_rentals/bike_rentals_cleaned.csv')

X_bike = df_bike.drop(['cnt'], axis=1)
y_bike = df_bike['cnt']

X_train, X_test, y_train, y_test = train_test_split(X_bike, y_bike, 
                                                    random_state=2)


# ----------------------------------------------------------------------------
# building a gradient boosting model from scratch
# ----------------------------------------------------------------------------
# the 1st tree
tree_1 = DecisionTreeRegressor(max_depth=2, random_state=2)
y1_train = y_train

tree_1.fit(X_train, y1_train)
y1_pred = tree_1.predict(X_train)

# the 1st tree's residual is the target for the 2nd tree
y2_train = y1_train - y1_pred

# the 2nd tree
tree_2 = DecisionTreeRegressor(max_depth=2, random_state=2)
tree_2.fit(X_train, y2_train)
y2_pred = tree_2.predict(X_train)
y3_train = y2_train - y2_pred


# the 3rd tree
tree_3 = DecisionTreeRegressor(max_depth=2, random_state=2)
tree_3.fit(X_train, y3_train)
y3_pred = tree_3.predict(X_train)
y4_train = y3_train - y3_pred


# sum the result
y_pred = tree_1.predict(X_test) \
        + tree_2.predict(X_test) \
        + tree_3.predict(X_test)


# measure the RMSE
rmse = root_mean_squared_error(y_pred, y_test)
print('The RMSE of the model (test) is {:.2f}.'.format(rmse))


# ----------------------------------------------------------------------------
# function for building my own gradient boosting regression
# ----------------------------------------------------------------------------
def my_gradient_boosting(n_estimator, X_train, y_train):
    
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import root_mean_squared_error

    residual_lst = [y_train]
    tree_pred_sum = 0

    for i in range(n_estimator):
        tree = DecisionTreeRegressor(max_depth=2, random_state=2)
        tree.fit(X_train, residual_lst[i])

        residual = tree.predict(X_train) - residual_lst[i]
        residual_lst.append(residual)

        tree_pred_sum += tree.predict(X_test)

    rmse = root_mean_squared_error(y_pred, y_test)
    print('The RMSE of the model (test) is {:.2f}.'.format(rmse))
    
    return rmse


my_gradient_boosting(3, X_train, y_train)


# ----------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------










