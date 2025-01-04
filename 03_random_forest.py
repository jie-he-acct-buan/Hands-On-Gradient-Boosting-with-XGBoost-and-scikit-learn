import numpy as np
import pandas as pd
pd.options.display.max_rows = 300
pd.options.display.max_columns = None

import os
os.chdir('E:/Data_Science_Study/' + 
         'Hands-On Gradient Boosting with XGBoost and scikit-learn/')

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# random forest classifier
# ----------------------------------------------------------------------------
df_census = pd.read_csv('dataset_2_census/census_cleaned.csv')

X_census = df_census.drop(['income_ >50K'], axis=1)
y_census = df_census['income_ >50K']

rf_clf = RandomForestClassifier(n_estimators=10,
                                random_state=2,
                                n_jobs=-1)

accuracy = cross_val_score(estimator=rf_clf,
                         X=X_census,
                         y=y_census,
                         cv=5,
                         n_jobs=-1,
                         scoring='accuracy')

print('The accuracy of the model is {:.2%}.'.format(accuracy.mean()))


# ----------------------------------------------------------------------------
# randome forest regression
# ----------------------------------------------------------------------------
df_bike = pd.read_csv('dataset_1_bike_rentals/bike_rentals_cleaned.csv')
X_bike = df_bike.drop(['cnt'], axis=1)
y_bike = df_bike['cnt']

rf_reg = RandomForestRegressor(n_estimators=10,
                               random_state=2,
                               n_jobs=-1)

neg_rmse = cross_val_score(estimator=rf_reg,
                           X=X_bike,
                           y=y_bike,
                           cv=10,
                           n_jobs=-1,
                           scoring='neg_root_mean_squared_error')

print('The RMSE of the model is {:.2f}'.format(-neg_rmse.mean()))


# ----------------------------------------------------------------------------
# randome forest hyperparameters
# ----------------------------------------------------------------------------
# oob - directly score with unused test data
rf_clf = RandomForestClassifier(n_estimators=10,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=2)
rf_clf.fit(X_census, y_census)
rf_clf.oob_score_


# n_estimator - larger n leads to better performance
rf_clf = RandomForestClassifier(n_estimators=50,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=2)
rf_clf.fit(X_census, y_census)
rf_clf.oob_score_


# warm_start - when True adding more trees does not require starting over
rf_clf = RandomForestClassifier(n_estimators=100,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=2,
                                warm_start=True)
rf_clf.fit(X_census, y_census)
rf_clf.oob_score_


# ----------------------------------------------------------------------------
# the relation between n_estimator (number of trees) and the performance
# ----------------------------------------------------------------------------
performance = {}
for i in range(1, 11, 1):
    performance[str(i)] = {}
    rf_clf = RandomForestClassifier(n_estimators=(50 * i),
                                    oob_score=True,
                                    n_jobs=-1,
                                    random_state=2,
                                    warm_start=True,
                                    )
    
    rf_clf.fit(X_census, y_census)
    performance[str(i)]['n_estimator'] = 50 * i 
    performance[str(i)]['accuracy'] = rf_clf.oob_score_

df_perf = pd.DataFrame(performance).T.reset_index(drop=True)

plt.figure(figsize=(20, 20))
plt.plot(df_perf['n_estimator'], df_perf['accuracy'])
plt.xlabel('n_estimator')
plt.ylabel('accuracy')
plt.title('Random Forest: n_estimator_vs_accuracy')
plt.grid(visible=True)
plt.savefig('data_other/03_n_estimator_vs_accuracy.png')
plt.show()





# ----------------------------------------------------------------------------

# we use bike_rentals to learn more about Random Forest Regression

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# read the data
# ----------------------------------------------------------------------------
df_bike = pd.read_csv('dataset_1_bike_rentals/bike_rentals_cleaned.csv')
df_bike.head()
X_bike = df_bike.drop(['cnt'], axis=1)
y_bike = df_bike['cnt']

X_train, X_test, y_train, y_test = train_test_split(X_bike, y_bike, 
                                                    random_state=2)


# ----------------------------------------------------------------------------
# build the Random Forest Regression
# ----------------------------------------------------------------------------
rf_reg = RandomForestRegressor(n_estimators=50,
                               n_jobs=-1,
                               random_state=2,
                               warm_start=True)

rmse_all = cross_val_score(estimator=rf_reg,
                           X=X_bike,
                           y=y_bike,
                           cv=10,
                           scoring='neg_root_mean_squared_error')

print('The RMSE of the model (all) is {:.2f}.'.format(-rmse_all.mean()))


# ----------------------------------------------------------------------------
# tune hyperparameters
# ----------------------------------------------------------------------------
rf_reg = RandomForestRegressor(n_jobs=-1, random_state=2)

param_distributions = {
    'min_weight_fraction_leaf': [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05],
    'min_samples_split': [2, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1],
    'min_samples_leaf': [1,2,4,6,8,10,20,30],
    'min_impurity_decrease': [0.0, 0.01, 0.05, 0.10, 0.15, 0.2],
    'max_leaf_nodes': [10, 15, 20, 25, 30, 35, 40, 45, 50, None],
    'max_features': ['auto', 0.8, 0.7, 0.6, 0.5, 0.4],
    'max_depth': [None,2,4,6,8,10,20]
    }

rand_search_rf_reg = RandomizedSearchCV(
        estimator=rf_reg, 
        n_iter=16,
        param_distributions=param_distributions,
        scoring='neg_mean_squared_error',
        cv=10,
        n_jobs=-1,
        random_state=2)

rand_search_rf_reg.fit(X_train, y_train)

best_estimator = rand_search_rf_reg.best_estimator_

best_params = rand_search_rf_reg.best_params_
print('The best params is', '\n', best_params)

rmse_train = np.power(-rand_search_rf_reg.best_score_, 1/2)
print('The RMSE of the model (train) is {:.2f}'.format(rmse_train))

y_pred = best_estimator.predict(X_test)
rmse_test = root_mean_squared_error(y_pred, y_test)
print('The RMSE of the model (test) is {:.2f}'.format(rmse_test))


# ----------------------------------------------------------------------------
# convert previous tuning to a function
# ----------------------------------------------------------------------------
def rand_search_cv_rt_reg(n_iter, param_distributions):
    # define the model to be tuned
    rf_reg = RandomForestRegressor(n_jobs=-1, random_state=2)

    rand_search_rf_reg = RandomizedSearchCV(
            estimator=rf_reg, 
            n_iter=n_iter,
            param_distributions=param_distributions,
            scoring='neg_mean_squared_error',
            cv=10,
            n_jobs=-1,
            random_state=2)

    rand_search_rf_reg.fit(X_train, y_train)

    best_estimator = rand_search_rf_reg.best_estimator_

    best_params = rand_search_rf_reg.best_params_
    print('The best params is', '\n', best_params)

    rmse_train = np.power(-rand_search_rf_reg.best_score_, 1/2)
    print('The RMSE of the model (train) is {:.2f}'.format(rmse_train))

    y_pred = best_estimator.predict(X_test)
    rmse_test = root_mean_squared_error(y_pred, y_test)
    print('The RMSE of the model (test) is {:.2f}'.format(rmse_test))


param_distributions = {
    'min_weight_fraction_leaf': [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05],
    'min_samples_split': [2, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.01, 0.05, 0.10, 0.15, 0.2],
    'max_leaf_nodes': [10, 15, 20, 25, 30, 35, 40, 45, 50, None],
    'max_features': ['auto', 0.8, 0.7, 0.6, 0.5, 0.4],
    'max_depth': [None, 2, 4, 6, 8, 10, 20]
    }

rand_search_cv_rt_reg(16, param_distributions)

# ----------------------------------------------------------------------------
# narrowing the range
# ----------------------------------------------------------------------------
param_distributions = {
    'min_samples_leaf': [1, 2, 4, 6, 8, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.01, 0.05, 0.10, 0.15, 0.2],
    'max_features': ['auto', 0.8, 0.7, 0.6, 0.5, 0.4],
    'max_depth': [None, 2, 4, 6, 8, 10, 20]
    }

rand_search_cv_rt_reg(16, param_distributions)


param_distributions = {
    'min_samples_leaf': [1, 2, 3],
    'min_impurity_decrease': [0.0, 0.01, 0.05, 0.10],
    'max_features': [0.7, 0.6, 0.5],
    'max_depth': [10, 12, 14]
    }

rand_search_cv_rt_reg(16, param_distributions)


