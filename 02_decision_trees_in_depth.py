import numpy as np
import pandas as pd
import sklearn.model_selection
pd.options.display.max_rows = 300
pd.options.display.max_columns = None
import sklearn
import xgboost
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir('E:/Data_Science_Study/' + 
         'Hands-On Gradient Boosting with XGBoost and scikit-learn/')


# ----------------------------------------------------------------------------
# read the data generated in chapter 1 dataset 2
# ----------------------------------------------------------------------------
df_census = pd.read_csv('dataset_2_census/census_cleaned.csv')
df_census
df_census.dtypes


# ----------------------------------------------------------------------------
# define X and y
# ----------------------------------------------------------------------------
X = df_census.drop(['income_ >50K'], axis=1)
y = df_census['income_ >50K']

X_train, X_test = sklearn.model_selection.train_test_split(X, random_state=2)
y_train, y_test = sklearn.model_selection.train_test_split(y, random_state=2)


# ----------------------------------------------------------------------------
# build the model
# ----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(random_state=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
print('The accuracy of the model is {:.2%}'.format(accuracy))


# ----------------------------------------------------------------------------
# tree visualization
# ----------------------------------------------------------------------------
from sklearn.tree import export_graphviz
import graphviz
os.environ['PATH'] += os.pathsep + 'C:/Program Files/Graphviz/bin'

dot_data = export_graphviz(clf,
                           max_depth=2,
                           feature_names=X.columns,
                           class_names=True,
                           filled=True,
                           rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)

graph.render('02_tree_1', format='png', cleanup=True)






# ----------------------------------------------------------------------------

# now we are going to tune hyper-parameters

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# read cleaned bike rentals data generated in chapter 1
# ----------------------------------------------------------------------------
df_bikes = pd.read_csv('dataset_1_bike_rentals/bike_rentals_cleaned.csv')
df_bikes.dtypes
df_bikes

X_bikes = df_bikes.drop(['cnt'], axis=1)
y_bikes = df_bikes['cnt']

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X_bikes, random_state=2)
y_train, y_test = train_test_split(y_bikes, random_state=2)


# ----------------------------------------------------------------------------
# build a decition tree model
# and find the max_depth that gives the lowest RMSE
# with GridSerchCV
# ----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

estimator = DecisionTreeRegressor(random_state=2)
param_grid = {'max_depth': [None, 2, 3, 4, 6, 8, 10, 20]}
grid_reg = GridSearchCV(estimator=estimator, param_grid=param_grid, 
                        scoring='neg_root_mean_squared_error',
                        cv=5, n_jobs=-1)

grid_reg.fit(X_train, y_train)

best_params = grid_reg.best_params_
print('Best params:', best_params)
print('RMSE of the model (training) is {:.2f}'.format(-grid_reg.best_score_))


best_model = grid_reg.best_estimator_
y_pred = best_model.predict(X_test)
RMSE_test = np.power(mean_squared_error(y_pred, y_test), 1/2)
print('RMSE of the model (testing) is {:.2f}'.format(RMSE_test))


# ----------------------------------------------------------------------------
# define a grid_search function
# ----------------------------------------------------------------------------
def grid_search(estimator, param_grid, cv, n_jobs,
                X_train, y_train,
                X_test, y_test):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import root_mean_squared_error

    grid_reg = GridSearchCV(estimator=estimator, param_grid=param_grid, 
                            cv=cv, n_jobs=n_jobs,
                            scoring='neg_root_mean_squared_error')
    grid_reg.fit(X_train, y_train)

    best_params = grid_reg.best_params_
    print('The best prarms are', best_params)

    rmse_train = -grid_reg.best_score_
    print('The RMSE of the model (training) is {:.2f}'.format(rmse_train))

    best_reg = grid_reg.best_estimator_
    y_pred = best_reg.predict(X_test)
    rmse_test = root_mean_squared_error(y_pred, y_test)
    print('The RMSE of the model (testing) is {:.2f}'.format(rmse_test))


grid_search(estimator=DecisionTreeRegressor(random_state=2), 
            param_grid={'min_samples_leaf': [1, 2, 4, 6, 8, 10, 20, 30]},
            cv=5, n_jobs=-1,
            X_train=X_train, y_train=y_train, 
            X_test=X_test, y_test=y_test)


grid_search(estimator=DecisionTreeRegressor(random_state=2), 
            param_grid={'max_depth': [None, 2, 3, 4, 6, 8, 10, 20],
                        'min_samples_leaf': [1, 2, 4, 6, 8, 10, 20, 30]},
            cv=5, n_jobs=-1,
            X_train=X_train, y_train=y_train, 
            X_test=X_test, y_test=y_test)


grid_search(estimator=DecisionTreeRegressor(random_state=2), 
            param_grid={'max_depth': [6, 7, 8, 9, 10],
                        'min_samples_leaf': [3, 5, 7, 9]},
            cv=5, n_jobs=-1,
            X_train=X_train, y_train=y_train, 
            X_test=X_test, y_test=y_test)







# ----------------------------------------------------------------------------

# case study

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# read the data
# ----------------------------------------------------------------------------
df_heart = pd.read_csv('data_other/02_heart_disease.csv')
df_heart.dtypes
df_heart


X = df_heart.drop(['target'], axis=1)
y = df_heart['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)


# ----------------------------------------------------------------------------
# baseline decision tree classifier
# ----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

base_model = DecisionTreeClassifier(random_state=2)
scores = cross_val_score(estimator=base_model,
                         X=X,
                         y=y,
                         scoring='accuracy',
                         cv=5)
accuracy = scores.mean()

print('Accuracy of the model is {:.2%}'.format(accuracy))


# ----------------------------------------------------------------------------
# tune hyper-parameters with random search
# ----------------------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score


def rand_search_cv(estimator, param_distributions, n_iter, random_state,
                   cv, n_jobs):
    rand_clf = RandomizedSearchCV(estimator=estimator, 
                                  param_distributions=param_distributions,
                                  scoring='accuracy',
                                  n_iter=n_iter, 
                                  random_state=random_state,
                                  cv=cv,
                                  n_jobs=n_jobs)
    rand_clf.fit(X_train, y_train)

    best_model = rand_clf.best_estimator_
    print('The best model is', best_model)

    best_params = rand_clf.best_params_
    print('The best params is', best_params)

    accuracy_train = rand_clf.best_score_
    print('The accuracy of the model (train) is {:.2%}'.format(accuracy_train))

    y_pred = best_model.predict(X_test)
    accuracy_test = accuracy_score(y_pred, y_test)
    print('The accuracy of the model (test) is {:.2%}'.format(accuracy_test))


from sklearn.tree import DecisionTreeClassifier
rand_search_cv(estimator=DecisionTreeClassifier(random_state=2),
               param_distributions={
    'criterion': ['entropy', 'gini'],
    'splitter': ['random', 'best'],
    'min_samples_split': [2, 3, 4, 5, 6, 8, 10],
    'min_samples_leaf': [1, 0.01, 0.02, 0.03, 0.04],
    'min_impurity_decrease': [0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
    'max_leaf_nodes': [10, 15, 20, 25, 30, 35, 40, 45, 50, None],
    'max_features': ['auto', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
    'max_depth': [None, 2,4,6,8],
    'min_weight_fraction_leaf': [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05]
    },
               n_iter=20,
               random_state=2,
               cv=5,
               n_jobs=-1
                )


# ----------------------------------------------------------------------------
# narrow the range based on previous results
# ----------------------------------------------------------------------------
rand_search_cv(estimator=DecisionTreeClassifier(random_state=2),
               param_distributions={
    'criterion': ['entropy', 'gini'],
    'splitter': ['random', 'best'],
    'min_weight_fraction_leaf': [0.04, 0.05, 0.06],
    'min_samples_split': [8, 9, 10, 11, 12],
    'min_samples_leaf': [0.03, 0.04, 0.05],
    'min_impurity_decrease': [0.0, 0.0005, 0.001],
    'max_leaf_nodes': [35, 40, 45, 50, None],
    'max_features': ['auto', 0.85, 0.80, 0.75],
    'max_depth': [None, 6, 7, 8, 9],
    'min_weight_fraction_leaf': [0.04, 0.05, 0.06],
    },
               n_iter=100,
               random_state=2,
               cv=5,
               n_jobs=-1
                )


# ----------------------------------------------------------------------------
# confirm the model and measure the performance with cross-validation
# ----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


model = DecisionTreeClassifier(
                                splitter='best', 
                                min_weight_fraction_leaf=0.06, 
                                min_samples_split=10, 
                                min_samples_leaf=0.03, 
                                min_impurity_decrease=0.0, 
                                max_leaf_nodes=45, 
                                max_features=0.8, 
                                max_depth=None, 
                                criterion='gini'    
                                )


scores = cross_val_score(estimator=model, X=X_train, y=y_train,
                         scoring='accuracy', cv=5)

accuracy_mean = scores.mean()
print('The accuracy of the model (train) is {:.2%}'.format(accuracy_mean))


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_mean_test = accuracy_score(y_pred, y_test)
print('The accuracy of the model (test) is {:.2%}'.format(accuracy_mean_test))



# ----------------------------------------------------------------------------
# feature importances
# ----------------------------------------------------------------------------
model.feature_importances_

dict_feature = dict(zip(X.columns, model.feature_importances_))

dict_feature_sorted = {k: v for k, v in 
                       sorted(dict_feature.items(), 
                              key=lambda item: item[1],
                              reverse=True)}

top_3_feature = list(dict_feature_sorted.items())[:3]
print('The top 3 important features are', top_3_feature)

