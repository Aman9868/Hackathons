#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##LIbrary###
#LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,RandomForestRegressor,BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import sklearn.metrics as metrics
from sklearn.metrics import r2_score,roc_auc_score,classification_report,mean_squared_error,accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


##########TUNNING PARAMTERES FOR REGRESSSION MODELS USING RANDOMISED SEARCH CV ########

#1.LIGHT GBM

lgbm = LGBMRegressor()
lgb_grid = {
    'n_estimators': [100, 200, 400, 500],
    'colsample_bytree': [0.9, 1.0],
    'max_depth': [5,10,15,20,25,35,None],
    'num_leaves': [20, 30, 50, 100],
    'reg_alpha': [1.0, 1.1, 1.2, 1.3],
    'reg_lambda': [1.0, 1.1, 1.2, 1.3],
    'min_split_gain': [0.2, 0.3, 0.4],
    'subsample': [0.8, 0.9, 1.0],
    'learning_rate': [0.05, 0.1]
}

search = RandomizedSearchCV(lgbm,lgb_grid,scoring='neg_mean_squared_error',cv=3, verbose=2, n_jobs=6, n_iter = 100)
search.fit(X,y)

print(search.best_params_)
print(search.best_estimator_)
print(search.cv_results_)
print(search.best_score_)


# In[ ]:


#2.RANDOMFOREST

rf = RandomForestRegressor()

grid = {'n_estimators' : [100,200,500,800,1000,1200],
           'max_depth' : [3,5,7,10,15,25,40,None],
           'min_samples_split':[2,4,6,10],
           'min_samples_leaf':[2,4,6,8]   
           }

search = RandomizedSearchCV(rf,grid,scoring='neg_mean_squared_error',cv=3, verbose=2, n_jobs=6, n_iter = 50)
search.fit(X,y)


print(search.best_params_)
print(search.best_estimator_)
print(search.cv_results_)
print(search.best_score_)


# In[ ]:


#3.SVM
from sklearn.svm import SVR
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','poly','linear']}  
  

sv = SVR()

    
grid = GridSearchCV(sv,param_grid,
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           n_jobs = -1)
grid.fit(X_train, y_train)
grid.best_params_
print(grid.best_params_)  


# In[ ]:


#4.GBM

gbmr = GradientBoostingRegressor()
gb_grid = {
    'n_estimators'     : range(100,1000,100),
    'max_depth'        : [5,10,15,20,25,35,None],
    'loss'             :['ls','lad','huber','quantile'],
    'subsample'        : [0.8, 0.9, 1.0],
    'min_samples_leaf' : [1,2,5,10],
    'min_samples_split': [2,5,10,15,100],
    'learning_rate'    : [0.1,0.03,0.4,0.5,0.7]

}

search = RandomizedSearchCV(gbmr,gb_grid,scoring='neg_mean_squared_error',cv=3, verbose=2, n_jobs=-1, n_iter = 100)
search.fit(x_train,y_train)
print(search.best_params_)


# In[ ]:


#5.XG BOOST

## Hyper Parameter Optimization

booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)

random_cv.fit(x_train,y_train)

