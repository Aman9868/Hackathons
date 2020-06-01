#!/usr/bin/env python
# coding: utf-8

# In[2]:


##SVM####
# defining parameter range SVM TUNNING

from sklearn.svm import SVC
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','poly','linear']}  
  

classifier = SVC()

    
grid = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid.fit(X_train, y_train)
grid.best_params_


# In[3]:


###XGB CLASSIFIER### using RANDOMISED SEARCH CV
import xgboost
## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='accuracy',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_trains,y_trains)
random_search.best_estimator_


# In[9]:


#### Random Forest##

#####FINDING BEST PARAMETER######
# Create the parameter grid based on the results of random search 
param_grid = {
    "criterion" : ["gini", "entropy"],
    'bootstrap': [True],
    'max_depth': [5,8,15,25,30],
    'max_features': [2,3],
    'min_samples_leaf': [1,2,5,10],
    'min_samples_split': [2,5,10,15,100],
    'n_estimators': [100,300,500,800,1200]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 3)

# Fit the grid search to the data
grid_search.fit(X_trains, y_trains)
grid_search.best_params_

