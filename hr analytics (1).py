#!/usr/bin/env python
# coding: utf-8

# In[377]:


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
import warnings
warnings.filterwarnings('ignore')


# In[378]:


#READING
train=pd.read_csv('D:/R/hr analys/train.csv')
test=pd.read_csv('D:/R/hr analys/test.csv')


# In[379]:


# BINDING
master=pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,master.shape)
master.head()


# In[380]:


### Chck Dtypes
master.info()


# In[358]:


# Check column names
print(master.columns)


# In[381]:


# check na
master.isnull().sum()/len(master)*100


# In[382]:


# CHECK UNIQUE VALUES

master.apply(lambda x : len(x.unique()))


# In[383]:


##SEPERATION##
cat=['education','gender','recruitment_channel','department','previous_year_rating','KPIs_met80','awards_won','no_of_trainings']
num=['age','avg_training_score','length_of_service']
final=master[cat+num]


# In[384]:


#CAT VARIABLE
fig, ax=plt.subplots(3,2,figsize=(20,20))
for variable,subplot in zip(cat,ax.flatten()):
    sns.countplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[385]:


#NUM VARIABLE

fig, ax=plt.subplots(3,figsize=(20,20))
for variable,subplot in zip(num,ax.flatten()):
    sns.countplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[386]:


# VISUALISING BOXPLOT
fig, ax=plt.subplots(3,figsize=(10,15))
for variable,subplot in zip(num,ax.flatten()):
    sns.boxplot(final[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[387]:



# Category vs Target
fig,axes = plt.subplots(4,2,figsize=(20,20))
for idx,cat_col in enumerate(cat):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=master,hue='is_promoted',ax=axes[row,col])


plt.subplots_adjust(hspace=1)


# In[388]:


#CATEGORY VS NUMERIC vs target
g=sns.FacetGrid(master,hue="is_promoted",col='gender',size=7)
g.map(sns.distplot,"age").add_legend()
plt.show()

g=sns.FacetGrid(master,hue="is_promoted",col='gender',size=7)
g.map(sns.distplot,"avg_training_score").add_legend()
plt.show()

g=sns.FacetGrid(master,hue="is_promoted",col='gender',size=7)
g.map(sns.distplot,"length_of_service").add_legend()
plt.show()


# In[389]:


# MISSING IMPUTATIONS##

#1.EDUCATION
mis=master[master.education.isna()]
mis
master.education=master.education.fillna('Primary')
master.education.value_counts()


# In[390]:


#2.PREVIOUS YEAR RATING
misd=master[master.previous_year_rating.isna()]
misd
master.previous_year_rating=master.previous_year_rating.fillna(0)
master.previous_year_rating.value_counts()


# In[391]:


###########OUTLIER TREATMENTS#############

#AGE
sorted(master.age)
quantile1,quantile3=np.percentile(master.age,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)

# TREATMENT
master.age.loc[master.age > up]=up
sns.boxplot(master['age'])


# In[392]:


# length of esrvice

sorted(master['length_of_service'])
quantile1,quantile3=np.percentile(master.length_of_service,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)

# TREATMENT
master.length_of_service.loc[master.length_of_service > up]=up
sns.boxplot(master['length_of_service'])


# In[393]:


#### FEATURE ENGINEERING#########
master['tot_traning']=master.no_of_trainings*master.avg_training_score
master['performance']=master.awards_won + master.KPIs_met80
master['starts_at']=master.age-master.length_of_service
master['work_frac']=master.length_of_service / master.age


# In[394]:


###MAPPING#####3
#1.Gender
g={'f':0,'m':1}
master.gender=master.gender.map(g)
master.gender=master.gender.astype(int)
master.gender.value_counts()


# In[395]:


e={'Sales & Marketing':0,'Operations':1,'Procurement':2,'Technology':3,'Analytics':4,'Finance':5,'HR':6,'Legal':7,'R&D':8}
master.department=master.department.map(e)
master.department=master.department.astype(int)
master.department.value_counts()


# In[396]:


# recruitment channel
d={'other':0,'sourcing':1,'referred':2}
master.recruitment_channel=master.recruitment_channel.map(d)
master.recruitment_channel=master.recruitment_channel.astype(int)
master.recruitment_channel.value_counts()


# In[397]:


master.info()

contvars=master[['work_frac','age','starts_at','avg_training_score','tot_traning','length_of_service','performance','awards_won','previous_year_rating']]
#####CORRELATION MATRIX######
#correlation matrix
corrmat = contvars.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,annot = True,square=True);


# In[398]:


#REGION
master.region=master.region.str.extract('(\d+)')
master.region=master.region.astype(int)
master.head()


# In[399]:


##########MODEL BUILDING###########

master= pd.get_dummies(master, columns=cat)
master.head()

X = master[master['is_promoted'].isnull()!=True].drop(['employee_id','is_promoted'], axis=1)
y = master[master['is_promoted'].isnull()!=True]['is_promoted']

X_test = master[master['is_promoted'].isnull()==True].drop(['employee_id','is_promoted'], axis=1)

X.shape, y.shape, X_test.shape


# In[400]:


###########SMOTE############3
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)


# In[401]:


seed=0
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics

X=X_smote
y=y_smote
X_trains, X_vals, y_trains, y_vals = train_test_split(X,y, test_size = 0.2, random_state =1)


###SCALING##

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_trains = scaler.fit_transform(X_trains)
X_vals = scaler.transform(X_vals)


# In[402]:



#declare the models
lr = LogisticRegression()
rf=RandomForestClassifier()
adb=ensemble.AdaBoostClassifier()
bgc=ensemble.BaggingClassifier()
gnb = GaussianNB()
knn=KNeighborsClassifier()
dt = DecisionTreeClassifier()
# ab_rf = AdaBoostClassifier(base_estimator=rf,random_state=0)
# ab_dt = AdaBoostClassifier(base_estimator=dt,random_state=0)
# ab_nb=  AdaBoostClassifier(base_estimator=gnb,random_state=0)
# ab_lr=  AdaBoostClassifier(base_estimator=lr,random_state=0)
bgcl_lr = BaggingClassifier(base_estimator=lr, random_state=0)

# ,ab_rf,ab_dt,ab_nb,ab_lr,bgcl_lr

models=[lr,rf,adb,bgc,gnb,knn,dt,bgcl_lr]
sctr,scte,auc,ps,rs=[],[],[],[],[]
def ens(X_trains,X_vals, y_trains, y_vals):
    for model in models:
            model.fit(X_trains, y_trains)
            y_test_pred = model.predict(X_vals)
            y_test_pred_new=model.predict_proba(X_vals)
            y_test_pred_new=y_test_pred_new[:,1]
            train_score=model.score(X_trains,y_trains)
            test_score=model.score(X_vals,y_vals)
            p_score=metrics.precision_score(y_vals,y_test_pred)
            r_score=metrics.recall_score(y_vals,y_test_pred)
            
            ac=metrics.roc_auc_score(y_vals,y_test_pred_new)
            
            sctr.append(train_score)
            scte.append(test_score)
            ps.append(p_score)
            rs.append(r_score)
            auc.append(ac)
    return sctr,scte,auc,ps,rs
ens(X_trains,X_vals, y_trains, y_vals)
# 'ab_rf','ab_dt','ab_nb','ab_lr','bgcl_lr'
ensemble=pd.DataFrame({'names':['Logistic Regression','Random Forest','Ada boost','Bagging',
                                'Naive-Bayes','KNN','Decistion Tree',
                                'bagged LR'],
                       'auc_score':auc,'training':sctr,'testing':scte,'precision':ps,'recall':rs})
ensemble=ensemble.sort_values(by='auc_score',ascending=False).reset_index(drop=True)
ensemble


# In[403]:


### RANDOM FOREST###
random_forest = RandomForestClassifier(n_estimators=1000, oob_score=True)
random_forest.fit(X_trains, y_trains)

Y_prediction = random_forest.predict(X_vals)

random_forest.score(X_trains, y_trains)
acc_random_forest = round(random_forest.score(X_trains, y_trains) * 100, 2)
print(acc_random_forest)
print(classification_report(y_vals,Y_prediction))
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# In[422]:


col_sorted_by_importance=random_forest.feature_importances_.argsort()
feat_imp=pd.DataFrame({
    'cols':X.columns[col_sorted_by_importance],
    'imps':random_forest.feature_importances_[col_sorted_by_importance]
})

import plotly_express as px
px.bar(feat_imp, x='cols', y='imps')


# In[412]:


##########   PREDICTIONS#####
prediction = random_forest.predict(X_test)
submission = pd.DataFrame()
submission['employee_id'] = master[master['is_promoted'].isnull()==True]['employee_id']
submission['is_promoted'] = prediction
submission.to_csv('rt.csv', index=False, header=True)
submission.shape


# In[411]:


## TUNNING###
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

