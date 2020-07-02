#!/usr/bin/env python
# coding: utf-8

# In[34]:



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
from sklearn.metrics import r2_score,roc_auc_score,classification_report,mean_squared_error,accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
import warnings
warnings.filterwarnings('ignore')


# In[35]:


#READING
train=pd.read_csv('D:/R/Banking Av/train.csv')
test=pd.read_csv('D:/R/Banking Av/test.csv')


# In[36]:


# BINDING
master=pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,master.shape)
master.head()


# In[37]:


### Chck Dtypes
master.info()


# In[38]:


# check na
master.isnull().sum()/len(master)*100


# In[39]:


# CHECK UNIQUE VALUES

master.apply(lambda x : len(x.unique()))


# In[40]:


#SEPERATING VARIABLES
cat=['Length_Employed','Home_Owner','Income_Verified','Purpose_Of_Loan','Gender','Inquiries_Last_6Mo']
num=['Loan_Amount_Requested','Annual_Income','Debt_To_Income','Number_Open_Accounts','Total_Accounts']
final=master[cat+num]


# In[41]:


# CAT VARIABLES
fig, ax=plt.subplots(3,2,figsize=(40,40))
for variable,subplot in zip(cat,ax.flatten()):
    sns.countplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[42]:


#Numerical
fig, ax=plt.subplots(5,figsize=(20,20))
for variable,subplot in zip(num,ax.flatten()):
    sns.distplot(final[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[43]:


# VISUALISING BOXPLOT
fig, ax=plt.subplots(5,figsize=(10,30))
for variable,subplot in zip(num,ax.flatten()):
    sns.boxplot(final[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[44]:


# Category vs Target
fig,axes = plt.subplots(3,2,figsize=(20,30))
for idx,cat_col in enumerate(cat):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=master,hue='Interest_Rate',ax=axes[row,col])


plt.subplots_adjust(hspace=1)


# In[45]:


##### DATA CLEANING###

#1.Length_Employed
missing=master[master.Length_Employed.isna()]
missing.head()

master['Length_Employed']=master['Length_Employed'].replace({'10+ YEARS':'10','2 YEARS':'2','< 1 YEAR':'0','5 YEARS':'5','1 YEAR':'1','4 YEARS':'4','7 YEARS':'7','6 YEARS':'6','8 YEARS':'8','9 YEARS':'9'},inplace=True)
master['Length_Employed']=master['Length_Employed'].fillna('10')
master['Length_Employed']=master['Length_Employed'].astype(int)


# In[46]:


#2.Home oWNER
missing=master[master.Home_Owner.isna()]
missing.head()
master.Home_Owner=master.Home_Owner.fillna('Mortgage')
ge={'Mortgage':0,'Rent':1,'Own':2,'Other':3,"None":4}
master.Home_Owner=master.Home_Owner.map(ge)
master.Home_Owner.value_counts()


# In[47]:


#3.ANNUAL INCOME
master.Annual_Income.describe()
# REPLACE LOAN AMT WITH RANDOM VALUE WITH RESPECT TO MEAN STD AND ISNULL
master.Annual_Income.describe()
mean = master["Annual_Income"].mean()
std = master["Annual_Income"].std()
is_null = master["Annual_Income"].isnull().sum()
# compute random numbers between the mean, std and is_null
rand_Annual_Income = np.random.randint(mean - std, mean + std, size = is_null)
 # fill NaN values in Annual_Income column with random values generated
Annual_Income_slice = master["Annual_Income"].copy()
Annual_Income_slice[np.isnan(Annual_Income_slice)] = rand_Annual_Income
master["Annual_Income"] = Annual_Income_slice
master["Annual_Income"] = master["Annual_Income"].astype(int)

###OUTLIER IDENTIFICATION###

sns.boxplot(master['Annual_Income'])

sorted(master['Annual_Income'])
quantile1,quantile3=np.percentile(master.Annual_Income,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)


# In[48]:


# TREATMENT
master.Annual_Income.loc[master.Annual_Income > up]=up
sns.boxplot(master['Annual_Income'])


# In[49]:


#4.MONTH SINCE DELIQUENCY
missing=master[master.Months_Since_Deliquency.isna()]
missing.head()

# REPLACE LOAN AMT WITH RANDOM VALUE WITH RESPECT TO MEAN STD AND ISNULL
master.Months_Since_Deliquency.describe()
mean = master["Months_Since_Deliquency"].mean()
std = master["Months_Since_Deliquency"].std()
is_null = master["Months_Since_Deliquency"].isnull().sum()
# compute random numbers between the mean, std and is_null
rand_Months_Since_Deliquency = np.random.randint(mean - std, mean + std, size = is_null)
 # fill NaN values in Months_Since_Deliquency column with random values generated
Months_Since_Deliquency_slice = master["Months_Since_Deliquency"].copy()
Months_Since_Deliquency_slice[np.isnan(Months_Since_Deliquency_slice)] = rand_Months_Since_Deliquency
master["Months_Since_Deliquency"] = Months_Since_Deliquency_slice

###OUTLIER IDENTIFICATION###

sns.boxplot(master['Months_Since_Deliquency'])

sorted(master['Months_Since_Deliquency'])
quantile1,quantile3=np.percentile(master.Months_Since_Deliquency,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)


# In[50]:


# TREATMENT
master.Months_Since_Deliquency.loc[master.Months_Since_Deliquency > up]=up
sns.boxplot(master['Months_Since_Deliquency'])


# In[51]:


###OUTLIER IDENTIFICATION###

sns.boxplot(master['Total_Accounts'])

sorted(master['Total_Accounts'])
quantile1,quantile3=np.percentile(master.Total_Accounts,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)

# TREATMENT
master.Total_Accounts.loc[master.Total_Accounts > 55]=55
sns.boxplot(master['Total_Accounts'])


# In[52]:


# NUmber of Open Accounts

###OUTLIER IDENTIFICATION###


sorted(master['Number_Open_Accounts'])
quantile1,quantile3=np.percentile(master.Number_Open_Accounts,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)

# TREATMENT
master.Number_Open_Accounts.loc[master.Number_Open_Accounts > up]=up
sns.boxplot(master['Number_Open_Accounts'])


# In[53]:


# GENDER MAPPING
gs={'Female':0,'Male':1}
master.Gender=master.Gender.map(gs)
master.Gender=master.Gender.astype(int)
master.Gender.value_counts()


# In[54]:


# Income verified
gt={'not verified':0,'VERIFIED - income':1,'VERIFIED - income source':2}
master.Income_Verified=master.Income_Verified.map(gt)
master.Income_Verified=master.Income_Verified.astype(int)


# In[55]:


# Purpose of Loan
gr={'debt_consolidation':0,'credit_card':1,'home_improvement':2,'other':3,'major_purchase':4,'small_business':5,'car':6,'medical':7,'moving':8,'vacation':9,'wedding':10,'house':11,'renewable_energy':12,'educational':13}
master.Purpose_Of_Loan=master.Purpose_Of_Loan.map(gr)
master.Purpose_Of_Loan=master.Purpose_Of_Loan.astype(int)
master.Purpose_Of_Loan.value_counts()


# In[56]:


# CHECK TARGET VARIABLE###

miss=master[master.Interest_Rate.isna()]
miss

master.Months_Since_Deliquency=master.Months_Since_Deliquency.astype(int)


# In[57]:


#####FEATURE ENGINEERING##
master['Debt_AIncome']=master.Loan_Amount_Requested / master.Annual_Income *100
master['Active_Accts']=master.Total_Accounts - master.Number_Open_Accounts


# In[58]:


contvars=master[['Loan_Amount_Requested','Annual_Income','Debt_To_Income','Months_Since_Deliquency','Number_Open_Accounts','Active_Accts','Debt_AIncome']]
#####CORRELATION MATRIX######
#correlation matrix
corrmat = contvars.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,annot = True,square=True);


# In[59]:


##########MODEL BUILDING###########

final=master.copy()

Xs = final[final['Interest_Rate'].isnull()!=True].drop(['Loan_ID','Interest_Rate'], axis=1)
Ys = final[final['Interest_Rate'].isnull()!=True]['Interest_Rate']                              ##COPY WITHOUT DUMMY



#WITH DUMMY
master= pd.get_dummies(master, columns=cat)
master.head()

X = master[master['Interest_Rate'].isnull()!=True].drop(['Loan_ID','Interest_Rate'], axis=1)  
y = master[master['Interest_Rate'].isnull()!=True]['Interest_Rate']

X_test = master[master['Interest_Rate'].isnull()==True].drop(['Loan_ID','Interest_Rate'], axis=1)

X.shape, y.shape, X_test.shape


# In[60]:


###########SMOTE############3
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_smote, Y_smote = oversample.fit_resample(X, y) # copy

seed=0
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics

X=X_smote   # dummy
Y=Y_smote # dummy


X_trains, X_vals, y_trains, y_vals = train_test_split(X,Y, test_size = 0.2, random_state =1) # dummy


# In[61]:


##COPY SMOTE

oversample = SMOTE()
Xs_smote, Ys_smote = oversample.fit_resample(Xs, Ys) #Dummy
xs=Xs_smote  # copy
ys=Ys_smote   # copy
x_trainc, x_valc, y_trainc, y_valc = train_test_split(xs,ys, test_size = 0.2, random_state =1)   # copy


# In[62]:


### RANDOM FOREST###
random_forest = RandomForestClassifier(n_estimators=1000, oob_score=True)
random_forest.fit(x_trainc, y_trainc)

Y_prediction = random_forest.predict(x_valc)

random_forest.score(x_trainc, y_trainc)
acc_random_forest = round(random_forest.score(x_trainc, y_trainc) * 100, 2)
print(acc_random_forest)
print(classification_report(y_valc,Y_prediction))
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# In[291]:


#######

from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_trains, y_trains, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[293]:


#######R FOREST IMPORTANCE####
importances = pd.DataFrame({'feature':X_trains.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)


# In[300]:


###XGB CLASSIFIER###
import xgboost
## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[301]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[297]:


classifier=xgboost.XGBClassifier()


# In[302]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='accuracy',n_jobs=-1,cv=5,verbose=3)


# In[303]:


from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_trains,y_trains)
timer(start_time) # timing ends here for "start_time" variable


# In[305]:


random_search.best_estimator_


# In[306]:


xg=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.3,
              learning_rate=0.2, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)






xg.fit(X_trains, y_trains)
Y_prtus = xg.predict(X_vals)
xg.score(X_trains, y_trains)
from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(xg, X_trains, y_trains, cv=3)
confusion_matrix(y_trains, predictions)
print(classification_report(y_vals,Y_prtus))
acc_xg = round(xg.score(X_trains, y_trains) * 100, 2)
print(acc_xg)


# In[63]:


from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

est = RandomForestClassifier(n_jobs=-1)
rf_p_dist={'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200,300,400,500],
              'max_features':randint(1,3),
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,4),
              }
def hypertuning_rscv(est, p_distr, nbr_iter,X_trains,y_trains):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=9)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X_trains,y_trains)
    rdmsearch.best_params_
    
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score

rf_parameters, rf_ht_score = hypertuning_rscv(est, rf_p_dist, 40, X_trains, y_trains)

