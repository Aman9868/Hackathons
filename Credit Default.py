#!/usr/bin/env python
# coding: utf-8

# In[178]:


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
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
from lightgbm import LGBMClassifier
import xgboost
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier


# In[179]:


#READING
train=pd.read_csv('D:/R/JANTA CURFEW/train.csv')
test=pd.read_csv('D:/R/JANTA CURFEW/test.csv')


# In[180]:


# BINDING
master=pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,master.shape)
master.head()


# In[181]:


master.info()


# In[182]:


#CHECK NA
master.isnull().sum()/len(master)*100


# In[183]:


#VISUALISATION#
fig, ax=plt.subplots(3,1)
sns.countplot(x='SEX',data=master,hue='default_payment_next_month', ax=ax[0])
sns.countplot(x='EDUCATION',data=master,hue='default_payment_next_month', ax=ax[1])
sns.countplot(x='MARRIAGE',data=master,hue='default_payment_next_month', ax=ax[2])
fig.show()


# In[184]:


#1.Married AS 0 IS UNDOCUMENTED IN PROBLEM STATEMENT
master.MARRIAGE.value_counts()
master.loc[master.MARRIAGE == 0, 'MARRIAGE'] = 3

#2.Education 0,6,5 are also undocumented
fil = (master.EDUCATION == 5) | (master.EDUCATION == 6) | (master.EDUCATION == 0)
master.loc[fil, 'EDUCATION'] = 4


# In[168]:


#FIX THE UNDOCUMENTED ACCORDING TO PROBLEM STATEMENT# i.e Payed Duly

ge={'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'}
gs={-2:0,-1:0,0:0}
for col in ge:
    master[col]=master[col].replace(gs)


# In[195]:


num={'LIMIT_BAL','AGE',
       'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'}

fig, ax=plt.subplots(3,5,figsize=(15,30))
for variable,subplot in zip(num,ax.flatten()):
    sns.boxplot(master[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[196]:


#########FLOORING & CAPPING############

def cap_outliers(series, iqr_threshold=1.5, verbose=False):
    '''Caps outliers to closest existing value within threshold (IQR).'''
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lbound = Q1 - iqr_threshold * IQR
    ubound = Q3 + iqr_threshold * IQR

    outliers = (series < lbound) | (series > ubound)

    series = series.copy()
    series.loc[series < lbound] = series.loc[~outliers].min()
    series.loc[series > ubound] = series.loc[~outliers].max()

    # For comparison purposes.
    if verbose:
            print('\n'.join(
                ['Capping outliers by the IQR method:',
                 f'   IQR threshold: {iqr_threshold}',
                 f'   Lower bound: {lbound}',
                 f'   Upper bound: {ubound}\n']))

    return series

#CAPPING OULIIERS##
for i in num:
    master[i]=cap_outliers(master[i],verbose=True)


# In[197]:


####FEATURE ENGINEERING###
#To me it seems that it goes like that:
#I have a BILL of X, I pay Y
#The month after I have to pay X-Y + X', being X' my new expenses, I pay Y'
#The month after I have to pay X+X' - Y - Y' + X'' , I pay Y''
#So on so forth
#On top of that I may or may not have months of delay.
#It seems that if by september I have a bill too close to my limit, I generally fail. However, I can already see some dramatic exceptions.
#Moreover, I can spot some clients that joined our dataset at a later month: they have 0 in BILL and PAY AMT for a while and then they start. I have to keep that in mind as well.
#Now I want to see how the month of delay gets assigned. To this end, I will consider only people with no delays 6 months ago and see how their payments go.

master['Avg_exp_5'] = ((master['BILL_AMT5'] - (master['BILL_AMT6'] - master['PAY_AMT5']))) / master['LIMIT_BAL']

master['Avg_exp_4'] = (((master['BILL_AMT5'] - (master['BILL_AMT6'] - master['PAY_AMT5'])) +
                 (master['BILL_AMT4'] - (master['BILL_AMT5'] - master['PAY_AMT4']))) / 2) / master['LIMIT_BAL']

master['Avg_exp_3'] = (((master['BILL_AMT5'] - (master['BILL_AMT6'] - master['PAY_AMT5'])) +
                 (master['BILL_AMT4'] - (master['BILL_AMT5'] - master['PAY_AMT4'])) +
                 (master['BILL_AMT3'] - (master['BILL_AMT4'] - master['PAY_AMT3']))) / 3) / master['LIMIT_BAL']

master['Avg_exp_2'] = (((master['BILL_AMT5'] - (master['BILL_AMT6'] - master['PAY_AMT5'])) +
                 (master['BILL_AMT4'] - (master['BILL_AMT5'] - master['PAY_AMT4'])) +
                 (master['BILL_AMT3'] - (master['BILL_AMT4'] - master['PAY_AMT3'])) +
                 (master['BILL_AMT2'] - (master['BILL_AMT3'] - master['PAY_AMT2']))) / 4) / master['LIMIT_BAL']

master['Avg_exp_1'] = (((master['BILL_AMT5'] - (master['BILL_AMT6'] - master['PAY_AMT5'])) +
                 (master['BILL_AMT4'] - (master['BILL_AMT5'] - master['PAY_AMT4'])) +
                 (master['BILL_AMT3'] - (master['BILL_AMT4'] - master['PAY_AMT3'])) +
                 (master['BILL_AMT2'] - (master['BILL_AMT3'] - master['PAY_AMT2'])) +
                 (master['BILL_AMT1'] - (master['BILL_AMT2'] - master['PAY_AMT1']))) / 5) / master['LIMIT_BAL']


master['Closeness_6'] = (master.LIMIT_BAL - master.BILL_AMT6) / master.LIMIT_BAL
master['Closeness_5'] = (master.LIMIT_BAL - master.BILL_AMT5) / master.LIMIT_BAL
master['Closeness_4'] = (master.LIMIT_BAL - master.BILL_AMT4) / master.LIMIT_BAL
master['Closeness_3'] = (master.LIMIT_BAL - master.BILL_AMT3) / master.LIMIT_BAL
master['Closeness_2'] = (master.LIMIT_BAL - master.BILL_AMT2) / master.LIMIT_BAL
master['Closeness_1'] = (master.LIMIT_BAL - master.BILL_AMT1) / master.LIMIT_BAL

ct={'MARRIAGE','EDUCATION','SEX','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'}
for col in ct:
    master[col]=master[col].astype(int)


# In[154]:


#CHCK TARGET VARIABLE###
sns.catplot(x='default_payment_next_month',kind='count',data=master)


# In[198]:


#####MODEL BUILDING####

master=pd.get_dummies(master,columns=ct) 
X=master[master['default_payment_next_month'].isnull()!=True].drop(['ID','default_payment_next_month'],axis=1)
y = master[master['default_payment_next_month'].isnull()!=True]['default_payment_next_month']
X_test=master[master['default_payment_next_month'].isnull()==True].drop(['ID','default_payment_next_month'],axis=1)

print(X.shape,y.shape,X_test.shape)


# In[199]:


###########SMOTE############3
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)
X=X_smote
y=y_smote
X_trains, X_vals, y_trains, y_vals = train_test_split(X,y, test_size = 0.2, random_state =1)


# In[200]:


#declare the models
lr = LogisticRegression()
rf=RandomForestClassifier()
adb=ensemble.AdaBoostClassifier()
bgc=ensemble.BaggingClassifier()
gnb = GaussianNB()
knn=KNeighborsClassifier()
dt = DecisionTreeClassifier()
lg = LGBMClassifier()
xg=xgboost.XGBClassifier()
ct = CatBoostClassifier()
gb=GradientBoostingClassifier()
# ab_rf = AdaBoostClassifier(base_estimator=rf,random_state=0)
# ab_dt = AdaBoostClassifier(base_estimator=dt,random_state=0)
# ab_nb=  AdaBoostClassifier(base_estimator=gnb,random_state=0)
# ab_lr=  AdaBoostClassifier(base_estimator=lr,random_state=0)
bgcl_lr = BaggingClassifier(base_estimator=lr, random_state=0)

# ,ab_rf,ab_dt,ab_nb,ab_lr,bgcl_lr

models=[lr,rf,adb,bgc,gnb,knn,dt,bgcl_lr,lg,xg,ct,gb]
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
                                'Naive-Bayes','KNN','Decistion Tree','LGBMClassifier','XGBClassifier',
                                'bagged LR','CatBoostClassifier','GradientBoostingClassifier'],
                       'auc_score':auc,'training':sctr,'testing':scte,'precision':ps,'recall':rs})
ensemble=ensemble.sort_values(by='auc_score',ascending=False).reset_index(drop=True)
ensemble


# In[202]:


##RANDOM FOREST##
randomforest = RandomForestClassifier(n_estimators=1000, oob_score=True)
randomforest.fit(X_trains,y_trains)
pred=randomforest.predict(X_vals)
randomforest.score(X_trains,y_trains)
accuracy=round(randomforest.score(X_trains, y_trains) * 100, 2)
print(accuracy)
print(classification_report(y_vals,pred))
print("oob score:", round(randomforest.oob_score_, 4)*100, "%")


# In[203]:


##RANDOM SEARCH CV###

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rfs = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rfs, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_trains, y_trains)
rf_random.best_params_


# In[204]:


# MAKE RF BASE ON THAT PARAMETER####
rfs = RandomForestClassifier(n_estimators=1400,
 min_samples_split= 2,
 min_samples_leaf= 1,
 max_features= 'auto',
 max_depth= 40,
 bootstrap= False)


rfs.fit(X_trains, y_trains)
Y_prtus = rfs.predict(X_vals)
rfs.score(X_trains, y_trains)
from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(rfs, X_trains, y_trains, cv=3)
confusion_matrix(y_trains, predictions)
print(classification_report(y_vals,Y_prtus))
acc_rf = round(rfs.score(X_trains, y_trains) * 100, 2)
print(acc_rf)
print("oob score:", round(rf.oob_score_, 4)*100, "%")


# In[205]:


#Print Feature Importance:
#######GBM IMPORTANCES####
importances = pd.DataFrame({'feature':X_trains.columns,'importance':np.round(rfs.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)


# In[209]:


prediction = rfs.predict(X_test)
submission = pd.DataFrame()
submission['ID'] = master[master['default_payment_next_month'].isnull()==True]['ID']
submission['default_payment_next_month'] = prediction
submission.to_csv('fisk.csv', index=False, header=True)
submission.shape


# In[ ]:





# In[ ]:




