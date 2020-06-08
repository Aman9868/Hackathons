#!/usr/bin/env python
# coding: utf-8

# In[81]:


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


# In[82]:


#READING
train=pd.read_csv('D:/R/upvotes prediction/train.csv')
test=pd.read_csv('D:/R/upvotes prediction/test.csv')


# In[83]:


# BINDING
master=pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,master.shape)
master.head()


# In[84]:


### Chck Dtypes
master.info()


# In[85]:


# Check column names
print(master.columns)


# In[86]:


# check na
master.isnull().sum()/len(master)*100


# In[87]:


# CHECK UNIQUE VALUES

master.apply(lambda x : len(x.unique()))


# In[88]:


# Check Tag
sns.catplot(x='Tag',kind='count',data=master)
a={'a':0,'c':1,'r':2,'j':3,'p':4,'s':5,'h':6,'o':7,'i':8,'x':9}
master.Tag=master.Tag.map(a)
master.Tag=master.Tag.astype(int)


# In[89]:


# Check Numerical Variables

num=['Reputation','Answers','Views','Upvotes']
final=master[num]

#Numerical
fig, ax=plt.subplots(3,figsize=(10,20))
for variable,subplot in zip(num,ax.flatten()):
    sns.distplot(final[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[77]:


######skewness check####
#CHECK TARGET VARIABLE BEFORE TRANSFORMATION
sns.distplot(master.Upvotes)


# In[90]:


#DROP OUT COLOUMNS

master=master.drop(['Username'],axis=1)
master.head()


# In[91]:


#FEATURE ENGINEERIN#
master['Watched']=master.Answers * master.Views
master['Tag']=master['Tag'].astype(str)


# In[92]:


master.info()
contvars=master[['Answers','Reputation','Views','Upvotes','Watched']]
#####CORRELATION MATRIX######
#correlation matrix
corrmat = contvars.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,annot = True,square=True);


# In[94]:


######SKEWNESS#####
from scipy import stats
from scipy.stats import norm, skew

ct={'Answers','Reputation','Views','Upvotes','Watched'}
numeric_feats = ct
#Check the skew of all numerical features
skewed_feats = master[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head()


# In[95]:


#########SKEWNESS TREATMENT###############
############TRANSFORMATION USING BOXCOX########

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for i in skewed_features:
    #master[feat] += 1
    master[i] = boxcox1p(master[i], lam)


# In[44]:


sns.distplot(master.Upvotes)


# In[96]:


##########MODEL BUILDING###########

master= pd.get_dummies(master, columns=['Tag'])
master.head()

X = master[master['Upvotes'].isnull()!=True].drop(['ID','Upvotes'], axis=1)
y = master[master['Upvotes'].isnull()!=True]['Upvotes']

X_test = master[master['Upvotes'].isnull()==True].drop(['ID','Upvotes'], axis=1)

X.shape, y.shape, X_test.shape


# In[97]:


########SPLITTING#####
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[98]:


###SCALING##

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)


# In[99]:


##########RIDGE REGRESSION##########

list1=[0.0018,0.002,0.005,0.08,0.09,0.1,0.5]
list2=[]
for i in list1:
    ridge_reg = Ridge(alpha=i,normalize=True)
    ridge_reg.fit(x_train,y_train)
    y_pred_r=ridge_reg.predict(x_val)
    r2score_r= r2_score(y_val,y_pred_r)
    list2.append(r2score_r)

ridge_rscore_df=pd.DataFrame({"ALPHA":list1,"R2SCORE":list2})
ridge_rscore_df 


# In[100]:


#####LINEAR REGRESSION###
logmodel = LinearRegression()
logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_val)

model_score = logmodel.score(x_train,y_train)
# Have a look at R sq to give an idea of the fit ,
# Explained variance score: 1 is perfect prediction
print('R2 sq: ',model_score)
# Accuracy Score
acc_log = round(logmodel.score(x_train, y_train) * 100, 2)
print('Acc: ',acc_log)


# In[101]:


### GBM###
from sklearn import ensemble
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)

model.fit(x_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score
model_score = model.score(x_train,y_train)

# Have a look at R sq to give an idea of the fit ,
# Explained variance score: 1 is perfect prediction
print('R2 sq: ',model_score)
y_predicted = model.predict(x_val)

# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y_val, y_predicted))
# Explained variance score: 1 is perfect prediction
print('Test Variance score: %.2f' % r2_score(y_val, y_predicted))
# Accuracy Score
acc_log = round(model.score(x_train, y_train) * 100, 2)
print('Acc: ',acc_log)


# In[102]:


fig, ax = plt.subplots()
ax.scatter(y_val, y_predicted, edgecolors=(0, 0, 0))
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()


# In[103]:


prediction = model.predict(X_test)
submission = pd.DataFrame()
submission['ID'] = master[master['Upvotes'].isnull()==True]['ID']
submission['Upvotes'] = prediction
submission.to_csv('upvs.csv', index=False, header=True)
submission.shape

