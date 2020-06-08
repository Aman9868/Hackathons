#!/usr/bin/env python
# coding: utf-8

# In[918]:


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


# In[919]:


# READING
train = pd.read_csv("D:/R/House Prices kaggle/train.csv")
test = pd.read_csv("D:/R/House Prices kaggle/test.csv")


# In[920]:


# BINDING
master=pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,master.shape)
master.head()


# In[921]:


### Chck Dtypes
master.info()


# In[922]:


# Check column names
print(master.columns)


# In[923]:


# check na
sn=master.isnull().sum()/len(master)*100
sn=pd.DataFrame(sn)
sn.head(50)


# In[924]:


# CHECK UNIQUE VALUES

master.apply(lambda x : len(x.unique()))


# In[925]:


## SEPEARTION CATEGORY VS NUMERIC

cat=['MSZoning','Street','Alley','LotShape','LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2','Heating',
       'HeatingQC', 'CentralAir', 'Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual',
       'GarageCond', 'PavedDrive','PoolQC','MSSubClass',
       'Fence', 'MiscFeature','SaleType','SaleCondition','OverallQual', 'OverallCond','GarageCars']


num=['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF', '2ndFlrSF','LowQualFinSF','GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath','1stFlrSF', '2ndFlrSF','WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea','SalePrice']



final=master[num+cat]


# In[926]:


### VALUE COUNTS MULTIPLE COLUMN##
for i in master.columns:
    x=master[i].value_counts()
    print("Column name is:",i,"and it value is:",x)


# In[927]:


# DEALING WITH IMPUTAION CHECKPOINTS
tu={'MasVnrArea','GarageArea','LotFrontage','GarageYrBlt','GarageCars','BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF'}
for d in tu:
    y=master[d].describe()
    print(y)
    


# In[928]:


#MSVNR MISSING
dd=master[master['MasVnrType'].isnull()][['MasVnrType','MasVnrArea']]
dd


# In[929]:


# LOT MISSING
sd=master[master['LotFrontage'].isnull()][['LotFrontage','LotArea','LotShape']]
sd


# In[930]:


# BSMT missing
sd=master[master['BsmtQual'].isnull()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1',
                        'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']]
sd


# In[931]:


##CHECK GARAGES MISSING###

sy=master[master['GarageYrBlt'].isnull()][['GarageType', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']]
sy


# In[932]:



master.MasVnrArea.value_counts()


# In[933]:


############################MISSING VALUE TREATMENT##############################################

           ### GIVEN IN DATA DICTIONRAY THAT NA MEANS None FOR SOME VARIABLES
           ## IMPUTE 2-3 NA WITH mode i.e For some Categorical Variables##
           ### IMPUTE 4-5 NA i.e Numeric some with mean some with median
### IMPUTE GARAGE BUILD 0 As all are garage variable are no and zero so imput garge yr built 0 means not built###
## IMPUTE LOT FRONTAGE WITH RESPECT TO ITS NEIGHBOURHOOD as since area of each street connected to....
##                                         house property likely have same area with respect to its neigbour

#1.MSVNR AND MIS FEATURE.....

ln={'MiscFeature','MasVnrType','Alley','PoolQC','Fence'}
for i in ln:
    master[i].fillna('None',inplace=True)
    
#DEFINING DEICTIONARIES 
Bsmt = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
Grg = ['GarageType','GarageFinish','GarageQual','GarageCond']

#2.BSMTS
for col in Bsmt:
    master[col].fillna('No_Bsmt',inplace=True)
    
#3.GRGES    
for col in Grg:
    master[col].fillna('No_Grg',inplace=True)    
    

#4.FIRE PLACES
master['FireplaceQu'] = master['FireplaceQu'].fillna('NotAvailable')

#5.IMPUTE WITH MEDIAN or 0  DUE TO 1-3 NA IN EACH VARIABLRS AS THESE ARE IN CATEGORIES 

tt={'BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF'}
for c in tt:
    master[c] = master[c].fillna(0)

    
    
#6.IMPUTE WITH MODE WHICH LEADS TO HIGH FREQUENCY VARIABLES 

ts={'Exterior1st','Exterior2nd','KitchenQual','Functional','MSZoning','Utilities','Electrical','SaleType','MSZoning'}
for i in ts:
    master[i].fillna(master[i].mode,inplace=True)
 

#9.MSVNR AREA # DUE TO 0 MEAN AND 0 MEDIAN
master.MasVnrArea = master.MasVnrArea.fillna(0)

    
#11.GarageYrBlt    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    master[col] = master[col].fillna(0)

#12.LOT FRONTAGE

master["LotFrontage"] = master.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[934]:



## Categorical Variables

fig, ax=plt.subplots(2,3,figsize=(20,20))
for variable,subplot in zip(Bsmt,ax.flatten()):
    sns.countplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
        
fig, ax=plt.subplots(2,2,figsize=(20,20))
for variable,subplot in zip(Grg,ax.flatten()):
    sns.countplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90) 
        
        
        
fig, ax=plt.subplots(3,3,figsize=(20,20))
for variable,subplot in zip(ts,ax.flatten()):
    sns.countplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90) 
        
        
        
        


fig, ax=plt.subplots(2,3,figsize=(20,20))
for variable,subplot in zip(ln,ax.flatten()):
    sns.countplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[935]:


##################### CATEGORY VS TRAGET VISUALISATIONS###################################### 
for cat in Grg:
    print("SalePrice in Thousands ('000)")
    print()
    print("-"*20 + cat + '  vs' + '  SalePrice' + "-"*20)
    output = final[[cat,'SalePrice']].groupby([cat]).apply(lambda x: x['SalePrice'].sum()/1000).sort_values(ascending=False)
    output = pd.DataFrame(output)
    output.columns = ['SalePrice']
    ax = sns.barplot(output.index,'SalePrice', data =output)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width()/ 2., p.get_height()),ha='center', va='center', rotation=90, xytext=(0,40), textcoords='offset points')  #vertical bars
    plt.tight_layout()
    plt.show()
    print()
    print("Maximum Sales : ")
    print(output.head(1))
    print()
    print("-" *50)  
    
    
for cat in Bsmt:
    print("SalePrice in Thousands ('000)")
    print()
    print("-"*20 + cat + '  vs' + '  SalePrice' + "-"*20)
    output = final[[cat,'SalePrice']].groupby([cat]).apply(lambda x: x['SalePrice'].sum()/1000).sort_values(ascending=False)
    output = pd.DataFrame(output)
    output.columns = ['SalePrice']
    ax = sns.barplot(output.index,'SalePrice', data =output)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width()/ 2., p.get_height()),ha='center', va='center', rotation=90, xytext=(0,40), textcoords='offset points')  #vertical bars
    plt.tight_layout()
    plt.show()
    print()
    print("Maximum Sales : ")
    print(output.head(1))
    print()
    print("-" *50)   
    
    
    
    
for cat in ts:
    print("SalePrice in Thousands ('000)")
    print()
    print("-"*20 + cat + '  vs' + '  SalePrice' + "-"*20)
    output = final[[cat,'SalePrice']].groupby([cat]).apply(lambda x: x['SalePrice'].sum()/1000).sort_values(ascending=False)
    output = pd.DataFrame(output)
    output.columns = ['SalePrice']
    ax = sns.barplot(output.index,'SalePrice', data =output)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width()/ 2., p.get_height()),ha='center', va='center', rotation=90, xytext=(0,40), textcoords='offset points')  #vertical bars
    plt.tight_layout()
    plt.show()
    print()
    print("Maximum Sales : ")
    print(output.head(1))
    print()
    print("-" *50)        
    
    
for cat in ln:
    print("SalePrice in Thousands ('000)")
    print()
    print("-"*20 + cat + '  vs' + '  SalePrice' + "-"*20)
    output = final[[cat,'SalePrice']].groupby([cat]).apply(lambda x: x['SalePrice'].sum()/1000).sort_values(ascending=False)
    output = pd.DataFrame(output)
    output.columns = ['SalePrice']
    ax = sns.barplot(output.index,'SalePrice', data =output)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width()/ 2., p.get_height()),ha='center', va='center', rotation=90, xytext=(0,40), textcoords='offset points')  #vertical bars
    plt.tight_layout()
    plt.show()
    print()
    print("Maximum Sales : ")
    print(output.head(1))
    print()
    print("-" *50)          
    
    
    
  ############ALL DATA ARE CLEANED NOW#####################  


# In[942]:


########CHANGING DATA TYPE FOR SOME VARIABLES#######

dd=['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond','BldgType','HouseStyle','Condition2','MSZoning',
'HeatingQC','ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1','Condition1','RoofMatl','Foundation', 
'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope','Neighborhood','Exterior2nd','GarageType',
'Heating','LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond','RoofStyle','Exterior1st', 
'MoSold','OverallQual','LandContour','Utilities','MiscFeature','SaleType','SaleCondition','MasVnrType','LotConfig','Electrical']


for i in dd:
    master[i]=master[i].astype(str)


# In[937]:


###DISTPLOT VISUALISATION  OF NUMERICAL VARIABLES#######

num=['LotFrontage', 'LotArea','MasVnrArea','1stFlrSF', '2ndFlrSF','GrLivArea','WoodDeckSF', 'OpenPorchSF',
         'SalePrice']


fig, ax=plt.subplots(3,3,figsize=(20,50))
for var,subplot in zip(num,ax.flatten()):
    sns.distplot(master[var],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[938]:


#####################BOXPLOT VISUALISATUONS###################

fig, ax=plt.subplots(3,3,figsize=(20,20))
for var,subplot in zip(num,ax.flatten()):
    sns.boxplot(master[var],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[939]:


################################FEATURE ENGINEERING######################################

master['Lot_Length']=master['LotArea'] / master['LotFrontage']
master['Life_Sold']=master['YrSold'] - master['YearBuilt']
master['Yr_Sin_Radd']=master['YrSold'] - master['YearRemodAdd']
master['Total_SF'] = master['TotalBsmtSF'] + master['1stFlrSF'] + master['2ndFlrSF']
master['Total_Area']=master['GarageArea'] + master['LotArea'] + master['GrLivArea'] + master['MasVnrArea'] + master['PoolArea']
master['Total_Sf_Outdoor']=master['WoodDeckSF'] + master['OpenPorchSF']


# In[940]:


#####CORRELATION MATRIX######
#correlation matrix
corrmat = master.corr()
f, ax = plt.subplots(figsize=(70, 20))
sns.heatmap(corrmat, vmax=.8,annot = True,square=True);


# In[941]:


##LABEL ENCODING##
ds=['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond','BldgType','HouseStyle','Condition2','MSZoning',
'HeatingQC','ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1','Condition1','RoofMatl','Foundation', 
'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope','Neighborhood','Exterior2nd','GarageType',
'Heating','LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond','RoofStyle','Exterior1st', 
'MoSold','OverallQual','LandContour','Utilities','MiscFeature','SaleType','SaleCondition','MasVnrType','LotConfig','Electrical']


le=LabelEncoder()
for i in ds:
    master[i]=le.fit_transform(master[i])
    
    
    
##TARGET VARIABLE TRANSFORMATION####
master["SalePrice"] = np.log1p(master["SalePrice"])

#COPY
masters=master.copy()


# In[943]:


##############SKEWNESSS##################
from scipy import stats
from scipy.stats import norm, skew

numeric_feats = master.dtypes[master.dtypes != "object"].index
#Check the skew of all numerical features
skewed_feats = master[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(50)


# In[944]:


############TRANSFORMATION USING BOXCOX########

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for i in skewed_features:
    #master[feat] += 1
    master[i] = boxcox1p(master[i], lam)


# In[831]:


#CHECK TARGET VARIABLE AFTER TRANSFORMATION
sns.distplot(master.SalePrice)


# In[945]:


#####MODEL BUILDINGS###

master= pd.get_dummies(master, columns=ds)
master.head()

X = master[master['SalePrice'].isnull()!=True].drop(['Id','SalePrice'], axis=1)
y = master[master['SalePrice'].isnull()!=True]['SalePrice']

X_test = master[master['SalePrice'].isnull()==True].drop(['Id','SalePrice'], axis=1)

X.shape, y.shape, X_test.shape


# In[946]:


########SPLITTING#####
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[947]:


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


# In[948]:


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


# In[949]:


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


# In[950]:


fig, ax = plt.subplots()
ax.scatter(y_val, y_predicted, edgecolors=(0, 0, 0))
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()


# In[964]:


#Print Feature Importance:
#######GBM IMPORTANCES####
importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(model.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)

