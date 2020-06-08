#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#READING
train=pd.read_csv('D:/R/loan prediction/train.csv')
test=pd.read_csv('D:/R/loan prediction/test.csv')


# In[3]:


# BINDING
master=pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,master.shape)
master.head()


# In[4]:


### Chck Dtypes
master.info()


# In[5]:


# Check column names
print(master.columns)


# In[6]:


# check na
master.isnull().sum()/len(master)*100


# In[7]:


# CHECK UNIQUE VALUES

master.apply(lambda x : len(x.unique()))


# In[8]:


cat=['Gender','Married','Credit_History','Self_Employed','Property_Area','Loan_Amount_Term','Education','Dependents']
num=['CoapplicantIncome','LoanAmount','ApplicantIncome']
final=master[cat+num]


# In[9]:


#CAT VARIABLE
fig, ax=plt.subplots(4,2,figsize=(20,20))
for variable,subplot in zip(cat,ax.flatten()):
    sns.countplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[10]:


#Numerical
fig, ax=plt.subplots(3,figsize=(10,10))
for variable,subplot in zip(num,ax.flatten()):
    sns.distplot(final[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[11]:


# VISUALISING BOXPLOT
fig, ax=plt.subplots(3,figsize=(10,15))
for variable,subplot in zip(num,ax.flatten()):
    sns.boxplot(final[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[12]:


# Category vs Target
fig,axes = plt.subplots(4,2,figsize=(20,20))
for idx,cat_col in enumerate(cat):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=master,hue='Loan_Status',ax=axes[row,col])


plt.subplots_adjust(hspace=1)


# In[13]:


# Check LOan Status with Loan AMt
##AGE AROUND 20-40 HAS HIGHEST SURVIVAL RATE
sns.FacetGrid(master,hue="Loan_Status",size=7).map(sns.distplot,"LoanAmount").add_legend()


# In[14]:


#CHECK LOAN STATUS UPON LOAN AMOUNT UPON THE GENDER  #AROUND 50 TO 200 L.AMT MALE HAS HIGHEST APPROVAL OF LOAN

status = 'Y'
not_status = 'N'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train[train['Gender']=='Female']
men = train[train['Gender']=='Male']
ax = sns.distplot(women[women['Loan_Status']=='Y'].LoanAmount.dropna(),label=status ,bins=18, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Loan_Status']=='N'].LoanAmount.dropna(),label=not_status, bins=40, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Loan_Status']=='Y'].LoanAmount.dropna(),label=status, bins=18, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Loan_Status']=='N'].LoanAmount.dropna(),label=not_status, bins=40, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# In[15]:


#############################MISSING TREATMENT##################################################
#1.GENDER
master.Gender.value_counts(dropna=False)
master.Gender=master.Gender.fillna('Male')
genders = {"Male": 0, "Female": 1}
master.Gender=master.Gender.map(genders)


# In[16]:


#2.Married
master.Married.value_counts(dropna=False)
master.Married=master.Married.fillna('Yes')
marr= {"Yes":1,"No":0}
master.Married=master.Married.map(marr)
master.Married=master.Married.astype(int)


# In[17]:


#3. DEpendents
master.Dependents.value_counts(dropna=False)
master['Dependents'] = master['Dependents'].str.rstrip('+')
master.Dependents.value_counts()
master['Dependents'] = master['Dependents'].fillna( master['Dependents'].dropna().mode().values[0] )
master.Dependents=master.Dependents.astype('str').astype(int)


# In[18]:


#4.SELF EMPLOYED
master.Self_Employed.value_counts(dropna=False)
master['Self_Employed'] = master['Self_Employed'].fillna( master['Self_Employed'].dropna().mode().values[0] )
de= {'No':0,'Yes':1}
master.Self_Employed=master.Self_Employed.map(de)
master.Self_Employed.value_counts()


# In[19]:


#5.EDUCtion
master.Education.value_counts()
ed= {'Graduate':1,'Not Graduate':0}
master.Education=master.Education.map(ed)
master.Education=master.Education.astype(int)


# In[20]:


#6.LOANAMT
missing=master[master.LoanAmount.isna()]
missing.head()

# REPLACE LOAN AMT WITH RANDOM VALUE WITH RESPECT TO MEAN STD AND ISNULL
master.LoanAmount.describe()
mean = master["LoanAmount"].mean()
std = master["LoanAmount"].std()
is_null = master["LoanAmount"].isnull().sum()
# compute random numbers between the mean, std and is_null
rand_LoanAmount = np.random.randint(mean - std, mean + std, size = is_null)
 # fill NaN values in LoanAmount column with random values generated
LoanAmount_slice = master["LoanAmount"].copy()
LoanAmount_slice[np.isnan(LoanAmount_slice)] = rand_LoanAmount
master["LoanAmount"] = LoanAmount_slice
master["LoanAmount"] = master["LoanAmount"].astype(int)

###OUTLIER IDENTIFICATION###

sns.boxplot(master['LoanAmount'])

sorted(master['LoanAmount'])
quantile1,quantile3=np.percentile(master.LoanAmount,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)


# In[21]:


##TRATMENT##

master.LoanAmount.loc[master.LoanAmount > up]=up
sns.boxplot(master['LoanAmount'])


# In[22]:


#LOAN_AMT
master['Loan_Amount_Term'] = master['Loan_Amount_Term'].fillna( master['Loan_Amount_Term'].dropna().mode().values[0] )


# In[23]:


master['Loan_Amount_Term']=master['Loan_Amount_Term'].round().astype(str)
master.info()


# In[24]:


####### CHANGING THE 6.0 TO 60.0 ANND 350.0 TO 360.0 AS LOAN CAN BE OF 6 MONTHS
master.head()
master['Loan_Amount_Term'].replace({'6.0':60.0,'350.0':360.0},inplace=True)
master['Loan_Amount_Term']=master['Loan_Amount_Term'].astype(int)


# In[25]:


master.Loan_Amount_Term.value_counts()


# In[26]:


# CREDIT_HISTORY
master.Credit_History.value_counts(dropna=False)
master.Credit_History=master.Credit_History.fillna('Not.Av')
master.Credit_History.value_counts()


# In[27]:


############################OUTLIER TREATMENT######################
#1.CO APPLICASNT INCOME\
sorted(master['CoapplicantIncome'])
quantile1,quantile3=np.percentile(master.CoapplicantIncome,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)

##TRATMENT##

master.CoapplicantIncome.loc[master.CoapplicantIncome > up]=up
sns.boxplot(master['CoapplicantIncome'])


# In[28]:


#2.Applicant_Income

sorted(master['ApplicantIncome'])
quantile1,quantile3=np.percentile(master.ApplicantIncome,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)

##TRATMENT##

master.ApplicantIncome.loc[master.ApplicantIncome > up]=up
sns.boxplot(master['ApplicantIncome'])


# In[29]:


##############FEATURE ENGINEERING########### ## iir income to installment ratio $#icr-income coverge ratio

master['Total_Income']=master.ApplicantIncome+master.CoapplicantIncome
master['Installment']=master.LoanAmount / master.Loan_Amount_Term * 1000
master['Debt_Income']=master.LoanAmount / master.Total_Income * 1000
master['IIR_Ratio']=master.Installment / master.Total_Income * 1000
master['Interest']=master.LoanAmount * 0.9 * master.Loan_Amount_Term / 12
master['TotalAmt_Paid']=master.LoanAmount + master.Interest
master['ICR']=master.Total_Income / master.Interest *1000
master.head()


# In[30]:


master.Family_Size =np.where((master.CoapplicantIncome > 0 | master.Married== 0),master.Dependents + 2,master.Dependents + 1)


# In[ ]:


master.info()


# In[31]:


master.info()
contvars=master[['LoanAmount','ApplicantIncome','CoapplicantIncome','Loan_Amount_Term','Installment','Total_Income','ICR','IIR_Ratio']]
#####CORRELATION MATRIX######
#correlation matrix
corrmat = contvars.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,annot = True,square=True);


# In[32]:


##########MODEL BUILDING###########

master= pd.get_dummies(master, columns=cat)
master.head()

X = master[master['Loan_Status'].isnull()!=True].drop(['Loan_ID','Loan_Status'], axis=1)
y = master[master['Loan_Status'].isnull()!=True]['Loan_Status']

X_test = master[master['Loan_Status'].isnull()==True].drop(['Loan_ID','Loan_Status'], axis=1)

X.shape, y.shape, X_test.shape


# In[33]:


########SPLITTING#####
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print(X.shape)
y_val.head()


# In[34]:


#LABEL ENCODER#
number = LabelEncoder()
y_val=number.fit_transform(y_val.astype('str'))
y_train=number.fit_transform(y_train.astype('str'))


# In[35]:


###SCALING##

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)


# In[36]:


#declare the models\

seed=0
import sklearn.ensemble as ensemble

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
def ens(x_train,x_val, y_train, y_val):
    for model in models:
            model.fit(x_train, y_train)
            y_test_pred = model.predict(x_val)
            y_test_pred_new=model.predict_proba(x_val)
            y_test_pred_new=y_test_pred_new[:,1]
            train_score=model.score(x_train,y_train)
            test_score=model.score(x_val,y_val)
            p_score=metrics.precision_score(y_val,y_test_pred)
            r_score=metrics.recall_score(y_val,y_test_pred)
            
            ac=metrics.roc_auc_score(y_val,y_test_pred_new)
            
            sctr.append(train_score)
            scte.append(test_score)
            ps.append(p_score)
            rs.append(r_score)
            auc.append(ac)
    return sctr,scte,auc,ps,rs
ens(x_train,x_val, y_train, y_val)
# 'ab_rf','ab_dt','ab_nb','ab_lr','bgcl_lr'
ensemble=pd.DataFrame({'names':['Logistic Regression','Random Forest','Ada boost','Bagging',
                                'Naive-Bayes','KNN','Decistion Tree',
                                'bagged LR'],
                       'auc_score':auc,'training':sctr,'testing':scte,'precision':ps,'recall':rs})
ensemble=ensemble.sort_values(by='auc_score',ascending=False).reset_index(drop=True)
ensemble


# In[37]:


###########SMOTE############3
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)


# In[38]:


seed=0
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics

X=X_smote
y=y_smote
X_trains, X_vals, y_trains, y_vals = train_test_split(X,y, test_size = 0.2, random_state =1)



#LABEL ENCODER#
number = LabelEncoder()
y_vals=number.fit_transform(y_vals.astype('str'))
y_trains=number.fit_transform(y_trains.astype('str'))


###SCALING##

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_trains = scaler.fit_transform(X_trains)
X_vals = scaler.transform(X_vals)


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


# In[1521]:


### RANDOM FOREST###
random_forest = RandomForestClassifier(n_estimators=1000, oob_score=True)
random_forest.fit(X_trains, y_trains)

Y_prediction = random_forest.predict(X_vals)

random_forest.score(X_trains, y_trains)
acc_random_forest = round(random_forest.score(X_trains, y_trains) * 100, 2)
print(acc_random_forest)
print(classification_report(y_vals,Y_prediction))
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# In[1175]:


# Use the forest's predict method on the test data
predictions = random_forest.predict(X_test)
submission = pd.DataFrame()
submission['Loan_ID'] = master[master['Loan_Status'].isnull()==True]['Loan_ID']
submission['Loan_Status'] = predictions
submission.to_csv('rfss.csv', index=False, header=True)
submission.shape


# In[1522]:


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


# In[1524]:


# MAKE RF BASE ON THAT PARAMETER####
rf2 = RandomForestClassifier(criterion = "entropy", 
                                       min_samples_leaf = 1, 
                                       max_features= 3,
                                       max_depth=30,
                                       min_samples_split = 2,
                                       n_estimators=500,oob_score=True,  
                                       n_jobs=-1)

rf2.fit(X_trains, y_trains)
Y_prtu = rf2.predict(X_vals)
rf2.score(X_trains, y_trains)
from sklearn.model_selection import cross_val_predict
predictions = cross_val_predict(rf2, X_trains, y_trains, cv=3)
confusion_matrix(y_trains, predictions)
print(classification_report(y_vals,Y_prtu))
acc_rf = round(rf2.score(X_trains, y_trains) * 100, 2)
print(acc_rf)
print("oob score:", round(rf.oob_score_, 4)*100, "%")


# In[1475]:


prediction = rf.predict(X_test)
submission = pd.DataFrame()
submission['Loan_ID'] = master[master['Loan_Status'].isnull()==True]['Loan_ID']
submission['Loan_Status'] = prediction
submission.to_csv('rf.csv', index=False, header=True)
submission.shape


# In[1184]:


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


# In[1185]:


rf_random.best_params_


# In[1192]:


# MAKE RF BASE ON THAT PARAMETER####
rfs = RandomForestClassifier(n_estimators=1200,
 min_samples_split= 2,
 min_samples_leaf= 4,
 max_features= 'sqrt',
 max_depth= 70,
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


# In[1199]:


##GBM###

##GBM##

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    ypreds=gb_clf.fit(X_trains, y_trains)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(ypreds.score(X_trains, y_trains)))
    print("Accuracy score (validation): {0:.3f}".format(ypreds.score(X_vals, y_vals)))
    
    
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_trains, y_trains)
predictions = gb_clf2.predict(X_vals)

acc_gbm = round(gb_clf2.score(X_trains, y_trains) * 100, 2)
print(acc_gbm)
print("Confusion Matrix:")
print(confusion_matrix(y_vals, predictions))

print("Classification Report")
print(classification_report(y_vals, predictions))


# In[1200]:


prediction = gb_clf2.predict(X_test)
submission = pd.DataFrame()
submission['Loan_ID'] = master[master['Loan_Status'].isnull()==True]['Loan_ID']
submission['Loan_Status'] = prediction
submission.to_csv('rfsd.csv', index=False, header=True)
submission.shape


# In[1267]:


###XGB CLASSIFIER###
import xgboost
params = {
    'learning_rate'   : [0.05,0.3,0.10,0.15,0.20],
    'max_depth'       : [3,4,5,6,8,10],
    'gamma'           : [0.0,0.1,0.2,0.3,0.4],
    'n_estimators'    : range(100,1000,100),
    'colsample_bytree': [0.3,0.4,0.5,0.7]
}


model_xg2 = xgboost.XGBClassifier()
xgb_rand_cv = RandomizedSearchCV(estimator=model_xg2,
                             param_distributions=params,n_iter=5,
                            scoring='accuracy',cv=5,n_jobs=-1)

xgb_rand_cv.fit(X_trains,y_trains)

pred_xgb = xgb_rand_cv.predict(X_vals)
print(classification_report(y_vals,pred_xgb))
print('\n')
print(confusion_matrix(y_vals,pred_xgb))


# In[ ]:


# defining parameter range SVM TUNNING
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','poly','linear']}  
  
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_trains, y_trains) 
grid_search.best_params_


# In[39]:


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

rf_parameters, rf_ht_score = hypertuning_rscv(est, rf_p_dist, 40, X_trains, y_trains)


def hypertuning_rscv(est, p_distr, nbr_iter,X_trains,y_trains):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=9)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X_trains,y_trains)
    rdmsearch.best_params_
    
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score


# In[42]:


rf_parameters


# In[44]:


# MAKE RF BASE ON THAT PARAMETER####
rfs = RandomForestClassifier(n_estimators=300,
 criterion='entropy',
 min_samples_leaf= 2,
 max_features= 1,
 max_depth= None,
 bootstrap= True)


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

