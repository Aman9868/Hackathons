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
from sklearn.metrics import r2_score,roc_auc_score,classification_report,mean_squared_error,accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#READING
train=pd.read_csv('D:/R/Mobility Analytics/train.csv')
test=pd.read_csv('D:/R/Mobility Analytics/test.csv')


# In[3]:


#Combine test and train into one file
master= pd.concat([train, test],ignore_index=True)
print(train.shape, test.shape, master.shape)
master.head()


# In[4]:


# DATA TYPES
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


# VISUALISING CATEGORICAL
categorical_columns=['Type_of_Cab','Confidence_Life_Style_Index','Destination_Type','Cancellation_Last_1Month','Gender']
numerical=['Customer_Rating','Trip_Distance','Life_Style_Index','Var1','Var2','Var3']
final=master[categorical_columns+numerical]


# In[9]:


#VISUALISING NUMERICAL
fig, ax=plt.subplots(3,2,figsize=(10,20))
for variable,subplot in zip(numerical,ax.flatten()):
    sns.distplot(final[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[10]:


#CATEGORICAL
fig, ax=plt.subplots(6,figsize=(10,20))
for variable,subplot in zip(categorical_columns,ax.flatten()):
    sns.countplot(final[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[11]:


# VISUALISING BOXPLOT
fig, ax=plt.subplots(3,2,figsize=(20,20))
for variable,subplot in zip(numerical,ax.flatten()):
    sns.boxplot(final[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[12]:


#### VISUALIZING CATEGORICAL WITH TARGET #####

fig,axes = plt.subplots(3,2,figsize=(10,20))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=train,hue='Surge_Pricing_Type',ax=axes[row,col])


plt.subplots_adjust(hspace=1)


# In[13]:


#SURGE 3 ARE SEEMS TO BE HIGH 
sns.FacetGrid(master,hue="Surge_Pricing_Type",size=5).map(sns.distplot,"Trip_Distance").add_legend()


# In[14]:


#############DATA CLEANING#############

#1.TYPE OF CAB
master['Type_of_Cab'].value_counts(dropna=False)
Missing_cab = master[master.Type_of_Cab.isna()]
Missing_cab.head()

#FILL NA
master['Type_of_Cab']=master['Type_of_Cab'].fillna('B')


# In[15]:


#2. CUSTOMER SINCE MONTHS
master['Customer_Since_Months'].value_counts(dropna=False)
Missing_c = master[master.Customer_Since_Months.isna()]
Missing_c.head()
master.Customer_Since_Months.fillna(master.Customer_Since_Months.median(),inplace=True)
master.Customer_Since_Months = master.Customer_Since_Months.astype(int)

sns.catplot(x="Customer_Since_Months",kind="count",data=master)


# In[16]:


#LIFESTYLEINDEX
master.Life_Style_Index.fillna(master.Life_Style_Index.mean(),inplace=True)

#CONFIDENCE LIFESTYLE
master.Confidence_Life_Style_Index.value_counts(dropna=False)
master.Confidence_Life_Style_Index.fillna('D',inplace=True)
sns.catplot(x="Confidence_Life_Style_Index",kind="count",data=master)


# In[17]:


###VAR1######
master.Var1.fillna(master.Var1.median(),inplace=True)


# In[18]:


###################OUTLIER TREATMENT###############  # z
#LIFESTYLE
sorted(master['Life_Style_Index'])
quantile1,quantile3=np.percentile(master.Life_Style_Index,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)


# In[19]:


#TREATMENT
master.Life_Style_Index.loc[master.Life_Style_Index > 3.257]=3.2527
master.Life_Style_Index.loc[master.Life_Style_Index < 2.34296]=2.34296

sns.boxplot(master.Life_Style_Index)


# In[20]:


# REMOVING COLOUMN#
master=master.drop(['Var1', 'Var2','Var3'], axis = 1)
master.head()

##########################ALL DATA ARE CLEANED AND TRANSFORMED NOW##############


# In[21]:


master.info()
contvars=master[['Trip_Distance','Customer_Since_Months','Life_Style_Index','Customer_Rating']]
#####CORRELATION MATRIX######
#correlation matrix
corrmat = contvars.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,annot = True,square=True);


# In[22]:



####FEATURE ENGINEERING#####
master['Trpper_Lif']=master['Trip_Distance'] / master['Life_Style_Index']


# In[23]:


# CHECK TARGET VARIABLE###

master['Surge_Pricing_Type'].value_counts(normalize=True)*100


# In[24]:


sns.countplot(master.Surge_Pricing_Type)


# In[25]:


################BUILDING DUMMY##########
master= pd.get_dummies(master, columns=categorical_columns)
master.head()


# In[26]:


##########3MODEL############3
###########MODEL BUILDING###########3
X = master[master['Surge_Pricing_Type'].isnull()!=True].drop(['Trip_ID','Surge_Pricing_Type'], axis=1)
y = master[master['Surge_Pricing_Type'].isnull()!=True]['Surge_Pricing_Type']

X_test = master[master['Surge_Pricing_Type'].isnull()==True].drop(['Trip_ID','Surge_Pricing_Type'], axis=1)

X.shape, y.shape, X_test.shape

########SPLITTING#####
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[27]:


###DECISION TREE####
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(x_train, y_train)  
Y_pred = decision_tree.predict(x_val)  
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
print(acc_decision_tree)
print(classification_report(y_val,Y_pred))


# In[28]:


### RANDOM FOREST###
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)

Y_prediction = random_forest.predict(x_val)

random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
print(acc_random_forest)
print(classification_report(y_val,Y_prediction))


# In[ ]:


####SVM###
from sklearn import svm
linear_svc = svm.SVC(decision_function_shape='ovo')
linear_svc.fit(x_train, y_train)

Y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)
print(acc_linear_svc)
print(classification_report(y_val,Y_pred))


# In[78]:


########NAIVE BYES#########3
gaussian = GaussianNB() 
gaussian.fit(x_train,y_train)
y_pred=gaussian.predict(x_val)
acc_gaussian_nb=round(gaussian.score(x_train,y_train) * 100, 2)
print(acc_gaussian_nb)
print(classification_report(y_val,y_pred))


# In[82]:


print(x_train.shape)#USE FOR CHOOSING K I.E SQRT(X_TRAIN)


# In[84]:


##########KNN#########
knn=KNeighborsClassifier(n_neighbors=325)
knn.fit(x_train,y_train)
y_p=knn.predict(x_val)
acc_knn=round(knn.score(x_train,y_train) *100, 2)
print(acc_knn)
print(classification_report(y_val,y_p))


# In[85]:


#KFOLD CROSS VALIDATION
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, x_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[86]:


#FOREST IMPT
importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)


# In[88]:


#############TUNNING PARAMETER################ WITH HIS OUT OF BAG SCORE
random_forest = RandomForestClassifier(n_estimators=1000, oob_score = True)
random_forest.fit(x_train, y_train)
Y_prediction = random_forest.predict(x_val)

random_forest.score(x_train, y_train)

acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
print(classification_report(y_val,Y_prediction))


# In[91]:


#####FINDING BEST PARAMETER######
param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}
from sklearn.model_selection import GridSearchCV, cross_val_score
rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=1)
clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=1)
clf.fit(x_train, y_train)
print(clf.best_estimator_)

