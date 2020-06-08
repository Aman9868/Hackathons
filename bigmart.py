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
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# READING
train = pd.read_csv("D:/R/BIG MART SALES/train.csv")
test = pd.read_csv("D:/R/BIG MART SALES/test.csv")


# In[3]:


#BINDING
master=pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,master.shape)
master.head()


# In[4]:


# CHECK DTYPES 
master.info()


# In[5]:


#Numerical data summary:
master.describe()


# In[6]:


# check na
master.isnull().sum()/len(master)*100


# In[7]:


#SEPERATING NUMERICAL VS CATEGORICAL
num=['Item_Weight','Item_Visibility','Item_MRP','Item_Outlet_Sales']
cat_data=['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']
final=master[num+cat_data]


# In[8]:


## Categorical Variables

fig, ax=plt.subplots(3,2,figsize=(20,20))
for variable,subplot in zip(cat_data,ax.flatten()):
    sns.countplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[9]:


# NUMERIC
fig, ax=plt.subplots(2,2,figsize=(20,20))
for var,subplot in zip(num,ax.flatten()):
    sns.distplot(final[var],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[10]:


# BOXPLOT

fig, ax=plt.subplots(2,2,figsize=(20,20))
for var,subplot in zip(num,ax.flatten()):
    sns.boxplot(final[var],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[11]:


## NUMERICAL VS CATEGORY

for cat in cat_data:
    print("Item_Outlet_Sales in Thousands ('000)")
    print()
    print("-"*20 + cat + '  vs' + '  Item_Outlet_Sales' + "-"*20)
    output = final[[cat,'Item_Outlet_Sales']].groupby([cat]).apply(lambda x: x['Item_Outlet_Sales'].sum()/1000).sort_values(ascending=False)
    output = pd.DataFrame(output)
    output.columns = ['Item_Outlet_Sales']
    ax = sns.barplot(output.index,'Item_Outlet_Sales', data =output)
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
    
 


######CONCLUSION#########
#SALES OF FRUITS AND VEGITABLE IS HIGH
#SALES OF LOW FAT CONTENT IS HIGH
# SALES OF OUTLET 27 IS HIGH
# SALES OF OUTLET SIZE MEDIUM IS HIGH
# SALES OF LOACTION TIER1 IS HIGH
# SALES OF SUPERMARKETTYPE1 IS HIGH


# In[12]:


##BIINING ITEM FAT CONTENT###
ge=['Item_Fat_Content']
gs={'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'}
for col in ge:
    master[col]=master[col].replace(gs)
    
    master.Item_Fat_Content.value_counts()


# In[13]:


## OULET SALES ACCORDING TO OUTLET IDENTIFIER,OUTLET ESTABLISHMENT YEAR

year_store_sales = master[['Outlet_Identifier','Outlet_Establishment_Year','Item_Outlet_Sales']].groupby(['Outlet_Identifier','Outlet_Establishment_Year']).apply(lambda x: x['Item_Outlet_Sales'].sum()/1000).sort_values(ascending=False)
year_store_sales = pd.DataFrame(year_store_sales)
year_store_sales.columns = ['Outlet_Sales']
year_store_sales


# In[14]:


#OUTLET SALES ACCORDING TO ITEM FAT CONTENT,ITEM_TYPE
item_store_sales = master[['Item_Type','Item_Fat_Content','Item_Outlet_Sales']].groupby(['Item_Type','Item_Fat_Content']).apply(lambda x: x['Item_Outlet_Sales'].sum()/1000).sort_values(ascending=False)
item_store_sales = pd.DataFrame(item_store_sales)
item_store_sales.columns = ['Outlet_Sales']
item_store_sales.head(10)


# In[15]:


##MISSSING VALUE TRATMENT#####

#1.Item Weight
mis=master[master.Item_Weight.isna()]
mis
# REPLACE LOAN AMT WITH RANDOM VALUE WITH RESPECT TO MEAN STD AND ISNULL
master.Item_Weight.describe()
mean = master["Item_Weight"].mean()
std = master["Item_Weight"].std()
is_null = master["Item_Weight"].isnull().sum()
# compute random numbers between the mean, std and is_null
rand_Item_Weight = np.random.randint(mean - std, mean + std, size = is_null)
 # fill NaN values in Item_Weight column with random values generated
Item_Weight_slice = master["Item_Weight"].copy()
Item_Weight_slice[np.isnan(Item_Weight_slice)] = rand_Item_Weight
master["Item_Weight"] = Item_Weight_slice
master["Item_Weight"] = master["Item_Weight"].astype(int)

###OUTLIER IDENTIFICATION###

sorted(master['Item_Weight'])
quantile1,quantile3=np.percentile(master.Item_Weight,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)

###TREATMENT###

master.Item_Weight.loc[master.Item_Weight > up]=up
sns.boxplot(master['Item_Weight'])


# In[16]:


#OUTLIER TREATMENT###

#.Item_Visibility

###OUTLIER IDENTIFICATION###

sorted(master['Item_Visibility'])
quantile1,quantile3=np.percentile(master.Item_Visibility,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)

###TREATMENT###

master.Item_Visibility.loc[master.Item_Visibility > up]=up
sns.boxplot(master['Item_Visibility'])


# In[17]:


#2.Outlet SIZE
mt=master[master.Outlet_Size.isna()]
mt
master['Outlet_Size'] = master['Outlet_Size'].fillna( master['Outlet_Size'].dropna().mode().values[0] )
master.Outlet_Size.value_counts()


# In[18]:


### FEATURE ENGINEERING##

perishable = ["Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood"]
non_perishable = ["Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene", "Household", "Soft Drinks"]
item_list =[] 
for i in master['Item_Type']:
    if i in perishable:
        item_list.append('perishable')
    elif (i in non_perishable):
        item_list.append('non_perishable')
    else:
        item_list.append('not_sure')
        
master['Item_Type_new'] = item_list
master['Price_per_Unit']=master.Item_MRP / master.Item_Weight
master['Year_Diff']=2013 - master.Outlet_Establishment_Year
master.head()


# In[19]:


# Labels Encoding

cat=['Item_Fat_Content','Outlet_Size','Outlet_Location_Type','Item_Type_new','Outlet_Type','Item_Type','Outlet_Identifier']

le=LabelEncoder()
for i in cat:
    master[i]=le.fit_transform(master[i])
    


# In[20]:


#LOG TRANSFORMATION OF TARGET VARIABLE

master['Item_Outlet_Sales']=np.log1p(master['Item_Outlet_Sales'])
master.head()


# In[21]:


master.skew(axis=0,skipna=True)


# In[22]:


#####CORRELATION MATRIX######
#correlation matrix
corrmat = master.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,annot = True,square=True);


# In[23]:


##########MODEL BUILDING###########

master= pd.get_dummies(master, columns=cat)
X = master[master['Item_Outlet_Sales'].isnull()!=True].drop(['Item_Identifier','Item_Outlet_Sales'], axis=1)
y = master[master['Item_Outlet_Sales'].isnull()!=True]['Item_Outlet_Sales']

X_test = master[master['Item_Outlet_Sales'].isnull()==True].drop(['Item_Identifier','Item_Outlet_Sales'], axis=1)

X.shape, y.shape, X_test.shape


# In[24]:


########SPLITTING#####
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[25]:


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


# In[26]:


#####LINEAR REGRESSION###
#LOGISTIC####
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


# In[26]:


## SELECTING THE BEST MODEL###
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
model_df = {'Name':['LR', 'Ridge', 'Lasso', 'E_Net','SVR','Dec_Tree','RF','Bagging_Reg','AdaBoost','Grad_Boost','Lgbm'],
             'Model' : [LinearRegression(), Ridge(alpha=0.05,solver='cholesky'), Lasso(alpha=0.01) ,ElasticNet(alpha=0.01,l1_ratio=0.5),
                     SVR(epsilon=15,kernel='linear'),DecisionTreeRegressor(),
                     RandomForestRegressor(),BaggingRegressor(max_samples=70),AdaBoostRegressor(),GradientBoostingRegressor(),LGBMRegressor()]}

model_df = pd.DataFrame(model_df)
model_df['Cross_val_score_mean'], model_df['Cross_val_score_STD'] = 0,0
model_df


# In[27]:


for m in range(0,model_df.shape[0]):
    print(model_df['Name'][m])
    score=cross_val_score(model_df['Model'][m] , x_train,y_train , cv=10 , scoring='neg_mean_squared_error')
    score_cross=np.sqrt(-score)
    model_df['Cross_val_score_mean'][m] = np.mean(score_cross)
    model_df['Cross_val_score_STD'][m] = np.std(score_cross)
    
model_df


# In[28]:


model_df.sort_values(by=['Cross_val_score_mean'])


# In[27]:


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


# In[28]:


fig, ax = plt.subplots()
ax.scatter(y_val, y_predicted, edgecolors=(0, 0, 0))
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()


# In[ ]:





# In[ ]:


##########TUNNING#########
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


gbs=GradientBoostingRegressor()


