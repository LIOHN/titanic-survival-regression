#!/usr/bin/env python
# coding: utf-8

# ## Predicting Survival on the Titanic
# 
# ### History
# Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time.
# 
# ### Assignment:
# 
# Transform the code in this Jupyter notebook into Procedural Programming like scripts. In the accompanying folder you will find templates of the scripts that you need to fill in with the code you find in this notebook.
# 
# Keep in mind, that the code may need a few tweaks here and there.
# 
# **Before running this notebook, you need to create the dataset. To create the data set run the script load_and_save_dataset.py** which you can also find in this folder.

# In[1]:


# to handle datasets
import pandas as pd
import numpy as np

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib


# ## Load the data set

# In[2]:


# load the data - it is available open source and online

data = pd.read_csv('titanic.csv')

# display data
data.head()


# ## Data Exploration
# 
# ### Find numerical and categorical variables

# In[3]:


target = 'survived'


# In[4]:


vars_num = [c for c in data.columns if data[c].dtypes!='O' and c!=target]

vars_cat = [c for c in data.columns if data[c].dtypes=='O']

print('Number of numerical variables: {}'.format(len(vars_num)))
print('Number of categorical variables: {}'.format(len(vars_cat)))


# In[5]:


vars_cat


# ## Separate data into train and test

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape


# ## Feature Engineering
# 
# ### Extract only the letter (and drop the number) from the variable Cabin

# In[7]:


X_train['cabin'] = X_train['cabin'].str[0] # captures the first letter
X_test['cabin'] = X_test['cabin'].str[0] # captures the first letter

X_train['cabin'].unique()


# ### Fill in Missing data in numerical variables:
# 
# - Add a binary missing indicator
# - Fill NA in original variable with the median

# In[8]:


for var in ['age', 'fare']:

    # add missing indicator
    X_train[var+'_NA'] = np.where(X_train[var].isnull(), 1, 0)
    X_test[var+'_NA'] = np.where(X_test[var].isnull(), 1, 0)

    # replace NaN by median
    median_val = X_train[var].median()
    print(var, median_val)

    X_train[var].fillna(median_val, inplace=True)
    X_test[var].fillna(median_val, inplace=True)

X_train[['age', 'fare']].isnull().sum()


# ### Replace Missing data in categorical variables with the string **Missing**

# In[9]:


X_train[vars_cat] = X_train[vars_cat].fillna('Missing')
X_test[vars_cat] = X_test[vars_cat].fillna('Missing')


# ### Remove rare labels in categorical variables
# 
# - remove labels present in less than 5 % of the passengers

# In[10]:


def find_frequent_labels(df, var, rare_perc):
    
    # function finds the labels that are shared by more than
    # a certain % of the passengers in the dataset
    
    df = df.copy()
    
    tmp = df.groupby(var)[var].count() / len(df)
    
    return tmp[tmp > rare_perc].index


for var in vars_cat:
    
    # find the frequent categories
    frequent_ls = find_frequent_labels(X_train, var, 0.05)
    print(var)
    print(frequent_ls)
    
    # replace rare categories by the string "Rare"
    X_train[var] = np.where(X_train[var].isin(
        frequent_ls), X_train[var], 'Rare')
    
    X_test[var] = np.where(X_test[var].isin(
        frequent_ls), X_test[var], 'Rare')


# ### Perform one hot encoding of categorical variables into k-1 binary variables
# 
# - k-1, means that if the variable contains 9 different categories, we create 8 different binary variables
# - Remember to drop the original categorical variable (the one with the strings) after the encoding

# In[10]:


for var in vars_cat:
    
    # to create the binary variables, we use get_dummies from pandas
    
    X_train = pd.concat([X_train,
                         pd.get_dummies(X_train[var], prefix=var, drop_first=True)
                         ], axis=1)
    
    X_test = pd.concat([X_test,
                        pd.get_dummies(X_test[var], prefix=var, drop_first=True)
                        ], axis=1)
    

X_train.drop(labels=vars_cat, axis=1, inplace=True)
X_test.drop(labels=vars_cat, axis=1, inplace=True)

X_train.shape, X_test.shape


# In[11]:


X_train.head()


# In[12]:


# we add 0 as values for all the observations, as Rare
# was not present in the test set

X_test['embarked_Rare'] = 0


# ### Scale the variables
# 
# - Use the standard scaler from Scikit-learn

# In[13]:


# create scaler
scaler = StandardScaler()

#  fit  the scaler to the train set
scaler.fit(X_train) 

# transform the train and test set
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# ## Train the Logistic Regression model
# 
# - Set the regularization parameter to 0.0005
# - Set the seed to 0

# In[14]:


# set up the model
# remember to set the random_state / seed

model = LogisticRegression(C=0.0005, random_state=0)

# train the model
model.fit(X_train, y_train)


# ## Make predictions and evaluate model performance
# 
# Determine:
# - roc-auc
# - accuracy
# 
# **Important, remember that to determine the accuracy, you need the outcome 0, 1, referring to survived or not. But to determine the roc-auc you need the probability of survival.**

# In[15]:


# make predictions for test set
class_ = model.predict(X_train)
pred = model.predict_proba(X_train)[:,1]

# determine mse and rmse
print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
print()

# make predictions for test set
class_ = model.predict(X_test)
pred = model.predict_proba(X_test)[:,1]

# determine mse and rmse
print('test roc-auc: {}'.format(roc_auc_score(y_test, pred)))
print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
print()

