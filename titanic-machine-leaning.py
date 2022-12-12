#!/usr/bin/env python
# coding: utf-8

# # Titanic Survivor Prediction

# **Objective:**  use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
# 
# **Dataset Link:** https://www.kaggle.com/competitions/titanic/data
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, the aim to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

# ### Downloading libraries and dataset

# In[1]:


get_ipython().system('pip install numpy pandas-profiling matplotlib seaborn --quiet')
get_ipython().system('pip install jovian opendatasets xgboost graphviz lightgbm scikit-learn xgboost lightgbm --upgrade --quiet')
import os
import opendatasets as od
import pandas as pd
od.download('https://www.kaggle.com/competitions/titanic/data')
data_dir='./titanic'
os.listdir(data_dir)
raw_train_data=pd.read_csv(data_dir+'/train.csv')
raw_test_data=pd.read_csv(data_dir+'/test.csv')


# In[2]:


raw_train_data


# In[3]:


raw_test_data


# In[4]:


raw_train_data.info()


# ### Feature Engineering and Data Preparation

# In[5]:


t=raw_train_data["Ticket"].nunique()
n=raw_train_data["Name"].nunique()
c=raw_train_data["Cabin"].nunique()
print("Unique Ticket Values:{0}\nUnique Name Values:{1}\nUnique Cabin Values:{2}".format(t,n,c))


# As number of unique values are very high in these 3 columns, they will have little or no effect on the predictions. Hence we can ignore these columns. 

# In[6]:


target_col='Survived'
input_col=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
num_cols=['Pclass','Age', 'SibSp', 'Parch', 'Fare']


# In[7]:


raw_train_data[input_col].isna().sum()


# In[8]:


raw_test_data[input_col].isna().sum()


# In[9]:


from sklearn.impute import SimpleImputer
Imputer=SimpleImputer(strategy='mean').fit(raw_train_data[num_cols].append(raw_test_data[num_cols]))
raw_train_data[num_cols] = Imputer.transform(raw_train_data[num_cols])
raw_test_data[num_cols] = Imputer.transform(raw_test_data[num_cols])

raw_train_data['Sex']=raw_train_data['Sex'].map({"male":1,"female":0})
raw_test_data['Sex']=raw_test_data['Sex'].map({"male":1,"female":0})
raw_train_data['Embarked']=raw_train_data['Embarked'].map({"Q":1,"S":0,"C":0.5})
raw_test_data['Embarked']=raw_test_data['Embarked'].map({"Q":1,"S":0,"C":0.5})

X=raw_train_data[['PassengerId']+input_col].fillna(0)
Test=raw_test_data[['PassengerId']+input_col].fillna(0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X[input_col].append(Test[input_col]))
X[input_col] = scaler.transform(X[input_col])
Test[input_col] = scaler.transform(Test[input_col])


# In[10]:


X.sample(5)


# In[11]:


Test.sample(5)


# In[12]:


target=raw_train_data[[target_col]]
target.sample(5)


# In[13]:


from sklearn.model_selection import train_test_split
train_input,val_input,target_input,target_val=train_test_split(X,target,test_size=0.2)


# ### Training Models

# In[14]:


get_ipython().system('pip install xgboost --quiet')
from xgboost import XGBClassifier
XGBmodel=XGBClassifier(eval_metric='mlogloss',learning_rate=None, max_depth=15, max_leaves=None,n_estimators=10, n_jobs=-1,random_state=42)
XGBmodel.fit(train_input,target_input)
XGBmodel.score(val_input,target_val)


# In[15]:


XGBmodel=XGBClassifier(eval_metric='mlogloss',learning_rate=None, max_depth=25, max_leaves=3,n_estimators=10, n_jobs=-1,random_state=42).fit(X[input_col],target)
Test["Survival"]=XGBmodel.predict(Test[input_col])
Submission=Test[["PassengerId","Survival"]]
Submission.to_csv(r'C:\Users\91981\Desktop\Submission.csv', index=False, header=True)
print(Submission)

