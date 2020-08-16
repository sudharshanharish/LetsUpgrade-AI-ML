#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset=pd.read_csv(r'C:\Users\User\Downloads\train.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.columns


# In[5]:


from sklearn import preprocessing


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


dataset.drop(['Cabin'],axis=1)


# In[8]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[9]:


le=preprocessing.LabelEncoder()
le.fit(dataset["Sex"])
print(le.classes_)


# In[10]:


dataset["Sex"]=le.fit_transform(dataset["Sex"])
dataset["Name"]=le.fit_transform(dataset["Name"])


# In[11]:


y=dataset["Sex"]
X=dataset.drop(["Sex","PassengerId"],axis=1)


# In[12]:


dataset.isna().sum()


# In[13]:


y.count()


# In[14]:


dataset.corr()


# In[15]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[16]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

GNBclassifier=GaussianNB()


# In[17]:


GNBclassifier.fit(X_train, y_train)


# In[ ]:




