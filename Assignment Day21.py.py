#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# In[2]:


df=pd.read_excel(r'C:/Users/User/Downloads/Bank_Personal_Loan_Modelling.xlsx',sheet_name='Data')


# In[3]:


df.head()


# In[4]:


df1 = df.drop("ID", axis=1)
df1


# In[5]:


df2=df1.drop("ZIP Code",axis=1)
df2


# In[6]:


y=df2['Personal Loan']


# In[7]:


x=df2[['Age','Experience','Income','Family','Education','Online']]


# In[8]:


import statsmodels.api as sm


# In[9]:


x1=sm.add_constant(x)


# In[10]:


logistic_regression=sm.Logit(y,x1)


# In[11]:


result=logistic_regression.fit()


# In[12]:


result.summary()


# # Inferences from result

# In[13]:


#since p value is deciding factor more for Age and online feature it affects more the Dependent variable
#p value is less for other features like const,Experience,Income,Family


# #     Attrition analysis

# In[14]:


dataset=pd.read_csv(r'C:/Users/User/Downloads/general_data (2).csv')


# In[15]:


dataset.head()


# In[16]:


y3=dataset['Attrition']


# In[17]:


x3=dataset[['Age','BusinessTravel','Department','Education','DistanceFromHome','EmployeeCount','EmployeeID','Gender']]


# In[18]:


import statsmodels.api as sm1


# In[19]:


x2=sm1.add_constant(x3)


# logistic_regression1=sm1.Logit(y3,x2)

# In[20]:


logistic_regression1=sm1.Logit(y3,x2)


# # Multiple linear Regression 

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[22]:


from sklearn import linear_model


# In[23]:


dataset2=pd.read_csv(r'C:/Users/User/Downloads/Linear Regression.xlsx')


# In[ ]:




