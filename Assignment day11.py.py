#!/usr/bin/env python
# coding: utf-8

# In[2]:


#  comparing Attrition with other columns

import os


# In[3]:


os.path.isfile('Pictures/general_data.csv')


# In[5]:


import numpy as np


# In[22]:


import pandas as pd
import matplotlib.pyplot as plt


# In[25]:


dataset=pd.read_csv(r'C:\Users\User\Downloads\general_data.csv')


# In[26]:


dataset.head()


# In[10]:


from scipy.stats import pearsonr


# In[16]:


stats,p=pearsonr(df.Age,df.YearsAtCompany)


# In[18]:


print(stats,p)


# In[27]:


from sklearn import preprocessing


# In[28]:


lb = preprocessing.LabelBinarizer()


# In[31]:


dataset["Attrition"]=lb.fit_transform(dataset["Attrition"])


# In[33]:


#for Attriton and DistanceFromHome
from scipy.stats import pearsonr
stats,p=pearsonr(dataset.Attrition,dataset.DistanceFromHome)


# In[34]:


print(stats,p)


# In[35]:


dataset[['Attrition','DistanceFromHome']].skew()


# In[36]:


plt.boxplot(dataset.Attrition)


# In[37]:


plt.boxplot(dataset.DistanceFromHome)


# In[38]:


stats,p=pearsonr(dataset.Attrition,dataset.Education)


# In[39]:


print(stats,p)


# In[41]:


#since p value is greater than 0.05 we can accept the null hypothesis
plt.scatter(dataset.Attrition,dataset.Education)


# In[42]:


dataset.corr()


# In[45]:


#with Attrition and EmployeeID

stats,p=pearsonr(dataset.Attrition,dataset.EmployeeID)
print(stats,p)


# In[48]:


#since p value is gretaer than 0.05 we can accept the null hypothesis

#With Attrition and job level

stats,p=pearsonr(dataset.Attrition,dataset.JobLevel)
print(stats,p)
#you can aceept null hypothesis


# In[50]:


stats,p=pearsonr(dataset.Attrition,dataset.MonthlyIncome)
print(stats,p)


# In[51]:


#since p value is less than 0.05 
#you can reject null hypothesis


# In[54]:


stats,p=pearsonr(dataset.Attrition,dataset.YearsAtCompany)
print(stats,p)


# In[55]:


dataset[['Attrition','YearsAtCompany']].skew()


# In[58]:


plt.boxplot(dataset.YearsAtCompany)
#Since it is below the center point it is positively skewed and mean is greater than median


# In[60]:


stats,p=pearsonr(dataset.Attrition,dataset.PercentSalaryHike)
print(stats,p)

#since p value is less than 0.05 you can reject null hypothesis 


# In[62]:


plt.boxplot(dataset.PercentSalaryHike)


# In[65]:


stats,p=pearsonr(dataset.Attrition,dataset.StockOptionLevel)
print(stats,p)


# In[66]:


#since p value is greater than you can accept null hypothesis


# In[71]:


stats,p=pearsonr(dataset.Attrition,dataset.TrainingTimesLastYear)
print(stats,p)


# In[ ]:


#since p value is less than 0.05 you can reject null and go for alternate hypothesis

