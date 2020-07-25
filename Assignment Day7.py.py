#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


dataset=pd.read_csv(r'C:\Users\User\Downloads\general_data.csv')


# In[5]:


dataset.head()


# # checking and removing duplicates in dataset

# In[6]:


dataset.duplicated()


# In[7]:


dataset.drop_duplicates()


# In[8]:


dataset.describe()


# In[9]:


dataset.median()


# In[10]:


dataset.mean()


# In[11]:


dataset.mode()


# In[12]:


dataset.skew()


# # conclusions based on these skewness

# In[13]:


#all the above mentioned variables show positive skewness
#Age and Distance from home are leptokurtic 
#other variables are platyokurtic


# # box plots for outliers

# In[14]:


box_plot=dataset.Age


# In[15]:


plt.boxplot(dataset.Age)


# # Analysis from the box plot

# In[16]:


#It is proven that Age is normally distributed without outliers


# In[17]:


plt.boxplot(dataset.YearsAtCompany)


# #Analysis 

# In[18]:


#Years At company has several outliers and it is right skewed
#Median>mean is proved since it is below the center point


# In[19]:


plt.boxplot(dataset.MonthlyIncome)


# In[20]:


#Monthly income has outliers and right skewed


# In[21]:


plt.hist(dataset.Age)


# In[22]:


import matplotlib.pyplot as plt
plt.scatter(dataset.Attrition,dataset.Gender)


# In[31]:


from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
dataset["Attrition"]=lb.fit_transform(dataset["Attrition"])


# In[32]:


#for Attriton and DistanceFromHome
from scipy.stats import pearsonr
stats,p=pearsonr(dataset.Attrition,dataset.DistanceFromHome)


# In[33]:


print(stats,p)


# In[34]:


#As p value is 0.5 it is greater than0.05 we can aceept null hypothesis


# In[35]:


#With Attrition and job level

stats,p=pearsonr(dataset.Attrition,dataset.JobLevel)
print(stats,p)
#you can accept null hypothesis


# In[36]:


stats,p=pearsonr(dataset.Attrition,dataset.YearsAtCompany)
print(stats,p)


# In[37]:


plt.boxplot(dataset.YearsAtCompany)
#Since it is below the center point it is positively skewed and mean is greater than median


# # checking some parametric and non parametric test on variables

# In[38]:


from scipy.stats import mannwhitneyu


# In[39]:


stats,p=mannwhitneyu(dataset.TotalWorkingYears,dataset.YearsAtCompany)


# In[40]:


print(stats,p)


# In[41]:


#from the above value we can conclude that alternate hypothesis can be selected


# In[42]:


from scipy.stats import friedmanchisquare


# In[43]:




stats,p=friedmanchisquare(dataset.YearsAtCompany,dataset.YearsSinceLastPromotion,dataset.TrainingTimesLastYear)


# In[44]:


print(stats,p)


# In[45]:


#we can eject null hypothesis


# In[46]:


from scipy.stats import chi2_contingency


# In[47]:


chitable=pd.crosstab(dataset.Gender,dataset.Attrition)


# In[48]:


stats,p,dof,expected=chi2_contingency(chitable)


# In[49]:


print(stats,p)


# In[50]:


dataset.columns


# In[51]:


from scipy.stats import ttest_ind


# In[52]:


stats,p=ttest_ind(dataset.DistanceFromHome,dataset.YearsWithCurrManager)


# In[53]:


print(stats,p)


# In[54]:


#we can accept null hypothesis


# In[ ]:




