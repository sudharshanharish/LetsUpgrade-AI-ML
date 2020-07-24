#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset=pd.read_csv(r'C:\Users\User\Downloads\general_data.csv')


# In[3]:


dataset.describe()


# In[4]:


dataset.head()


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


from scipy.stats import wilcoxon


# In[7]:


from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
dataset["Attrition"]=lb.fit_transform(dataset["Attrition"])


# In[8]:


stats,p=wilcoxon(dataset.Attrition,dataset.EmployeeID)


# In[9]:


print(stats,p)


# In[10]:


from scipy.stats import friedmanchisquare


# In[11]:


stats,p=friedmanchisquare(dataset.YearsAtCompany,dataset.YearsSinceLastPromotion,dataset.TrainingTimesLastYear)


# In[12]:


print(stats,p)


# In[13]:


if(p<0.05):
    print("we can reject null hypothesis")
else:
    print("we can accept null hypothesis")


# In[14]:


from scipy.stats import mannwhitneyu


# In[15]:


stats,p=mannwhitneyu(dataset.YearsSinceLastPromotion,dataset.YearsSinceLastPromotion)


# In[16]:


print(stats,p)


# In[17]:


if(p<0.05):
    print("we can reject null hypothesis")
else:
    print("we can accept null hypothesis")


# In[18]:


dataset.corr()


# In[19]:


from scipy.stats import mannwhitneyu


# In[20]:


stats,p=mannwhitneyu(dataset.TotalWorkingYears,dataset.YearsAtCompany)


# In[21]:


print(stats,p)


# In[22]:


#we can accept alternate hypothesis

dataset[['TotalWorkingYears','YearsAtCompany']].skew()


# In[23]:


import matplotlib.pyplot as plt
plt.scatter(dataset.TotalWorkingYears,dataset.YearsAtCompany)


# In[24]:


dataset.dropna()


# In[25]:


from scipy.stats import chi2_contingency


# In[26]:


chitable=pd.crosstab(dataset.Gender,dataset.Attrition)


# In[27]:


stats,p,dof,expected=chi2_contingency(chitable)


# In[28]:


print(stats,p)


# In[29]:


#since p value is greater than 0.05 we can accept null hypothesis statement 


# In[30]:


from scipy.stats import ttest_ind


# In[31]:


dataset.columns


# In[39]:


a1=dataset.DistanceFromHome
a2=dataset.YearsWithCurrManager


# In[40]:


stats,p=ttest_ind(a1,a2)
print(stats,p)


# In[ ]:


#since p value is greater than 0.05 we can accept null hypothesis

