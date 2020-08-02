#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset1=pd.read_csv(r'C:\Users\User\Downloads\titles.csv')


# 

# In[3]:


dataset1.head()


# # How many movies are listed in the titles dataframe?
# 

# In[4]:


print(len(dataset1))


# # What are the earliest two films listed in the titles dataframe?

# In[5]:


print(dataset1.sort_values('year').head(2))


# # How many movies have the title "Hamlet"?

# In[6]:


print(len(dataset1[dataset1.title=="Hamlet"]))


# # How many movies were made in the year 1950?

# In[7]:


print(len(dataset1[dataset1.year==1950]))


# # How many movies are titled "North by Northwest"?

# In[8]:


count_north=0
for i in dataset1:
    if "North by NorthWest" in i:
        count_north+=1
print(f"{count_north} movie is titled")


# # When was the first movie titled "Hamlet" made?

# In[ ]:





# # How many movies were made in the year 1950?

# In[9]:


moviesIn_1950=dataset1_title.year==1950
print(dataset1_title[moviesIn_1950]).shape[0],"movies were made in the year 1950")


# # How many movies were made from 1950 through 1959?
# 

# In[11]:


mov=(dataset1['year']>1949)& (dataset1['year']<1960)
x=dataset1.loc[mov]
print(x.count())


# # In what years has a movie titled "Batman" been released?

# In[13]:


search=dataset1.batman['year']
print(search)


# # How many roles were there in the movie "Inception"?

# In[17]:


dataset2=pd.read_csv(r'C:\Users\User\Downloads\cast.csv')
dataset2.head()


# In[21]:


dataset2=pd.read_csv(r'C:\Users\User\Downloads\cast.csv')
dataset2.head(1)


# # But how many roles in the movie "Inception" did receive an "n" value?

# In[22]:


d=dataset2[dataset2["n"].isnull()]
print("roles",d.iloc[:,5:].size)


# # Display the cast of "North by Northwest" in their correct "n"-value order, ignoring roles that did not earn a numeric "n" value.
# 

# In[25]:


cast2=(dataset2['title']=='North by NorthWest')
x=dataset2.loc[cast2]
print(x['name'])


# # How many roles were credited in the silent 1921 version of Hamlet?
# 

# In[27]:


df=dataset2.set_index('title').loc["Hamlet"]
df=df.set_index('year').loc[1921]
df[df['character']=='the silent'].count()


# # How many roles were credited in Branagh’s 1996 Hamlet?

# In[26]:


de=dataset2[(dataset2["title"]=="Hamlet")& (dataset2["year"]==1996)]
de


# In[ ]:




