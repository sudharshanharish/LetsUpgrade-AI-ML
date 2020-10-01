#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import numpy as np
import pandas as pd


# 

# In[2]:


dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)
dataset.head()


# ### Inspecting Data

# In[3]:


dataset.info()


# In[4]:


dataset.dtypes


# - Dataset have only 2 columns : one is in textual format and other is binary categorical ( 0 or 1 )

# In[5]:


dataset.groupby('Liked').size()


# -  Dataset is Balanced 
# - It doesn't have any null values

# In[6]:


dataset.isnull().sum()


# #### Objective : 
# - Clean and Preprocess a single review then create a for loop for cleaning all 1000 reveiws
# 
# 
# 
# #### First review 

# In[7]:


dataset['Review'][0]


# In[8]:


# Removing Numbers and Punctuations with the help of Rgular Expressions

import re

review = re.sub( '[^a-zA-Z]', ' ', dataset['Review'][0] )
print(review)


# 

# In[9]:


# Convert the string to lower 

review = review.lower()
review


# In[13]:


import nltk
nltk.download('stopwords') #------- download stopwords

from nltk.corpus import stopwords


# In[14]:


# stopwords.words('english')
len(stopwords.words('english'))


# - There are total 179 stopwords in english language

# In[16]:


review = review.split()
review


# In[17]:


# By list comprehension, we tried to remove the stop word 

review1 = [ word for word in review if not word in set(stopwords.words('english')) ]
review1


# # Stemming:
# - Convert word to its root word
# 
# Eg:
# loved ----> love

# In[18]:


# Use Stemming to take word it to its Root form

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

review1 = [ ps.stem(word) for word in review1 ]
review1


# In[19]:


# Convert list to string 

review2 = ' '.join(review1)
review2


# ### Count-Vectorizer( )
# - This will construct the vocabulary of the bag-of-words model and transform the sentences into sparse feature vectors

# In[20]:


corpus1 = []

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3) # max-feature=3 means take only top 3 columns into consideration
print(review2)

corpus1.append(review2)
print(corpus1)

X = cv.fit_transform(corpus1)
print(X.toarray())


# - Now the textual data is preprocessed and converted into numerical format, which we can use for ML model
# 
# #### Preprocessing all the rows :

# In[20]:


dataset.shape


# In[21]:


dataset.tail()


# - There are 1000 rows (from 0 to 999)
# 
# #### Preprocessing 1000 rows

# In[21]:


corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    review = ' '.join(review)
    # print(review)
    corpus.append(review)


# In[22]:


print("Review Type: ",type(review))
print("Corpus Type: ",type(corpus))


# #### Creating DataFrame for Preprocessed Reviews

# In[23]:


corpus_dataset = pd.DataFrame(corpus)
corpus_dataset.head()


# In[24]:


corpus_dataset['corpus'] = corpus_dataset
corpus_dataset = corpus_dataset.drop([0], axis=1)
corpus_dataset.head()


# In[25]:


# Saving pre-processed dataset for future reference: 
corpus_dataset.to_csv("corpus_dataset.csv")


# ### Bag of Words Model for whole data

# In[26]:


# Create a Bag of Words Model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)


# In[27]:


X = cv.fit_transform(corpus).toarray()
X[0]


# In[28]:


X


# - Sparse matrix is created for top 1500 columns

# In[29]:


# To see  all the top 1500 seleceted feature names: 
# cv.get_feature_names()
len(cv.get_feature_names())


# In[30]:


# As our input data is in numpy format so changing y(target variable) in numpy array
y = dataset.iloc[:,1].values


# ### Splitting Data into 80-20 ratio

# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# ### Naive Bayes

# In[49]:


from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)


# In[50]:


y_pred = classifier.predict(X_test)


# In[51]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[52]:


confusion_matrix(y_test,y_pred)


# In[53]:


accuracy_score(y_test,y_pred)


# - Model is not too good nor too bad as it is 77% accurate in predicting review either positive or negative

# ### Check it on Unseen Data

# In[54]:


Review = "nice service"
input1 = [Review]

input_data = cv.transform(input1).toarray()

input_pred = classifier.predict(input_data)

if input_pred[0]==1:
    print("Review is Positive")
else:
    print("Review is Negative")


# In[55]:


Review = "long waiting time"
input1 = [Review]

input_data = cv.transform(input1).toarray()

input_pred = classifier.predict(input_data)

if input_pred[0]==1:
    print("Review is Positive")
else:
    print("Review is Negative")


# In[ ]:




