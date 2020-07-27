#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# creating 3*3*3 array


# In[1]:


import numpy as np
x = np.random.random((3,3,3))
print(x)


# In[2]:


#creating 5*5 matrix
a = np.diag(1+np.arange(4), k = -1)
print (a)


# In[4]:


#Create a 8x8 matrix and fill it with a checkerboard pattern

Z = np.zeros ((8,8), dtype=int)
Z[1::2, ::2]= 1
Z[::2, 1::2] = 1
print (Z)


# In[5]:


#Normalize a 5x5 random matrix

Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z= (Z-Zmin)/(Zmax-Zmin)
print (Z)


# In[6]:


#How to find common values between two arrays?
import numpy as np
array1 = np.array([0, 10, 2, 4, 6])
print("Array1: ",array1)
array2 = [10, 3, 4]
print("Array2: ",array2)
print("Common values between two arrays:")
print(np.intersect1d(array1, array2))


# In[7]:


#finding dates 
import numpy as np
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
print("Yestraday: ",yesterday)
today     = np.datetime64('today', 'D')
print("Today: ",today)
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print("Tomorrow: ",tomorrow)


# In[8]:


#checking the arrays
import numpy as np
x = np.random.randint(0,2,6)
print("First array:")
print(x)
y = np.random.randint(0,2,6)
print("Second array:")
print(y)
print("Test above two arrays are equal or not!")
array_equal = np.allclose(x, y)
print(array_equal)


# In[9]:


#Create random vector of size 10 and replace the maximum value by 0
import numpy as np
x = np.random.random(15)
print("Original array:")
print(x)
x[x.argmax()] = -1
print("Maximum value replaced by -1:")
print(x)


# In[14]:


#printing all values in array

#Subtract the mean of each row of a matrix

import numpy as np
print("Original matrix:\n")
X = np.random.rand(5, 10)
print(X)
print("\nSubtract the mean of each row of the said matrix:\n")
Y = X - X.mean(axis=1, keepdims=True)
print(Y)


# In[15]:


#.Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with 

#repeated indices)?

#How to get the diagonal of a dot product?
arr = [1, 5, 2, 1, 3, 2, 1]  
n = len(arr) 
print(mostFrequent(arr, n)) 


# In[16]:


#How to get the n largest values of an array

import numpy as np
x = np.arange(10)
print("Original array:")
print(x)
np.random.shuffle(x)
n = 1
print (x[np.argsort(x)[-n:]])


# In[17]:


#creating record array

import numpy as np
arra1 = np.array([("Yasemin Rayner", 88.5, 90),
                 ("Ayaana Mcnamara", 87, 99),
             ("Jody Preece", 85.5, 91)])
print("Original arrays:")
print(arra1)
print("\nRecord array;")
result = np.core.records.fromarrays(arra1.T,
                              names='col1, col2, col3',
                              formats = 'S80, f8, i8')
print(result)


# In[ ]:




