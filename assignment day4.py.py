#!/usr/bin/env python
# coding: utf-8

# In[1]:


#operations on complex numbers

a = 6+4j
b = 3+6j
print(a+b)

print(a-b) 
a = 6+4j
b = 3+2j
c = a*b
c = (6+4j)*(3+2j)
print(c)

a = 2+4j
b = 1-2j
print(a/b)

a=1+5j
b=6+8j
c=a//b
print(c)


s=1+2j
print(abs(s))


# In[3]:


#range function

#range() is a built in function in python and used to generate sequence of numbers to the given range 
#for example x=range(7) means it will generate numbers from 0 to 6 excluding 7. we can also give the starting range and the ending range as paramters 

sum=0

for i in range(1,6):
    sum=sum+i
print(sum)


# In[6]:


# operation on numbers
num1=10
num2=20
num3=num2-num1
if num3>25:
    print(num2*num1)
else:
    print(num2/num1)
    
    


# In[18]:


#list

list=[100,101,102,103,104,105,106,107,108,109]
num1=2 
new_list=[i/num1 for i in list]
print("square of that number minus 2")
    


# In[ ]:




