#!/usr/bin/env python
# coding: utf-8

# In[3]:


#program  to find first 20 non even prime natural numbers
from sympy import isprime

count=0
for i in range(100):
    if (isprime(i) and count <20):
        if i%2 !=0:
            print(i)
            count+=1


# In[21]:


#functions of a string

text="hello welcome"

x=text.capitalize()
print(x)


txt = "I love apples, apple are my favorite fruit"

x = txt.count("apple", 10, 24)

print(x)


y = text.index("welcome")

print(y)

y=text.isdigit()
print(y)

y=text.isdecimal()
print(y)

y=text.isupper()
print(y)

text1='98'
z=text1.isnumeric()
print(z)

str='hello everyone'
a=str.replace("hello", "all")
print(a)

a=str.startswith('h')
print(a)

txt = "Company12"

x = txt.isalnum()

print(x)


# In[1]:


#palindrome or anagram

string1=input("Enter the word")
rev_string=string1[::-1]
if rev_string==string1:
    print("it is palindrome")
else:
    string2=input("Enter the word:")
    str1=sorted(string1)
    str2=sorted(string2)
    if str1==str2:
        print("Anagram")
    else:
        print("none")
    


# In[9]:


#user defined function to remove special characters

def replace_char(x):
    list=['!','@','#','$','%','^','&','*']
    for i in list:
        x=x.replace(i,'')
        x=x.lower()
    print("After removing all special characters from string")
    print("Resultant string is",x)
string_s='Dr.darshan ingle @AI-ML Trainer'
print("String is ",string_s)
replace_char(string_s)
    


# In[ ]:




