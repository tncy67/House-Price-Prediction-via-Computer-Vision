#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import glob
import random


# In[1]:


pwd


# In[3]:


path= 'C:/Users/tuncay/OneDrive/Coding_Projects/image classification REI/socal_pics'
os.chdir(path)


# In[4]:


pwd


# In[8]:


pics=(glob.glob("*.jpg"))


# In[10]:


len(pics)


# In[15]:


random.shuffle(pics)


# In[18]:


split_1 = int(0.8 * len(pics))


# In[22]:


train_pics = pics[:split_1]


# In[24]:


len(train_pics)


# In[25]:


split_2 = int(0.9 * len(pics))


# In[26]:


test_pics = pics[split_2:]


# In[28]:


len(test_pics)


# In[30]:


dev_pics = pics[split_1:split_2]


# In[31]:


len(test_pics)+len(train_pics)+len(dev_pics)


# In[ ]:




