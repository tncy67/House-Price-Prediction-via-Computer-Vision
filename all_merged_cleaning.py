#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df=pd.read_csv("all_socal_merged.csv")


# In[77]:


df.head()


# In[73]:


df.dtypes


# In[5]:


df.columns[df.isnull().any()]


# In[10]:


df["price"]=df.price.str.replace("$","")
df["price"]=df.price.str.replace(",","")


# In[46]:


df["bed2"]=(df["bed"].str.split(pat = "\n",expand=True))[0]


# In[54]:


df["bath2"]=(df["bath"].str.split(pat = "\n",expand=True))[0]


# In[57]:


df["sqft2"]=(df["sqft"].str.split(pat = "\n",expand=True))[0]


# In[58]:


df["sqft2"]=df.sqft2.str.replace(",","")


# In[60]:


del df["bed"]
del df["bath"]
del df["sqft"]


# In[63]:


df.columns = ["image_id","pic_link","street","citi","price","bed","bath","sqft"]


# In[82]:


df2.to_csv("socal.csv",index=False)


# In[68]:


df["citi"].nunique()


# In[78]:


df["n_citi"].nunique()


# In[74]:


# from sklearn.preprocessing import LabelEncoder


# In[72]:


df["citi"]=df["citi"].astype("category")


# In[75]:


df["n_citi"]=df["citi"].cat.codes


# In[79]:


df2=df[["image_id","pic_link","street","citi","n_citi","bed","bath","sqft","price"]]


# In[80]:


df2.head()


# In[ ]:





# In[ ]:




