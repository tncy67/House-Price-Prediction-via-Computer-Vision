#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import os
import glob
import urllib.request


# In[4]:


os.getcwd()


# In[6]:


path="C:/Users/tuncay/OneDrive/Coding_Projects/image classification REI/csvs"


# In[7]:


os.chdir(path)


# In[8]:


pwd


# In[18]:


#glob
# print(glob.glob("*.csv"))


# In[39]:


# frames=glob.glob("*.csv")


# In[37]:


# [x.strip("") for x in frames]

# [i.replace("'", "") for i in frames]

# ','.join(frames)


# In[40]:


path = r'C:/Users/tuncay/OneDrive/Coding_Projects/image classification REI/csvs' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)


# In[43]:


frame.sample(10)


# In[45]:


frame.to_csv("all_socal_merged.csv",index=False)


# In[46]:


socal=pd.read_csv("all_socal_merged.csv")


# In[50]:


# define the name of the directory to be created
path = "C:/Users/tuncay/OneDrive/Coding_Projects/image classification REI/socal_pics"
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
    
os.chdir(path)


# In[51]:


pwd


# In[52]:


socal.loc[:,"pic"]


# In[55]:


x=0
for line in socal.loc[:,"pic"]:
    URL= line
    urllib.request.urlretrieve(URL, str(x) + ".jpg")
    x+=1


# In[ ]:




