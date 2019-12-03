#!/usr/bin/env python
# coding: utf-8

# In[4]:


from selenium import webdriver
import time
import re
import urllib
import urllib.request
import pandas as pd
import os


# In[1033]:


num1 = input("enter a min: ")
num2 = input("enter a max: ")
city = str(input("enter a city: "))


# In[1034]:


path = "C:/Users/tuncay/OneDrive/Coding_Projects/image classification REI"
os.chdir(path)

driver=webdriver.Chrome()

# ptypeid=32  is for single family housing

# https://www.weichert.com/CA/Orange/?ptypeid=32&minpr=200&maxpr=500

# https://www.weichert.com/CA/Orange/
# https://www.weichert.com/CA/Losangeles/
# https://www.weichert.com/CA/Sandiego/
# https://www.weichert.com/CA/Imperial/
# https://www.weichert.com/CA/Ventura/
# https://www.weichert.com/CA/Sanbernardino/
# https://www.weichert.com/CA/Riverside/
# https://www.weichert.com/CA/Santabarbara/
# https://www.weichert.com/CA/Sanluisobispo/
# https://www.weichert.com/CA/Kern/





url="https://www.weichert.com/CA/"+city+"/?ptypeid=32&minpr="+num1+"&maxpr="+num2
driver.get(url)


# In[1035]:


index=0

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

cols=["pic","street","citi","price","bed","bath","sqft"]

lst=[]
while index <=400:
    try:
#         print("Scraping Page number " + str(index))
        index = index + 1
        reviews = driver.find_elements_by_xpath('//div[@class="listing"]')        
        for review in reviews:            
           
            review_dict = []
            
            try:
                pic = review.find_element_by_xpath('.//div[@class="item active"]//img[1]').get_attribute("src")
            except:
                continue
#             print('pic = {}'.format(pic))
            
#             pic_urls=open("pic_urls.txt","w+")

#             urllib.request.urlretrieve(pic1, "pic1.png")
            
#             try:
#                 pic2 = review.find_element_by_xpath('.//div[@class ="carousel-inner"]//img').get_attribute("src")
#             except:
#                 continue
#             print('pic2 = {}'.format(pic2))  
            
            try:
                street = review.find_element_by_xpath('.//div[@class="card-text-address"]//p[1]').text
            except:
                continue
#             print('street = {}'.format(street))  

            try:
                citi = review.find_element_by_xpath('.//div[@class="card-text-address"]//p[2]').text
            except:
                continue
#             print('citi = {}'.format(citi))  
            
            try:
                price = review.find_element_by_xpath('.//div[@class="card-text-left"]').text
            except:
                continue
#             print('price = {}'.format(price))  

            try:
                bed = review.find_element_by_xpath('.//div[@class="card-pair bed"]').text
            except:
                continue
#             print('bed = {}'.format(bed))  

            try:
                bath = review.find_element_by_xpath('.//div[@class="card-pair bath"]').text
            except:
                continue
#             print('bath = {}'.format(bath))  
            
            try:
                sqft = review.find_element_by_xpath('.//div[@class="card-pair sq"]').text
            except:
                continue
            lst.append([pic,street,citi,price,bed,bath,sqft])

#             print('sqft = {}'.format(sqft)) 
            
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        button = driver.find_element_by_xpath('//li[@class="next"]/a[1]')
        button.click()
        time.sleep(2)
        
        
    except Exception as e:
        print(e)
        #driver.close()
        break
df1 = pd.DataFrame(lst, columns=cols)
df1


# In[1036]:


df1.to_csv(city+"_"+num1+"_"+num2+".csv",index=False)


# In[1037]:


# pd.options.display.max_rows = 4000
# pd.options.display.max_columns = 4000

# text_file = open("Output.txt", "w")
# text_file.write(str(df1.loc[:,"pic"]))
# text_file.close()
# (df1.loc[:,"pic"])


# In[1038]:


df_oc=pd.read_csv(city+"_"+num1+"_"+num2+".csv")
df_oc.loc[:,"pic"].head()


# In[1039]:


df_oc.head()


# In[1040]:


# import os
# detect the current working directory and print it
path = os.getcwd()
print ("The current working directory is %s" % path)


# In[1041]:


# define the name of the directory to be created
path = "C:/Users/tuncay/OneDrive/Coding_Projects/image classification REI/"+city+"_"+num1+"_"+num2
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
    
os.chdir(path)


# In[1042]:


pwd


# In[1043]:


x=0
for line in df_oc.loc[:,"pic"]:
    URL= line
    urllib.request.urlretrieve(URL, str(x) + ".jpg")
    x+=1


# In[ ]:





# In[ ]:




