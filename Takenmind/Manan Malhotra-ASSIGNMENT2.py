#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd


# In[30]:


excelfile=pd.ExcelFile("Manan Malhotra-ASSIGNMENT2.xlsx")


# In[31]:


df1=excelfile.parse("Sheet1")
df1.head()


# In[42]:


df1.to_csv("Sheet1.csv",sep=",")


# In[33]:


df2=excelfile.parse("Sheet2")
df2.head()


# In[ ]:


df2.to_csv("Sheet2.csv",sep=",")


# In[34]:


df3=excelfile.parse("Sheet3")
df3.head()


# In[ ]:


df3.to_csv("Sheet3.csv",sep=",")


# In[35]:


df4=excelfile.parse("Sheet4")
df4.head()


# In[ ]:


df4.to_csv("Sheet4.csv",sep=",")


# In[36]:


df5=excelfile.parse("Sheet5")
df5.head()


# In[ ]:


df5.to_csv("Sheet5.csv",sep=",")


# In[37]:


df6=excelfile.parse("Sheet6")
df6.head()


# In[ ]:


df6.to_csv("Sheet6.csv",sep=",")


# In[38]:


df7=excelfile.parse("Sheet7")
df7.head()


# In[ ]:


df7.to_csv("Sheet7.csv",sep=",")


# In[39]:


df8=excelfile.parse("Sheet8")
df8.head()


# In[ ]:


df8.to_csv("Sheet8.csv",sep=",")


# In[40]:


df9=excelfile.parse("Sheet9")
df9.head()


# In[ ]:


df9.to_csv("Sheet9.csv",sep=",")


# In[41]:


df10=excelfile.parse("Sheet10")
df10.head()


# In[ ]:


df10.to_csv("Sheet10.csv",sep=",")

