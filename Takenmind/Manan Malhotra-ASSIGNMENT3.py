#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Manan Malhotra-ASSIGNMENT3.csv")
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df.shape


# In[ ]:



# df = sns.load_dataset('flights')
# df2 = df.pivot('year','month','passengers')
# print df2

# #sns.heatmap(df2).get_figure().savefig('heatmap1.png')
# #sns.heatmap(df2,annot=True,fmt='d').get_figure().savefig('heatmap2.png')

# sns.heatmap(df2, center=df2.loc[1955,'January']).get_figure().savefig('heatmap3.png')


# In[11]:


df2 = df.pivot('country','year','lifeExp')
print(df2)


# In[30]:


sns.heatmap(df2).get_figure().savefig('heatmap1.png')
plt.show()

