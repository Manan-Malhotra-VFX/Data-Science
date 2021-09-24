#!/usr/bin/env python
# coding: utf-8

# # Accessing Database From Python

# In[1]:


#import pandas as pd
#import os
import warnings
warnings.filterwarnings('ignore')


# ### Install the pymql package. This needs to be done only once

# In[ ]:


get_ipython().system('pip install pymysql')


# In[2]:


import pymysql


# In[ ]:


#conn = pymysql.connect(host = 'localhost',port=3306,user='root', passwd='secret',db='excelr')


# In[3]:


conn = pymysql.connect(host = 'localhost',port=3306, user='root',  db='practicedb', autocommit=True)


# ### Using the Cursor Object

# In[4]:


# prepare a cursor object using cursor() method
cursor = conn.cursor()


# In[5]:


# Drop table if it already exist using execute() method.
cursor.execute("DROP TABLE IF EXISTS PyBooks")


# In[6]:


# Create table as per requirement
sql = """CREATE TABLE PyBooks (
   bookid  INT PRIMARY KEY AUTO_INCREMENT,
   bookname  CHAR(50) unique,
   authorname CHAR(30))"""


# In[7]:


cursor.execute(sql)


# ### Insert Records

# In[8]:


sql1 = """INSERT INTO PyBooks(bookname, authorname) 
          VALUES ('Master Python','Geethika')"""

cursor.execute(sql1)  


# ### Transaction Control

# In[10]:



sql1 = """INSERT INTO PyBooks(bookname, authorname) 
          VALUES ('Master Python','Geethika')"""

sql2 = """INSERT INTO PyBooks(bookname, authorname) 
          VALUES ('Python Bible','Geethika')"""
sql3 = """INSERT INTO PyBooks(bookname, authorname) 
          VALUES ('Python Bible','Geethika')"""

try:  
    # Start a transaction
    conn.begin() 
    # Execute the command
    cursor.execute(sql1)  
    cursor.execute(sql2)  
    cursor.execute(sql3)  
    
    # Commit the transaction
    conn.commit()
except:
    # Rollback transaction in case of errors
    print("Duplicate Entry .. Rolling Back")
    conn.rollback()
    
print("Done")


# In[11]:


sql = """SELECT * FROM books 
      WHERE aid = 5 """ 


# In[12]:


# Execute the command
cursor.execute(sql)


# In[13]:


# Fetch all the rows in a list of lists.
results = cursor.fetchall()


# In[14]:


results


# In[15]:


for row in results:
    bookid = row[0]
    title = row[1]
    authorid = row[2]
  
    # Now print fetched result
    print ("bookid = %d,title = %s,author is = %d" %      (bookid, title , authorid ))


# In[ ]:


#results


# ## Invoking a procedure

# In[ ]:


cursor.callproc('myproc')

 
# print out the result
results = cursor.fetchall()
for result in results:
    print(result)


# In[ ]:


sql = """select bookid, title, auth_name(authorid) from books; """ 


# In[ ]:


# Execute the command
cursor.execute(sql)
# print out the result
for result in cursor.fetchall():
    print(result)


# In[ ]:


# disconnect from server
conn.close()
