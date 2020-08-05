#!/usr/bin/env python
# coding: utf-8

# In[116]:


# import libraries

import pandas as pd
from sqlalchemy import create_engine
import sys


# In[117]:


# load messages dataset

messages = pd.read_csv('messages.csv')
messages.head()


# In[118]:


# load categories dataset

categories = pd.read_csv('categories.csv')
categories.head()


# In[119]:


# merge datasets

df = pd.merge(messages,categories, on =["id"])
df.head()


# In[120]:


# create a dataframe of the 36 individual category columns

categories = df['categories'].str.split(';',expand=True)
categories.head()


# In[121]:


# select the first row of the categories dataframe
row = categories.iloc[0]
category_colnames = row
print(category_colnames)


# In[122]:


# rename the columns of `categories`
categories.columns = category_colnames
categories.head()


# In[123]:


#  set each value to be the last character of the string


for column in categories:
        # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: x[-1:]).values

        # convert column from string to numeric
    categories[column] = categories[column].astype(int)
category_colnames = row.apply(lambda x: x[:-2]).values
categories.columns = category_colnames
print(categories.rename(columns={'':'related'}, inplace=True))
# convert column from string to numeric


# In[125]:


categories.columns


# In[126]:


# concatenate the original dataframe with the new `categories` dataframe
df = df.join(categories,how='outer')
df.head()


# In[127]:


# check number of duplicates
df.duplicated().sum()


# In[128]:


# drop duplicates
df = df.drop_duplicates(keep='first')


# In[129]:


# check number of duplicates
df.duplicated().sum()


# In[130]:


engine = create_engine('sqlite:///DisasterResponse.db')
df.to_sql('Disaster_Table', engine, index=False, if_exists = 'replace')


# In[ ]:




