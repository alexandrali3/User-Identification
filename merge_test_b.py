
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')


# In[2]:


import sys
import numpy as np
import pandas as pd
import re as re
import matplotlib.pyplot as plt
import xgboost as xgb


# In[20]:


test_b_df = pd.read_csv('../data/Test-B/test_b_feature_21to27.csv', header = None, index_col=0)
test_b_df.index.name = 'uid'


# In[21]:


print(test_b_df)


# In[23]:


id_series = test_b_df.index


# In[24]:


cnt = long(85)


# In[25]:


tmp = pd.read_csv('../data/Test-B/test_b_feature_28to33.csv', header = None, index_col = 0)
tmp.index.name = 'uid'
for i in range(0, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1


# In[26]:


print(tmp)


# In[27]:


test_b_df = test_b_df.join(tmp)


# In[28]:


tmp = pd.read_csv('../data/Test-B/test_b_feature_34to41.csv', header = None, index_col = 0)
tmp.index.name = 'uid'
for i in range(0, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1


# In[29]:


print(tmp)


# In[30]:


test_b_df = test_b_df.join(tmp)


# In[33]:


test_b_df.to_csv('../data/Test-B/test_b_feature_21to41.csv', index=True)

