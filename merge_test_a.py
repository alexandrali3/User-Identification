
# coding: utf-8

# In[2]:


get_ipython().magic(u'matplotlib inline')


# In[3]:


import sys
import numpy as np
import pandas as pd
import re as re
import matplotlib.pyplot as plt
import xgboost as xgb


# In[3]:


def myPrint(x):
    print(x.head(15))
    print(x.info())
def myPrint2(x):
    print(x)


# In[5]:


test_a_df = pd.read_csv('../data/Test-A/test_a_feature_1.csv', header = None, names = ['uid', '1'])
id_series = test_a_df['uid']
test_a_df = test_a_df.set_index('uid')


# In[6]:


tmp = pd.read_csv('../data/Test-A/test_a_feature_2to5.csv', header = 0, names = ['uid', '2', '3', '4', '5'])
tmp = tmp.set_index('uid')


# In[7]:


test_a_df = test_a_df.join(tmp[['2', '3', '4', '5']])


# In[8]:


tmp = pd.read_csv('../data/Test-A/test_a_feature_6.csv', header = 0, names = ['uid', '6'])
tmp = tmp.set_index('uid')


# In[9]:


test_a_df = test_a_df.join(tmp)


# In[10]:


for i in range(7, 21):
    tmp = pd.read_csv('../data/Test-A/test_a_feature_' + str(i) + '.csv', header = None, names = ['uid', str(i)])
    tmp = tmp.set_index('uid')
    test_a_df = test_a_df.join(tmp)


# In[11]:


test_a_df.to_csv('../data/Test-A/test_a_feature_1to20.csv', index=True)


# In[12]:


myPrint(test_a_df)


# # Merge Feature 21 to 33

# In[4]:


test_a_df = pd.read_csv('../data/Test-A/test_a_feature_21to24.csv', header = None)
test_a_df.rename(columns = {test_a_df.columns[0]: 'uid'}, inplace=True)


# In[5]:


id_series = test_a_df['uid']


# In[6]:


test_a_df = test_a_df.set_index('uid')


# In[7]:


cnt = long(25)


# In[8]:


tmp = pd.read_csv('../data/Test-A/test_a_feature_25to27.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[9]:


print(tmp)


# In[10]:


test_a_df = test_a_df.join(tmp)


# In[11]:


tmp = pd.read_csv('../data/Test-A/test_a_feature_28.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[12]:


test_a_df = test_a_df.join(tmp)


# In[13]:


tmp = pd.read_csv('../data/Test-A/test_a_feature_29to33.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[14]:


print(tmp)


# In[15]:


test_a_df = test_a_df.join(tmp)


# In[16]:


tmp = pd.read_csv('../data/Test-A/test_a_feature_14.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[17]:


test_a_df = test_a_df.join(tmp)


# In[18]:


tmp = pd.read_csv('../data/Test-A/test_a_feature_17.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[19]:


test_a_df = test_a_df.join(tmp)


# In[20]:


tmp = pd.read_csv('../data/Test-A/test_a_feature_19.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[21]:


test_a_df = test_a_df.join(tmp)


# In[22]:


tmp = pd.read_csv('../data/Test-A/test_a_feature_20.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[23]:


test_a_df = test_a_df.join(tmp)


# In[24]:


test_a_df = test_a_df.drop([16, 38, 39], axis=1)


# In[25]:


test_a_df.to_csv('../data/Test-A/test_a_feature_21to33_14_17_19_20.csv', index=True)


# # Drop some features which is not that important

# In[5]:


test_a_df = pd.read_csv('../data/Test-A/test_a_feature_21to33_14_17_19_20_norm.csv', header=0)


# In[6]:


for i in [24, 8, 23, 32, 36, 15, 33, 52, 27, 17, 14, 19, 29, 34, 50, 10, 4, 31, 22]:
    tmp = str(i)
    test_a_df = test_a_df.drop(tmp, axis = 1)


# In[7]:


test_a_df.to_csv('../data/Test-A/test_a_feature_21to33_14_17_19_20_norm_2.csv', index=False)

