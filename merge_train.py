
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


# In[9]:


train_df = pd.read_csv('../data/train/train_feature_num_records.csv', header = None, names = ['uid', '1'])
id_series = train_df['uid']
train_df = train_df.set_index('uid')


# In[10]:


def myPrint(x):
    print(x.head(15))
    print(x.info())


# In[11]:


myPrint(train_df)


# In[6]:


tmp = pd.read_csv('../data/train/train_feature_2to5.csv', header = 0, names = ['uid', '2', '3', '4', '5'])
tmp = tmp.set_index('uid')


# In[7]:


myPrint(tmp)


# In[8]:


train_df = train_df.join(tmp[['2', '3', '4', '5']])


# In[9]:


myPrint(train_df)


# In[10]:


print(train_df.index)


# In[11]:


tmp = pd.read_csv('../data/train/train_feature_6.csv', header = 0, names = ['uid', '6'])
tmp = tmp.set_index('uid')


# In[12]:


myPrint(tmp)


# In[13]:


train_df = train_df.join(tmp)


# In[14]:


myPrint(train_df)


# In[15]:


for i in range(7, 21):
    tmp = pd.read_csv('../data/train/train_feature_' + str(i) + '.csv', header = None, names = ['uid', str(i)])
    tmp = tmp.set_index('uid')
    train_df = train_df.join(tmp)


# In[16]:


myPrint(train_df)


# In[17]:


train_uid_df = pd.read_csv('../data/train/train_uid_df.csv', header = 0)


# In[18]:


train_uid_df = train_uid_df.set_index('uid')


# In[19]:


train_df = train_df.join(train_uid_df)


# In[20]:


myPrint(train_df)


# In[21]:


train_df.to_csv('../data/train/train_feature_1to20_label.csv', index=True)


# # Merge Feature 21 to 33

# In[3]:


train_df = pd.read_csv('../data/train/train_feature_21to24.csv', header = None)
train_df.rename(columns = {train_df.columns[0]: 'uid'}, inplace=True)


# In[4]:


print(train_df)


# In[5]:


id_series = train_df['uid']


# In[6]:


train_df = train_df.set_index('uid')


# In[7]:


cnt = long(25)


# In[8]:


tmp = pd.read_csv('../data/train/train_feature_25to27.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[9]:


print(tmp)


# In[10]:


train_df = train_df.join(tmp)


# In[11]:


tmp = pd.read_csv('../data/train/train_feature_28.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[12]:


print(tmp)


# In[13]:


train_df = train_df.join(tmp)


# In[14]:


tmp = pd.read_csv('../data/train/train_feature_29to33.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[15]:


print(tmp)


# In[16]:


train_df = train_df.join(tmp)


# In[17]:


print(train_df)


# In[18]:


tmp = pd.read_csv('../data/train/train_feature_14.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[19]:


print(tmp)


# In[20]:


train_df = train_df.join(tmp)


# In[21]:


tmp = pd.read_csv('../data/train/train_feature_17.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[22]:


train_df = train_df.join(tmp)


# In[23]:


tmp = pd.read_csv('../data/train/train_feature_19.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[24]:


train_df = train_df.join(tmp)


# In[25]:


tmp = pd.read_csv('../data/train/train_feature_20.csv', header = None)
tmp.rename(columns = {tmp.columns[0]: 'uid'}, inplace=True)
for i in range(1, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1
tmp = tmp.set_index('uid')


# In[26]:


train_df = train_df.join(tmp)


# In[27]:


train_uid_df = pd.read_csv('../data/train/train_uid_df.csv', header = 0)


# In[28]:


train_uid_df = train_uid_df.set_index('uid')


# In[29]:


train_df = train_df.join(train_uid_df)


# In[30]:


print(train_df)


# In[32]:


train_df = train_df.drop([38, 39], axis=1)


# In[33]:


print(train_df)


# In[31]:


train_df = train_df.drop(16, axis = 1)


# In[35]:


train_df.to_csv('../data/train/train_feature_21to33_14_17_19_20.csv', index=True)


# # Drop some features which is not that important

# In[3]:


train_df = pd.read_csv('../data/train/train_feature_21to33_14_17_19_20_norm.csv', header=0)


# In[4]:


print(train_df.columns)


# In[5]:


for i in [10, 4, 31, 22]:
    tmp = str(i)
    train_df = train_df.drop(tmp, axis = 1)


# In[6]:


print(train_df.columns)


# In[7]:


train_df.to_csv('../data/train/train_feature_21to33_14_17_19_20_norm_2.csv', index=False)


# # Merge 21 to 41

# In[3]:


train_df = pd.read_csv('../data/train/train_feature_21to27.csv', header = None, index_col=0)
train_df.index.name = 'uid'


# In[4]:


print(train_df)


# In[5]:


id_series = train_df.index


# In[6]:


cnt = long(85)


# In[7]:


tmp = pd.read_csv('../data/train/train_feature_28to33.csv', header = None, index_col = 0)
tmp.index.name = 'uid'
for i in range(0, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1


# In[8]:


print(tmp)


# In[9]:


train_df = train_df.join(tmp)


# In[10]:


tmp = pd.read_csv('../data/train/train_feature_34to41.csv', header = None, index_col = 0)
tmp.index.name = 'uid'
for i in range(0, tmp.columns.size):
    tmp.rename(columns = {tmp.columns[i]: cnt}, inplace=True)
    cnt = cnt + 1


# In[11]:


print(tmp)


# In[12]:


train_df = train_df.join(tmp)


# In[13]:


train_uid_df = pd.read_csv('../data/train/train_uid_df.csv', header = 0)


# In[14]:


train_uid_df = train_uid_df.set_index('uid')


# In[15]:


train_df = train_df.join(train_uid_df)


# In[16]:


print(train_df)


# In[17]:


train_df.to_csv('../data/train/train_feature_21to41.csv', index=True)

