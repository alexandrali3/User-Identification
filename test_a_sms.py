
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


# In[3]:


def myPrint(x):
    print(x.head(15))
    print(x.info())
def myPrint2(x):
    print(x)


# In[4]:


test_a_sms_df = pd.read_csv('../data/Test-A/test_a_sms_df.csv', header = 0, dtype = {'opp_num': str})


# # Feature 7

# In[5]:


t1 = test_a_sms_df[test_a_sms_df['in_out'] == 0]


# In[6]:


tmp = t1['uid'].value_counts().sort_index()


# In[7]:


dict = tmp.to_dict()
tmp2 = pd.Series(index = range(5000, 7000), data = dict)


# In[8]:


tmp2 = tmp2.fillna(0)
tmp2 = tmp2.astype('int64')


# In[9]:


tmp2.to_csv('../data/Test-A/test_a_feature_7.csv', index=True)


# # Feature 8

# In[10]:


t2 = test_a_sms_df[test_a_sms_df['in_out'] == 1]


# In[11]:


tmp = t2['uid'].value_counts().sort_index()


# In[12]:


dict = tmp.to_dict()
tmp2 = pd.Series(index = range(5000, 7000), data = dict)


# In[13]:


tmp2 = tmp2.fillna(0)
tmp2 = tmp2.astype('int64')


# In[14]:


tmp2.to_csv('../data/Test-A/test_a_feature_8.csv', index=True)


# # Feature 9

# In[15]:


def do(x):
    hour = int(x % 1000000 / 10000)
    return hour
tmp = test_a_sms_df
test_a_sms_df.loc[:, 'timestamp'] = tmp['start_time'].apply(do)


# In[16]:


test_a_1 = test_a_sms_df[test_a_sms_df['in_out'] == 0]
test_a_2 = test_a_sms_df[test_a_sms_df['in_out'] == 1]


# In[17]:


myPrint(test_a_1)


# In[18]:


ww = test_a_1[(test_a_1['timestamp'] == 22) | (test_a_1['timestamp'] == 23) | (test_a_1['timestamp'] == 0)]


# In[19]:


ans = ww['uid'].value_counts().sort_index()


# In[20]:


myPrint2(ans)


# In[21]:


dict = ans.to_dict()
ans2 = pd.Series(index = range(5000, 7000), data = dict)


# In[22]:


ans2 = ans2.fillna(0)
ans2 = ans2.astype('int64')


# In[23]:


ans2.to_csv('../data/Test-A/test_a_feature_9.csv', index=True)


# # Feature 20

# In[5]:


def do(x):
    return int(x['start_time'] / 1000000)
tmp = test_a_sms_df
test_a_sms_df.loc[:, 'day'] = tmp.apply(do, axis = 1)


# In[6]:


test_a_1 = test_a_sms_df[test_a_sms_df['in_out'] == 0]
test_a_2 = test_a_sms_df[test_a_sms_df['in_out'] == 1]


# In[8]:


group = test_a_1.groupby('uid')
item_dict = {}
for index,g in group:
    tmp = g['day'].value_counts()
    item_dict[index] = tmp.std()


# In[9]:


ans = pd.Series(index = range(5000, 7000), data = item_dict)


# In[10]:


ans = ans.fillna(0)


# In[11]:


ans.to_csv('../data/Test-A/test_a_feature_20.csv', index=True)


# # Feature 25

# In[5]:


sms_opp_num = test_a_sms_df.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_')


# In[6]:


print(sms_opp_num)


# # Feature 26

# In[7]:


sms_opp_head=test_a_sms_df.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_')


# In[8]:


print(sms_opp_head)


# # Feature 27

# In[13]:


sms_opp_len=test_a_sms_df.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').fillna(0)


# In[14]:


sms_opp_len.columns.names = ['']


# In[16]:


sms_opp_len = sms_opp_len[['sms_opp_len_7', 'sms_opp_len_11', 'sms_opp_len_12', 'sms_opp_len_13']]


# In[17]:


print(sms_opp_len)


# # Feature 28

# In[18]:


sms_in_out = test_a_sms_df.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').fillna(0)


# In[19]:


sms_in_out.columns.names = ['']


# In[20]:


print(sms_in_out)


# # Merge Feature 25 to 28

# In[21]:


agg = sms_opp_num


# In[22]:


agg = agg.join(sms_opp_head)


# In[23]:


agg = agg.join(sms_opp_len)


# In[24]:


agg = agg.join(sms_in_out)


# In[25]:


print(agg)


# In[26]:


dict = agg.to_dict()
agg2 = pd.DataFrame(index = range(5000, 7000), data = dict)


# In[27]:


agg2 = agg2.fillna(0)
agg2 = agg2.astype('int64')


# In[28]:


agg2.to_csv('../data/Test-A/test_a_feature_25to27.csv', index=True, header=None)


# # Feature 28

# In[29]:


def do(x):
    hour = int(x % 1000000 / 10000)
    return hour
tmp = test_a_sms_df
test_a_sms_df.loc[:, 'timestamp'] = tmp['start_time'].apply(do)


# In[30]:


test_a_1 = test_a_sms_df[test_a_sms_df['in_out'] == 0]
test_a_2 = test_a_sms_df[test_a_sms_df['in_out'] == 1]


# In[31]:


ww = test_a_1[(test_a_1['timestamp'] == 22) | (test_a_1['timestamp'] == 23) | (test_a_1['timestamp'] == 0)]


# In[32]:


ans = ww['uid'].value_counts().sort_index()


# In[33]:


dict = ans.to_dict()
ans2 = pd.Series(index = range(5000, 7000), data = dict)


# In[34]:


ans2 = ans2.fillna(0)
ans2 = ans2.astype('int64')


# In[35]:


ans2.to_csv('../data/Test-A/test_a_feature_28.csv', index=True)

