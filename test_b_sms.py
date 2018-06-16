
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


test_b_sms_df = pd.read_csv('../data/Test-B/test_b_sms_df.csv', header = 0, dtype = {'opp_num':str})


# # Feature 28

# In[4]:


sms_opp_num = test_b_sms_df.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_')


# In[5]:


print(sms_opp_num)


# # Feature 29

# In[6]:


sms_opp_head=test_b_sms_df.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_')


# In[7]:


print(sms_opp_head)


# # Feature 30

# In[8]:


sms_opp_len=test_b_sms_df.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').fillna(0)


# In[9]:


sms_opp_len.columns.names = ['']


# In[10]:


print(sms_opp_len)


# # Feature 31

# In[11]:


sms_in_out = test_b_sms_df.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').fillna(0)


# In[12]:


sms_in_out.columns.names = ['']


# In[13]:


print(sms_in_out)


# # Feature 32

# In[14]:


def do(x):
    hour = int(x % 1000000 / 10000)
    return hour
tmp = test_b_sms_df
test_b_sms_df.loc[:, 'timestamp'] = tmp['start_time'].apply(do)


# In[15]:


test_b_1 = test_b_sms_df[test_b_sms_df['in_out'] == 0]
test_b_2 = test_b_sms_df[test_b_sms_df['in_out'] == 1]


# In[16]:


ww = test_b_1[(test_b_1['timestamp'] == 22) | (test_b_1['timestamp'] == 23) | (test_b_1['timestamp'] == 0)]


# In[17]:


ans = ww['uid'].value_counts().sort_index()


# In[18]:


ans_df = pd.DataFrame(data = ans)
ans_df.index.name = 'uid'
ans_df.columns = ['night_count']


# In[19]:


print(ans_df)


# # Feature 33

# In[20]:


def do(x):
    hour = int(x / 1000000)
    return hour
tmp = test_b_sms_df
test_b_sms_df.loc[:, 'date'] = tmp['start_time'].apply(do)


# In[21]:


print(test_b_sms_df)


# In[22]:


date = test_b_sms_df.groupby(['uid','date'])['uid'].count().unstack().add_prefix('date_').fillna(0)


# In[23]:


print(date)


# # Merge Feature 28 to 33

# In[24]:


agg = sms_opp_num
agg = agg.join(sms_opp_head)
agg = agg.join(sms_opp_len)
agg = agg.join(sms_in_out)
agg = agg.join(ans_df)
agg = agg.join(date)


# In[27]:


agg = agg.drop(u'sms_opp_len_3', axis = 1)


# In[29]:


print(agg)


# In[30]:


dict = agg.to_dict()


# In[31]:


agg2 = pd.DataFrame(index = range(7000, 10000), data = dict)


# In[32]:


agg2 = agg2.fillna(0)
agg2 = agg2.astype('int64')


# In[33]:


agg2.to_csv('../data/Test-B/test_b_feature_28to33.csv', index=True, header=None)

