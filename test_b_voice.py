
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


test_b_voice_df = pd.read_csv('../data/Test-B/test_b_voice_df.csv', header = 0, dtype = {'opp_num':str})


# # Feature 34

# In[5]:


voice_opp_num = test_b_voice_df.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_')


# In[6]:


print(voice_opp_num)


# # Feature 35

# In[7]:


voice_opp_head=test_b_voice_df.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_')


# In[8]:


print(voice_opp_head)


# # Feature 36

# In[9]:


voice_opp_len=test_b_voice_df.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').fillna(0)


# In[10]:


voice_opp_len.columns.name = None


# In[11]:


print(voice_opp_len)


# # Feature 37

# In[12]:


voice_call_type = test_b_voice_df.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').fillna(0)


# In[13]:


voice_call_type.columns.name = None


# In[14]:


print(voice_call_type)


# # Feature 38

# In[15]:


voice_in_out = test_b_voice_df.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').fillna(0)


# In[16]:


voice_in_out.columns.name = None


# In[17]:


print(voice_in_out)


# # Feature 39

# In[18]:


test_b_1 = test_b_voice_df[test_b_voice_df['in_out'] == 0]
test_b_2 = test_b_voice_df[test_b_voice_df['in_out'] == 1]


# In[19]:


t1 = test_b_1['uid'].value_counts()
t2 = test_b_2['uid'].value_counts()
dict = t1.to_dict()
tt1 = pd.Series(index = range(7000, 10000), data = dict)
tt1 = tt1.fillna(0)
dict = t2.to_dict()
tt2 = pd.Series(index = range(7000, 10000), data = dict)
tt2 = tt2.fillna(0)
tt3 = tt1 / (tt1 + tt2)


# In[20]:


print(tt3)


# In[21]:


tt3 = tt3.fillna(0.5)


# In[24]:


voice_in_out_rate = pd.DataFrame(tt3)
voice_in_out_rate.columns = ['in_out_rate']


# In[25]:


print(voice_in_out_rate)


# # Feature 40

# In[27]:


def do(x):
    start_day = int(x['start_time'] % 100000000 / 1000000)
    start_hour = int(x['start_time'] % 1000000 / 10000)
    start_min = int(x['start_time'] % 10000 / 100)
    start_sec = int(x['start_time'] % 100)
    start = (start_day * 24 * 60 + start_hour * 60 + start_min) * 60 + start_sec
    
    end_day = int(x['end_time'] % 100000000 / 1000000)
    end_hour = int(x['end_time'] % 1000000 / 10000)
    end_min = int(x['end_time'] % 10000 / 100)
    end_sec = int(x['end_time'] % 100)
    end = (end_day * 24 * 60 + end_hour * 60 + end_min) * 60 + end_sec
    
    # duration in seconds
    return end - start + 1
tmp = test_b_voice_df
test_b_voice_df.loc[:, 'dura_second'] = tmp.apply(do, axis = 1)


# In[29]:


test_b_1 = test_b_voice_df[test_b_voice_df['in_out'] == 0]
test_b_2 = test_b_voice_df[test_b_voice_df['in_out'] == 1]


# In[30]:


g = test_b_1[test_b_1['dura_second'] <= 10]


# In[33]:


cnt = g['uid'].value_counts().sort_index()


# In[42]:


cnt.index.name = 'uid'
voice_short_dura_count = pd.DataFrame(cnt)
voice_short_dura_count.columns = ['voice_short_dura_count']


# In[44]:


print(voice_short_dura_count)


# # Feature 41

# In[46]:


def do(x):
    return int(x['start_time'] / 1000000)
tmp = test_b_voice_df
test_b_voice_df.loc[:, 'date'] = tmp.apply(do, axis = 1)


# In[47]:


print(test_b_voice_df[['uid', 'start_time', 'date']])


# In[48]:


voice_date = test_b_voice_df.groupby(['uid','date'])['uid'].count().unstack().add_prefix('date_').fillna(0)


# In[49]:


print(voice_date)


# # Merge Feature 34 to 41

# In[50]:


agg = voice_opp_num
agg = agg.join(voice_opp_len)
agg = agg.join(voice_opp_head)
agg = agg.join(voice_call_type)
agg = agg.join(voice_in_out)
agg = agg.join(voice_in_out_rate)
agg = agg.join(voice_short_dura_count)
agg = agg.join(voice_date)


# In[57]:


agg = agg.drop(u'voice_opp_len_18', axis = 1)


# In[59]:


print(agg)


# In[60]:


dict = agg.to_dict()
agg2 = pd.DataFrame(index = range(7000, 10000), data = dict)


# In[61]:


agg2 = agg2.fillna(0)
agg2 = agg2.astype('int64')


# In[62]:


agg2.to_csv('../data/Test-B/test_b_feature_34to41.csv', index=True, header=None)

