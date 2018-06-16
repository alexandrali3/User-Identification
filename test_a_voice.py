
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


test_a_voice_df = pd.read_csv('../data/Test-A/test_a_voice_df.csv', header = 0, dtype = {'opp_num':str})


# # Feature 10

# In[45]:


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

def do2(x):
    start_day = int(x['start_time'] % 100000000 / 1000000)
    start_hour = int(x['start_time'] % 1000000 / 10000)
    start_min = int(x['start_time'] % 10000 / 100)
    start = start_day * 24 * 60 + start_hour * 60 + start_min
    
    end_day = int(x['end_time'] % 100000000 / 1000000)
    end_hour = int(x['end_time'] % 1000000 / 10000)
    end_min = int(x['end_time'] % 10000 / 100)
    end = end_day * 24 * 60 + end_hour * 60 + end_min
    
    # duration in minutes
    return end - start + 1

tmp = test_a_voice_df
test_a_voice_df.loc[:, 'dura'] = tmp.apply(do, axis = 1)


# In[46]:


test_a_1 = test_a_voice_df[test_a_voice_df['in_out'] == 0]
test_a_2 = test_a_voice_df[test_a_voice_df['in_out'] == 1]


# In[8]:


tt = test_a_1['uid'].value_counts().sort_index()


# In[9]:


dict = tt.to_dict()
ans = pd.Series(index = range(5000, 7000), data = dict)


# In[10]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[11]:


ans.to_csv('../data/Test-A/test_a_feature_10.csv', index=True)


# # Feature 11

# In[12]:


tt = test_a_2['uid'].value_counts().sort_index()


# In[13]:


dict = tt.to_dict()
ans = pd.Series(index = range(5000, 7000), data = dict)


# In[14]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[15]:


ans.to_csv('../data/Test-A/test_a_feature_11.csv', index=True)


# # Feature 12

# In[38]:


tt = test_a_1.groupby('uid').dura.sum()
dict = tt.to_dict()
ans = pd.Series(index = range(5000, 7000), data = dict)


# In[39]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[40]:


ans.to_csv('../data/Test-A/test_a_feature_12.csv', index=True)


# # Feature 13

# In[41]:


tt = test_a_2.groupby('uid').dura.sum()
dict = tt.to_dict()
ans = pd.Series(index = range(5000, 7000), data = dict)


# In[42]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[43]:


ans.to_csv('../data/Test-A/test_a_feature_13.csv', index=True)


# # Feature 14

# In[23]:


t1 = test_a_1['uid'].value_counts()
t2 = test_a_2['uid'].value_counts()
dict = t1.to_dict()
tt1 = pd.Series(index = range(5000, 7000), data = dict)
tt1 = tt1.fillna(0)
dict = t2.to_dict()
tt2 = pd.Series(index = range(5000, 7000), data = dict)
tt2 = tt2.fillna(0)
tt3 = tt1 / (tt1 + tt2)


# In[24]:


myPrint2(tt3)


# In[25]:


tt3 = tt3.fillna(0.5)


# In[26]:


tt3.to_csv('../data/Test-A/test_a_feature_14.csv', index=True)


# # Feature 15

# In[27]:


group = test_a_1.groupby('uid')['opp_num']
agg = group.aggregate({'opp_num': lambda x: x.nunique()})


# In[29]:


agg.rename(columns=lambda x:x.replace('opp_num','opp_cnt'), inplace=True)


# In[30]:


dict = pd.Series(agg['opp_cnt']).to_dict()
tmp = pd.Series(index = range(5000, 7000), data = dict)


# In[32]:


tmp = tmp.fillna(0)
tmp = tmp.astype('int64')


# In[33]:


tmp.to_csv('../data/Test-A/test_a_feature_15.csv', index=True)


# # Feature 16

# In[47]:


group = test_a_1.groupby('uid')['dura']
agg = group.mean()


# In[50]:


dict = agg.to_dict()
ans = pd.Series(index = range(5000, 7000), data = dict)


# In[51]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[53]:


ans.to_csv('../data/Test-A/test_a_feature_16.csv', index=True)


# # Feature 17

# In[54]:


g = test_a_1[test_a_1['dura'] <= 10]


# In[55]:


ans = g['uid'].value_counts()


# In[56]:


dict = ans.to_dict()
ans2 = pd.Series(index = range(5000, 7000), data = dict)


# In[57]:


ans2 = ans2.fillna(0)
ans2 = ans2.astype('int64')


# In[58]:


ans2.to_csv('../data/Test-A/test_a_feature_17.csv', index=True)


# # Feature 18

# In[59]:


g = test_a_1[test_a_1['opp_len'] <= 8]


# In[60]:


tmp = g['uid'].value_counts()
tmp = tmp.sort_index()


# In[61]:


dict = tmp.to_dict()
ans = pd.Series(index = range(5000, 7000), data = dict)


# In[62]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[63]:


ans.to_csv('../data/Test-A/test_a_feature_18.csv', index=True)


# # Feature 19

# In[64]:


def do(x):
    return int(x['start_time'] / 1000000)

tmp = test_a_voice_df
test_a_voice_df.loc[:, 'day'] = tmp.apply(do, axis = 1)


# In[65]:


test_a_1 = test_a_voice_df[test_a_voice_df['in_out'] == 0]
test_a_2 = test_a_voice_df[test_a_voice_df['in_out'] == 1]


# In[66]:


group = test_a_1.groupby('uid')
item_dict = {}
for index,g in group:
    tmp = g['day'].value_counts()
    item_dict[index] = tmp.std()


# In[67]:


ans = pd.Series(index = range(5000, 7000), data = item_dict)


# In[68]:


ans = ans.fillna(0)


# In[69]:


ans.to_csv('../data/Test-A/test_a_feature_19.csv', index=True)


# # Feature 29 to 33

# In[5]:


voice_opp_num = test_a_voice_df.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_')


# In[6]:


print(voice_opp_num)


# # Feature 30

# In[7]:


voice_opp_head=test_a_voice_df.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_')


# In[8]:


print(voice_opp_head)


# # Feature 31

# In[9]:


voice_opp_len=test_a_voice_df.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').fillna(0)


# In[10]:


voice_opp_len.columns.name = None


# In[11]:


voice_opp_len = voice_opp_len[['voice_opp_len_5', 'voice_opp_len_8',  'voice_opp_len_11', 'voice_opp_len_12']]


# In[12]:


print(voice_opp_len)


# # Feature 32

# In[13]:


voice_call_type = test_a_voice_df.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').fillna(0)


# In[14]:


voice_call_type.columns.name = None


# # Feature 33

# In[15]:


voice_in_out = test_a_voice_df.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').fillna(0)


# In[16]:


voice_in_out.columns.name = None


# # Merge Feature 29 to 33

# In[17]:


agg = voice_opp_num


# In[18]:


agg = agg.join(voice_opp_len)


# In[19]:


agg = agg.join(voice_opp_head)


# In[20]:


agg = agg.join(voice_call_type)


# In[21]:


agg = agg.join(voice_in_out)


# In[25]:


dict = agg.to_dict()
agg2 = pd.DataFrame(index = range(5000, 7000), data = dict)


# In[26]:


agg2 = agg2.fillna(0)
agg2 = agg2.astype('int64')


# In[27]:


agg2.to_csv('../data/Test-A/test_a_feature_29to33.csv', index=True, header=None)

