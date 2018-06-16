
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


train_uid_df = pd.read_csv('../data/train/train_uid_df.csv', header = 0)


# In[4]:


train_sms_df = pd.read_csv('../data/train/train_sms_df.csv', header = 0, dtype = {'opp_num':str})


# In[5]:


def do(x):
    hour = int(x % 1000000 / 10000)
    return hour
tmp = train_sms_df
train_sms_df.loc[:, 'timestamp'] = tmp['start_time'].apply(do)


# In[6]:


print(train_sms_df)


# In[5]:


train1 = train_sms_df[train_sms_df['in_out'] == 0]
train2 = train_sms_df[train_sms_df['in_out'] == 1]


# In[6]:


tmp1 = train1[train1['uid'] < 4100]
tmp2 = train1[train1['uid'] >= 4100]
tmp3 = train1[train1['uid'] < 2000]


# In[52]:


print(tmp1)


# In[53]:


print(tmp2)


# In[46]:


t1 = tmp1['timestamp'].value_counts().sort_index()
t1.describe()


# In[47]:


t2 = tmp2['timestamp'].value_counts().sort_index()
t2.describe()


# In[48]:


print(t1)


# In[49]:


t1.plot(kind='bar')


# In[50]:


t2.plot(kind='bar')


# In[51]:


t3 = tmp3['timestamp'].value_counts().sort_index()
t3.describe()
t3.plot(kind='bar')


# In[55]:


ww = train1[(train1['timestamp'] == 22) | (train1['timestamp'] == 23) | (train1['timestamp'] == 0)]


# In[57]:


t1 = ww[ww['uid'] < 4100]
t2 = ww[ww['uid'] >= 4100]


# In[63]:


tt1 = t1['uid'].value_counts().sort_index()
tt2 = t2['uid'].value_counts().sort_index()


# In[64]:


print(tt1)


# In[65]:


tt1.describe()


# In[66]:


tt2.describe()


# In[70]:


ans = ww['uid'].value_counts().sort_index()


# In[71]:


print(ans)


# In[72]:


dict = ans.to_dict()
ans2 = pd.Series(index = range(1, 5000), data = dict)


# In[73]:


ans2 = ans2.fillna(0)
ans2 = ans2.astype('int64')


# In[74]:


ans2.to_csv('../data/train/train_feature_9.csv', index=True)


# In[16]:


tt1 = tmp1[tmp1['opp_len'] != 11].opp_len.value_counts()


# In[17]:


tt1.plot(kind='bar')


# In[18]:


tt2 = tmp2[tmp2['opp_len'] != 11].opp_len.value_counts()
tt2.plot(kind='bar')


# In[19]:


tt3 = tmp3[tmp3['opp_len'] != 11].opp_len.value_counts()
tt3.plot(kind='bar')


# In[20]:


tmp1['opp_num'].value_counts()


# In[24]:


tt1 = tmp1[tmp1['opp_num'] == 'DE768264C1209BFD2FCBE076BC757FDA']
tt2 = tmp2[tmp2['opp_num'] == 'DE768264C1209BFD2FCBE076BC757FDA']


# In[27]:


w1 = tt1['uid'].value_counts()
w1.plot(kind='bar')
w1.describe()


# In[28]:


w2 = tt2['uid'].value_counts()
w2.plot(kind='bar')
w2.describe()


# In[5]:


def do(x):
    return int(x['start_time'] / 1000000)
tmp = train_sms_df
train_sms_df.loc[:, 'day'] = tmp.apply(do, axis = 1)


# In[6]:


print(train_sms_df)


# In[7]:


train1 = train_sms_df[train_sms_df['in_out'] == 0]
train2 = train_sms_df[train_sms_df['in_out'] == 1]


# In[8]:


group = train1.groupby('uid')
item_dict = {}
for index,g in group:
    tmp = g['day'].value_counts()
    item_dict[index] = tmp.std()


# In[9]:


ans = pd.Series(index = range(1, 5000), data = item_dict)


# In[11]:


ans = ans.fillna(0)


# In[12]:


print(ans)


# In[13]:


ans.to_csv('../data/train/train_feature_20.csv', index=True)


# In[14]:


tt1 = ans[ans.index < 4100]
tt2 = ans[ans.index >= 4100]
tt3 = ans[ans.index < 2000]


# In[15]:


tt1.plot(kind='bar')


# In[16]:


tt2.plot(kind='bar')


# In[17]:


tt3.plot(kind='bar')


# # Feature 28

# In[5]:


sms_opp_num = train_sms_df.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_')


# In[6]:


print(sms_opp_num)


# # Feature 29

# In[7]:


sms_opp_head=train_sms_df.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_')


# In[8]:


print(sms_opp_head)


# # Feature 30

# In[9]:


sms_opp_len=train_sms_df.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').fillna(0)


# In[10]:


sms_opp_len.columns.names = ['']


# In[11]:


print(sms_opp_len.columns[0])


# In[12]:


'''
ori = sms_opp_len
for i in range(0, ori.columns.size):
    if len(pd.unique(ori.iloc[:, i])) < 50:
        sms_opp_len = sms_opp_len.drop(columns = ori.columns[i], axis=1)
'''


# In[13]:


print(sms_opp_len)


# # Feature 31

# In[14]:


sms_in_out = train_sms_df.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').fillna(0)


# In[15]:


sms_in_out.columns.names = ['']


# In[16]:


print(sms_in_out)


# # Feature 32

# In[17]:


def do(x):
    hour = int(x % 1000000 / 10000)
    return hour
tmp = train_sms_df
train_sms_df.loc[:, 'timestamp'] = tmp['start_time'].apply(do)


# In[18]:


train_1 = train_sms_df[train_sms_df['in_out'] == 0]
train_2 = train_sms_df[train_sms_df['in_out'] == 1]


# In[19]:


ww = train_1[(train_1['timestamp'] == 22) | (train_1['timestamp'] == 23) | (train_1['timestamp'] == 0)]


# In[20]:


ans = ww['uid'].value_counts().sort_index()


# In[21]:


ans_df = pd.DataFrame(data = ans)
ans_df.index.name = 'uid'
ans_df.columns = ['night_count']


# In[22]:


print(ans_df)


# # Feature 33

# In[23]:


def do(x):
    hour = int(x / 1000000)
    return hour
tmp = train_sms_df
train_sms_df.loc[:, 'date'] = tmp['start_time'].apply(do)


# In[24]:


date = train_sms_df.groupby(['uid','date'])['uid'].count().unstack().add_prefix('date_').fillna(0)


# In[25]:


print(date)


# # Merge Feature 28 to 33

# In[26]:


agg = sms_opp_num
agg = agg.join(sms_opp_head)
agg = agg.join(sms_opp_len)
agg = agg.join(sms_in_out)
agg = agg.join(ans_df)
agg = agg.join(date)


# In[29]:


print(agg)


# In[28]:


print(agg.columns)


# In[32]:


dict = agg.to_dict()


# In[33]:


agg2 = pd.DataFrame(index = range(1, 5000), data = dict)


# In[34]:


agg2 = agg2.fillna(0)
agg2 = agg2.astype('int64')


# In[35]:


agg2.to_csv('../data/train/train_feature_28to33.csv', index=True, header=None)

