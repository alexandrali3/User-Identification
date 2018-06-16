
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


# In[3]:


train_voice_df = pd.read_csv('../data/train/train_voice_df.csv', header = 0, dtype = {'opp_num':str})


# In[4]:


train_voice_df.info()


# In[7]:


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
tmp = train_voice_df
train_voice_df.loc[:, 'dura'] = tmp.apply(do, axis = 1)


# In[8]:


print(train_voice_df)


# In[61]:


train1 = train_voice_df[train_voice_df['in_out'] == 0]
train2 = train_voice_df[train_voice_df['in_out'] == 1]

tmp1 = train1[train1['uid'] < 4100]
tmp2 = train1[train1['uid'] >= 4100]
tmp3 = train1[train1['uid'] < 2000]


# In[34]:


tt1 = tmp1['uid'].value_counts()
tt2 = tmp2['uid'].value_counts()
tt3 = tmp3['uid'].value_counts()


# In[35]:


tt1.describe()


# In[36]:


tt2.describe()


# In[37]:


tt3.describe()


# In[39]:


tt = train1['uid'].value_counts().sort_index()


# In[40]:


dict = tt.to_dict()
ans = pd.Series(index = range(1, 5000), data = dict)


# In[41]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[42]:


print(ans)


# In[43]:


ans.to_csv('../data/train/train_feature_10.csv', index=True)


# In[44]:


tmp1 = train2[train2['uid'] < 4100]
tmp2 = train2[train2['uid'] >= 4100]
tmp3 = train2[train2['uid'] < 2000]


# In[45]:


tt1 = tmp1['uid'].value_counts()
tt2 = tmp2['uid'].value_counts()
tt3 = tmp3['uid'].value_counts()


# In[46]:


tt1.describe()


# In[47]:


tt2.describe()


# In[48]:


tt3.describe()


# In[49]:


tt = train2['uid'].value_counts().sort_index()


# In[50]:


dict = tt.to_dict()
ans = pd.Series(index = range(1, 5000), data = dict)


# In[51]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[52]:


print(ans)


# In[53]:


ans.to_csv('../data/train/train_feature_11.csv', index=True)


# In[62]:


tmp1 = train1[train1['uid'] < 4100]
tmp2 = train1[train1['uid'] >= 4100]
tmp3 = train1[train1['uid'] < 2000]


# In[65]:


tmp1['dura'].value_counts().sort_index().plot(kind='bar')


# In[66]:


tmp2['dura'].value_counts().sort_index().plot(kind='bar')


# In[67]:


tmp1 = train2[train2['uid'] < 4100]
tmp2 = train2[train2['uid'] >= 4100]
tmp3 = train2[train2['uid'] < 2000]


# In[68]:


tmp1['dura'].value_counts().sort_index().plot(kind='bar')


# In[69]:


tmp2['dura'].value_counts().sort_index().plot(kind='bar')


# In[70]:


tmp1['dura'].describe()


# In[71]:


tmp2['dura'].describe()


# In[72]:


tmp3['dura'].describe()


# In[73]:


tmp1 = train1[train1['uid'] < 4100]
tmp2 = train1[train1['uid'] >= 4100]
tmp3 = train1[train1['uid'] < 2000]


# In[76]:


tt1 = tmp1['call_type'].value_counts()
tt2 = tmp2['call_type'].value_counts()
print(tt1)
print(tt2)


# In[87]:


tt1 = tmp1.groupby('uid').dura.sum()
tt2 = tmp2.groupby('uid').dura.sum()
tt3 = tmp3.groupby('uid').dura.sum()


# In[83]:


tt1.plot(kind='bar')
plt.show()
tt2.plot(kind='bar')
plt.show()


# In[84]:


tt1.describe()


# In[85]:


tt2.describe()


# In[88]:


tt3.describe()


# In[89]:


tt = train1.groupby('uid').dura.sum()
dict = tt.to_dict()
ans = pd.Series(index = range(1, 5000), data = dict)


# In[90]:


print(ans)


# In[93]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[94]:


ans.to_csv('../data/train/train_feature_12.csv', index=True)


# In[95]:


tmp1 = train2[train2['uid'] < 4100]
tmp2 = train2[train2['uid'] >= 4100]
tmp3 = train2[train2['uid'] < 2000]


# In[96]:


tt1 = tmp1.groupby('uid').dura.sum()
tt2 = tmp2.groupby('uid').dura.sum()
tt3 = tmp3.groupby('uid').dura.sum()


# In[99]:


tt1.describe()


# In[100]:


tt2.describe()


# In[101]:


tt3.describe()


# In[102]:


tt = train2.groupby('uid').dura.sum()
dict = tt.to_dict()
ans = pd.Series(index = range(1, 5000), data = dict)


# In[103]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[104]:


ans.to_csv('../data/train/train_feature_13.csv', index=True)


# 统计：主叫次数 / 被叫次数

# In[71]:


train1 = train_voice_df[train_voice_df['in_out'] == 0]
train2 = train_voice_df[train_voice_df['in_out'] == 1]

tmp1 = train1[train1['uid'] < 4100]
tmp2 = train1[train1['uid'] >= 4100]
tmp3 = train1[train1['uid'] < 2000]


# In[72]:


t1 = train1['uid'].value_counts()
t2 = train2['uid'].value_counts()
dict = t1.to_dict()
tt1 = pd.Series(index = range(1, 5000), data = dict)
tt1 = tt1.fillna(0)
dict = t2.to_dict()
tt2 = pd.Series(index = range(1, 5000), data = dict)
tt2 = tt2.fillna(0)
tt3 = tt1 / (tt1 + tt2)


# In[73]:


print(tt3)


# In[74]:


aa1 = tt3[tt3.index < 4100]
aa2 = tt3[tt3.index >= 4100]
aa3 = tt3[tt3.index < 2000]


# In[75]:


aa1.describe()


# In[76]:


aa2.describe()


# In[77]:


aa3.describe()


# In[78]:


print(tt3[tt3.index == 354])


# In[80]:


tt3 = tt3.fillna(0.5)


# In[81]:


tt3.to_csv('../data/train/train_feature_14.csv', index=True)


# In[38]:


train1 = train_voice_df[train_voice_df['in_out'] == 0]
train2 = train_voice_df[train_voice_df['in_out'] == 1]

tmp1 = train1[train1['uid'] < 4100]
tmp2 = train1[train1['uid'] >= 4100]
tmp3 = train1[train1['uid'] < 2000]


# In[49]:


group = train1.groupby('uid')['opp_num']
agg = group.aggregate({'opp_num': lambda x: x.nunique()})


# In[50]:


print(agg)


# In[51]:


agg.rename(columns=lambda x:x.replace('opp_num','opp_cnt'), inplace=True)


# In[52]:


print(agg)


# In[56]:


tt1 = agg[agg.index < 4100]
tt2 = agg[agg.index >= 4100]
tt3 = agg[agg.index < 2000]


# In[57]:


tt1.describe()


# In[58]:


tt2.describe()


# In[59]:


tt3.describe()


# In[66]:


dict = pd.Series(agg['opp_cnt']).to_dict()
tmp = pd.Series(index = range(1, 5000), data = dict)
print(tmp)


# In[67]:


tmp = tmp.fillna(0)
tmp = tmp.astype('int64')


# In[68]:


print(tmp)


# In[70]:


tmp.to_csv('../data/train/train_feature_15.csv', index=True)


# In[9]:


train1 = train_voice_df[train_voice_df['in_out'] == 0]
train2 = train_voice_df[train_voice_df['in_out'] == 1]

tmp1 = train1[train1['uid'] < 4100]
tmp2 = train1[train1['uid'] >= 4100]
tmp3 = train1[train1['uid'] < 2000]


# In[10]:


group = train1.groupby('uid')['dura']
agg = group.mean()
print(agg)


# In[11]:


tt1 = agg[agg.index < 4100]
tt2 = agg[agg.index >= 4100]


# In[12]:


tt1.describe()


# In[13]:


tt2.describe()


# In[14]:


dict = agg.to_dict()
ans = pd.Series(index = range(1, 5000), data = dict)


# In[15]:


ans = ans.fillna(0)
ans = ans.astype('int64')


# In[16]:


ans.to_csv('../data/train/train_feature_16.csv', index=True)


# In[100]:


g = train1[train1['dura'] <= 10]
tmp1 = g[g['uid'] < 4100]
tmp2 = g[g['uid'] >= 4100]
tmp3 = g[g['uid'] < 2000]


# In[101]:


print(tmp1)


# In[103]:


ww1 = tmp1['uid'].value_counts()
ww2 = tmp2['uid'].value_counts()
ww3 = tmp3['uid'].value_counts()


# In[104]:


ww1.describe()


# In[105]:


ww2.describe()


# In[106]:


ww3.describe()


# In[108]:


ans = g['uid'].value_counts()


# In[109]:


dict = ans.to_dict()
ans2 = pd.Series(index = range(1, 5000), data = dict)


# In[110]:


ans2 = ans2.fillna(0)
ans2 = ans2.astype('int64')


# In[111]:


print(ans2)


# In[114]:


ans2.to_csv('../data/train/train_feature_17.csv', index=True)


# In[194]:


g = train1[train1['opp_len'] <= 8]


# In[195]:


tmp = g['uid'].value_counts()


# In[196]:


tmp1 = tmp[tmp.index < 4100]
tmp2 = tmp[tmp.index >= 4100]
tmp3 = tmp[tmp.index < 2000]


# In[197]:


# tmp1.plot(kind='bar')


# In[198]:


# tmp2.plot(kind='bar')


# In[199]:


tmp1.describe()


# In[200]:


tmp2.describe()


# In[201]:


tmp3.describe()


# In[203]:


tmp = tmp.sort_index()


# In[204]:


print(tmp)


# In[205]:


dict = tmp.to_dict()
ans = pd.Series(index = range(1, 5000), data = dict)


# In[206]:


ans = ans.fillna(0)


# In[207]:


ans = ans.astype('int64')


# In[208]:


print(ans)


# In[209]:


ans.to_csv('../data/train/train_feature_18.csv', index=True)


# In[214]:


def do(x):
    return int(x['start_time'] / 1000000)
tmp = train_voice_df
train_voice_df.loc[:, 'day'] = tmp.apply(do, axis = 1)


# In[215]:


print(train_voice_df[['uid', 'start_time', 'day']])


# In[217]:


train1 = train_voice_df[train_voice_df['in_out'] == 0]
train2 = train_voice_df[train_voice_df['in_out'] == 1]


# In[225]:


group = train1.groupby('uid')
item_dict = {}
for index,g in group:
    tmp = g['day'].value_counts()
    item_dict[index] = tmp.std()


# In[230]:


ans = pd.Series(index = range(1, 5000), data = item_dict)


# In[231]:


print(ans)


# In[232]:


ans = ans.fillna(0)


# In[234]:


ans.to_csv('../data/train/train_feature_19.csv', index=True)


# In[235]:


tt1 = ans[ans.index < 4100]
tt2 = ans[ans.index >= 4100]
tt3 = ans[ans.index < 2000]


# In[236]:


tt1.plot(kind='bar')


# In[237]:


tt2.plot(kind='bar')


# # Feature 34

# In[5]:


voice_opp_num = train_voice_df.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_')


# In[6]:


print(voice_opp_num)


# # Feature 35

# In[7]:


voice_opp_head=train_voice_df.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_')


# In[8]:


print(voice_opp_head)


# # Feature 36

# In[9]:


voice_opp_len=train_voice_df.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').fillna(0)


# In[10]:


voice_opp_len.columns.name = None


# In[52]:


'''
ori = voice_opp_len
for i in range(0, ori.columns.size):
    # print(len(pd.unique(ori.iloc[:, i])))
    if len(pd.unique(ori.iloc[:, i])) < 30:
        voice_opp_len = voice_opp_len.drop(columns = ori.columns[i], axis=1)
'''


# In[11]:


print(voice_opp_len)


# # Feature 37

# In[12]:


voice_call_type = train_voice_df.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').fillna(0)


# In[13]:


voice_call_type.columns.name = None


# In[14]:


print(voice_call_type)


# # Feature 38

# In[15]:


voice_in_out = train_voice_df.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').fillna(0)


# In[16]:


voice_in_out.columns.name = None


# In[17]:


print(voice_in_out)


# # Feature 39

# In[18]:


train_1 = train_voice_df[train_voice_df['in_out'] == 0]
train_2 = train_voice_df[train_voice_df['in_out'] == 1]


# In[20]:


t1 = train_1['uid'].value_counts()
t2 = train_2['uid'].value_counts()
dict = t1.to_dict()
tt1 = pd.Series(index = range(1, 5000), data = dict)
tt1 = tt1.fillna(0)
dict = t2.to_dict()
tt2 = pd.Series(index = range(1, 5000), data = dict)
tt2 = tt2.fillna(0)
tt3 = tt1 / (tt1 + tt2)


# In[21]:


print(tt3)


# In[22]:


tt3 = tt3.fillna(0.5)


# In[23]:


voice_in_out_rate = pd.DataFrame(tt3)
voice_in_out_rate.columns = ['in_out_rate']


# In[24]:


print(voice_in_out_rate)


# # Feature 40

# In[25]:


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
tmp = train_voice_df
train_voice_df.loc[:, 'dura_second'] = tmp.apply(do, axis = 1)


# In[26]:


train_1 = train_voice_df[train_voice_df['in_out'] == 0]
train_2 = train_voice_df[train_voice_df['in_out'] == 1]


# In[27]:


g = train_1[train_1['dura_second'] <= 10]


# In[28]:


cnt = g['uid'].value_counts().sort_index()


# In[29]:


cnt.index.name = 'uid'
voice_short_dura_count = pd.DataFrame(cnt)
voice_short_dura_count.columns = ['voice_short_dura_count']


# In[30]:


print(voice_short_dura_count)


# # Feature 41

# In[31]:


def do(x):
    return int(x['start_time'] / 1000000)
tmp = train_voice_df
train_voice_df.loc[:, 'date'] = tmp.apply(do, axis = 1)


# In[32]:


print(train_voice_df[['uid', 'start_time', 'date']])


# In[33]:


voice_date = train_voice_df.groupby(['uid','date'])['uid'].count().unstack().add_prefix('date_').fillna(0)


# In[34]:


print(voice_date)


# # Merge Feature 34 to 41

# In[35]:


agg = voice_opp_num
agg = agg.join(voice_opp_len)
agg = agg.join(voice_opp_head)
agg = agg.join(voice_call_type)
agg = agg.join(voice_in_out)
agg = agg.join(voice_in_out_rate)
agg = agg.join(voice_short_dura_count)
agg = agg.join(voice_date)


# In[38]:


agg = agg.drop('voice_call_type_4', axis = 1)


# In[40]:


agg = agg.drop(u'date_0', axis = 1)


# In[42]:


print(agg)


# In[43]:


dict = agg.to_dict()
agg2 = pd.DataFrame(index = range(1, 5000), data = dict)


# In[44]:


agg2 = agg2.fillna(0)
agg2 = agg2.astype('int64')


# In[46]:


agg2.to_csv('../data/train/train_feature_34to41.csv', index=True, header=None)

