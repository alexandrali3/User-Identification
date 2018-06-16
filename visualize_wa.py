
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


print(train_uid_df.head(10))


# In[5]:


train_voice_df = pd.read_csv('../data/train/train_voice_df.csv', header = 0, dtype = {'opp_num':str})


# In[6]:


print(train_voice_df.head(10))


# In[7]:


train_sms_df = pd.read_csv('../data/train/train_sms_df.csv', header = 0, dtype = {'opp_num':str})


# In[8]:


train_wa_df = pd.read_csv('../data/train/train_wa_df.csv', header = 0)


# In[9]:


print(train_sms_df.head(10))


# In[10]:


print(train_wa_df.head(15))


# In[11]:


tmp = train_voice_df['opp_head'].value_counts()


# In[12]:


tmp


# In[13]:


tmp.value_counts()


# In[14]:


train_voice_df['uid'].value_counts()


# In[15]:


train_uid_df.label.value_counts()


# In[16]:


plt.scatter(train_uid_df.uid, train_uid_df.label)
plt.ylabel(u"label")
plt.xlabel(u"uid")
plt.title(u"label-uid(1:danger)")
plt.show()


# In[18]:


print(train_wa_df.info())


# In[19]:


print(train_wa_df.head(20))


# In[12]:


tmp = train_wa_df.uid.value_counts()
tmp = tmp.sort_index()
print(tmp)


# In[13]:


tmp.plot(kind='bar')


# In[15]:


tmp[tmp.index >= 4100].plot(kind='bar')


# In[16]:


tmp[tmp.index < 4100].plot(kind='bar')


# In[17]:


tmp1 = tmp[tmp.index < 4100]
tmp2 = tmp[tmp.index >= 4100]


# In[20]:


print(tmp1.describe())


# In[21]:


print(tmp2.describe())


# In[24]:


tmp.to_csv('../data/train/train_feature_num_records.csv', index=True)


# In[30]:


dict = tmp.to_dict()


# In[31]:


print(dict)


# In[42]:


tmp2 = pd.Series(index = range(1, 5000), data = dict, dtype = np.int)


# In[54]:


tmp2 = tmp2.fillna(0)


# In[56]:


tmp2 = tmp2.astype('int')


# In[57]:


print(tmp2)


# In[58]:


tmp2.to_csv('../data/train/train_feature_num_records.csv', index=True)


# In[59]:


tmp1 = train_wa_df[train_wa_df['uid'] < 4100]


# In[60]:


tmp2 = train_wa_df[train_wa_df['uid'] >= 4100]


# In[61]:


print(tmp1.info())


# In[70]:


tmp = train_wa_df[['uid', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow']].groupby('uid').sum()


# In[76]:


print(tmp.head(40))


# In[83]:


tmp[tmp.index < 4100].describe()


# In[82]:


tmp[tmp.index >= 4100].describe()


# In[85]:


dict = tmp.to_dict()


# In[86]:


print(dict)


# In[95]:


tmp2 = pd.DataFrame(index = range(1, 5000), data = dict)


# In[96]:


tmp2 = tmp2.fillna(0)
print(tmp2[tmp2.index == 35])


# In[97]:


tmp2.loc[:, ['down_flow', 'up_flow', 'visit_dura', 'visit_cnt']] = tmp2.loc[:, ['down_flow', 'up_flow', 'visit_dura', 'visit_cnt']].astype('int64')


# In[98]:


print(tmp2)


# In[100]:


tmp2.to_csv('../data/train/train_feature_2to5.csv', index=True)


# In[116]:


group = train_wa_df.groupby('uid')['wa_name']
print(group.unique())


# In[113]:


agg = group.aggregate({'wa_name': lambda x: x.nunique()})


# In[117]:


print(type(agg))


# In[119]:


print(agg[agg.index < 4100].describe())


# In[121]:


print(agg[agg.index >= 4100].describe())


# In[129]:


print(agg)


# In[132]:


agg.rename(columns=lambda x:x.replace('wa_name','wa_cnt'), inplace=True)


# In[133]:


print(agg)


# In[134]:


dict = agg.to_dict()


# In[135]:


print(dict)


# In[136]:


agg2 = pd.DataFrame(index = range(1, 5000), data = dict)
agg2 = agg2.fillna(0)


# In[137]:


print(agg2)


# In[141]:


agg2 = agg2.astype('int64')


# In[142]:


print(agg2.info())


# In[143]:


agg2.to_csv('../data/train/train_feature_6.csv', index=True)


# In[148]:


tmp = train_wa_df['date'].value_counts()
tmp = tmp.sort_index()


# In[149]:


print(tmp)


# In[156]:


tmp = train_sms_df['uid'].value_counts().sort_index()
print(tmp)


# In[163]:


t1 = train_sms_df[train_sms_df['in_out'] == 0]


# In[169]:


tmp = t1['uid'].value_counts().sort_index()
print(tmp)


# In[165]:


tmp1 = tmp[tmp.index < 4100]
tmp2 = tmp[tmp.index >= 4100]


# In[166]:


tmp1.describe()


# In[167]:


tmp2.describe()


# In[168]:


print(tmp2)


# In[170]:


dict = tmp.to_dict()
tmp2 = pd.Series(index = range(1, 5000), data = dict)


# In[171]:


tmp2 = tmp2.fillna(0)


# In[172]:


tmp2 = tmp2.astype('int64')


# In[173]:


tmp2.to_csv('../data/train/train_feature_7.csv', index=True)


# In[174]:


t2 = train_sms_df[train_sms_df['in_out'] == 1]


# In[175]:


tmp = t2['uid'].value_counts().sort_index()


# In[176]:


tmp1 = tmp[tmp.index < 4100]
tmp2 = tmp[tmp.index >= 4100]


# In[177]:


tmp1.describe()


# In[178]:


tmp2.describe()


# In[179]:


dict = tmp.to_dict()
tmp2 = pd.Series(index = range(1, 5000), data = dict)


# In[180]:


tmp2 = tmp2.fillna(0)


# In[181]:


tmp2 = tmp2.astype('int64')


# In[182]:


tmp2.to_csv('../data/train/train_feature_8.csv', index=True)


# # Feature 21

# In[11]:


group = train_wa_df.groupby('uid')['visit_cnt']
print(group.unique())


# In[18]:


visit_cnt = group.agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_')


# In[19]:


print(agg)


# # Feature 22

# In[21]:


visit_dura = train_wa_df.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_')


# In[22]:


print(visit_dura)


# # Feature 23

# In[23]:


up_flow = train_wa_df.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_')


# In[24]:


print(up_flow)


# # Feature 24

# In[40]:


down_flow = train_wa_df.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_')


# In[41]:


print(down_flow)


# Merge feature 21 to 24

# In[42]:


agg = visit_cnt
agg = agg.join(visit_dura)
agg = agg.join(up_flow)
agg = agg.join(down_flow)


# In[43]:


print(agg)


# In[44]:


dict = agg.to_dict()


# In[45]:


agg2 = pd.DataFrame(index = range(1, 5000), data = dict)


# In[46]:


agg2 = agg2.fillna(0)
agg2 = agg2.astype('int64')


# In[48]:


agg2.to_csv('../data/train/train_feature_21to24.csv', index=True, header=None)

