
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


test_a_wa_df = pd.read_csv('../data/Test-A/test_a_wa_df.csv', header = 0)


# In[5]:


myPrint(test_a_wa_df)


# # Feature 1

# In[8]:


tmp = test_a_wa_df.uid.value_counts()
tmp = tmp.sort_index()


# In[11]:


myPrint2(tmp)


# In[14]:


dict = tmp.to_dict()
tmp2 = pd.Series(index = range(5000, 7000), data = dict)


# In[15]:


tmp2 = tmp2.fillna(0)
tmp2 = tmp2.astype('int64')


# In[16]:


myPrint2(tmp2)


# In[17]:


tmp2.to_csv('../data/Test-A/test_a_feature_1.csv', index=True)


# # Feature 2 to 5

# In[18]:


tmp = test_a_wa_df[['uid', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow']].groupby('uid').sum()


# In[19]:


dict = tmp.to_dict()
tmp2 = pd.DataFrame(index = range(5000, 7000), data = dict)


# In[20]:


tmp2 = tmp2.fillna(0)
tmp2.loc[:, ['down_flow', 'up_flow', 'visit_dura', 'visit_cnt']] = tmp2.loc[:, ['down_flow', 'up_flow', 'visit_dura', 'visit_cnt']].astype('int64')


# In[22]:


tmp2.to_csv('../data/Test-A/test_a_feature_2to5.csv', index=True)


# # Feature 6

# In[23]:


group = test_a_wa_df.groupby('uid')['wa_name']


# In[24]:


agg = group.aggregate({'wa_name': lambda x: x.nunique()})


# In[25]:


agg.rename(columns=lambda x:x.replace('wa_name','wa_cnt'), inplace=True)


# In[26]:


dict = agg.to_dict()
agg2 = pd.DataFrame(index = range(5000, 7000), data = dict)


# In[27]:


agg2 = agg2.fillna(0)
agg2 = agg2.astype('int64')


# In[28]:


agg2.to_csv('../data/Test-A/test_a_feature_6.csv', index=True)


# # Feature 21

# In[6]:


group = test_a_wa_df.groupby('uid')['visit_cnt']
print(group.unique())


# In[7]:


visit_cnt = group.agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_')


# In[8]:


print(visit_cnt)


# # Feature 22

# In[9]:


visit_dura = test_a_wa_df.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_')


# In[10]:


print(visit_dura)


# # Feature 23

# In[11]:


up_flow = test_a_wa_df.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_')


# In[12]:


print(up_flow)


# # Feature 24

# In[13]:


down_flow = test_a_wa_df.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_')


# # Merge Feature 21 to 24

# In[14]:


agg = visit_cnt
agg = agg.join(visit_dura)
agg = agg.join(up_flow)
agg = agg.join(down_flow)


# In[15]:


print(agg)


# In[16]:


dict = agg.to_dict()
agg2 = pd.DataFrame(index = range(5000, 7000), data = dict)


# In[17]:


agg2 = agg2.fillna(0)
agg2 = agg2.astype('int64')


# In[18]:


agg2.to_csv('../data/Test-A/test_a_feature_21to24.csv', index=True, header=None)

