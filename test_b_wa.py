
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


test_b_wa_df = pd.read_csv('../data/Test-B/test_b_wa_df.csv', header = 0)


# In[5]:


myPrint(test_b_wa_df)


# # Feature 21

# In[6]:


group = test_b_wa_df.groupby('uid')['visit_cnt']
print(group.unique())


# In[28]:


visit_cnt = group.agg(['std', 'var', 'max','min','median','mean','sum', 'count', 'mad']).add_prefix('wa_visit_cnt_')


# In[29]:


print(visit_cnt)


# # Feature 22

# In[30]:


visit_dura = test_b_wa_df.groupby(['uid'])['visit_dura'].agg(['std', 'var', 'max','min','median','mean','sum', 'count', 'mad']).add_prefix('wa_visit_dura_')


# In[31]:


print(visit_dura)


# # Feature 23

# In[32]:


up_flow = test_b_wa_df.groupby(['uid'])['up_flow'].agg(['std', 'var', 'max','min','median','mean','sum', 'count', 'mad']).add_prefix('wa_up_flow_')


# In[33]:


print(up_flow)


# # Feature 24

# In[34]:


down_flow = test_b_wa_df.groupby(['uid'])['down_flow'].agg(['std', 'var', 'max','min','median','mean','sum', 'count', 'mad']).add_prefix('wa_down_flow_')


# In[35]:


print(down_flow)


# # Feature 25

# In[36]:


group = test_b_wa_df.groupby('uid')['wa_name']


# In[37]:


agg = group.aggregate({'wa_name': lambda x: x.nunique()})


# In[38]:


agg.rename(columns=lambda x:x.replace('wa_name','wa_name_count'), inplace=True)


# In[39]:


print(agg)


# In[40]:


wa_name = agg


# # Feature 26

# In[43]:


wa_type = test_b_wa_df.groupby(['uid','wa_type'])['uid'].count().unstack().add_prefix('wa_type_').fillna(0)


# In[44]:


wa_type.columns.names = ['']


# In[45]:


print(wa_type)


# # Feature 27

# In[46]:


date = test_b_wa_df.groupby(['uid','date'])['uid'].count().unstack().add_prefix('date_').fillna(0)


# In[47]:


date.columns.names = ['']


# In[48]:


print(date)


# # Merge Feature 21 to 27

# In[49]:


agg = visit_cnt
agg = agg.join(visit_dura)
agg = agg.join(up_flow)
agg = agg.join(down_flow)
agg = agg.join(wa_name)
agg = agg.join(wa_type)
agg = agg.join(date)


# In[50]:


print(agg)


# In[51]:


dict = agg.to_dict()
agg2 = pd.DataFrame(index = range(7000, 10000), data = dict)


# In[52]:


agg2 = agg2.fillna(0)
agg2 = agg2.astype('int64')


# In[54]:


agg2.to_csv('../data/Test-B/test_b_feature_21to27.csv', index=True, header=None)

