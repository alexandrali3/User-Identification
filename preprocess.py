
# coding: utf-8

# In[3]:


get_ipython().magic(u'matplotlib inline')


# In[4]:


import sys
import numpy as np
import pandas as pd
import re as re
import matplotlib.pyplot as plt
import xgboost as xgb


# In[3]:


train_uid = pd.read_table('../data/train/uid_train.txt', header = None, names = ['uid', 'label'])


# In[4]:


print(train_uid.info())


# In[51]:


train_voice = pd.read_table('../data/train/voice_train.txt', header = None, names = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out'], dtype = {'opp_num':str})


# In[52]:


print(train_voice.head())


# In[7]:


train_sms = pd.read_table('../data/train/sms_train.txt', header = None, names = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out'], dtype = {'opp_num':str})


# In[8]:


print(train_sms.info())


# In[9]:


train_wa = pd.read_table('../data/train/wa_train.txt', header = None, names = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'wa_type', 'date'])


# In[10]:


print(train_wa.info())


# In[11]:


train_wa


# In[17]:


train_wa_df = train_wa[True^train_wa['wa_name'].isnull()]


# In[21]:


train_wa_df


# In[24]:


train_wa_df.loc[:,'date'] = train_wa_df.loc[:,'date'].astype(int)


# In[26]:


print(train_wa_df.info())


# In[28]:


train_wa_df.loc[:,'visit_cnt'] = train_wa_df.loc[:,'visit_cnt'].astype(int)


# In[29]:


print(train_wa_df.info())


# In[30]:


train_wa_df.loc[:,['up_flow', 'down_flow']] = train_wa_df.loc[:,['up_flow','down_flow']].astype(int)


# In[31]:


print(train_wa_df.info())


# In[32]:


train_wa_df.loc[:,['visit_dura', 'wa_type']] = train_wa_df.loc[:,['visit_dura', 'wa_type']].astype(int)


# In[33]:


print(train_wa_df.info())


# In[34]:


train_wa_df.loc[:,'wa_name'] = train_wa_df.loc[:,'wa_name'].astype(str)


# In[36]:


print(train_wa_df.info())


# In[41]:


print(train_wa_df.head(10))


# In[42]:


train_wa_df.to_csv('../data/train_wa_df.csv', index=False)


# In[62]:


train_voice_df = train_voice


# In[63]:


train_voice_df.loc[:,'in_out'] = train_voice_df.loc[:,'in_out'].astype(int)


# In[64]:


print(train_voice_df.info())


# In[65]:


train_voice_df.to_csv('../data/train_voice_df.csv', index=False)


# In[67]:


print(train_sms.info())


# In[69]:


train_sms_df = train_sms


# In[70]:


train_sms_df.to_csv('../data/train/train_sms_df.csv', index=False)


# In[19]:


def do2(x):
    x = int(x.replace('u', ''))
    return x
def do(dataset):
    ans = dataset
    ans.loc[:, 'uid'] = ans['uid'].apply(do2)
    
    return ans


# In[20]:


train_uid_df = pd.read_csv('../data/train/train_uid_df.csv', header = 0)
tmp = train_uid_df
train_uid_df_2 = do(tmp)


# In[21]:


print(train_uid_df_2)


# In[22]:


train_voice_df = pd.read_csv('../data/train/train_voice_df.csv', header = 0, dtype = {'opp_num':str})
train_sms_df = pd.read_csv('../data/train/train_sms_df.csv', header = 0, dtype = {'opp_num':str})
train_wa_df = pd.read_csv('../data/train/train_wa_df.csv', header = 0)


# In[23]:


train_voice_df_2 = do(train_voice_df)


# In[25]:


print(train_voice_df)


# In[26]:


train_sms_df_2 = do(train_sms_df)
train_wa_df_2 = do(train_wa_df)


# In[27]:


print(train_sms_df_2)


# In[28]:


print(train_wa_df_2)


# In[29]:


train_uid_df_2.to_csv('../data/train/train_uid_df.csv', index=False)
train_voice_df_2.to_csv('../data/train/train_voice_df.csv', index=False)
train_sms_df_2.to_csv('../data/train/train_sms_df.csv', index=False)
train_wa_df_2.to_csv('../data/train/train_wa_df.csv', index=False)

