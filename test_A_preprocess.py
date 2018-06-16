
# coding: utf-8

# In[2]:


get_ipython().magic(u'matplotlib inline')


# In[3]:


import sys
import numpy as np
import pandas as pd
import re as re
import matplotlib.pyplot as plt
import xgboost as xgb


# In[5]:


def myPrint(x):
    print(x.head(15))
    print(x.info())


# In[6]:


test_a_wa = pd.read_table('../data/Test-A/wa_test_a.txt', header = None, names = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'wa_type', 'date'])


# In[7]:


myPrint(test_a_wa)


# In[8]:


test_a_voice = pd.read_table('../data/Test-A/voice_test_a.txt', header = None, names = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out'], dtype = {'opp_num':str})


# In[9]:


myPrint(test_a_voice)


# In[10]:


test_a_sms = pd.read_table('../data/Test-A/sms_test_a.txt', header = None, names = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out'], dtype = {'opp_num':str})


# In[11]:


myPrint(test_a_sms)


# In[13]:


test_a_wa[test_a_wa['wa_name'].isnull()]


# In[16]:


test_a_wa_df = test_a_wa[True^test_a_wa['wa_name'].isnull()]


# In[17]:


myPrint(test_a_wa_df)


# In[22]:


test_a_wa_df.loc[:,'date'] = test_a_wa_df.loc[:,'date'].astype('int64')


# In[23]:


test_a_wa_df.loc[:,['up_flow', 'down_flow']] = test_a_wa_df.loc[:,['up_flow','down_flow']].astype('int64')


# In[26]:


test_a_wa_df.loc[:,['visit_dura', 'wa_type']] = test_a_wa_df.loc[:,['visit_dura', 'wa_type']].astype('int64')


# In[28]:


test_a_wa_df.loc[:,'visit_cnt'] = test_a_wa_df.loc[:,'visit_cnt'].astype('int64')


# In[29]:


myPrint(test_a_wa_df)


# # voice

# In[30]:


test_a_voice_df = test_a_voice


# In[31]:


test_a_voice_df.loc[:,'in_out'] = test_a_voice_df.loc[:,'in_out'].astype('int64')


# In[32]:


myPrint(test_a_voice_df)


# # sms

# In[33]:


test_a_sms_df = test_a_sms


# In[34]:


def do2(x):
    x = int(x.replace('u', ''))
    return x
def do(dataset):
    ans = dataset
    ans.loc[:, 'uid'] = ans['uid'].apply(do2)
    
    return ans


# In[35]:


test_a_voice_df_2 = do(test_a_voice_df)


# In[36]:


test_a_sms_df_2 = do(test_a_sms_df)


# In[37]:


test_a_wa_df_2 = do(test_a_wa_df)


# In[38]:


myPrint(test_a_wa_df_2)


# In[39]:


myPrint(test_a_voice_df_2)


# In[40]:


test_a_voice_df_2.to_csv('../data/Test-A/test_a_voice_df.csv', index=False)
test_a_sms_df_2.to_csv('../data/Test-A/test_a_sms_df.csv', index=False)
test_a_wa_df_2.to_csv('../data/Test-A/test_a_wa_df.csv', index=False)

