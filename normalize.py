
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


# In[33]:


def myPrint(x):
    print(x.head(15))
    print(x.info())
def myPrint2(x):
    print(x)


# In[34]:


train_df = pd.read_csv('../data/train/train_feature_21to33_14_17_19_20.csv', header = 0, encoding = 'utf-8')


# In[35]:


test_a_df = pd.read_csv('../data/Test-A/test_a_feature_21to33_14_17_19_20.csv', header = 0, encoding = 'utf-8')


# In[36]:


myPrint(train_df)


# In[37]:


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()


# In[38]:


train_2 = train_df
for i in [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 18, 19, 20, 21, 23, 24]:
    tmp = str(i)
    X_train_norm = mms.fit_transform(train_df[tmp].reshape(-1, 1))
    train_2[tmp] = X_train_norm.ravel()


# In[39]:


myPrint(train_2)


# In[40]:


train_2.to_csv('../data/train/train_feature_21to33_14_17_19_20_norm.csv', index=False)


# In[41]:


test_a_2 = test_a_df
for i in [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 18, 19, 20, 21, 23, 24]:
    tmp = str(i)
    X_test_norm = mms.fit_transform(test_a_df[tmp].reshape(-1, 1))
    test_a_2[tmp] = X_test_norm.ravel()


# In[42]:


myPrint(test_a_2)


# In[43]:


test_a_2.to_csv('../data/Test-A/test_a_feature_21to33_14_17_19_20_norm.csv', index=False)

