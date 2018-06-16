
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


# In[4]:


def myPrint(x):
    print(x.head(15))
    print(x.info())
def myPrint2(x):
    print(x)


# In[5]:


train_df = pd.read_csv('../data/train/train_feature_1to20_label.csv', header = 0, encoding = 'utf-8')


# In[6]:


test_a_df = pd.read_csv('../data/Test-A/test_a_feature_1to20.csv', header = 0, encoding = 'utf-8')


# In[7]:


train_df = train_df.set_index(u'uid')
test_a_df = test_a_df.set_index(u'uid')


# In[8]:


myPrint(train_df)


# In[9]:


myPrint(test_a_df)


# In[10]:


import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.cross_validation import KFold

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


import warnings
warnings.filterwarnings('ignore')


# In[11]:


# Some useful parameters which will come in handy later on
ntrain = train_df.shape[0]
ntest = test_a_df.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 4 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED)


# In[12]:


# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        
    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self, x, y):
        return self.clf.fit(x, y)
    
    def feature_importances(self, x, y):
        print (self.clf.fit(x, y).feature_importances_)


# In[13]:


import time
def get_time_stamp():
    now = int(time.time())
    return now


# In[14]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        time_before_train = get_time_stamp()
        clf.train(x_tr, y_tr)
        print("time for training:")
        print(get_time_stamp() - time_before_train)

        time_before_predict = get_time_stamp()
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
        print("time for predicting")
        print(get_time_stamp() - time_before_predict)

    oof_test[:] = oof_test_skf.mean(axis=0)
    print(cal_precision(oof_train.reshape(-1, 1), y_train.reshape(-1, 1)))
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[15]:


# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0,
    'loss': 'deviance'
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# In[16]:


# Create 5 objects that represent our 5 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# In[17]:


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train_df['label'].ravel()
train_noy = train_df.drop(['label'], axis=1)
x_train = train_noy.values # Creates an array of the train data
x_test = test_a_df.values # Creats an array of the test data
print (x_train.shape, x_test.shape, y_train.shape)


# In[18]:


print(x_train)


# In[19]:


def cal_precision(a, b):
    return 1 - float(np.sum(abs(a - b))) / a.size


# In[24]:


print(y_train, ntrain, ntest)


# In[27]:


# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof2(et, x_train, y_train, x_test) # Extra Trees
print("Training is complete")

# rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
# print("Training is complete")

ada_oof_train, ada_oof_test = get_oof2(ada, x_train, y_train, x_test) # AdaBoost 
print("Training is complete")

gb_oof_train, gb_oof_test = get_oof2(gb,x_train, y_train, x_test) # Gradient Boost
print("Training is complete")

svc_oof_train, svc_oof_test = get_oof2(svc,x_train, y_train, x_test) # Support Vector Classifier
print("Training is complete")

print("Training is complete!!!")


# In[26]:


def get_oof2(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    
    time_before_train = get_time_stamp()
    clf.train(x_train, y_train)
    print("time for training:")
    print(get_time_stamp() - time_before_train)
    
    time_before_predict = get_time_stamp()
    oof_train = clf.predict(x_train)
    oof_test = clf.predict(x_test)
    print("time for predicting")
    print(get_time_stamp() - time_before_predict)
    
    # oof_train = oof_train.reshape(-1, 1)
    oof_test = oof_test.reshape(-1, 1)
    # print(cal_precision(oof_train, y_train.reshape(-1, 1)))
    return oof_train.reshape(-1, 1), oof_test


# In[20]:


gb_oof_train, gb_oof_test = get_oof2(gb, x_train, y_train, x_test)
print("Training is complete")


# In[21]:


print(gb_oof_test)


# In[32]:


id_series = pd.DataFrame({'uid': test_a_df.index})


# In[33]:


def do3(x):
    tmp = "%04d" % x['uid']
    return 'u' + str(tmp)
id_series.loc[:, 'uid'] = id_series.apply(do3, axis = 1)


# In[34]:


print(id_series['uid'])


# In[28]:


ans = pd.DataFrame({'uid': id_series['uid'], 'label': gb_oof_test.ravel()}, columns = ['uid', 'label'])


# In[29]:


print(ans)


# In[30]:


ans = ans.sort_values(by='label', ascending=False)


# In[ ]:


ans.to_csv('../data/Test-A/ans_2_df.csv', index=False, header = False)


# In[29]:


x_train = np.concatenate(( et_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


# In[30]:


gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


# In[31]:


print(predictions)


# In[35]:


ans = pd.DataFrame({'uid': id_series['uid'], 'label': predictions}, columns = ['uid', 'label'])


# In[36]:


print(ans)


# In[38]:


ans = ans.sort_values(by='label', ascending=False)


# In[40]:


ans.to_csv('../data/Test-A/ans_3_df.csv', index=False, header = False)

