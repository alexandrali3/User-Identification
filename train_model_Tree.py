
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')


# In[2]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


# In[43]:


train_df = pd.read_csv('../data/train/train_feature_21to33_14_17_19_20_norm.csv', header = 0, encoding = 'utf-8')
test_a_df = pd.read_csv('../data/Test-A/test_a_feature_21to33_14_17_19_20_norm.csv', header = 0, encoding = 'utf-8')


# In[44]:


dtrain = lgb.Dataset(train_df.drop(['uid','label'],axis=1),label=train_df.label)
dtest = lgb.Dataset(test_a_df.drop(['uid'],axis=1))


# In[45]:


id_series = pd.DataFrame({'uid': test_a_df.uid})
def do3(x):
    tmp = "%04d" % x['uid']
    return 'u' + str(tmp)
id_series.loc[:, 'uid'] = id_series.apply(do3, axis = 1)


# In[61]:


lgb_params =  {
   'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'metric_freq': 100,
    'is_training_metric': True,
    'min_data_in_leaf': 360,
    'num_leaves': 60,
    'learning_rate': 0.085,
    'is_unbalance': True,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'min_hessian': 0.05,
     'verbosity':-1
#    'gpu_device_id':2,
#     'device':'gpu'
 #   'lambda_l1': 0.001,
 #   'skip_drop': 0.95,
 #   'max_drop' : 10,
# 'lambda_l2': 0.005,
 #'num_threads': 18,
}


# In[62]:


def evalMetric(preds,dtrain):
    
    label = dtrain.get_label()
    
    
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre= pre.sort_values(by='preds',ascending=False)
    
    auc = metrics.roc_auc_score(pre.label,pre.preds)

    pre.preds=pre.preds.map(lambda x: 1 if x>=0.5 else 0)

    f1 = metrics.f1_score(pre.label,pre.preds)
    
    
    res = 0.6*auc +0.4*f1
    
    return 'res',res,True


# In[63]:


cv_results = lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=5,num_boost_round=10000,nfold=3,metrics=['evalMetric'])


# In[64]:


res_mean = pd.Series(cv_results['res-mean']).max()
print(res_mean)


# In[65]:


model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=5,num_boost_round=300,valid_sets=[dtrain])


# In[66]:


pred=model.predict(test_a_df.drop(['uid'],axis=1))


# In[67]:


res =pd.DataFrame({'uid':id_series['uid'],'label':pred})


# In[68]:


res=res.sort_values(by='label',ascending=False)
res.label=res.label.map(lambda x: 1 if x>=0.5 else 0)
# res.label = res.label.map(lambda x: int(x))


# In[69]:


print(res)


# In[70]:


res.to_csv('../result/lgb-baseline-7.csv',index=False,header=False,sep=',',columns=['uid','label'])


# # 2 layer model

# In[503]:


train_df = pd.read_csv('../data/train/train_feature_21to33.csv', header = 0, encoding = 'utf-8')


# In[504]:


test_a_df = pd.read_csv('../data/Test-A/test_a_feature_21to33.csv', header = 0, encoding = 'utf-8')


# In[505]:


train_df = train_df.set_index(u'uid')
test_a_df = test_a_df.set_index(u'uid')


# In[506]:


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


# In[507]:


# Some useful parameters which will come in handy later on
ntrain = train_df.shape[0]
ntest = test_a_df.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 4 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds = NFOLDS, random_state = SEED)


# In[508]:


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


# In[509]:


import time
def get_time_stamp():
    now = int(time.time())
    return now


# In[530]:


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


# In[531]:


# Create 5 objects that represent our 5 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# In[532]:


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train_df['label'].ravel()
train_noy = train_df.drop(['label'], axis=1)
x_train = train_noy.values # Creates an array of the train data
x_test = test_a_df.values # Creats an array of the test data
print (x_train.shape, x_test.shape, y_train.shape)


# In[533]:


dtrain = train_df


# In[534]:


def cal_precision(preds, dtrain):
    label = dtrain['label']
    
    pre = pd.DataFrame({'preds':preds,'label':label})
    print(pre)
    pre= pre.sort_values(by='preds',ascending=False)
    
    auc = metrics.roc_auc_score(pre.label,pre.preds)

    pre.preds=pre.preds.map(lambda x: 1 if x>=0.5 else 0)

    f1 = metrics.f1_score(pre.label,pre.preds)
    
    
    res = 0.6*auc +0.4*f1
    
    return 'res',res,True


# In[535]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        # print(x_tr, y_tr, x_te)

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
    pd.Series(oof_train).to_csv('../data/oof_train.csv', header = None)
    print(cal_precision(pd.Series(oof_train), dtrain))
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[536]:


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


# In[537]:


# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof2(et, x_train, y_train, x_test) # Extra Trees
print("Training is complete")

# rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
# print("Training is complete")

ada_oof_train, ada_oof_test = get_oof2(ada, x_train, y_train, x_test) # AdaBoost 
print("Training is complete")

gb_oof_train, gb_oof_test = get_oof2(gb,x_train, y_train, x_test) # Gradient Boost
print("Training is complete")

# svc_oof_train, svc_oof_test = get_oof2(svc,x_train, y_train, x_test) # Support Vector Classifier
# print("Training is complete")

print("Training is complete!!!")


# In[538]:


x_train = np.concatenate(( et_oof_train, ada_oof_train, gb_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, ada_oof_test, gb_oof_test), axis=1)


# In[539]:


import xgboost as xgb

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


# In[540]:


id_series = pd.DataFrame({'uid': test_a_df.index})


# In[541]:


def do3(x):
    tmp = "%04d" % x['uid']
    return 'u' + str(tmp)
id_series.loc[:, 'uid'] = id_series.apply(do3, axis = 1)


# In[542]:


print(id_series['uid'])


# In[543]:


ans = pd.DataFrame({'uid': id_series['uid'], 'label': predictions}, columns = ['uid', 'label'])


# In[544]:


ans = ans.sort_values(by='label', ascending=False)


# In[545]:


ans.to_csv('../result/ans_4_df.csv', index=False, header = False)

