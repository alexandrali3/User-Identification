
# coding: utf-8

# In[183]:


get_ipython().magic(u'matplotlib inline')


# In[184]:


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics


# In[388]:


train_df = pd.read_csv('../data/train/train_feature_21to33_14_17_19_20_norm_2.csv', header = 0, encoding = 'utf-8')
test_a_df = pd.read_csv('../data/Test-A/test_a_feature_21to33_14_17_19_20_norm_2.csv', header = 0, encoding = 'utf-8')


# In[389]:


trains = train_df


# In[390]:


trains.drop_duplicates(inplace=True)


# In[391]:


online_test = test_a_df


# In[392]:


from sklearn.model_selection import train_test_split
train_xy,offline_test = train_test_split(trains,test_size = 0.2,random_state=21)
train,val = train_test_split(train_xy,test_size = 0.2,random_state=21)


# In[393]:


y_train = train.label
X_train = train.drop(['uid','label'],axis=1)


# In[394]:


y_val = val.label
X_val = val.drop(['uid','label'],axis=1)


# In[395]:


offline_test_X = offline_test.drop(['uid','label'],axis=1)
online_test_X  = online_test.drop(['uid'],axis=1)


# # 数据转换

# In[396]:


lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,free_raw_data=False)


# # 设置参数

# In[397]:


params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
        'metric_freq': 100,
        'is_training_metric': True,
        'min_data_in_leaf': 360,
        'num_leaves': 50,
        'learning_rate': 0.08,
        'is_unbalance': True,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_hessian': 0.05,
         'verbosity':-1
}


# # 交叉验证

# In[143]:


def evalMetric(preds,dtrain):
    
    label = dtrain.get_label()
    
    
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre= pre.sort_values(by='preds',ascending=False)
    
    auc = metrics.roc_auc_score(pre.label,pre.preds)

    pre.preds=pre.preds.map(lambda x: 1 if x>=0.5 else 0)

    f1 = metrics.f1_score(pre.label,pre.preds)
    
    
    res = 0.6*auc +0.4*f1
    
    return 'res',res,True


# In[144]:


print(X_train)


# # 调参1

# In[150]:


max_mean = float(0)
best_params = {}


# In[151]:


for num_leaves in range(50, 55, 5):
    for learning_rate in np.arange(0.05, 1.0, 0.05):
        params['num_leaves'] = num_leaves
        # params['max_depth'] = max_depth
        params['learning_rate'] = learning_rate
        
        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=2018,
                            nfold=3,
                            feval=evalMetric,
                            metrics=['evalMetric'],
                            early_stopping_rounds=100,
                            verbose_eval=5,
                            num_boost_round=10000,
                            )

        res_mean = pd.Series(cv_results['res-mean']).max()
        # boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
        print(res_mean)

        if res_mean > max_mean:
            print(num_leaves, learning_rate)
            max_mean = res_mean
            best_params['num_leaves'] = num_leaves
            # best_params['max_depth'] = max_depth
            best_params['learning_rate'] = learning_rate

params['num_leaves'] = best_params['num_leaves']
# params['max_depth'] = best_params['max_depth']
params['learning_rate'] = best_params['learning_rate']


# In[152]:


print(params['num_leaves'], params['learning_rate'])


# # 调参2

# In[155]:


for max_bin in range(255,256,1):
    for min_data_in_leaf in range(300,500,10):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf
            
            cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=42,
                            nfold=3,
                            feval=evalMetric,
                            metrics=['evalMetric'],
                            early_stopping_rounds=100,
                            verbose_eval=5,
                            num_boost_round=10000,
                            )
                    
            res_mean = pd.Series(cv_results['res-mean']).max()
            # boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
            print(res_mean)

            if res_mean > max_mean:
                print(max_bin, min_data_in_leaf)
                max_mean = res_mean
                best_params['max_bin']= max_bin
                best_params['min_data_in_leaf'] = min_data_in_leaf

params['min_data_in_leaf'] = best_params['min_data_in_leaf']
params['max_bin'] = best_params['max_bin']


# In[156]:


print(params['min_data_in_leaf'], params['max_bin'])


# # 调参3

# In[157]:


for feature_fraction in [0.5,0.6,0.7,0.8,0.9,1.0]:
    for bagging_fraction in [0.5,0.6,0.7,0.8,0.9,1.0]:
        for bagging_freq in range(0,11,5):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq
            
            cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=42,
                            nfold=3,
                            feval=evalMetric,
                            metrics=['evalMetric'],
                            early_stopping_rounds=100,
                            verbose_eval=5,
                            num_boost_round=10000,
                            )
                    
            res_mean = pd.Series(cv_results['res-mean']).max()
            # boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
            print(res_mean)

            if res_mean > max_mean:
                print(feature_fraction, bagging_fraction, bagging_freq)
                max_mean = res_mean
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq

params['feature_fraction'] = best_params['feature_fraction']
params['bagging_fraction'] = best_params['bagging_fraction']
params['bagging_freq'] = best_params['bagging_freq']


# In[158]:


print(params['feature_fraction'], params['bagging_fraction'], params['bagging_freq'])


# # 调参4

# In[160]:


for lambda_l1 in [0.0,0.1,0.2,0.3,0.4]:
    for lambda_l2 in [0.0,0.1,0.2,0.3,0.4]:
        for min_split_gain in [0.0,0.1,0.2,0.3,0.4]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            params['min_split_gain'] = min_split_gain
            
            cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=42,
                            nfold=3,
                            feval=evalMetric,
                            metrics=['evalMetric'],
                            early_stopping_rounds=100,
                            verbose_eval=5,
                            num_boost_round=10000,
                            )
                    
            res_mean = pd.Series(cv_results['res-mean']).max()
            # boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
            print(res_mean)

            if res_mean > max_mean:
                print(lambda_l1, lambda_l2, min_split_gain)
                max_mean = res_mean
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
                best_params['min_split_gain'] = min_split_gain

params['lambda_l1'] = best_params['lambda_l1']
params['lambda_l2'] = best_params['lambda_l2']
params['min_split_gain'] = best_params['min_split_gain']


# In[161]:


print(best_params)


# In[398]:


params['num_leaves'] = 50
params['lambda_l1'] = 0.0
params['bagging_freq'] = 0
params['learning_rate'] = 0.2
params['lambda_l2'] = 0.1
params['min_split_gain'] = 0.2
params['min_data_in_leaf'] = 360
params['max_bin'] = 255
params['bagging_fraction'] = 0.5
params['feature_fraction'] = 0.6


# # Train

# In[399]:


model = lgb.train(
          params,                     # 参数字典
          lgb_train,                  # 训练集
          valid_sets=lgb_eval,        # 验证集
          num_boost_round=2000,       # 迭代次数
          early_stopping_rounds=50,
        feval=evalMetric,
        verbose_eval=5
    )


# # 线下预测

# In[400]:


preds_offline = model.predict(offline_test_X, num_iteration=model.best_iteration)
offline=offline_test[['uid','label']]
offline.loc[:, 'preds']=preds_offline
offline.loc[:, 'label'] = offline['label'].astype(np.float64)
print('log_loss', metrics.log_loss(offline.label, offline.preds))


# # 线上预测

# In[381]:


preds_online =  model.predict(online_test_X, num_iteration=model.best_iteration)


# In[382]:


id_series = pd.DataFrame({'uid': test_a_df.uid})
def do3(x):
    tmp = "%04d" % x['uid']
    return 'u' + str(tmp)
id_series.loc[:, 'uid'] = id_series.apply(do3, axis = 1)


# In[383]:


res =pd.DataFrame({'uid':id_series['uid'],'label':preds_online})


# In[384]:


res=res.sort_values(by='label',ascending=False)
res.label=res.label.map(lambda x: 1 if x>=0.5 else 0)


# In[385]:


print(res)


# In[386]:


res.to_csv('../result/lgb-baseline-selfparam-4.csv',index=False,header=False,sep=',',columns=['uid','label'])


# # 保存模型

# In[281]:


from sklearn.externals import joblib
joblib.dump(model,'model.pkl')


# # Feature Importance

# In[387]:


df = pd.DataFrame(X_train.columns.tolist(), columns=['feature'])
df['importance']=list(model.feature_importance())
df = df.sort_values(by='importance',ascending=False)
df.to_csv("../result/feature_score_3.csv",index=None,encoding='gbk')

