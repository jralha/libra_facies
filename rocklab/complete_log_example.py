#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import itertools
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, zscore
plt.style.use('ggplot')

#%%%
well = '3-BRSA-944A-RJS_ELASTIC.csv'

df = pd.read_csv(well,sep=';')
df = df.iloc[1:,:]
for col in df.columns:
    if (pd.api.types.is_numeric_dtype(df[col]) == False):
        try:
            df[col] = np.float32(df[col].values)   
            df = df.mask(df < 0)                   
        except:
            df[col]=df[col]
    
df = df.replace(-9999, np.nan)
df = df.replace('-9999', np.nan)
df = df.dropna()

feats = ['MD','DTCO','DTSM','RHOB','PhiMac']
df_feats = df[feats].dropna()

#%%
plt.figure(figsize=[6,8])
nplot=1
cut_cols=[]
for feat in feats:
    if feat != 'MD':
        plt.subplot(1,4,nplot)
        plt.plot(df_feats[feat],df_feats['MD'])
        plt.xlabel(feat)
        plt.ylim(top=np.min(df_feats['MD'])-25,bottom=np.max(df_feats['MD'])+25)
        if nplot==1: plt.ylabel('MD')
        nplot+=1
plt.tight_layout()

#%%
plt.figure(figsize=[6,8])
nplot=1
cut_cols=[]
rnd_depths=[]
for feat in feats:
    if feat != 'MD':
        rand_depth = random.randint(int(np.min(df_feats['MD'])),int(np.max(df_feats['MD'])))
        missing_thick = 150
        rnd_depths.append(rand_depth)

        temp_df = df_feats.loc[~((df_feats['MD'] > rand_depth) & (df_feats['MD'] < (rand_depth+missing_thick)))]
        cut_cols.append(temp_df[['MD',feat]])

        plt.subplot(1,4,nplot)
        plt.plot(temp_df[feat],temp_df['MD'])
        plt.xlabel(feat)
        plt.ylim(top=np.min(df_feats['MD'])-25,bottom=np.max(df_feats['MD'])+25)
        if nplot==1: plt.ylabel('MD')
        nplot+=1
plt.tight_layout()

#%%
cut_df = pd.DataFrame()
cut_df['MD'] = cut_cols[0]['MD']
for col in cut_cols:
    cut_df = pd.merge(cut_df,col,how='outer',on='MD',)

cut_df = cut_df.dropna()

# cut_df = pd.concat(cut_cols,axis=1).dropna()
# cut_df = cut_df.loc[:,~cut_df.columns.duplicated()]

#%%
feats_to_model = cut_df.columns

scaler = MinMaxScaler()
nplot=1
plt.figure(figsize=[6,8])
preds_feat=[]
for feat in feats_to_model:
    if feat != 'MD':
        train,test = train_test_split(cut_df,test_size=0.5,shuffle = False, stratify = None)

        test = test.dropna()
        # test = test[(np.abs(zscore(test)) < 2).all(axis=1)]
        md_test = test['MD']
        test = test.drop('MD',1)
        X_test = test.drop(feat,1)
        Y_test = test[feat]

        train = train.dropna()
        # train = train[(np.abs(zscore(train)) < 3).all(axis=1)]
        md_train = train['MD']
        train = train.drop('MD',1)
        X_train = train.drop(feat,1)
        Y_train = train[feat]

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.01,
            max_depth=15,
            n_jobs=-1,
            n_estimators=1000
        )
        model.fit(X_train,Y_train)

        preds = model.predict(X_test)
        preds_feat.append(preds)
        real = Y_test
        r=pearsonr(real,preds)
        Mae = mean_absolute_error(real,preds)
        plt.subplot(1,5,nplot)
        plt.plot(preds,md_test,label='pred')
        plt.plot(real,md_test,label='real')
        plt.xlabel(feat)
        plt.title(str(np.round(r[0],2)))
        if nplot==4: plt.legend()
        nplot+=1
plt.tight_layout()
# %%
plt.figure(figsize=[6,8])
nplot=1
cut_cols=[]
for feat in feats:
    if feat != 'MD':
        rand_depth = rnd_depths[nplot-1]
        missing_thick = 150

        temp_df = df_feats.loc[((df_feats['MD'] > rand_depth) & (df_feats['MD'] < (rand_depth+missing_thick)))]
        cut_cols.append(temp_df[['MD',feat]])

        plt.subplot(1,4,nplot)
        plt.plot(df_feats[feat],df_feats['MD'])
        plt.plot(temp_df[feat],temp_df['MD'])
        plt.xlabel(feat)
        plt.ylim(top=np.min(df_feats['MD'])-25,bottom=np.max(df_feats['MD'])+25)
        if nplot==1: plt.ylabel('MD')
        nplot+=1
plt.tight_layout()

# %%
