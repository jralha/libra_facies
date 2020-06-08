#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random
plt.style.use('ggplot')
#%%%
well = '3-BRSA-923A-SPS_ELASTIC.csv'

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

feats = ['MD','DTCO','DTSM','PhiMac','RHOB']
df_feats = df[feats].dropna()
mds = df_feats['MD']
df_feats = df_feats.drop('MD',1)

plt.figure(figsize=[8,8])
nplot=1
cut_cols=[]
for feat in feats:
    if feat != 'MD':
        plt.subplot(1,5,nplot)
        plt.plot(df_feats[feat],mds)
        plt.xlabel(feat)
        plt.ylim(top=np.min(mds),bottom=np.max(mds))
        if nplot==1: plt.ylabel('MD')
        nplot+=1
plt.tight_layout()

#%%
scaler = StandardScaler()
data = scaler.fit_transform(df_feats)

max_k=11
wcss=[]
dbs=[]
sil=[]
k_range = range(2,max_k)
for k in k_range:
    nplot=1
    plt.figure(figsize=[8,8])
    kmeans = KMeans(n_clusters=k)
    temp_pred = kmeans.fit_predict(data)
    wcss_k = kmeans.inertia_
    wcss.append(wcss_k)
    dbs.append(davies_bouldin_score(data,temp_pred))
    sil.append(silhouette_score(data,temp_pred))

    for feat in feats:
        if feat != 'MD':
            plt.subplot(1,5,nplot)
            # plt.plot(df_feats[feat],mds)
            plt.scatter(df_feats[feat],mds,c=temp_pred,marker='.',s=5)
            plt.xlabel(feat)
            plt.ylim(top=np.min(mds),bottom=np.max(mds))
            if nplot==1: plt.ylabel('MD')
            nplot+=1
    plt.subplot(1,5,5)
    a = np.ones(len(mds)).reshape(len(mds),1)
    b = temp_pred.reshape(1,len(mds))
    m = np.dot(a,b)
    c_map = plt.get_cmap('viridis', k)
    plt.imshow(m.T[:,:200],cmap=c_map)
    plt.colorbar(ticks=range(k))
    plt.tight_layout()
    plt.show()



# %%
plt.figure(figsize=[8,8])
plt.subplot(3,1,1)
plt.plot(k_range,sil)
plt.ylabel("Silhueta")
plt.subplot(3,1,2)
plt.plot(k_range,dbs)
plt.ylabel("Davis-Bouldin")
plt.xlabel('Número Clusters')
db_std  = scaler.fit_transform(np.array(dbs).reshape(-1, 1))
sil_std = scaler.fit_transform(np.array(sil).reshape(-1, 1))
combined = db_std-sil_std
plt.subplot(3,1,3)
plt.plot(k_range,combined)
plt.ylabel("Silhueta - Davies-Bouldin (Escalado)")
plt.xlabel('Número Clusters')
plt.tight_layout()

# %%

# %%
