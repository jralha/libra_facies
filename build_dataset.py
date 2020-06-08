#%%
# from utils.datautils import merge_las, get_curves
import glob
import lasio
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
#%%
LASFOLDER = "data\\north_sea\\"
lasfiles = glob.glob(LASFOLDER+"*.las")

def merge_las(lasfiles,labels,columns=None):

    dfs=[]
    for n in tqdm(range(len(lasfiles))[50:55]):
        las = lasio.read(lasfiles[n])
        df = las.df()
        df = df.dropna(subset=labels)
        df = df.dropna(axis=1,how='all')
        # df = df.apply(lambda x: x.fillna(x.median()),axis=0)
        dfs.append(df)
        df['wellName'] = lasfiles[n].split('\\')[-1].split('.')[0]

    merged = pd.concat(dfs,axis=0,join="outer")

    for column in merged.columns:
        l0 = len(merged[column])
        l1 = np.sum(merged[column].isnull().values)
        if l1/l0 > 0.1:
            merged.drop([column],axis=1)

    return merged

test = merge_las(lasfiles,['LITHOLOGY_GEOLINK'])

# %%
