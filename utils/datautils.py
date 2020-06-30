import lasio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Get all las files in a folder and merge them into a single dataframe, also adds a columns with the well name.
def merge_las(lasfiles,labels=None,columns=None):


    dfs=[]
    for n in tqdm(range(len(lasfiles))):
        las = lasio.read(lasfiles[n])
        df = las.df()
        if labels != None:
            df = df.dropna(subset=labels)
        df = df.dropna(axis=1,how='all')
        dfs.append(df)
        df['wellName'] = lasfiles[n].split('\\')[-1].split('.')[0]

    merged = pd.concat(dfs,axis=0,join="outer")

    if columns != None:
        return merged[columns]
    else:
        return merged