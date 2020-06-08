import lasio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def merge_las(lasfiles,outpath):

    dfs=[]
    for n in tqdm(range(len(lasfiles))):
        las = lasio.read(lasfiles[n])
        df = las.df()
        df['wellName'] = lasfiles[n].split('\\')[-1].split('.')[0]
        # df = df.loc[df['Lithology_geolink'] >= 0]
        # df = df.replace()

        dfs.append(df)

    merged = pd.concat(dfs,axis=0,join="outer")

    return merged
