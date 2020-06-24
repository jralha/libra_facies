#%%
from utils.datautils import merge_las
import glob
import lasio
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from utils import datautils
import random
#%% Load Data
LASFOLDER = "data\\north_sea\\"
lasfiles = glob.glob(LASFOLDER+"*.las")

# Merge all las files into a single dataframe.
# Function drops all samples where there isn't a facies label
# Function also drops all rows with all NaN values.
merged = datautils.merge_las(lasfiles,labels=['LITHOLOGY_GEOLINK'])

# %% Check lenght of each column to check data availability
col_count = merged.count()

# %%Select columns for data output
cols = ['wellName','LITHOLOGY_GEOLINK','DTC','GR','RHOB','NPHI','RDEP','RMED','SP']

data = merged[cols]
data = data.dropna()

wells = np.unique(data['wellName'])
n_wells = len(wells)
val_size = 0.2
n_val_wells = int(np.ceil(val_size*n_wells))

val_wells = random.choices(wells,k=n_val_wells)

val = data[data['wellName'].isin(val_wells)]
train = data[~(data['wellName'].isin(val_wells))]

# %%
