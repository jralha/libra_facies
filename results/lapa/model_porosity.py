# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import lasio
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import argparse

# %%
#### Getting parameters from text file
parser = argparse.ArgumentParser()
parser.add_argument('--parfile',type=str,default='parameters.txt')
args = parser.parse_args()

f = open(args.parfile)
pars = f.read().split('\n')
pars = [ par.split('=')[1].strip() for par in pars ]

# %% Reading LAS file and setting curves to numpy arrays
las = lasio.read(pars[0])
depths = las[pars[1]]
dtc = las[pars[2]]
dts = las[pars[3]]
rhob = las[pars[4]]
merge = np.column_stack((dtc,dts,rhob))
df = pd.DataFrame(merge)
df.columns = ['dtc','dts','rhob']
df = df.dropna()
scaler = MinMaxScaler((0,1))
for col in df.columns:
    df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))

# %% Build input tensor
ten = np.expand_dims(df.values,axis=1)



# %% Load Models
m_macro = tf.keras.models.load_model('model_macro.hdf5')
m_micro = tf.keras.models.load_model('model_micro.hdf5')

# %% Inference
phimac = np.ravel(m_macro.predict(ten))
phimic = np.ravel(m_micro.predict(ten))
phit = np.ravel(phimac+phimic)

# %% Pass values to new LAS
out = lasio.LASFile()
out.well.DATE = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
out.well.WELL = las.well.WELL
out.well.STRT = las.well.STRT
out.well.STOP = las.well.STOP
out.well.STEP = las.well.STEP


out.add_curve('DTC',dtc,unit='us/ft')
out.add_curve('DTS',dts,unit='us/ft')
out.add_curve('RHOB',rhob,unit='g/cm3')
out.add_curve('PhiMacML',phimac,unit='v/v',descr='CNN-LSTM macroporosity')
out.add_curve('PhiMicML',phimic,unit='v/v',descr='CNN-LSTM microporosity')
out.add_curve('PHITML',phit,unit='v/v',descr='CNN-LSTM total porosity')

#%% Output new las
fname = pars[0].split('\\')[-1].split('.')[0]+"_ML_phi.las"
out.write(fname,version=2.0)

# %%
