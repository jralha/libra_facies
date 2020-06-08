#%%
import numpy as np
import pandas as pd
import os
os.chdir('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Activation, Flatten, MaxPooling1D, BatchNormalization, Dropout, LSTM, GRU, Input
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, zscore
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.style.use('ggplot')
import ipykernel

#%%
def make_df(csvs, path):
    all_dfs = []
    for csv in csvs:#train_csvs:
        df = pd.read_csv(os.path.join(path,csv))
        if len(df.columns) == 1:
            df = pd.read_csv(os.path.join(path,csv),sep=';')
        df = df.iloc[1:,:]
        #df = df.iloc[50:-50,:]
        for col in df.columns:
            if (pd.api.types.is_numeric_dtype(df[col]) == False) and (col != 'MD'):
                try:
                    # df[col]=df[col].astype(float)  
                    df[col] = np.float32(df[col].values)   
                    df = df.mask(df < 0)                   
                except:
                    df[col]=df[col]
            if col == 'RHOZ':
                df['RHOB'] = df[col]
                del df[col]
            elif col == 'DTCM' or col == 'DTCO':
                df['DTc'] = df[col]
                del df[col]
            elif col == 'DTSM' or col == 'DTSO':
                df['DTs'] = df[col]
                del df[col]   
    
        df = df.replace(-9999, np.nan)
        df = df.replace('-9999', np.nan)
        df = df.dropna()
        all_dfs.append(df)
    return pd.concat(all_dfs, axis=0, sort=False)

path0 = 'Dados_pre_treino'
path = 'DadosInacio_31-01-2020'
csvs0 = [x for x in os.listdir(path0) if x.endswith('csv')]
csvs0.sort()
csvs0 = np.array(csvs0)
csvs = [x for x in os.listdir(path) if x.endswith('ELASTIC.csv')]
csvs.sort()
csvs = np.array(csvs)

df_withcols0 = make_df(csvs0,path0)
df_withcols0 = df_withcols0.reset_index(drop=True)
df_withcols = make_df(csvs,path)
df_withcols = df_withcols.reset_index(drop=True)

test_wells = [ '1-SPS-50', '3-SPS-74', '3-SPS-85', '3-SPS-100']

train_columns = ['PhiMac', 'PhiMic', 'DTc', 'DTs', 'RHOB']

#%%
scaler = MinMaxScaler((0,1))

for well in test_wells: 
    # well = test_wells[0]
    test = df_withcols.loc[df_withcols['wellName'] == well]
    test_md = test['MD'].values
    test = test[train_columns]    
    train = df_withcols.loc[df_withcols['wellName'] != well][train_columns]
    train0 = df_withcols0[train_columns]
    train_list = [ train , train0 ]
    train = pd.concat(train_list,axis=0, sort=False)

    test = test.rolling(100).median()
    test = test.dropna()
    test = test[(np.abs(zscore(test)) < 3).all(axis=1)]

    train = train.rolling(100).median()
    train = train.dropna()
    train = train[(np.abs(zscore(train)) < 3).all(axis=1)]

    s_size = 1

    X_train0 = scaler.fit_transform(train.iloc[:,2:])
    Y_train0 = (train.iloc[:,0:1].values)*100

    n_batch = int(np.floor(X_train0.shape[0] / s_size))
    X_train = np.zeros((n_batch,s_size,X_train0.shape[1]))
    Y_train = np.zeros((n_batch,s_size,1))

    X_test0 = scaler.fit_transform(test.iloc[:,2:])
    Y_test0 = (test.iloc[:,0:1].values)*100

    n_batch_test = int(np.floor(X_test0.shape[0] / s_size))
    X_test = np.zeros((n_batch_test,s_size,X_test0.shape[1]))
    Y_test = np.zeros((n_batch_test,s_size,1))

    for num in range(n_batch):
        X_train[num] = X_train0[(num*s_size):((num+1)*s_size),:]
        Y_train[num] = Y_train0[(num*s_size):((num+1)*s_size)]

    for num in range(n_batch_test):
        X_test[num] = X_test0[(num*s_size):((num+1)*s_size),:]
        Y_test[num] = Y_test0[(num*s_size):((num+1)*s_size)]


    model = Sequential()

    model.add(Input(shape=(X_train.shape[1],X_train.shape[2])))

    n_conv = 2
    act = 'relu'
    # for i in range(n_conv):
        # model.add(Conv1D(20,15,padding='same'))
        # model.add(Activation(act))
        # model.add(MaxPooling1D(strides=2))
        # model.add(Dropout(0.5))
    # model.add(Dense(20, activation='relu'))
    # model.add(LSTM(10,return_sequences=True,activation='sigmoid'))
    # model.add(LSTM(10,return_sequences=True,activation='sigmoid'))
    model.add(LSTM(10,return_sequences=False,activation='sigmoid'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())

    model.add(Dense(X_train.shape[1]))
    model.add(Activation('linear'))
    model.compile(
        # loss=tf.keras.losses.Huber(),
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        # loss='mae',
        # loss='mse',
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        metrics=['mae']
    )

    mfile = "pororegression_joao\\weights_report_"+well+".hdf5"
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=mfile, verbose=1, save_best_only=True)

    history = model.fit(X_train,Y_train,epochs=100,validation_data=(X_test,Y_test),batch_size=5,verbose=1,shuffle=False,callbacks=[checkpointer])

    model = tf.keras.models.load_model(mfile)

    preds0 = model.predict(X_test)
    preds = np.concatenate(preds0,axis=0)
    real = np.ravel(np.concatenate(Y_test,axis=0))
    r=pearsonr(real,preds)
    r2=r2_score(real,preds)
    Mae = mean_absolute_error(real,preds)

    plot_md = np.float32(test_md[0:len(preds)])

    fig = plt.figure(figsize=[8,8])

    gridspec.GridSpec(2,2)

    plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=3)
    plt.plot(preds,plot_md,label='cnn')
    plt.plot(real,plot_md,label='real')
    plt.ylim(top=np.min(plot_md)-25,bottom=np.max(plot_md)+25)
    plt.xlim(0,30)
    plt.ylabel('MD (m)')
    plt.xlabel('Macroporosity (%)')
    title0 = well
    plt.title(title0)
    plt.legend()
    mae_string = 'MAE = '+str(np.round(Mae,2))+'%'
    plt.text(np.min(real)+0,-8+np.min(plot_md), mae_string, fontsize=15)

    plt.subplot2grid((2,2), (0,1),colspan=1,rowspan=1)
    plt.scatter(real,preds,marker='.',s=5,label='cnn')
    r_string='R = '+str(np.round(r[0],2))
    plt.title('Cross Plot')
    plt.ylabel('Predicted Macroporosity (%)')
    plt.xlabel('Real Macroporosity (%)')
    plt.text(np.max(real)-5,np.min(preds), r_string, fontsize=15)

    plt.subplot2grid((2,2), (1,1),colspan=1,rowspan=1)
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.legend()
    plt.title(mae_string)

    plt.tight_layout(pad=2)
    # plt.show()
    plt.savefig(well+'.png')

# %%
