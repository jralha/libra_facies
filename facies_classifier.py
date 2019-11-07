#%% Data loading libraries

import lasio
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

#General preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SVMSMOTE

#Models
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

#Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

#%% Load Data
file = 'libra\\2-ANP-2A-RJS.las'

las = lasio.read(file).df().iloc[50:-50].dropna()

facies_raw = pd.read_excel('libra\\input_testemunho.xlsx')[['Interv. (m)','fac_simp']]
facies_raw['d0'], facies_raw['d1'] = facies_raw['Interv. (m)'].str.split('-',1).str
facies_raw['d0'] = facies_raw['d0'].str.replace(',','.').astype(float) + 0.01
facies_raw['d1'] = facies_raw['d1'].str.replace(',','.').astype(float)
facies_raw = facies_raw.iloc[:,1:]

md_list=np.array(list(las.index))
assign_facies=np.zeros(md_list.shape[0]).astype(str)
for x,md in enumerate(md_list):
    for index,row in facies_raw.iterrows():
        
        if md >= row['d0'] and md < row['d1']:
            assign_facies[x] = row['fac_simp']
        elif assign_facies[x] == '0.0':
            assign_facies[x] = 'none'

las['fac'] = assign_facies

classified = las.loc[las['fac'] != 'none']


# %% Preparing data for training.
labels = classified['fac']
features = classified.iloc[:,:-1] 

t_range=range(1,6)
t_result=[]

for i in tqdm(t_range):
    t_size = i/10


    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=t_size, random_state=123)

    #Balancing dataset by oversampling labels
    x_col = X_train.columns
    smote = SVMSMOTE(random_state=123,n_jobs=-1,sampling_strategy='all')
    X_train, y_train = smote.fit_resample(X_train,y_train)
    X_train = pd.DataFrame(X_train)
    X_train.columns = x_col

    # %% Logistic Regression Model
    acc=[]
    logit = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=100000)
    logit.fit(X_train,y_train)
    logit_pred = logit.predict(X_test)
    acc.append("LOGIT: "+str(accuracy_score(y_test,logit_pred)))

    # XGBoost
    xgb = XGBClassifier()
    xgb.fit(X_train,y_train)
    xgb_pred = xgb.predict(X_test)
    acc.append("XGB: "+str(accuracy_score(y_test,xgb_pred)))

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=100000)
    mlp.fit(X_train,y_train)
    mlp_pred = mlp.predict(X_test)
    acc.append("MLP: "+str(accuracy_score(y_test,mlp_pred)))

    t_result.append(["Training with "+str(t_size)+" split.",acc])

# %%
