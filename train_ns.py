#%% Imports
import os
import sys
import tensorflow as tf
import glob
import random
import numpy as np
import datetime
import pandas as pd
from utils import make_gen
from utils import define_model
from utils import postutils
import argparse
import xgboost as xgb
from sklearn.metrics import log_loss, precision_score, recall_score

#%%Parsing args
#Parser when run on command line
if 'ipykernel' not in sys.argv[0]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='-1')
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--logs_dir', type=str, default='./logs')
    parser.add_argument('--format', type=str, default='las')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--epoch_count', type=int, default=1500)
    parser.add_argument('--init_epoch', type=int, default=0)
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--window_size', type=int, default=1)

    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--model', type=str, default='cnn1d')
    parser.add_argument('--datafile', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    args = parser.parse_args()
else:
    # Set default options when using Jupyter
    class Args():
        def __init__(self):
            self.gpu_ids = '-1'
            self.continue_training = False
            self.checkpoints_dir = './checkpoints'
            self.logs_dir = './logs'
            self.format = 'las'
            self.batch_size=10
            self.steps=100
            self.epoch_count=1500
            self.init_epoch=0
            self.model_file=None
            self.optimizer='adam'
            self.window_size=1
            self.run_name='test0'
            self.model='resnet1d'
            self.datafile='./data/north_sea/train.csv'
            self.labels='LITHOLOGY_GEOLINK'
    args = Args()

#%% Setting path to dataset and dataset properties.
##########################################################
checks = args.checkpoints_dir+"\\"
log_dir = args.logs_dir+"\\"
data_file_path = os.path.join(args.datafile)
labels = args.labels

data_file = pd.read_csv(data_file_path)
sample_count = len(data_file)
class_names = np.unique(data_file[labels])
features = data_file.columns[3:]

#%% Training parameters.
########################################
RUN_NAME = args.run_name
CONTINUE = args.continue_training
BATCH_SIZE = args.batch_size
if args.steps == None:
    STEPS_PER_EPOCH = np.ceil((sample_count/BATCH_SIZE)*0.01)
else:
    STEPS_PER_EPOCH = args.steps
epochs = args.epoch_count
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
window_size = args.window_size

#%%Data generator and model
#########################################

seed = random.randint(1,999)
gens = make_gen.from_dataframe(data_file,BATCH_SIZE=BATCH_SIZE,length=window_size)
train_data_gen = gens[0]
val_data_gen = gens[1]
data_file = None

if CONTINUE == False:
    FIRST_EPOCH = 1
    model = define_model.build_model(
        args.model,
        len(class_names),
        len(features),
        window_size)

    
elif CONTINUE == True:
    modelfile = args.model_file
    FIRST_EPOCH = int(modelfile.split('.')[0].split('-')[-1])
    model = tf.keras.models.load_model(checks+modelfile)
else:
    print('Either start or continue training')
    sys.exit()

#%% Compile Keras model, set callbacks and start/continue training
#########################################
if args.model != 'xgb':
    model.compile(
        optimizer=args.optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
        )

    #Callbacks
    #Save model if there is an increase in performance
    filepath_best=checks+RUN_NAME+"-{epoch}"+".hdf5"
    ckp_best=tf.keras.callbacks.ModelCheckpoint(filepath_best,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=False,
        save_freq='epoch'
        )

    #Log model history in csv
    logfile=RUN_NAME+'.csv'
    csv_log=tf.keras.callbacks.CSVLogger(filename=log_dir+logfile)

    #Early stopping, not using right now
    earlystopping=tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0,patience=10
    )

    # Metrics logging, still unstable, takes too long on small batch sizes
    # Currently after each epoch it calculates the metrics for each batch
    # So on small batch sizes this takes a lot of time
    # metrics = postutils.Metrics(val_data=val_data_gen,batch_size=BATCH_SIZE)

    callbacks_list = [ckp_best,csv_log]

    #Train or resume training
    model.fit(
        x=train_data_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=epochs,
        callbacks=callbacks_list,
        validation_data=val_data_gen,
        validation_steps=STEPS_PER_EPOCH,
        initial_epoch=FIRST_EPOCH
        )



# %% XGB model, still needs a way to save results
if args.model == 'xgb':
    print('Training XGB Model)')
    model.fit(train_data_gen.data,train_data_gen.targets,verbose=1)
    pred = model.predict(val_data_gen.data)
    pred_proba = model.predict_proba(val_data_gen.data)
    logloss = log_loss(val_data_gen.targets,pred_proba)
    prec = precision_score(val_data_gen.targets,pred,average='macro')
    rec = recall_score(val_data_gen.targets,pred,average='macro')

    print(logloss,prec,rec)

# %%
