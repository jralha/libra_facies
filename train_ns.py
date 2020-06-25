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
import argparse

#Parsing args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='-1')
parser.add_argument('--continue_training', type=bool, default=False)
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
parser.add_argument('--logs_dir', type=str, default='./logs')
parser.add_argument('--format', type=str, default='las')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epoch_count', type=int, default=1500)
parser.add_argument('--init_epoch', type=int, default=0)
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--optimizer', type=str, default='adam')

parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--model', type=str, default='1dcnn')
parser.add_argument('--datafile', type=str, required=True)
parser.add_argument('--labels', type=str, required=True)
args = parser.parse_args()

#%% Setting path to dataset and dataset properties.
##########################################################
checks = args.checkpoints_dir+"\\"
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
STEPS_PER_EPOCH = np.ceil(sample_count/BATCH_SIZE)
epochs = args.epoch_count
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#%%Data generator and model
#########################################

seed = random.randint(1,999)
gens = make_gen.from_dataframe(data_file,BATCH_SIZE=BATCH_SIZE)
train_data_gen = gens[0]
val_data_gen = gens[1]

if CONTINUE == False:
    FIRST_EPOCH = 1
    if args.model == '1dcnn':
        model = define_model.cnn_1d_classifier(len(class_names),len(features),1)
    # elif args.model == 'vgg':
    #     model = define_model.vgg_model(len(class_names),shape)
    # else:
    #     model = define_model.cnn_shallow(len(class_names),shape)
    
elif CONTINUE == True:
    modelfile = args.model_file
    FIRST_EPOCH = int(modelfile.split('.')[0].split('-')[-1])
    model = tf.keras.models.load_model(checks+modelfile)
else:
    print('Either start or continue training')
    sys.exit()

model.compile(
    optimizer=args.optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
    )


#%%Callbacks
#########################################
val_acc = '{val_accuracy}'
filepath_best=checks+RUN_NAME+"-{epoch}-"+val_acc+".hdf5"

ckp_best=tf.keras.callbacks.ModelCheckpoint(filepath_best,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_weights_only=False,
    save_freq='epoch'
    )

log_dir="logs\\"
# board=tf.keras.callbacks.TensorBoard(log_dir=log_dir,
#     histogram_freq=1,
#     write_graph=True
#     )

logfile=RUN_NAME+'.csv'
csv_log=tf.keras.callbacks.CSVLogger(filename=log_dir+logfile)


earlystopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0,patience=10
)


callbacks_list = [ckp_best,csv_log]

#%%Train or resume training
#########################################

model.fit_generator(
    generator=train_data_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=epochs,
    callbacks=callbacks_list,
    validation_data=val_data_gen,
    validation_steps=STEPS_PER_EPOCH,
    initial_epoch=FIRST_EPOCH
    )

# %%
  