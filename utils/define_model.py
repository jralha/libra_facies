#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Activation, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dropout, Dense, LSTM
from tensorflow.keras.layers import Conv1D, BatchNormalization
from tensorflow.keras.losses import MeanAbsolutePercentageError

def build_model(name,num_classes,num_feats,length,**kwargs):
    o = models(num_classes,num_feats,length,**kwargs)
    model = get_model(o,name)
    return model

def get_model(o,name):
    return getattr(o, name)()

class models():

    def __init__(self,num_classes,num_feats,length,**kwargs):
        self.num_feats = num_feats
        self.length = length
        self.num_classes = num_classes
        for key,value in kwargs.items():
            setattr(self, key, value)
        

    def cnn1d(self):
        try:
            n_conv = self.n_conv
        except:
            n_conv = 2
        length = self.length
        num_feats = self.num_feats
        num_classes = self.num_classes
        model = Sequential()
        model.add(Input(shape=(length,num_feats)))
        n=length
        for i in range(n_conv):
            model.add(Conv1D(64,8-(i*2),padding='same',activation='relu'))
            n=n/2
            if n >= 2:
                model.add(MaxPooling1D(strides=2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100,activation='relu'))
        model.add(Dense(num_classes,activation='softmax'))

        return model

    def lstm(self):
        length = self.length
        num_feats = self.num_feats
        num_classes = self.num_classes
        model = Sequential()
        model.add(Input(shape=(length,num_feats)))
        model.add(LSTM(100))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes,activation='softmax'))

        return model

    def mlp(self):
        length = self.length
        num_feats = self.num_feats
        num_classes = self.num_classes
        model = Sequential()
        model.add(Input(shape=(length,num_feats)))
        model.add(Flatten())
        model.add(Dense(100,activation='relu'))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(num_classes,activation='softmax'))

        return model


    def resnet1d(self):
        try:
            n_feature_maps = self.n_feature_maps
        except:
            n_feature_maps = 64
        nb_classes = self.num_classes
        length = self.length
        num_feats = self.num_feats
        input_shape = (length,num_feats)

        x = tf.keras.layers.Input(shape=input_shape)
        conv_x = tf.keras.layers.BatchNormalization()(x)
        conv_x = tf.keras.layers.Conv1D(n_feature_maps, 8, padding='same')(conv_x)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        conv_y = tf.keras.layers.Conv1D(n_feature_maps, 5, padding='same')(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)
        conv_z = tf.keras.layers.Conv1D(n_feature_maps, 3, padding='same')(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        is_expand_channels = not (input_shape[-1] == n_feature_maps)
        if is_expand_channels:
            shortcut_y = tf.keras.layers.Conv1D(n_feature_maps, 1,padding='same')(x)
            shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = tf.keras.layers.BatchNormalization()(x)

        y = tf.keras.layers.Add()([shortcut_y, conv_z])
        y = tf.keras.layers.Activation('relu')(y)

        x1 = y
        conv_x = tf.keras.layers.Conv1D(n_feature_maps*2, 8, padding='same')(x1)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        conv_y = tf.keras.layers.Conv1D(n_feature_maps*2, 5, padding='same')(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)
        conv_z = tf.keras.layers.Conv1D(n_feature_maps*2, 3, padding='same')(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)
        
        is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
        if is_expand_channels:
            shortcut_y = tf.keras.layers.Conv1D(n_feature_maps*2, 1, padding='same')(x1)
            shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = tf.keras.layers.BatchNormalization()(x1)

        y = tf.keras.layers.Add()([shortcut_y, conv_z])
        y = tf.keras.layers.Activation('relu')(y)
        
        x1 = y
        conv_x = tf.keras.layers.Conv1D(n_feature_maps*2, 8, padding='same')(x1)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)
        
        conv_y = tf.keras.layers.Conv1D(n_feature_maps*2, 5, padding='same')(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)
        
        conv_z = tf.keras.layers.Conv1D(n_feature_maps*2, 3, padding='same')(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
        if is_expand_channels:
            shortcut_y = tf.keras.layers.Conv1D(n_feature_maps*2, 1, padding='same')(x1)
            shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        else:
            shortcut_y = tf.keras.layers.BatchNormalization()(x1)

        y = tf.keras.layers.Add()([shortcut_y, conv_z])
        y = tf.keras.layers.Activation('relu')(y)
        
        full = tf.keras.layers.GlobalAveragePooling1D()(y)
        out = tf.keras.layers.Dense(nb_classes, activation='softmax')(full)
        model = tf.keras.models.Model(inputs=x, outputs=y)

        return model

    def xgb(params=None,cv=2,verbose=0):
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV

        estimator = xgb.XGBClassifier(objective='multi:softmax')
        if params == None:
            param_dict = {}
        else:
            param_dict = params

        model = GridSearchCV(
            estimator,
            param_dict,
            n_jobs=-1,
            cv=cv,
            verbose=verbose,
            refit='logloss'
            )
        
        return model

# %%
