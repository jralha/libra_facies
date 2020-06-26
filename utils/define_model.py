import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Activation, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dropout, Dense, LSTM
from tensorflow.keras.layers import Conv1D, BatchNormalization
from tensorflow.keras.losses import MeanAbsolutePercentageError

def cnn_1d_classifier(num_classes,num_feats,length,n_convs=2):
    
    model = Sequential()
    model.add(Input(shape=(length,num_feats)))
    n_conv = n_convs
    act = 'relu'
    n=length
    for i in range(n_conv):
        model.add(Conv1D(64,1,padding='same'))
        model.add(Activation(act))
        model.add(BatchNormalization())
        n=n/2
        if n >= 2:
            model.add(MaxPooling1D(strides=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    return model

def xgb_cv_model(params=None,cv=2,verbose=0):

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

def lstm_model(num_classes,num_feats,length):

    model = Sequential()
    model.add(Input(shape=(length,num_feats)))
    model.add(LSTM(5))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    return model

def resnet_1d(num_feats,length, n_feature_maps, nb_classes):
    input_shape = (length,num_feats)
    print ('build conv_x')
    x = tf.keras.layers.Input(shape=input_shape)
    conv_x = tf.keras.layers.BatchNormalization()(x)
    conv_x = tf.keras.layers.Conv1D(n_feature_maps, 8, padding='same')(conv_x)
    conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = tf.keras.layers.Conv1D(n_feature_maps, 5, padding='same')(conv_x)
    conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = tf.keras.layers.Conv1D(n_feature_maps, 3, padding='same')(conv_y)
    conv_z = tf.keras.layers.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = tf.keras.layers.Conv1D(n_feature_maps, 1,padding='same')(x)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = tf.keras.layers.BatchNormalization()(x)
    print ('Merging skip connection')
    y = tf.keras.layers.Add()([shortcut_y, conv_z])
    y = tf.keras.layers.Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = tf.keras.layers.Conv1D(n_feature_maps*2, 8, padding='same')(x1)
    conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = tf.keras.layers.Conv1D(n_feature_maps*2, 5, padding='same')(conv_x)
    conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = tf.keras.layers.Conv1D(n_feature_maps*2, 3, padding='same')(conv_y)
    conv_z = tf.keras.layers.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = tf.keras.layers.Conv1D(n_feature_maps*2, 1, padding='same')(x1)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = tf.keras.layers.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = tf.keras.layers.Add()([shortcut_y, conv_z])
    y = tf.keras.layers.Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = tf.keras.layers.Conv1D(n_feature_maps*2, 8, padding='same')(x1)
    conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = tf.keras.layers.Conv1D(n_feature_maps*2, 5, padding='same')(conv_x)
    conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = tf.keras.layers.Conv1D(n_feature_maps*2, 3, padding='same')(conv_y)
    conv_z = tf.keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = tf.keras.layers.Conv1D(n_feature_maps*2, 1, padding='same')(x1)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = tf.keras.layers.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = tf.keras.layers.Add()([shortcut_y, conv_z])
    y = tf.keras.layers.Activation('relu')(y)
     
    full = tf.keras.layers.GlobalAveragePooling1D()(y)
    out = tf.keras.layers.Dense(nb_classes, activation='softmax')(full)
    print ('        -- model was built.')
    model = tf.keras.models.Model(inputs=x, outputs=y)

    return model