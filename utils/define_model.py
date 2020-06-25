import tensorflow as tf
from tensorflow.keras.models import Sequential

def cnn_1d_classifier(num_classes,num_feats,lenght,n_convs=2):
    
    from tensorflow.keras.layers import Input, Conv1D, Activation, MaxPooling1D, Flatten, Dropout, Dense
    from tensorflow.keras.losses import MeanAbsolutePercentageError

    model = Sequential()
    model.add(Input(shape=(lenght,num_feats)))
    n_conv = n_convs
    act = 'relu'
    for i in range(n_conv):
        model.add(Conv1D(20,15,padding='same'))
        model.add(Activation(act))
        # model.add(MaxPooling1D(strides=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model