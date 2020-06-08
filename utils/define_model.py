import tensorflow as tf

def cnn_shallow(num_classes,shape):

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer, Dense, Conv1D, MaxPooling2D, Flatten, Activation, Dropout

    model = Sequential()
    model.add(Conv2D(32, (7, 7), input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def cnn_1d_classifier(num_classes,num_feats,loss=None):
    model = Sequential()

    model.add(Input(shape=(1,num_feats)))

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

    model.add(Dense(num_classes))
    model.add(Activation('linear'))
    if loss == None:
        loss = tf.keras.losses.MeanAbsolutePercentageError()
    else:
        loss = loss
    model.compile(
    loss=loss
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae']
    )