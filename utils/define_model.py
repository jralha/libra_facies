import tensorflow as tf
from tensorflow.keras.models import Sequential

def cnn_1d_classifier(num_classes,num_feats,length,n_convs=2):
    
    from tensorflow.keras.layers import Input, Conv1D, Activation, MaxPooling1D, Flatten, Dropout, Dense
    from tensorflow.keras.losses import MeanAbsolutePercentageError

    model = Sequential()
    model.add(Input(shape=(length,num_feats)))
    n_conv = n_convs
    act = 'relu'
    n=length
    for i in range(n_conv):
        model.add(Conv1D(20,15,padding='same'))
        model.add(Activation(act))
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
    from sklearn.metrics import log_loss, accuracy_score, make_scorer

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
