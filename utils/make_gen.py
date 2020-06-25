import tensorflow as tf
from sklearn.model_selection import train_test_split


def from_dataframe(df,BATCH_SIZE=128,lenght=1,sampling_rate=1,stride=1,val_split=0.2,header_cols=1,seed=123):

    target_col = 0+header_cols
    start_feats = target_col+1
    data=df.iloc[:,start_feats:].values
    targets=df.iloc[:,target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X=data,
        y=targets,
        test_size=0.2,
        random_state=seed,
        stratify=targets 
    )

    ts_gen_train = tf.keras.preprocessing.sequence.TimeSeriesGenerator(
        data=X_train,
        targets=y_train,
        batch_size=BATCH_SIZE,
        lenght=lenght,
        sampling_rate=sampling_rate,
        stride=stride
    )

    ts_gen_val = tf.keras.preprocessing.sequence.TimeSeriesGenerator(
        data=X_test,
        targets=y_test,
        batch_size=BATCH_SIZE,
        lenght=lenght,
        sampling_rate=sampling_rate,
        stride=stride
    )

    return ts_gen_train, ts_gen_val

def from_aray(arr,targets,BATCH_SIZE=128,lenght=1,sampling_rate=1,stride=1,val_split=0.2,seed=123)

    data=arr
    targets=targets

    X_train, X_test, y_train, y_test = train_test_split(
        X=data,
        y=targets,
        test_size=0.2,
        random_state=seed,
        stratify=targets 
    )

    ts_gen_train = tf.keras.preprocessing.sequence.TimeSeriesGenerator(
        data=X_train,
        targets=y_train,
        batch_size=BATCH_SIZE,
        lenght=lenght,
        sampling_rate=sampling_rate,
        stride=stride
    )

    ts_gen_val = tf.keras.preprocessing.sequence.TimeSeriesGenerator(
        data=X_test,
        targets=y_test,
        batch_size=BATCH_SIZE,
        lenght=lenght,
        sampling_rate=sampling_rate,
        stride=stride
    )

    return ts_gen_train, ts_gen_val