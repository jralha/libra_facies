import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


def from_dataframe(df,BATCH_SIZE=128,length=1,sampling_rate=1,stride=1,val_split=0.2,header_cols=2,seed=123):

    scaler = MinMaxScaler()
    encoder = LabelEncoder()

    target_col = header_cols
    start_feats = target_col+1

    data=df.iloc[:,start_feats:].values
    targets=df.iloc[:,target_col].values.reshape(-1,1)

    data = scaler.fit_transform(data)
    targets = encoder.fit_transform(targets)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        targets,
        test_size=0.2,
        random_state=seed,
        stratify=targets 
    )

    ts_gen_train = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data=X_train,
        targets=y_train,
        batch_size=BATCH_SIZE,
        length=length,
        sampling_rate=sampling_rate,
        stride=stride
    )

    ts_gen_val = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data=X_test,
        targets=y_test,
        batch_size=BATCH_SIZE,
        length=length,
        sampling_rate=sampling_rate,
        stride=stride
    )

    return ts_gen_train, ts_gen_val

def from_aray(arr,targets,BATCH_SIZE=128,length=1,sampling_rate=1,stride=1,val_split=0.2,seed=123):

    scaler = MinMaxScaler()
    encoder = LabelEncoder()
    
    data=arr
    targets=targets.reshape(-1,1)

    data = scaler.fit_transform(data)
    targets = encoder.fit_transform(targets)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        targets,
        test_size=0.2,
        random_state=seed,
        stratify=targets 
    )

    ts_gen_train = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data=X_train,
        targets=y_train,
        batch_size=BATCH_SIZE,
        length=length,
        sampling_rate=sampling_rate,
        stride=stride
    )

    ts_gen_val = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        data=X_test,
        targets=y_test,
        batch_size=BATCH_SIZE,
        length=length,
        sampling_rate=sampling_rate,
        stride=stride
    )

    return ts_gen_train, ts_gen_val