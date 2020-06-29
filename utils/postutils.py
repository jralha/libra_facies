import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):

    def __init__(self, val_data, batch_size):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs):
        for batch_index in range(0, len(self.validation_data)):
            temp_targ = self.validation_data[batch_index][1]
            temp_predict = (np.asarray(self.model.predict(
                                self.validation_data[batch_index][0]))).round()
            if(batch_index == 0):
                val_targ = temp_targ
                val_predict = temp_predict
            else:
                val_targ = np.vstack((val_targ, temp_targ))
                val_predict = np.vstack((val_predict, temp_predict))

        val_f1 = np.round(f1_score(val_targ, val_predict,average='macro'), 4)
        val_recall = np.round(recall_score(val_targ, val_predict,average='macro'), 4)
        val_precis = np.round(precision_score(val_targ, val_predict,average='macro'), 4)

        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precis)

        logs["val_f1"] = val_f1
        logs["val_recall"] = val_recall
        logs["val_precis"] = val_precis

        print("— val_f1: {} — val_precis: {} — val_recall {}".format(
                 val_f1, val_precis, val_recall))
        return
