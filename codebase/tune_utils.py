#
# A Keras Callback subclass used by Ray Tune
#

from tensorflow import keras
from ray import tune

class TuneReporterCallback(keras.callbacks.Callback):
    """Tune Callback for Keras. The call to tune.report defines what metrics 
       Tune will be aware of, and which metrics Tune will track in
       TensorBoard logs.
    """

    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        tune.report(train_loss=logs.get("loss"),
                    val_loss=logs.get("val_loss"),
                    train_accuracy=logs.get("accuracy"),
                    val_accuracy=logs.get('val_accuracy'),
                    train_auc=logs.get('auc'),
                    val_auc=logs.get('val_auc'),
                    val_precision=logs.get('val_precision'),
                    val_recall=logs.get('val_recall'))

