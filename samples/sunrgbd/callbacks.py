from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np

from samples.sunrgbd.sun_config import SunConfig

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        #Cast logs to np.float to fix error
        logs.update({'lr': np.float_(K.eval(self.model.optimizer.lr))})
        logs.update({'Augmentation strength': np.float_(SunConfig.AUGMENTATION_STRENGTH)})
        logs.update({'Augmentation num': np.float_(SunConfig.AUGMENTATION_NUM)})        

        super().on_epoch_end(epoch, logs)
