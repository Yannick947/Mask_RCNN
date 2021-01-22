from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np

from samples.sunrgbd.sun_config import SunConfig


class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, augm_num, augm_strength, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.augm_num = augm_num
        self.augm_strength = augm_strength

    def on_train_begin(self, logs):
        logs = logs or {}

        logs.update(
            {'Augmentation strength': np.float_(self.augm_strength)})
        logs.update({'Augmentation num': np.float_(self.augm_num)})
        super().on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Cast logs to np.float to fix error
        logs.update({'lr': np.float_(K.eval(self.model.optimizer.lr))})
        super().on_epoch_end(epoch, logs)
