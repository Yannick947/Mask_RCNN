import math
import os
import pandas as pd

from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np

from mrcnn import model as modellib, utils
from samples.sunrgbd.sun_config import SunConfig, InferenceConfig


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

class mAPEvaluator(object):

    def __init__(self, eval_dir, datasets): 
        self.eval_dir = eval_dir
        self.inference_config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", 
                                       config=self.inference_config,
                                       model_dir=os.path.dirname(self.eval_dir))
        self.results = pd.DataFrame(columns=['name', 'dataset', 'mAP',])
        self.datasets = datasets
        

    def evaluate_all(self, dataset_name: str = 'test'):
        print('Evaluating for IoU_min=0.5 on test dataset ...')
        for model_name in os.listdir(self.eval_dir):
            model_path = os.path.join(self.eval_dir, model_name)
            print("Loading weights from ", model_path)
            self.model.load_weights(model_path, by_name=True)
            self.evaluate_sun(dataset_name, model_name)

    def evaluate_sun(self, dataset_name, model_name):

        image_ids = self.datasets[dataset_name].image_ids
        APs = []

        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(self.datasets[dataset_name], self.inference_config,
                                    image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, self.inference_config), 0)
            # Run object detection
            results = self.model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, _, _, _ =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                r["rois"], r["class_ids"], r["scores"], r['masks'])

            # AP can be nan, check to not include those values
            if (type(AP) is float or type(AP) is np.float64) and not math.isnan(AP):
                APs.append(float(AP))

        mAP = sum(APs) / len(APs)
        print("Num aps analyzed: ", len(APs))
        print("mAP: ", mAP)

        # row = self.results.shape[0]
        self.results.append(pd.Series([model_name, dataset_name, mAP]), ignore_index=True)

    def save_results(self, save_dir=None):
        if save_dir is not None: 
            save_path = os.path.join(save_dir, 'results.csv')
            self.results.to_csv(save_path)
        else: 
            self.results.to_csv('results.csv', index=False)
