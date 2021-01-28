import math
import os

import numpy as np
import pandas as pd
from mrcnn import model as modellib
from mrcnn import utils
from samples.sunrgbd.dataset import SunDataset2D, SunDataset3D
from samples.sunrgbd.sun_config import InferenceConfig


class mAPEvaluator(object):

    """Wrapper for a mask rcnn model to evaluate mAPs and store them."""

    def __init__(self,
                 eval_dir: str,
                 depth_mode: bool,
                 dataset_dir: str,
                 model_names: list):
        """Create mAP evaluator that adds evaluation functionality to mrcnn model.

        Args:
            eval_dir (str): Directory where models are stored.
            depth_mode (bool): Flag whether depth models or rgb models shall be evaluated.
            dataset_dir (str): Name of dataset. ('train', 'val' or 'test')
            model_names (list): Model names to be evaluated.
        """
        self.depth_mode = depth_mode
        self.eval_dir = eval_dir
        self.inference_config = InferenceConfig(depth_mode)
        self.model = modellib.MaskRCNN(mode="inference",
                                       config=self.inference_config,
                                       model_dir=os.path.dirname(self.eval_dir))

        self.result_columns = ['sun_class', 'dataset_size',
                               'occurances_of_class', 'model_name', 'dataset', 'depth', 'mAP', 'mean_iou']
        self.model_names = model_names
        self.results = pd.DataFrame(columns=self.result_columns)

        self.datasets = dict()
        for dataset_name in ["val", "test"]:
            if depth_mode:
                self.datasets[dataset_name] = SunDataset3D(
                    config=self.inference_config)
            else:
                self.datasets[dataset_name] = SunDataset2D(
                    config=self.inference_config)
            self.datasets[dataset_name].load_sun(dataset_dir, dataset_name)
            self.datasets[dataset_name].prepare()

    def evaluate(self, per_class_eval: bool, dataset_name: str):
        """ Evaluate maskrcnn model.

        Args:
            per_class_eval (bool): Flag indicating whether evaluation shall
                                   be done seperated by classes or for all classes
            dataset_name (str): Name of the dataset (can be 'train', 'val', 'test')
        """
        print('Evaluating for IoU_min=0.5 on test dataset ...')

        for model_name in self.model_names:

            # Skip model if not the correct mode
            if self.depth_mode and ('depth' not in model_name):
                continue
            elif not self.depth_mode and 'depth' in model_name:
                continue

            model_path = os.path.join(self.eval_dir, model_name)
            print("Loading weights from ", model_path)

            self.model.load_weights(model_path, by_name=True)

            if not per_class_eval:
                APs, ious = self.evaluate_sun(
                    dataset_name, model_name, class_id=None)
                mAP = sum(APs) / len(APs)
                mean_iou = sum(ious) / len(ious)

                eval_data = pd.Series(index=self.result_columns,
                                      data=['all_classes', len(self.datasets[dataset_name].image_ids), len(APs), model_name, dataset_name, 'False', mAP, mean_iou])
                self.results = self.results.append(
                    eval_data, ignore_index=True)

            else:
                for sun_class_id, sun_class in zip(self.datasets[dataset_name].class_ids, self.datasets[dataset_name].class_names):
                    if sun_class == 'BG':
                        continue
                    print('Evaluating class ', sun_class)

                    APs, ious = self.evaluate_sun(
                        dataset_name, model_name, sun_class_id)
                    mAP = sum(APs) / len(APs)
                    mean_iou = sum(ious) / len(ious)

                    eval_data = pd.Series(data=[sun_class, len(self.datasets[dataset_name].image_ids), len(APs), model_name, dataset_name, 'False', mAP, mean_iou],
                                          index=self.result_columns)
                    self.results = self.results.append(
                        eval_data, ignore_index=True)

    def evaluate_sun(self, dataset_name, model_name, class_id=None):
        """Evaluate model on sun data for specific class."""
        image_ids = self.datasets[dataset_name].image_ids
        APs = []
        ious = []

        for iid in image_ids:
            # Load image and ground truth data
            _, class_ids = self.datasets[dataset_name].load_mask(iid)

            image, _, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(self.datasets[dataset_name],
                                       self.inference_config,
                                       iid, use_mini_mask=False)

            if (class_id and class_id not in class_ids) or len(gt_class_id) == 0:
                continue

            molded_images = np.expand_dims(
                modellib.mold_image(image, self.inference_config), 0)
            # Run object detection
            results = self.model.detect(molded_images, verbose=0)
            r = results[0]
            # Compute AP
            AP, _, _, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'],
                                 class_id=class_id)

            # AP can be nan, check to not include those values
            if (type(AP) is float or type(AP) is np.float64) and not math.isnan(AP):
                APs.append(float(AP))

                if type(overlaps) is np.ndarray and overlaps.shape[0] > 0:
                    iou_per_row = float(0)
                    for row_i in range(overlaps.shape[0]):
                        iou_per_row += max(overlaps[row_i])

                    ious.append(float(iou_per_row / len(overlaps)))

                else:
                    ious.append(float(0))

            # if len(APs) > 2:
            #     return APs, ious

        if len(APs) > 0 and len(ious) > 0:
            print("Num aps analyzed: ", len(APs))
            print("mAP: ", sum(APs) / len(APs))

            return APs, ious
        else:
            print("Num aps analyzed: ", len(APs))
            return [0], [0]

    def save_results(self, save_dir=None, save_name='results.csv'):
        """ Write results to csv file.

        Args:
            save_dir (str, optional): Directory where results shall be saved. Defaults to None.
            save_name (str, optional): Name of csv file. Defaults to 'results.csv'.
        """
        if save_dir is not None:
            save_path = os.path.join(save_dir, save_name)
        else:
            save_path = os.path.abspath('./' + save_name)

        if os.path.isfile(save_path):
            prior_results = pd.read_csv(save_path, index_col=False)
            self.results = prior_results.append(
                self.results, ignore_index=True)

        self.results.to_csv(save_path, index=False)

        # Reset results, so every change that is written to disk does not appear twice
        self.results = pd.DataFrame(columns=self.result_columns)
