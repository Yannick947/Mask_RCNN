import math
import os
import pandas as pd

import numpy as np

from mrcnn import model as modellib, utils
from samples.sunrgbd.sun_config import InferenceConfig, CLASSES
from samples.sunrgbd.dataset import SunDataset2D, SunDataset3D


class mAPEvaluator(object):

    def __init__(self, eval_dir, depth_mode, dataset_dir, model_names): 
        """ Evaluator for class mAP
        Arguments: 
            :param eval_dir: Directory with models which shall be evaluated
            :param datasets: All datasets which shall be evaluated
        """
        self.depth_mode = depth_mode
        self.eval_dir = eval_dir
        self.inference_config = InferenceConfig(depth_mode)
        self.model = modellib.MaskRCNN(mode="inference", 
                                       config=self.inference_config,
                                       model_dir=os.path.dirname(self.eval_dir))
        
        self.result_columns = ['sun_class', 'dataset_size', 'occurances_of_class', 'model_name', 'dataset', 'depth', 'mAP']
        self.model_names = model_names
        self.results = pd.Series(index=self.result_columns)

        self.datasets = dict()
        for dataset_name in ["val", "test"]:
            if depth_mode:
                self.datasets[dataset_name] = SunDataset3D(config=self.inference_config)
            else: 
                self.datasets[dataset_name] = SunDataset2D(config=self.inference_config)
            self.datasets[dataset_name].load_sun(dataset_dir, dataset_name)
            self.datasets[dataset_name].prepare()


    def evaluate(self, per_class_eval, dataset_name):
        print('Evaluating for IoU_min=0.5 on test dataset ...')

        for model_name in self.model_names:
            
            # Skip model if not the correct mode
            if self.depth_mode and not 'depth' in model_name:
                continue
            elif not self.depth_mode and 'depth' in model_name:
                continue

            model_path = os.path.join(self.eval_dir, model_name)
            print("Loading weights from ", model_path)

            self.model.load_weights(model_path, by_name=True)

            if not per_class_eval:
                APs = self.evaluate_sun(dataset_name, model_name, class_id=None)
                mAP = sum(APs) / len(APs)
                self.results = pd.Series(index=self.result_columns,
                                         data=['all_classes', len(self.datasets[dataset_name].image_ids), len(APs), model_name, dataset_name, 'False', mAP])

            else: 
                self.results = pd.DataFrame(columns=self.result_columns)
                for sun_class_id, sun_class in zip(self.datasets[dataset_name].class_ids, self.datasets[dataset_name].class_names): 
                    if sun_class == 'BG': 
                        continue
                    print('Evaluating class ', sun_class)

                    APs = self.evaluate_sun(dataset_name, model_name, sun_class_id)
                    mAP = sum(APs) / len(APs)
                    eval_data = pd.Series(data = [sun_class, len(self.datasets[dataset_name].image_ids), len(APs), model_name, dataset_name, 'False', mAP],
                                            index = self.result_columns)
                    self.results = self.results.append(eval_data, ignore_index=True)        


    def evaluate_sun(self, dataset_name, model_name, class_id):

        image_ids = self.datasets[dataset_name].image_ids
        APs = []

        for iid in image_ids:
            # Load image and ground truth data
            _, class_ids = self.datasets[dataset_name].load_mask(iid)
            if class_id not in class_ids: 
                continue
            image, _, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(self.datasets[dataset_name], self.inference_config,
                                    iid, use_mini_mask=False)
            
            molded_images = np.expand_dims(modellib.mold_image(image, self.inference_config), 0)
            # Run object detection
            results = self.model.detect(molded_images, verbose=0)
            r = results[0]
            # Compute AP
            AP, _, _, _ =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                r["rois"], r["class_ids"], r["scores"], r['masks'],
                                class_id)

            # AP can be nan, check to not include those values
            if (type(AP) is float or type(AP) is np.float64) and not math.isnan(AP):
                APs.append(float(AP))
            # if len(APs) == 2: 
            #     return APs

        if len(APs) > 0:
            print("Num aps analyzed: ", len(APs))
            print("mAP: ", sum(APs) / len(APs))

            return APs
        else: 
            print("Num aps analyzed: ", len(APs))
            return [0]


    def save_results(self, save_dir=None, save_name='results.csv'):
        
        if save_dir is not None: 
            save_path = os.path.join(save_dir, save_name)
        else: 
            save_path = os.path.abspath('./' + save_name)
        
        if os.path.isfile(save_path):
            prior_results = pd.read_csv(save_path, index_col=False)
            self.results = prior_results.append(self.results, ignore_index=True)

        self.results.to_csv(save_path)

