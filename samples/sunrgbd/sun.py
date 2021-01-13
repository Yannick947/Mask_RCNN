"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 sun.py train --dataset=/path/to/sun/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 sun.py train --dataset=/path/to/sun/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 sun.py train --dataset=/path/to/sun/dataset --weights=imagenet

    # Apply color splash to an image
    python3 sun.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 sun.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug

from samples.sunrgbd.sun_config import ROOT_DIR, ANNOTATION_FILENAME, IGNORE_IMAGES_PATH
from samples.sunrgbd.dataset import SunDataset2D, SunDataset3D
from samples.sunrgbd.eval_sun import mAPEvaluator
from mrcnn import model as modellib, utils
from samples.sunrgbd.sun_config import SunConfig, InferenceConfig, CLASSES

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

CVHCI_DATASET_PATH = "/cvhci/data/depth/SUNRGBD/"
LOCAL_PATH_DATASET = "C:/Users/Yannick/Downloads/SUNRGBD"

# Path to trained weights file
MASK_RCNN_PATH = "/home/practicum_WS2021_2/instance_segmentation/Mask_RCNN/"
VANILLA_MODEL_NAME = "mask_rcnn_coco.h5"
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "snapshots", VANILLA_MODEL_NAME)

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect images.')
    parser.add_argument("command",
                        default='train',
                        metavar="<command>",
                        help="'train' or 'evaluate' on SUNRGBD")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/sun/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        default=COCO_MODEL_PATH,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--snapshot-mode', required=False,
                        default='None',
                        metavar="'last', 'coco' or 'imagenet'",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--cvhci-mode', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Whether training is done in cvhci server.',
                        type=bool)
    parser.add_argument('--epochs', required=False,
                        default=4,
                        metavar="Num epochs",
                        help='Number of epochs for fine tuning all layers',
                        type=int)
    parser.add_argument('--lr', required=False,
                        default=None,
                        metavar="Learning rate",
                        help='Learning rate, for fine tuning all layers it is devided by 5.',
                        type=float)
    parser.add_argument('--depth-mode', required=False,
                        default=False,
                        metavar="Depth mode",
                        help='Whether to train with depth images',
                        type=bool)

    args = parser.parse_args()
    print('Arguments:\n ', args)
    if args.cvhci_mode is True:
        print('cvhci mode is ', args.cvhci_mode)
        args.dataset = CVHCI_DATASET_PATH
    elif not args.dataset: 
        args.dataset = LOCAL_PATH_DATASET

    assert args.dataset, "Argument --dataset is required"
    datasets = dict()

    # Configurations
    print('Depth mode ', args.depth_mode)
    config = SunConfig(depth_mode=args.depth_mode)
    config.display()


    if args.lr:
        config.LEARNING_RATE = float(args.lr)

    if args.command == "train":

        for dataset_name in ["train", "val", "test"]:
            if args.depth_mode:
                datasets[dataset_name] = SunDataset3D(config=config)
            else: 
                datasets[dataset_name] = SunDataset2D(config=config)

            datasets[dataset_name].load_sun(args.dataset, dataset_name)
            datasets[dataset_name].prepare()
        print('Length train: ', datasets['train'].num_images,
            '\nLength test', datasets['test'].num_images, 
            '\nLength val: ', datasets['val'].num_images)

        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

        weights_path = args.weights

        # Load weights
        print("Loading weights ", weights_path)
        try:
            model.load_weights(weights_path, by_name=True)
        except ValueError: 
            print('Dimensions of input and output layers did not match, exclude these layers..')
            exclude_layers = [
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"]
            if args.depth_mode: 
                exclude_layers.extend(["conv1"])
            model.load_weights(weights_path, by_name=True, exclude=exclude_layers)

        if args.depth_mode: 
            augmentation = None
        
        else:
            #RandAugment applies random augmentation techniques chosen from a wide range of augmentation
            augmentation = imgaug.augmenters.Sometimes(0.5, [
                imgaug.augmenters.RandAugment(n=config.AUGMENTATION_NUM, m=config.AUGMENTATION_STRENGTH)
            ])
            print(f'\n\n---AUGMENTATION---: \nStrength: {config.AUGMENTATION_STRENGTH}\nNumber: {config.AUGMENTATION_NUM}')
        
    if args.command == "train" and args.weights == COCO_MODEL_PATH:
        print("\n\n --- Training network heads --- \n\n")

        # Training - Stage 1
        
        model.train(datasets["train"], datasets["val"],
                    learning_rate=config.LEARNING_RATE,
                    epochs=4,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("\n\n --- Fine tune Resnet stage 4 and up --- \n\n")
        model.train(datasets["train"], datasets["val"],
                    learning_rate=config.LEARNING_RATE,
                    epochs=3,
                    layers='4+',
                    augmentation=augmentation)
    
    if args.command == 'train' and args.epochs > 0:
        print('Starting with fine tuning the whole network.')
        # Training - Stage 3
        # Fine tune all layers
        epochs = model.epoch + args.epochs

        print("\n\n --- Fine tune all layers --- \n\n")
        model.train(datasets["train"], datasets["val"],
                    learning_rate=config.LEARNING_RATE / 5,
                    epochs=epochs,
                    layers='all',
                    augmentation=augmentation)
                    
    if args.command == 'evaluate':

        eval_model_dir = os.path.join(args.logs, 'reduced_classes/best_models')
        model_names = list()

        for weights_name in os.listdir(eval_model_dir):
            if weights_name[-3:] == '.h5':
                model_names.append(weights_name)
        
        for depth_mode in [True, False]:
            evaluator = mAPEvaluator(eval_dir=eval_model_dir, depth_mode=depth_mode, dataset_dir=args.dataset, model_names=model_names)
            evaluator.evaluate(per_class_eval=True, dataset_name='test')
            evaluator.save_results()
