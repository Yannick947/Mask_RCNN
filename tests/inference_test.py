import os
import sys
import random
import math
import re
import time

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage

from samples.sunrgbd import sun, dataset, sun_config
from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn.visualize import display_images
from mrcnn import visualize
from mrcnn import utils

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
HOLOLENS_MODE = True
DEPTH_MODE = False
HOLOLENS_IMAGE_PATHS = os.path.abspath("./images")

CLASS_NAMES = ['BG']
CLASS_NAMES.extend(sun_config.CLASSES)

config = sun_config.SunConfig(depth_mode=DEPTH_MODE)
config.display()

SUN_DIR = 'C:/Users/Yannick/Downloads/SUNRGBD/'
SUN_WEIGHTS_PATH = os.path.join(
    ROOT_DIR, 'logs/reduced_classes/best_models/strength3_num2_0007.h5')
IGNORE_IMAGES_PATH = os.path.abspath('../skip_image_paths.txt')

sun.ROOT_DIR = ROOT_DIR
sun_config.ROOT_DIR = ROOT_DIR
dataset.ROOT_DIR = ROOT_DIR


DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def get_ax(rows=1, cols=1, size=14):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def main():

    assert not (
        DEPTH_MODE and HOLOLENS_MODE), "No depth channel for Hololens available"

    # Set up model
    config = InferenceConfig(depth_mode=DEPTH_MODE)
    config.BATCH_SIZE = 1
    config.DETECTION_MIN_CONFIDENCE = 0.8
    config.display()

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    model.load_weights(SUN_WEIGHTS_PATH, by_name=True)

    if HOLOLENS_MODE:
        visualize_hololens(model)
    else:
        vsiualize_sun(model)


def visualize_hololens(model):
    for image_name in os.listdir(HOLOLENS_IMAGE_PATHS):
        if image_name[-4:] == '.jpg':
            rgb_path = os.path.join(HOLOLENS_IMAGE_PATHS, image_name)
            image = skimage.io.imread(rgb_path, plugin='pil')

            results = model.detect([image], verbose=1)
            r = results[0]
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        CLASS_NAMES, r['scores'],
                                        title="Predictions")
            # log("gt_class_id", gt_class_id)
            # log("gt_bbox", gt_bbox)
            # log("gt_mask", gt_mask)


def visualize_sun(model):

    if DEPTH_MODE:
        sun_dataset = dataset.SunDataset3D(skip_images_path=IGNORE_IMAGES_PATH)
    else:
        sun_dataset = dataset.SunDataset2D(skip_images_path=IGNORE_IMAGES_PATH)
    sun_dataset.load_sun(SUN_DIR, subset="test")

    # Must call before using the dataset
    sun_dataset.prepare()

    print("Images: {}\nClasses: {}".format(
        len(sun_dataset.image_ids), sun_dataset.class_names))

    test_sample_ids = [684, 1065, 854, 717, 44]
    test_sample_ids.extend(sun_dataset.image_ids)

    for image_id in test_sample_ids:
        image_id = random.choice(sun_dataset.image_ids)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            sun_dataset, config, image_id, use_mini_mask=False)
        info = sun_dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               sun_dataset.image_reference(image_id)))
        results = model.detect([image], verbose=1)

        ax = get_ax(1)
        r = results[0]
        print(r['scores'])
        if depth_mode:
            image = image[:, :, :3]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    sun_dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)


if __name__ == '__main__':
    main()
