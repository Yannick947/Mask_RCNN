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

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.sunrgbd import sun, dataset, sun_config

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

depth_mode = True

config = sun_config.SunConfig(depth_mode=depth_mode)
config.display()

SUN_DIR = 'C:/Users/Yannick/Downloads/SUNRGBD/'
SUN_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'logs/local_best_models/depth_best.h5')  
IGNORE_IMAGES_PATH = os.path.abspath('../skip_image_paths.txt')

sun.ROOT_DIR = ROOT_DIR
sun_config.ROOT_DIR = ROOT_DIR
dataset.ROOT_DIR = ROOT_DIR
print(sun_config.ROOT_DIR)

# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

config = InferenceConfig(depth_mode=depth_mode)
config.BATCH_SIZE = 1
config.DETECTION_MIN_CONFIDENCE = 0.8

config.display()

DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=14):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

if depth_mode: 
    sun_dataset = dataset.SunDataset3D(skip_images_path=IGNORE_IMAGES_PATH)
else: 
    sun_dataset = dataset.SunDataset2D(skip_images_path=IGNORE_IMAGES_PATH)
sun_dataset.load_sun(SUN_DIR, subset="test")

# Must call before using the dataset
sun_dataset.prepare()

print("Images: {}\nClasses: {}".format(len(sun_dataset.image_ids), sun_dataset.class_names))

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
model.load_weights(SUN_WEIGHTS_PATH, by_name=True)

test_sample_ids = [684, 1065, 854, 717, 44].extend(sun_dataset.image_ids)

for image_id in test_sample_ids:
    image_id = random.choice(sun_dataset.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(sun_dataset, config, image_id, use_mini_mask=False)
    info = sun_dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                        sun_dataset.image_reference(image_id)))
    results = model.detect([image], verbose=1)

    ax = get_ax(1)
    r = results[0]
    print(r['scores'])
    if depth_mode:
        image = image[:,:,:3]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                sun_dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
