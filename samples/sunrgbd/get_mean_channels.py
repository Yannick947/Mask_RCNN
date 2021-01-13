import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath('./')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  
from mrcnn.config import Config
from mrcnn.model import DataGenerator 
from samples.sunrgbd.sun_config import SunConfig
from samples.sunrgbd.dataset import SunDataset2D, SunDataset3D
from samples.sunrgbd.sun_config import ROOT_DIR, ANNOTATION_FILENAME, IGNORE_IMAGES_PATH, CLASSES

LOCAL_PATH_DATASET = "C:/Users/Yannick/Downloads/SUNRGBD"

DEBUG_DATASET = "train" 
DEPTH_MODE = True

means = list()
def main():

    if DEPTH_MODE: 
        num_channels = 4
        dataset = SunDataset3D(SunConfig(depth_mode=DEPTH_MODE))
    else:
        num_channels = 3
        dataset = SunDataset2D(SunConfig())

    dataset.load_sun(LOCAL_PATH_DATASET, DEBUG_DATASET)
    dataset.prepare()

    for _ in range(num_channels):
        means.append(list())

    for idx in dataset._image_ids:
        try:
            image = dataset.load_image(idx)
        except: 
            continue
        for channel in range(num_channels):
            means[channel].append(np.mean(image[:,:,channel]))

    for channel in range(num_channels):
        print('Mean for channel ', channel, ' is ', np.mean(means[channel]))

if __name__ == '__main__':
    main()
