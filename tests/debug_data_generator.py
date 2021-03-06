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

DEBUG_DATASET = "test" # Can be 'train', 'val' or 'test
DEPTH_MODE = True
def main():
    if DEPTH_MODE: 
        dataset = SunDataset3D(SunConfig(depth_mode=DEPTH_MODE))
    else:
        dataset = SunDataset2D(SunConfig())

    dataset.load_sun(LOCAL_PATH_DATASET, DEBUG_DATASET)
    dataset.prepare()
    gen = DataGenerator(dataset=dataset,
                        config=SunConfig(depth_mode=DEPTH_MODE),
                        shuffle=True,
                        augmentation=False,
                        batch_size=2)

    for i in range(len(gen)):
        inputs, outputs = gen[i]

if __name__ == '__main__':
    main()
