import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Activate free gpu
# import nvgpu

# available_gpus = nvgpu.available_gpus()

# if type(available_gpus) is list and len(available_gpus) > 0:
#     os.environ["CUDA_VISIBLE_DEVICES"] = available_gpus[0]
#     print('Using GPU ', available_gpus[0])

# else: 
#     print('No free gpu found, try later..')
#     exit()

# Root directory of the project
ROOT_DIR = os.path.abspath('./')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  
from mrcnn.config import Config
from mrcnn.model import DataGenerator 
from samples.sunrgbd.sun import SunConfig, SunDataset

CLASSES = ['bed', 'tool', 'desk', 'chair', 'table', 'wardrobe', 'sofa', 'bookcase']
ANNOTATION_FILENAME = 'via_regions_sunrgbd.json'
CVHCI_DATASET_PATH = "/cvhci/data/depth/SUNRGBD/"
LOCAL_PATH_DATASET = "C:/Users/Yannick/Downloads/SUNRGBD"

IGNORE_IMAGES_PATH = os.path.join(LOCAL_PATH_DATASET, 'skip_image_paths.txt')
DEBUG_DATASET = "train" # Can be 'train', 'val' or 'test

def main():
    dataset = SunDataset(SunConfig())
    dataset.load_sun(LOCAL_PATH_DATASET, DEBUG_DATASET)
    dataset.prepare()
    gen = DataGenerator(dataset=dataset,
                        config=SunConfig(),
                        shuffle=True,
                        augmentation=False,
                        batch_size=2)

    for i in range(len(gen)):
        gen[i]

if __name__ == '__main__':
    main()
