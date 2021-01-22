import sys
import os

import numpy as np

from mrcnn.config import Config

ANNOTATION_FILENAME = 'via_regions_sunrgbd.json'

# removed 'tool', put 'wardrobe' and 'desk' to COMBINED_CLASSES
CLASSES = ['bed', 'chair', 'table', 'sofa', 'bookcase'] 

COMBINED_CLASSES = {'desk': 'table'}
 
IGNORE_IMAGES_PATH = os.path.abspath('./skip_image_paths.txt')

# Root directory of the project
ROOT_DIR = os.path.abspath('./')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

try:  
    print('Try to set gpu ressources ...')
    import nvgpu
    available_gpus = nvgpu.available_gpus()

    if type(available_gpus) is list and len(available_gpus) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = available_gpus[0]
        print('Using GPU ', available_gpus[0])

    else: 
        print('No free gpu found, try later..')
        exit()
except: 
    pass

class SunConfig(Config):
    """Configuration for training on the sun dataset.
    Derives from the base Config class and overrides some values.
    """
    # Mean of depth channel was calculated with tests/get_mean_channels
    # Other means of channels were calculated as well but are mostly similar to the provided ones
    # and are therefore not changed
    MEAN_DEPTH_VALUE = 61

    def __init__(self, depth_mode=False):
        self.depth_mode = depth_mode
        if depth_mode: 
            print('Depth mode enabled')
            self.IMAGE_CHANNEL_COUNT = 4
            self.MEAN_PIXEL = np.append(self.MEAN_PIXEL, self.MEAN_DEPTH_VALUE)
        super().__init__()
        
        print('Following classes are used: ', *CLASSES)
            
    # Give the configuration a recognizable name
    NAME = "sun"
    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Into 12GB GPU memory, can fit two images.
    IMAGES_PER_GPU = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CLASSES)  # Background + num_classes

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class InferenceConfig(SunConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    def __init__(self, depth_mode): 
        super().__init__(depth_mode=depth_mode)

