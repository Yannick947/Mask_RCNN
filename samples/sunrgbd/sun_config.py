import sys
import os

from mrcnn.config import Config

ANNOTATION_FILENAME = 'via_regions_sunrgbd.json'
CLASSES = ['bed', 'tool', 'desk', 'chair', 'table', 'wardrobe', 'sofa', 'bookcase']
IGNORE_IMAGES_PATH = './skip_image_paths.txt'

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

    def __init__(self, depth_mode=True):
        self.depth_mode = depth_mode
        if depth_mode: 
            self.IMAGE_CHANNEL_COUNT = 4
        super().__init__()
            
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

    #Augmentation Config
    AUGMENTATION_NUM = 2
    AUGMENTATION_STRENGTH = 3

class InferenceConfig(SunConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

