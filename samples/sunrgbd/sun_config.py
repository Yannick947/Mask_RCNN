from mrcnn.config import Config


CLASSES = ['bed', 'tool', 'desk', 'chair', 'table', 'wardrobe', 'sofa', 'bookcase']

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
    AUGMENTATION_NUM = 0
    AUGMENTATION_STRENGTH = 0

class InferenceConfig(SunConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1