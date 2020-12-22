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
# Activate free gpu

# Root directory of the project
ROOT_DIR = os.path.abspath('./')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib, utils
from samples.sunrgbd.sun_config import SunConfig, InferenceConfig, CLASSES
from samples.sunrgbd.eval_sun import evaluate_sun

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

ANNOTATION_FILENAME = 'via_regions_sunrgbd.json'
CVHCI_DATASET_PATH = "/cvhci/data/depth/SUNRGBD/"
LOCAL_PATH_DATASET = "C:/Users/Yannick/Downloads/SUNRGBD"

# Path to trained weights file
MASK_RCNN_PATH = "/home/practicum_WS2021_2/instance_segmentation/Mask_RCNN/"
VANILLA_MODEL_NAME = "mask_rcnn_coco.h5"
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "snapshots", VANILLA_MODEL_NAME)

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
IGNORE_IMAGES_PATH = './skip_image_paths.txt'

############################################################
#  Dataset
############################################################

class SunDataset(utils.Dataset):
    def __init__(self, config=None):
        self.config = config
        self.class_to_id_map = dict()
        for class_id_i, class_name in enumerate(CLASSES): 
            self.class_to_id_map[class_name] = class_id_i + 1

        self.ignore_paths = self.__get_ignore_pahts()
        super().__init__()

    def __get_ignore_pahts(self):
        with open(IGNORE_IMAGES_PATH, 'r') as f:
            content = f.readlines()
        paths = [x.strip() for x in content] 
        return paths

    def load_sun(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        split_info = json.load(open(os.path.join(ROOT_DIR, 
                                                 'samples',
                                                 'sunrgbd',
                                                 'train_test_split.json')))
        # Add classes. 
        for class_name in CLASSES:
            self.add_class('sun', self.class_to_id_map[class_name], class_name)

        annotations = json.load(open(os.path.join(ROOT_DIR, 'samples', 'sunrgbd', ANNOTATION_FILENAME)))
        annotations = annotations['labels']

        # annotations. Skip unannotated images.
        for a_key in annotations.keys():
            if annotations[a_key].get('regions') is None:
                annotations.pop(a_key, None)

        # Add images
        for a_id, a_val in annotations.items():

            assert split_info[a_id] in ["train", "val", 'test']

            # If the image is not part of the currently initialized split -> continue and handle this image later on
            # Skip if image caused problems during earlier training sessions
            if split_info[a_id] != subset or a_val['path_to_image'] in self.ignore_paths:
                continue

            image_path = os.path.join(dataset_dir, a_val['path_to_image'])
            class_ids = list()
            for class_name in a_val['classes']:
                class_ids.append(self.class_to_id_map[class_name])

            self.add_image(
                source='sun',
                image_id=a_id,  # The key of the annotations dict is unique and is used as image_id
                path=image_path,
                width=a_val['image_width'], height=a_val['image_height'],
                polygons=a_val['regions'], 
                class_ids=class_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            
            # Skip polygons which are invalid
            if (type(p['all_points_y']) is list and type(p['all_points_x']) is list) and\
                (not p['all_points_y'] or not p['all_points_x'] or\
                len(p['all_points_y']) < 3 or len(p['all_points_x']) < 3 or\
                (len(p['all_points_y']) != len(p['all_points_x']))):
                # print('Ignore points y: ', p['all_points_y'], ' and x ', p['all_points_x'])
                continue
            
            y_vals = np.clip(p['all_points_y'], a_min=0, a_max=info["height"])
            x_vals = np.clip(p['all_points_x'], a_min=0, a_max=info["width"])
            rr, cc = skimage.draw.polygon(y_vals, x_vals)
            mask[rr, cc, i] = 1

        class_ids_annotation = np.zeros([len(info["polygons"])])

        # When initializing the dataset the ids were added as parameter "class_ids"
        class_ids = info["class_ids"]
        for i, class_id in enumerate(class_ids):
            class_ids_annotation[i] = class_id

        class_ids_annotation = class_ids_annotation.astype(int)

        return mask.astype(np.bool), class_ids_annotation

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sun":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image, plugin='pil')
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

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

    args = parser.parse_args()
    print('Arguments:\n ', args)
    if args.cvhci_mode is True:
        print('cvhci mode is ', args.cvhci_mode)
        args.dataset = CVHCI_DATASET_PATH
    elif not args.dataset: 
        args.dataset = LOCAL_PATH_DATASET

    assert args.dataset, "Argument --dataset is required for training"

    # Configurations
    config = SunConfig()
    if args.lr:
        config.LEARNING_RATE = float(args.lr)

    config.display()

    datasets = dict()
    for dataset_name in ["train", "val", "test"]:
        datasets[dataset_name] = SunDataset(config=config)
        datasets[dataset_name].load_sun(args.dataset, dataset_name)
        datasets[dataset_name].prepare()

    if args.command == "train":

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
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])

        #RandAugment applies random augmentation techniques chosen from a wide range of augmentation
        augmentation = imgaug.augmenters.Sometimes(0.4, [
            imgaug.augmenters.RandAugment(n=config.AUGMENTATION_NUM, m=config.AUGMENTATION_STRENGTH)
        ])    # Train or evaluate
        print(f'\n\n---AUGMENTATION---: \nStrength: {config.AUGMENTATION_STRENGTH}\nNumber: {config.AUGMENTATION_NUM}')
        # augmentation = None
        
    if args.command == "train" and args.weights == COCO_MODEL_PATH:
        # *** This training schedule is an example. Update to your needs ***
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.
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
        evaluate_sun(args, datasets['test'])

