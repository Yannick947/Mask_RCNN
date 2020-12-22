import json
from abc import abstractmethod
import os

import numpy as np
import skimage

from mrcnn import model as modellib, utils
from samples.sunrgbd.sun_config import CLASSES, IGNORE_IMAGES_PATH, ANNOTATION_FILENAME, ROOT_DIR


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

class SunDataset2D(SunDataset):
    def __init_(self, **kwargs):
        super().__init__(**kwargs)

class SunDataset3D(SunDataset):
    def __init_(self, **kwargs):
        super().__init__(**kwargs)

    def rgb_to_depth_path(self, rgb_path):
        parent_path = os.path.dirname(os.path.dirname(rgb_path))
        depth_dir = os.path.join(parent_path, 'depth')
        try: 
            # If there is no file in this dir then raise an error in the exception
            depth_path = os.path.join(depth_dir, os.listdir(depth_dir)[0])
            return depth_path
        except Exception as e: 
            raise FileNotFoundError(f'No depth file found in {depth_dir}')

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,4] Numpy array.
        """
        # Load image
        rgb_path = self.image_info[image_id]['path']
        depth_path = self.rgb_to_depth_path(rgb_path)

        rgb_image = skimage.io.imread(rgb_path, plugin='pil')
        # If has an alpha channel, remove it 
        if rgb_image.shape[-1] == 4:
            rgb_image = rgb_image[..., :3]

        depth_image = skimage.io.imread(depth_path, plugin='pil')
        #depth image needs third dimension with length 1 so that numpy can stack it with rgb
        depth_image = np.expand_dims(depth_image, 2)
        
        image = np.concatenate([rgb_image, depth_image], axis=2)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim < 3:
            image = skimage.color.gray2rgb(image)

        return image                                                                     
                                                                                                                                              