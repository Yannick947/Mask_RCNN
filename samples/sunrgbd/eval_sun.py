from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np

from mrcnn import model as modellib, utils
from samples.sunrgbd.sun_config import SunConfig, InferenceConfig


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        logs.update({'Augmentation strength': SunConfig.AUGMENTATION_STRENGTH})
        logs.update({'Augmentation num': SunConfig.AUGMENTATION_NUM})
        super().on_epoch_end(epoch, logs)


def evaluate_sun(args, dataset):

    print('Evaluating for IoU_min=0.5 on test dataset ...')

    # Create inference model
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=args.logs)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    if not args.weights:
        model_path = model.find_last()
    else: 
        model_path = args.weights

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    image_ids = dataset.image_ids
    APs = []
    IoUs = []

    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, inference_config,
                                image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, _, _, _ =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            r["rois"], r["class_ids"], r["scores"], r['masks'])
        if type(AP) is float or type(AP) is np.float64:
            APs.append(AP)
        elif type(AP) is list and len(AP) > 0: 
            APs.append(sum(AP) / len(AP))

    print("mAP: ", np.mean(APs))