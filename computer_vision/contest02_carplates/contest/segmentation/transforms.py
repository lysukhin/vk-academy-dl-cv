import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask, force_apply=False):
        image_, mask_ = image.copy(), mask.copy()
        if image_.shape[0] != self.size[1] or image_.shape[1] != self.size[0]:
            image_ = cv2.resize(image_, self.size)
            mask_ = cv2.resize(mask_, self.size)
        return dict(image=image_, mask=mask_)

    
class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)):
        self.mean = np.asarray(mean).reshape((1, 1, 3)).astype(np.float32)
        self.std = np.asarray(std).reshape((1, 1, 3)).astype(np.float32)

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        return image, mask


# TODO TIP: Is default image size (256) enough for segmentation of car license plates?
def get_train_transforms(image_size):
    return A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.),
        Resize(size=(image_size, image_size)),
        ToTensorV2(),
    ])
