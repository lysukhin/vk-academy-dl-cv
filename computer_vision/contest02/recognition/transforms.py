import cv2
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item):
        item_ = item.copy()
        for t in self.transforms:
            item_ = t(item_)
        return item_


class Rotate(object):
    def __init__(self, max_angle=20, fill_value=0.0, p=0.5):
        self.max_angle = max_angle
        self.fill_value = fill_value
        self.p = p

    def __call__(self, item):
        if np.random.uniform(0.0, 1.0) > self.p:
            return item
        item_ = item.copy()
        h, w, _ = item_['image'].shape
        image = item_['image']
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        image = self.rotate_and_scale(image, angle=angle)
        item_['image'] = image
        return item_

    def rotate_and_scale(self, image, scale_factor=1.0, angle=30):
        old_h, old_w = image.shape[:2]
        m = cv2.getRotationMatrix2D(center=(old_w / 2, old_h / 2), angle=angle, scale=scale_factor)

        new_w, new_h = old_w * scale_factor, old_h * scale_factor
        r = np.deg2rad(angle)
        sin_r = np.sin(r)
        cos_r = np.cos(r)
        new_w, new_h = (abs(sin_r * new_h) + abs(cos_r * new_w), abs(sin_r * new_w) + abs(cos_r * new_h))

        (tx, ty) = ((new_w - old_w) / 2, (new_h - old_h) / 2)
        m[0, 2] += tx
        m[1, 2] += ty
        rotated_img = cv2.warpAffine(image, m, dsize=(int(new_w), int(new_h)))
        return rotated_img


class Resize(object):
    def __init__(self, size=(320, 32)):
        self.size = size

    def __call__(self, item):
        item['image'] = cv2.resize(item['image'], self.size)
        return item


class Pad(object):
    def __init__(self, max_size=0.1, p=0.1):
        self.max_size = max_size
        self.p = p
        self.border_styles = ('replicate', 'zeroes', 'colour')

    def __call__(self, item):
        if np.random.uniform(0.0, 1.0) > self.p:
            return item
        item_ = item.copy()
        image = item_['image'].copy()

        h, w, _ = image.shape
        top = int(np.random.uniform(0, self.max_size) * h)
        bottom = int(np.random.uniform(0, self.max_size) * h)
        left = int(np.random.uniform(0, self.max_size) * w)
        right = int(np.random.uniform(0, self.max_size) * w)
        border_style = np.random.choice(self.border_styles)
        if border_style == 'replicate':
            image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
        else:
            value = np.random.uniform(size=(3,)) if border_style == 'colour' else 0.0  # zeroes
            image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=value)

        item_['image'] = image
        return item_


class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)):
        self.mean = np.asarray(mean).reshape((1, 1, 3)).astype(np.float32)
        self.std = np.asarray(std).reshape((1, 1, 3)).astype(np.float32)

    def __call__(self, item):
        item["image"] = (item["image"] - self.mean) / self.std
        return item


def get_train_transforms(image_size, augs):
    return Compose([
        Normalize(),
        Rotate(max_angle=augs * 7.5, p=0.5),
        Pad(max_size=augs / 10, p=0.1),
        Resize(size=image_size),
    ])


def get_val_transforms(image_size):
    return Compose([
        Normalize(),
        Resize(size=image_size)
    ])
