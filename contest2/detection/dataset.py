import os, json
import cv2
import numpy as np
from torch.utils.data import Dataset


class DetectionDataset(Dataset):

    def __init__(self, data_path, config_file=None, transforms=None):
        super(DetectionDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.image_names, self.mask_names = [], []
        
        if config_file is not None: # config = None only for the sake of dirty hack in train.py
            self.image_names, self.mask_names = self._parse_root_(config_file)

    def _parse_root_(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        image_names, mask_names = [], []
        for item in config:
            if 'mask' in item: # handling bad files during transfer
                image_names.append(item['file'])
                mask_names.append(item['mask'])

        assert len(image_names) == len(mask_names), 'Images and masks length mismatch'
        return image_names, mask_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image_name = os.path.join(self.data_path, self.image_names[item])
        mask_name = os.path.join(self.data_path, self.mask_names[item])

        image = cv2.imread(image_name).astype(np.float32) / 255.
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image.transpose(2, 0, 1), mask
