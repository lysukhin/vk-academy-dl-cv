import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import cv2
import json


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict_file, transforms=None):
        self.transforms = transforms

        with open(data_dict_file, 'r') as f:
            self.data_dict = json.load(f)

        self.imgs = list(self.data_dict.keys())

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        h, w = image.shape[:2]
#         img = Image.open(img_path).convert("RGB")
#         w, h = img.size

        num_objs = len(self.data_dict[img_path])
        boxes = []
        for i in range(num_objs):
            bbox = self.data_dict[img_path][i]
            xmin = bbox[0] * w
            xmax = bbox[1] * w
            ymin = bbox[2] * h
            ymax = bbox[3] * h
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = np.asarray(boxes)  # N x 4
        annot = np.zeros((len(boxes), 1))  # N x 1
        annot = np.concatenate((boxes, annot), axis=1)  # N x 5

        sample = {'img': image, 'annot': annot}
        
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.imgs)

    def image_aspect_ratio(self, image_index):
        img_path = self.imgs[image_index]
        img = Image.open(img_path).convert("RGB") # slow
        w, h = img.size
        return w / h

    def num_classes(self):
        return 2
