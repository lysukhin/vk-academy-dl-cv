import json, os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from common import abc, is_valid_str, convert_to_eng



class RecognitionDataset(Dataset):

    def __init__(self, data_path, config_file=None, abc=abc, transforms=None):
        super(RecognitionDataset, self).__init__()
        self.data_path = data_path
        self.abc = abc
        self.transforms = transforms
        self.image_names, self.texts = [], []

        if config_file is not None:
            self.image_names, self.texts = self._parse_root_(config_file)

    def _parse_root_(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        image_names, texts = [], []
    
        for item in config:
            image_name = item['file']
            text = item['text']
            text = convert_to_eng(text.upper())  # samples can have russian characters or lower case
            if is_valid_str(text):
                texts.append(text)
                image_names.append(image_name)

        return image_names, texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        image_name = os.path.join(self.data_path, self.image_names[item])
        image = cv2.imread(image_name).astype(np.float32) / 255.

        text = self.texts[item]
        seq = self.text_to_seq(text)
        seq_len = len(seq)
        output = dict(image=image, seq=seq, seq_len=seq_len, text=text)
        
        if self.transforms is not None:
            output = self.transforms(output)
            
        return output

    def text_to_seq(self, text):
        seq = [self.abc.find(c) + 1 for c in text]
        return seq

    @staticmethod
    def collate_fn(batch):
        images = list()
        seqs = list()
        seq_lens = list()
        for sample in batch:
            images.append(torch.from_numpy(sample["image"].transpose((2, 0, 1))).float())
            seqs.extend(sample["seq"])
            seq_lens.append(sample["seq_len"])
        images = torch.stack(images)
        seqs = torch.Tensor(seqs).int()
        seq_lens = torch.Tensor(seq_lens).int()
        batch = {"images": images, "seqs": seqs, "seq_lens": seq_lens}
        return batch
