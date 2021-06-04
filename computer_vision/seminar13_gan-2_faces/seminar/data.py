"""Custom utils to get training data."""

import glob
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_SIZE = 64
IMAGE_EXTS = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}


def is_image(filename):
    return os.path.splitext(filename)[1] in IMAGE_EXTS


class ImageFolderNoLabels(Dataset):
    """Much like ImageFolder from torchvision, but returns only images without labels."""

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        image_filenames = glob.glob(os.path.join(self.root, "*.*"))
        self._image_filenames = list(filter(is_image, image_filenames))

    def __len__(self):
        return len(self._image_filenames)

    def __getitem__(self, item):
        image_filename = self._image_filenames[item]
        image = Image.open(image_filename)
        if self.transform is not None:
            image = self.transform(image)
        return image


def get_data(data_root, batch_size=1, num_workers=1, image_size=IMAGE_SIZE):
    dataset = ImageFolderNoLabels(root=data_root,
                                  transform=transforms.Compose([
                                      transforms.Resize(image_size),
                                      transforms.CenterCrop(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
                                  ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                            pin_memory=True)

    return dataset, dataloader
