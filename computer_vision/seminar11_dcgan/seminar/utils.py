"""Custom utils to handle data batches for showing / saving."""

import random

import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def get_device(cpu=False):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and not cpu) else "cpu")
    return device


def reproduce(seed=1234):
    manual_seed = seed
    print("Random Seed:", manual_seed)

    np.random.seed(0)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def batch_to_image(batch):
    grid = vutils.make_grid(batch.cpu(), padding=2, normalize=True).permute(1, 2, 0)
    image = (grid.numpy() * 255).clip(0, 255).astype(np.uint8)
    return image
    
    
def show_data_batch(batch, max_images=64):
    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.title("Images")
    image = batch_to_image(batch[:max_images])
    plt.imshow(image)
    plt.show()
    
    
def save_data_batch(batch, filename, max_images=64):
    image = batch_to_image(batch[:max_images])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, image)
