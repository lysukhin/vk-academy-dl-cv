"""Plotting tools for PyTorch."""
import random

import torchvision
from matplotlib import pyplot as plt


def show_images_dataset(dataset, n=5, collate_fn=lambda x: x[0]):
    """Plot images from dataset."""
    images = [collate_fn(random.choice(dataset)) for _ in range(n)]
    grid = torchvision.utils.make_grid(images)
    grid -= grid.min()
    grid /= grid.max()
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
