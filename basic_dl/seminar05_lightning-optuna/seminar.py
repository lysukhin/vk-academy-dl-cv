from collections import defaultdict

import cv2
import numpy as np
import sqlite3 as sq
import torch
import torchvision
from matplotlib import pyplot as plt


def add_image_label(image, label, output_size=256):
    image = cv2.resize(image, (output_size, output_size))
    text_height = int(cv2.getTextSize("X", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1] * 1.1)
    label_image = np.full((text_height, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(label_image, str(label), (0, label_image.shape[0] - 1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return np.concatenate([image, label_image], 0)


def make_grid(images, predictions, labels, size=9):
    """Create grid view of images and model predictions."""
    predictions = predictions[:size].detach().cpu().numpy()
    images = (images[:size] * 255).type(torch.uint8)
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    images = [add_image_label(image, labels[prediction])
              for image, prediction in zip(images, predictions)]
    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    grid = torchvision.utils.make_grid(images, nrow=3)
    return grid


def show_study(storage_path):
    conn = sq.connect(storage_path)
    c = conn.cursor()

    query = """
        select
            trial_id, param_name, param_value
        from trial_params
    """

    parameters = defaultdict(dict)
    for trial_id, key, value in c.execute(query):
        parameters[key][trial_id] = value

    query = """
    select
        trial_id, value
    from trial_values
    """

    metrics = {}
    for trial_id, value in c.execute(query):
        metrics[trial_id] = value

    def show_trials(xlabel, ylabel):
        x = []
        y = []
        size = []
        for trial_id in metrics:
            x.append(parameters[xlabel][trial_id])
            y.append(parameters[ylabel][trial_id])
            size.append(metrics[trial_id])

        size = np.asarray(size)
        size = 100 / (size - size.min() + 0.1)

        plt.xlabel(xlabel.split(".")[-1])
        plt.ylabel(ylabel.split(".")[-1])
        plt.xlim(np.min(x) * 0.9, 1.1 * np.max(x))
        plt.ylim(np.min(y) * 0.9, 1.1 * np.max(y))

        plt.scatter(x, y, s=size, alpha=0.5)
        plt.show()
    
    show_trials("module_params.optimizer_params.optimizer_params.weight_decay", "module_params.optimizer_params.optimizer_params.lr")
    show_trials("data_params.batch_size", "module_params.optimizer_params.optimizer_params.lr")
    show_trials("data_params.batch_size", "seed")
    show_trials("module_params.optimizer_params.optimizer_params.lr", "seed")
