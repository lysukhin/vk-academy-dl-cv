import logging
import torch
import cv2
import numpy as np


def get_logger(filename: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def dice_coeff(input, target):
    smooth = 1.

    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum()
    union = input_flat.sum() + target_flat.sum()

    return (2. * intersection + smooth) / (union + smooth)


def dice_loss(input, target):
    return - torch.log(dice_coeff(input, target))


def get_boxes_from_mask(mask, margin, clip=False):
    """
    Detect connected components on mask, calculate their bounding boxes (with margin) and return them (normalized).
    If clip is True, cutoff the values to (0.0, 1.0).
    :return np.ndarray boxes shaped (N, 4)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for j in range(1, num_labels):  # j = 0 == background component
        x, y, w, h = stats[j][:4]
        x1 = int(x - margin * w)
        y1 = int(y - margin * h)
        x2 = int(x + w + margin * w)
        y2 = int(y + h + margin * h)
        box = np.asarray([x1, y1, x2, y2])
        boxes.append(box)
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes).astype(np.float)
    boxes[:, [0, 2]] /= mask.shape[1]
    boxes[:, [1, 3]] /= mask.shape[0]
    if clip:
        boxes = boxes.clip(0.0, 1.0)
    return boxes


def prepare_for_inference(image, fit_size):
    """
    Scale proportionally image into fit_size and pad with zeroes to fit_size
    :return: np.ndarray image_padded shaped (*fit_size, 3), float k (scaling coef), float dw (x pad), dh (y pad)
    """
    # pretty much the same code as detection.transforms.Resize
    h, w = image.shape[:2]
    k = fit_size[0] / max(w, h)
    image_fitted = cv2.resize(image, dsize=None, fx=k, fy=k)
    h_, w_ = image_fitted.shape[:2]
    dw = (fit_size[0] - w_) // 2
    dh = (fit_size[1] - h_) // 2
    image_padded = cv2.copyMakeBorder(image_fitted, top=dh, bottom=dh, left=dw, right=dw,
                                      borderType=cv2.BORDER_CONSTANT, value=0.0)
    if image_padded.shape[0] != fit_size[1] or image_padded.shape[1] != fit_size[0]:
        image_padded = cv2.resize(image_padded, dsize=fit_size)
    return image_padded, k, dw, dh


def get_boxes_from_mask(mask, margin, clip=False):
    """
    Detect connected components on mask, calculate their bounding boxes (with margin) and return them (normalized).
    If clip is True, cutoff the values to (0.0, 1.0).
    :return np.ndarray boxes shaped (N, 4)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for j in range(1, num_labels):  # j = 0 == background component
        x, y, w, h = stats[j][:4]
        x1 = int(x - margin * w)
        y1 = int(y - margin * h)
        x2 = int(x + w + margin * w)
        y2 = int(y + h + margin * h)
        box = np.asarray([x1, y1, x2, y2])
        boxes.append(box)
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes).astype(np.float)
    boxes[:, [0, 2]] /= mask.shape[1]
    boxes[:, [1, 3]] /= mask.shape[0]
    if clip:
        boxes = boxes.clip(0.0, 1.0)
    return boxes
