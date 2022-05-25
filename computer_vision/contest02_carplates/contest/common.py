import logging

import cv2
import numpy as np
import torch


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


def prepare_for_segmentation(image, image_size):
    """
    Scale proportionally image into fit_size and pad with zeroes to fit_size
    :return: np.ndarray image_padded shaped (*fit_size, 3), float k (scaling coef), float dw (x pad), dh (y pad)
    """
    h, w = image.shape[:2]
    kx = w / image_size[0]
    ky = h / image_size[1]
    return cv2.resize(image, image_size, interpolation=cv2.INTER_AREA), (kx, ky)


def get_boxes_from_mask(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for j in range(1, num_labels):  # j = 0 == background component
        x, y, w, h = stats[j][:4]
        x1 = round(x)
        y1 = round(y)
        x2 = round(x + w)
        y2 = round(y + h)
        box = [x1, y1, x2, y2]
        boxes.append(box)
    boxes = np.asarray(boxes)
    return boxes


def prepare_for_recognition(image, output_size):
    image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float) / 255.
    return torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)


def get_box_points_in_correct_order(box):
    """
    Permute the points in box in following order: Top left -> Top right -> Bottom right -> Bottom Left.
    :return: np.ndarray box shaped (4, 2)
    """
    box_sorted_by_x = sorted(box.tolist(), key=lambda x: x[0])
    if box_sorted_by_x[0][1] < box_sorted_by_x[1][1]:
        top_left = box_sorted_by_x[0]
        bottom_left = box_sorted_by_x[1]
    else:
        top_left = box_sorted_by_x[1]
        bottom_left = box_sorted_by_x[0]
    if box_sorted_by_x[2][1] < box_sorted_by_x[3][1]:
        top_right = box_sorted_by_x[2]
        bottom_right = box_sorted_by_x[3]
    else:
        top_right = box_sorted_by_x[3]
        bottom_right = box_sorted_by_x[2]
    return np.asarray([top_left, top_right, bottom_right, bottom_left])


def apply_normalization(image, keypoints, to_size=(320, 64)):
    """
    Apply perspective transform to crop such that keypoints are moved to corners of rectangle with fit_size.
    :return np.ndarray image of shape (to_size, 3) (result crop)
    """
    dest_points = np.asarray([[0., 0.],
                              [to_size[0], 0.],
                              [to_size[0], to_size[1]],
                              [0., to_size[1]]])
    h, _ = cv2.findHomography(keypoints, dest_points)
    crop_normalized = cv2.warpPerspective(image, h, dsize=to_size, flags=cv2.INTER_LANCZOS4)
    return crop_normalized
