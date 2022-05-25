"""Create recognition crops from train.json (using car plates coordinates & texts)."""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import json
from argparse import ArgumentParser

import cv2
import numpy as np
import tqdm

from contest.common import get_box_points_in_correct_order, apply_normalization


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("data_dir", help="Path to dir containing 'train/', 'test/', 'train.json'.")
    return parser.parse_args()


def get_crop(image, box):
    # TODO TIP: Maybe useful to crop using corners
    # See cv2.findHomography & cv2.warpPerspective for more
    box = get_box_points_in_correct_order(box)

    # TODO TIP: Maybe adding some margin could help.
    x_min = np.clip(min(box[:, 0]), 0, image.shape[1])
    x_max = np.clip(max(box[:, 0]), 0, image.shape[1])
    y_min = np.clip(min(box[:, 1]), 0, image.shape[0])
    y_max = np.clip(max(box[:, 1]), 0, image.shape[0])

    crop = image[y_min: y_max, x_min: x_max]

    return crop


def main(args):
    config_filename = os.path.join(args.data_dir, "train.json")
    with open(config_filename, "rt") as fp:
        config = json.load(fp)

    config_recognition = []
    for i, item in enumerate(tqdm.tqdm(config)):

        image_filename = item["file"]
        image = cv2.imread(os.path.join(args.data_dir, image_filename))
        if image is None:
            continue

        image_base, ext = os.path.splitext(image_filename)

        nums = item["nums"]
        for j, num in enumerate(nums):
            text = num["text"]
            box = np.asarray(num["box"])
            crop_filename = image_base + ".box." + str(j).zfill(2) + ext
            new_item = {"file": crop_filename, "text": text}
            if os.path.exists(crop_filename):
                config_recognition.append(new_item)
                continue

            crop = get_crop(image, box)
            cv2.imwrite(os.path.join(args.data_dir, crop_filename), crop)
            config_recognition.append(new_item)

    output_config_filename = os.path.join(args.data_dir, "train_recognition.json")
    with open(output_config_filename, "wt") as fp:
        json.dump(config_recognition, fp)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
