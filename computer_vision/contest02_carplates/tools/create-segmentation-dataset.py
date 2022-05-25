"""Create segmentation masks from train.json (using car plates coordinates)."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import json
from argparse import ArgumentParser

import cv2
import numpy as np
import tqdm


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("data_dir", help="Path to dir containing 'train/', 'test/', 'train.json'.")
    return parser.parse_args()


def main(args):
    config_filename = os.path.join(args.data_dir, "train.json")
    with open(config_filename, "rt") as fp:
        config = json.load(fp)

    config_segmentation = []
    for item in tqdm.tqdm(config):
        new_item = {}
        new_item["file"] = item["file"]

        image_filename = item["file"]
        image_base, ext = os.path.splitext(image_filename)
        mask_filename = image_base + ".mask" + ext

        nums = item["nums"]
        if os.path.exists(os.path.join(args.data_dir, mask_filename)):
            raise FileExistsError(os.path.join(args.data_dir, mask_filename))

        image = cv2.imread(os.path.join(args.data_dir, image_filename))
        if image is None:
            continue
        mask = np.zeros(shape=image.shape[:2], dtype=np.uint8)
        for num in nums:
            bbox = np.asarray(num["box"])
            cv2.fillConvexPoly(mask, bbox, 255)
        cv2.imwrite(os.path.join(args.data_dir, mask_filename), mask)

        new_item["mask"] = mask_filename
        config_segmentation.append(new_item)

    output_config_filename = os.path.join(args.data_dir, "train_segmentation.json")
    with open(output_config_filename, "wt") as fp:
        json.dump(config_segmentation, fp)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
