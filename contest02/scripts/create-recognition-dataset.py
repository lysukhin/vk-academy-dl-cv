"""Create recognition crops from train.json (using car plates coordinates & texts)."""

import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import tqdm


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", help="Path to dir containing 'train/', 'test/', 'train.json'.")
    parser.add_argument("--normalize", help="If True, crop & transform box using 4 corner points;"
                                            "crop bounding box otherwise.", action="store_true")
    return parser.parse_args()


def get_crop(image, box, normalize=False):
    if normalize:
        # TODO TIP: Maybe useful to crop using corners
        # See cv2.getPerspectiveTransform for more
        raise NotImplementedError

    # TODO TIP: Maybe adding some margin could help.
    x_min = np.clip(min(box[:, 0]), 0, image.shape[1])
    x_max = np.clip(max(box[:, 0]), 0, image.shape[1])
    y_min = np.clip(min(box[:, 1]), 0, image.shape[0])
    y_max = np.clip(max(box[:, 1]), 0, image.shape[0])
    return image[y_min: y_max, x_min: x_max]


def main(args):
    config_filename = os.path.join(args.data_dir, "train.json")
    with open(config_filename, "rt") as fp:
        config = json.load(fp)

    config_recognition = []

    for item in tqdm.tqdm(config):

        image_filename = item["file"]
        image = cv2.imread(os.path.join(args.data_dir, image_filename))
        if image is None:
            continue

        image_base, ext = os.path.splitext(image_filename)

        nums = item["nums"]
        for i, num in enumerate(nums):
            text = num["text"]
            box = np.asarray(num["box"])
            crop_filename = image_base + ".box" + str(i).zfill(2) + ext
            new_item = {"file": crop_filename, "text": text}
            if os.path.exists(crop_filename):
                config_recognition.append(new_item)
                continue

            crop = get_crop(image, box, args.normalize)
            cv2.imwrite(os.path.join(args.data_dir, crop_filename), crop)
            config_recognition.append(new_item)

    output_config_filename = os.path.join(args.data_dir, "train_recognition.json")
    with open(output_config_filename, "wt") as fp:
        json.dump(config_recognition, fp)


if __name__ == "__main__":
    main(parse_arguments())
