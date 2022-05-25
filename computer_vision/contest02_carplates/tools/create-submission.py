import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import tqdm

from contest.segmentation.models import get_model as get_segmentation_model
from contest.recognition.model import get_model as get_recognition_model
from contest.common import prepare_for_segmentation, get_boxes_from_mask, prepare_for_recognition


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, default=None, help="path to the data")
    parser.add_argument("-s", "--seg-model", dest="seg_model", type=str, default=None,
                        help="path to a trained segmentation model")
    parser.add_argument("-r", "--rec-model", dest="rec_model", type=str, default=None,
                        help="path to a trained recognition model")
    parser.add_argument("-o", "--output_file", dest="output_file", default="baseline_submission.csv",
                        help="file to save predictions to")
    return parser.parse_args()


def main(args):
    print("Start inference")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    segmentation_model = get_segmentation_model()
    with open(args.seg_model, "rb") as fp:
        state_dict = torch.load(fp, map_location="cpu")
    segmentation_model.load_state_dict(state_dict)
    segmentation_model.to(device)
    segmentation_model.eval()

    recognition_model = get_recognition_model()
    with open(args.rec_model, "rb") as fp:
        state_dict = torch.load(fp, map_location="cpu")
    recognition_model.load_state_dict(state_dict)
    recognition_model.to(device)
    recognition_model.eval()

    test_images_dirname = os.path.join(args.data_path, "test")
    results = []
    files = os.listdir(test_images_dirname)
    for i, file_name in enumerate(tqdm.tqdm(files)):
        image_src = cv2.imread(os.path.join(test_images_dirname, file_name))
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

        # 1. Segmentation.
        image, (kx, ky) = prepare_for_segmentation(image_src.astype(np.float32) / 255., (256, 256))
        x = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            pred = torch.sigmoid(segmentation_model(x.to(device))).squeeze().cpu().numpy()
        mask = (pred >= 0.5).astype(np.uint8) * 255

        # 2. Extraction of detected regions.
        boxes = get_boxes_from_mask(mask)
        if len(boxes) == 0:
            results.append((file_name, []))
            continue
        boxes = boxes.astype(np.float32)
        boxes[:, [0, 2]] *= kx
        boxes[:, [1, 3]] *= ky
        boxes = boxes.astype(np.int32)

        # 3. Text recognition for every detected bbox.
        texts = []
        for box in boxes:
            x1, y1, x2, y2 = box
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                raise (Exception, str(box))
            if (y2 - y1) < 3 or (x2 - x1) < 3:  # skip small boxes
                continue
            crop = image_src[y1: y2, x1: x2, :]

            tensor = prepare_for_recognition(crop, (320, 64)).to(device)
            with torch.no_grad():
                text = recognition_model(tensor, decode=True)[0]
            texts.append((x1, text))

        # all predictions must be sorted by x1
        texts.sort(key=lambda x: x[0])
        results.append((file_name, [w[1] for w in texts]))

    # Generate a submission file
    with open(args.output_file, "wt") as wf:
        wf.write("file_name,plates_string\n")
        for file_name, texts in sorted(results, key=lambda x: int(os.path.splitext(x[0])[0])):
            wf.write(f"test/{file_name},{' '.join(texts)}\n")
    print('Done')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
