import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from segmentation.dataset import DetectionDataset
from segmentation.loss import dice_coeff, dice_loss
from segmentation.models import get_model
from segmentation.transforms import get_train_transforms, get_val_transforms
from inference_utils import get_logger


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, default=None, help="path to the data")
    parser.add_argument("-e", "--epochs", dest="epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=80, type=int, help="batch size")
    parser.add_argument("-s", "--image_size", dest="image_size", default=256, type=int, help="input image size")
    parser.add_argument("-lr", "--learning_rate", dest="lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("-lrs", "--learning_rate_step", dest="lr_step", default=None, type=int, help="learning rate step")
    parser.add_argument("-lrg", "--learning_rate_gamma", dest="lr_gamma", default=None, type=float,
                        help="learning rate gamma")
    parser.add_argument("-w", "--weight_bce", default=1, type=float, help="weight for BCE loss")
    parser.add_argument("-l", "--load", dest="load", default=None, help="load file model")
    parser.add_argument("-o", "--output_dir", dest="output_dir", default="runs/segmentation_baseline",
                        help="dir to save log and models")
    return parser.parse_args()


def validate(model, val_dataloader, device):
    model.eval().to(device)
    val_dice = []
    for batch in tqdm.tqdm(val_dataloader):
        images, true_masks = batch
        with torch.no_grad():
            masks_pred = model(images.to(device)).squeeze(1)  # (b, 1, h, w) -> (b, h, w)
        masks_pred = (torch.sigmoid(masks_pred) > 0.5).float()
        dice = dice_coeff(masks_pred.cpu(), true_masks).item()
        val_dice.append(dice)
    return np.mean(val_dice)


def train(model, optimizer, criterion, scheduler, train_dataloader, logger, device=None):
    model.train()

    epoch_losses = []
    epoch_bce_losses, epoch_dice_losses = [], []

    tqdm_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, batch in tqdm_iter:
        imgs, true_masks = batch
        masks_pred = model(imgs.to(device))
        masks_probs = torch.sigmoid(masks_pred)

        bce_loss_value, dice_loss_value = criterion(masks_probs.cpu().view(-1), true_masks.view(-1))
        loss = bce_loss_value + dice_loss_value

        epoch_bce_losses.append(bce_loss_value.item())
        epoch_dice_losses.append(dice_loss_value.item())
        epoch_losses.append(loss.item())
        tqdm_iter.set_description(f"mean loss: {np.mean(epoch_losses):.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    logger.info(
        f"Epoch finished! Loss: {np.mean(epoch_losses):.5f} ({np.mean(epoch_bce_losses):.5f} | {np.mean(epoch_dice_losses):.5f})")

    return np.mean(epoch_losses)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = get_logger(os.path.join(args.output_dir, "train.log"))
    logger.info("Start training with params:")
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    # TODO TIP: Try other models, either from smp package or from somewhere else.
    model = get_model()
    if args.load is not None:
        with open(args.load, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(state_dict)
    model.to(device)
    logger.info(f"Model type: {model.__class__.__name__}")

    # TODO TIP: Pure Adam(W) with Karpathy constant LR is great, but there's still room for improvements.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # TODO TIP: You can always try on plateau scheduler as a default option
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma) \
        if args.lr_step is not None else None

    # TODO TIP: Remember what are the problems of BCE for semantic segmentation?
    # Key words: 'background'.
    criterion = lambda x, y: (args.weight_bce * nn.BCELoss()(x, y), (1. - args.weight_bce) * dice_loss(x, y))

    train_transforms = get_train_transforms(args.image_size)
    train_dataset = DetectionDataset(args.data_path, os.path.join(args.data_path, "train_segmentation.json"),
                                     transforms=train_transforms, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8,
                                  pin_memory=True, shuffle=True, drop_last=True)

    val_transforms = get_val_transforms(args.image_size)
    val_dataset = DetectionDataset(args.data_path, os.path.join(args.data_path, "train_segmentation.json"),
                                   transforms=val_transforms, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4,
                                pin_memory=True, shuffle=False, drop_last=False)

    logger.info(f"Length of train / val = {len(train_dataset)} / {len(val_dataset)}")
    logger.info(f"Number of batches of train / val = {len(train_dataloader)} / {len(val_dataloader)}")

    best_model_info = {"epoch": -1, "val_dice": 0., "train_dice": 0., "train_loss": 0.}
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.epochs}.")

        train_loss = train(model, optimizer, criterion, scheduler, train_dataloader, logger, device)

        val_dice = validate(model, val_dataloader, device)
        if val_dice > best_model_info["val_dice"]:
            best_model_info["val_dice"] = val_dice
            best_model_info["train_loss"] = train_loss
            best_model_info["epoch"] = epoch
            with open(os.path.join(args.output_dir, "CP-best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)
            logger.info(f"Validation Dice Coeff: {val_dice:.3f} (best)")
        else:
            logger.info(f"Validation Dice Coeff: {val_dice:.5f} (best {best_model_info['val_dice']:.5f})")

    with open(os.path.join(args.output_dir, "CP-last.pth"), "wb") as fp:
        torch.save(model.state_dict(), fp)


if __name__ == "__main__":
    main(parse_arguments())
