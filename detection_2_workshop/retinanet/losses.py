import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    def __init__(self, device=None):
        super(FocalLoss, self).__init__()
        if device is None:
            device = torch.cuda.is_available() and 'cuda' or 'cpu'
        self.device = device

    def forward(self, classifications, regressions, anchors, annotations):
        device = self.device
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths # X coordinate of the center pixel
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights # Y coordinate of the center pixel

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().to(device))
                classification_losses.append(torch.tensor(0).float().to(device))
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            targets = targets.to(device)

            """Specifically, anchors are assigned to ground-truth object boxes using an intersection-over-union (IoU)
             threshold of 0.5; and to background if their IoU is in [0, 0.4).
             If an anchor is unassigned, which may happen with overlap in [0.4, 0.5), it is ignored during training"""

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            """As each anchor is assigned to at most one object box, we set the corresponding entry
             in its length K label vector to 1 and all other entries to 0"""
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1  # only top classes get 1, other -1

            """
            A common method for addressing class imbalance is to introduce a weighting factor 
            α ∈ [0, 1] for class 1 and 1−α for class −1.
            In practice α may be set by inverse class frequency or treated as a hyperparameter to set by cross validation
            We adopt this form in our experiments as it yields slightly improved accuracy over the non-α-balanced form
            """
            # alpha_t = y == 1 and alfa or 1 - alpha
            alpha_factor = torch.ones(targets.shape).to(device) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor) # even more balance
            # 1 - p_t
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            # alfa * (1 - p_t)** gamma
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # modified Cross Entropy
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(device)) # compute loss only for +1 targets

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                """we predict dx, dy, s_x, s_y, but annotations are bounding boxes x1,y1,x2,y2,
                and anchors are a grid of boxes
                so here we convert from absolute ground truth bboxes to relative of given anchors
                annotations -> dx,dy,sx,sy
                """
                assigned_annotations = assigned_annotations[positive_indices, :] # pick assigned anchors

                #w,h,centers of assigned anchors
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                # annotations: boxes -> h,w,centers
                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                # annotations -> dx,dy,sx,sy relative to assigned anchors = targets
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t() # transpose

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device) # std

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :]) # l1 loss

                # Smooth L1 Loss
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0 # subtract small number
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().to(device))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

    
