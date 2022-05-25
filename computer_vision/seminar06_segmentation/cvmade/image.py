import cv2
import torch
import numpy as np


ALIGNMENT_TOPLEFT = "topleft"
ALIGNMENT_CENTER = "center"


def image_to_numpy(image):
    """Convert Torch tensor to Numpy image."""
    if isinstance(image, np.ndarray):
        return image
    if image.dtype == torch.float32:
        image = image - image.min()
        image = 255 * image / max(1e-6, image.max())
    return image.to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def image_to_torch(image, device=None):
    """Convert Numpy image to Torch tensor."""
    if isinstance(image, torch.Tensor):
        return image
    if image.shape[-1] == 4:
        image = image[..., :3]
    result = torch.from_numpy(np.ascontiguousarray(image))
    if device is not None:
        result = result.to(device)
    result = result.permute(2, 0, 1)
    return result


def resize_image(image, size, preserve_aspect=False, pad_value=0,
                 interpolation=cv2.INTER_AREA, alignment=ALIGNMENT_TOPLEFT):
    """Resize image.

    Args:
        image: Numpy image in HWC format.
        size: Width and height.
        preserve_aspect: Preserve aspect and pad borders.
        pad_value: Value for padding when preserve_aspect is true.
        alignment: Type of image alignment when preserve_aspect is true.

    Returns:
        image: Resized image in HWC format or original image if no resize needed.
        scale: X and Y scale of the image.
        offset: X and Y offset.
    """
    nchannels = image.shape[2]
    if nchannels not in {1, 3, 4}:
        raise NotImplementedError("{} channels".format(nchannels))
    if size[0] == image.shape[1] and size[1] == image.shape[0]:
        return image
    x_scale = size[0] / image.shape[1]
    y_scale = size[1] / image.shape[0]
    if preserve_aspect:
        scale = min(x_scale, y_scale)
        x_scale, y_scale = scale, scale
        target_size = (int(image.shape[1] * x_scale), int(image.shape[0] * y_scale))
        if alignment == ALIGNMENT_TOPLEFT:
            x_offset = 0
            y_offset = 0
        elif alignment == ALIGNMENT_CENTER:
            x_offset = (size[0] - target_size[0]) // 2
            y_offset = (size[1] - target_size[1]) // 2
        else:
            raise ValueError("Unknown alignment: {}".format(alignment))
    else:
        target_size = size
        x_offset = 0
        y_offset = 0
    resized = np.empty((size[1], size[0], nchannels), dtype=np.uint8)
    if preserve_aspect:
        resized.fill(pad_value)
    cv2.resize(image, tuple(target_size), interpolation=interpolation,
               dst=resized[y_offset:y_offset + target_size[1], x_offset:x_offset + target_size[0]])
    return resized, (x_scale, y_scale), (x_offset, y_offset)
