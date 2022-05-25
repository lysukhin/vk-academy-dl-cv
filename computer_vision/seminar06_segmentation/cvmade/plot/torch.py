"""Plotting tools for PyTorch."""
import random

from matplotlib import pyplot as plt

from ..image import resize_image, image_to_numpy, image_to_torch, ALIGNMENT_CENTER


def show_images_dataset(dataset, n=5, collate_fn=lambda x: x[0], image_size=(128, 128)):
    """Plot images from dataset."""
    images = []
    for _ in range(n):
        image = collate_fn(random.choice(dataset))
        image = image_to_numpy(image)
        image = resize_image(image, image_size,
                             preserve_aspect=True,
                             alignment=ALIGNMENT_CENTER)[0]
        images.append(image_to_torch(image))
    print(images[0].dtype)
    grid = torch.cat(images, 2)
    grid = grid.permute(1, 2, 0)
    if (grid.dtype == torch.uint8) and (grid.shape[2] == 1):
        grid = grid.squeeze(2)
    plt.imshow(grid)
    plt.show()
