import random

import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def get_device(cpu=False):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and not cpu) else "cpu")
    return device


def reproduce(seed=1234):
    manual_seed = seed
    print("Random Seed:", manual_seed)

    np.random.seed(0)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def show_data_batch(batch, max_images=64):
    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(batch[0][:max_images], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
