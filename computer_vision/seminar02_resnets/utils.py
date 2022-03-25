import matplotlib.pyplot as plt
import numpy as np


def plot(arr, label=""):
    plt.plot(arr, label=label)
    plt.xlabel("iterations")
    plt.ylabel("CE loss")
    plt.grid(True)
    plt.legend()
    
    
def show_kernels(conv_layer, num_cols=8):
    kernels = conv_layer.weight.data
    num_rows = int(np.ceil(len(kernels) / num_cols))
    plt.figure(figsize=(12, 12 // num_cols * num_rows))
    
    for i, w in enumerate(kernels, 1):
        w = w.cpu().numpy().transpose(1, 2, 0)
        w -= w.min()
        w /= w.max()
        
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(w)
        plt.axis("off")
        
    plt.show()
