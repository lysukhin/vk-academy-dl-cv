"""Extra plot tools for Made CV demonstrations."""
import cv2
from matplotlib import pyplot as plt


class InteractivePlot(object):
    """Plot, that can be updated after creation.

    Ensure ```%matplotlib notebook``` header in Jupyter.

    """
    def __init__(self, min_x, min_y, max_x, max_y, xlabel=None, ylabel=None, grid=True):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=120)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        self._fig = fig
        self._ax = ax
        if grid:
            ax.grid(True)
        plt.show()

    def plot(self, xs, ys, **kwargs):
        self._ax.plot(xs, ys, **kwargs)
        self._fig.canvas.draw()

    def scatter(self, xs, ys, **kwargs):
        self._ax.scatter(xs, ys, **kwargs)
        self._fig.canvas.draw()


def show_image(filename, figsize=(6, 4), dpi=120):
    """Show image from file."""
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(num=None, figsize=figsize, dpi=dpi)
    plt.imshow(image_rgb)
    plt.show()
