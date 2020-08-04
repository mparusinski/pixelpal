import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from hdpixels.model.base import get_model, load_weights, augment


def display_file(image, augmentator=None, weights_file=None, save_fig=None, horizontal_flip=False, vertical_flip=False):
    image = mpimg.imread(image)
    if horizontal_flip:
        image = np.fliplr(image)
    if vertical_flip:
        image = np.flipud(image)

    if augmentator is None and weights_file:
        raise Exception("Option 'weights_file' only supported with a 'module_name' option")
    if augmentator:
        augmentation_model = get_model(augmentator)
        if weights_file:
            load_weights(augmentation_model, weights_file)
        image = augment(augmentation_model, image)[0]
    plt.imshow(image)
    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()

