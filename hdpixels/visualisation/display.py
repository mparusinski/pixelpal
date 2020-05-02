import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from model.base import get_model


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
            augmentation_model.load_weights(weights_file)
        image = augmentation_model.augment(image)[0]
    plt.imshow(image)
    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()


