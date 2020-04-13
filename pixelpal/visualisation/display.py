import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pixelpal.model.base import get_model


def display_file(image, augmentator=None):
    image = mpimg.imread(image)
    if augmentator:
        augmentation_model = get_model(augmentator)
        image = augmentation_model.augment(image)[0]
    plt.imshow(image)
    plt.show()


