import click
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pixelpal.model.base import get_model


@click.command()
@click.argument('image')
@click.option('--augmentator', help='Use model for augmentation')
def display_file(image, augmentator=None):
    image = mpimg.imread(image)
    if augmentator:
        augmentation_model = get_model(augmentator)
        image = augmentation_model.augment(image)[0]
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    display_file()
