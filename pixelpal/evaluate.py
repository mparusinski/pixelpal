import click
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf

from pixelpal.model.base import get_model


@click.command()
@click.argument('small_image')
@click.argument('large_image')
@click.argument('augmentator')
def display_file(small_image, large_image, augmentator):
    small_image = mpimg.imread(small_image)
    large_image = mpimg.imread(large_image)
    augmentation_model = get_model(augmentator)
    augmented_image = augmentation_model.augment(small_image)[0]
    print()
    print("==== METRICS ====")
    print("Peak Signal to Noise Ratio: {}".format(tf.image.psnr(large_image, augmented_image, max_val=1.0).numpy()))
    print("Structural Similarity     : {}".format(
        tf.image.ssim(
            tf.convert_to_tensor(large_image, dtype=tf.float32),
            tf.convert_to_tensor(augmented_image, dtype=tf.float32), max_val=1.0
        ).numpy())
    )
    print("=================")


if __name__ == '__main__':
    display_file()
