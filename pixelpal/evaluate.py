import click
import os
import os.path
import matplotlib.image as mpimg

import numpy as np
import tensorflow as tf

from pixelpal.model.base import get_model
from pixelpal.utils import fix_missing_alpha_channel


def __build_list_of_images__(folder):
    for file_in_folder in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, file_in_folder)
        extension = file_in_folder.split('.')[-1]
        if os.path.isfile(full_path) and extension in ('png', 'jpg'):
            yield full_path


@click.command()
@click.argument('small_image')
@click.argument('large_image')
@click.argument('augmentator')
def display_file(small_image, large_image, augmentator):
    if os.path.isdir(small_image) and os.path.isdir(large_image):
        small_images = __build_list_of_images__(small_image)
        large_images = __build_list_of_images__(large_image)
    elif os.path.isfile(small_image) and os.path.isfile(large_image):
        small_images = [small_image]
        large_images = [large_image]
    else:
        raise Exception("small_image and large_image need to be both either files or directories")

    accum_psnr_metric = []
    accum_ssim_metric = []

    for small_image_fp, large_image_fp in zip(small_images, large_images):
        small_image = fix_missing_alpha_channel(mpimg.imread(small_image_fp))
        large_image = fix_missing_alpha_channel(mpimg.imread(large_image_fp))
        augmentation_model = get_model(augmentator)
        augmented_image = augmentation_model.augment(small_image)[0]

        # Computing metrics
        psnr_metric = tf.image.psnr(large_image, augmented_image, max_val=1.0).numpy()
        ssim_metric = tf.image.ssim(
            tf.convert_to_tensor(large_image, dtype=tf.float32),
            tf.convert_to_tensor(augmented_image, dtype=tf.float32), max_val=1.0
        ).numpy()

        print()
        print("Similarity metrics between {} after augmentation and target image {} are :".format(
            os.path.basename(small_image_fp), os.path.basename(large_image_fp))
        )
        print("\tPeak Signal to Noise Ratio: {}".format(psnr_metric))
        print("\tStructural Similarity     : {}".format(ssim_metric))

        accum_psnr_metric.append(psnr_metric)
        accum_ssim_metric.append(ssim_metric)

    print()
    print("Overall")
    print("\tAverage Peak Signal to Noise Ratio: {}".format(np.average(accum_ssim_metric)))
    print("\tAverage Structural Similarity     : {}".format(np.average(accum_psnr_metric)))
    print("\tStd-dev Peak Signal to Noise Ratio: {}".format(np.std(accum_psnr_metric)))
    print("\tStd-dev Structural Similarity     : {}".format(np.std(accum_ssim_metric)))


if __name__ == '__main__':
    display_file()
