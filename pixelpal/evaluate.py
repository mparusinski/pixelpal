import os
import os.path
import matplotlib.image as mpimg
from itertools import islice

import numpy as np
import tensorflow as tf

from pixelpal.model.base import get_model, load_weights, augment
from pixelpal.utils import fix_missing_alpha_channel, build_list_of_images_in_dir


def batch_iterate(iterable, batch_size=32):
    i = iter(iterable)
    piece = list(islice(i, batch_size))
    while piece:
        yield piece
        piece = list(islice(i, batch_size))


def handle_batch(augmentation_model, batch, verbose):
    lowres_images_batch = []
    highres_images_batch = []
    for small_image_fp, large_image_fp in batch:
        lowres_images_batch.append(fix_missing_alpha_channel(mpimg.imread(small_image_fp)))
        highres_images_batch.append(fix_missing_alpha_channel(mpimg.imread(large_image_fp)))

    lowres_images_batch = np.array(lowres_images_batch)
    highres_images_batch = np.array(highres_images_batch)

    augmented_images = augment(augmentation_model, lowres_images_batch)

    for augmented_image, highres_image, batch_elem in zip(augmented_images, highres_images_batch, batch):
        small_image_fp, large_image_fp = batch_elem

        # Computing metrics
        psnr_metric = tf.image.psnr(highres_image, augmented_image, max_val=1.0).numpy()
        ssim_metric = tf.image.ssim(
            tf.convert_to_tensor(highres_image, dtype=tf.float32),
            tf.convert_to_tensor(augmented_image, dtype=tf.float32), max_val=1.0
        ).numpy()

        if verbose:
            print()
            print("Similarity metrics between {} after augmentation and target image {} are :".format(
                os.path.basename(small_image_fp), os.path.basename(large_image_fp))
            )
            print("\tPeak Signal to Noise Ratio: {}".format(psnr_metric))
            print("\tStructural Similarity     : {}".format(ssim_metric))

        yield psnr_metric, ssim_metric


def evaluate_augmentator(lowres_images, highres_images, augmentator, weights, verbose=True):
    if os.path.isdir(lowres_images) and os.path.isdir(highres_images):
        lowres_images = build_list_of_images_in_dir(lowres_images)
        highres_images = build_list_of_images_in_dir(highres_images)
    elif os.path.isfile(lowres_images) and os.path.isfile(highres_images):
        lowres_images = [lowres_images]
        highres_images = [highres_images]
    else:
        raise Exception("small_image and large_image need to be both either files or directories")

    accum_psnr_metric = []
    accum_ssim_metric = []

    augmentation_model = get_model(augmentator)
    if weights:
        load_weights(augmentation_model, weights)

    for batch in batch_iterate(zip(lowres_images, highres_images)):
        for psnr_metric, ssim_metric in handle_batch(augmentation_model, batch, verbose):
            accum_psnr_metric.append(psnr_metric)
            accum_ssim_metric.append(ssim_metric)

    print()
    print("Overall")
    print("\tAverage Peak Signal to Noise Ratio: {}".format(np.average(accum_psnr_metric)))
    print("\tAverage Structural Similarity     : {}".format(np.average(accum_ssim_metric)))
    print("\tStd-dev Peak Signal to Noise Ratio: {}".format(np.std(accum_psnr_metric)))
    print("\tStd-dev Structural Similarity     : {}".format(np.std(accum_ssim_metric)))

