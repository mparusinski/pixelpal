import os
import os.path
import matplotlib.image as mpimg

import numpy as np
import tensorflow as tf

from hdpixels.model.base import get_model
from hdpixels.utils import fix_missing_alpha_channel, build_list_of_images_in_dir


def evaluate_augmentator(lowres_images, highres_images, augmentator, weights):
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
        augmentation_model.load_weights(weights)

    for small_image_fp, large_image_fp in zip(lowres_images, highres_images):
        lowres_images = fix_missing_alpha_channel(mpimg.imread(small_image_fp))
        highres_images = fix_missing_alpha_channel(mpimg.imread(large_image_fp))
        augmented_image = augmentation_model.augment(lowres_images)[0]

        # Computing metrics
        psnr_metric = tf.image.psnr(highres_images, augmented_image, max_val=1.0).numpy()
        ssim_metric = tf.image.ssim(
            tf.convert_to_tensor(highres_images, dtype=tf.float32),
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
    print("\tAverage Peak Signal to Noise Ratio: {}".format(np.average(accum_psnr_metric)))
    print("\tAverage Structural Similarity     : {}".format(np.average(accum_ssim_metric)))
    print("\tStd-dev Peak Signal to Noise Ratio: {}".format(np.std(accum_psnr_metric)))
    print("\tStd-dev Structural Similarity     : {}".format(np.std(accum_ssim_metric)))

