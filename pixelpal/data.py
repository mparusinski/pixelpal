import os
from tqdm import tqdm
import numpy as np
import matplotlib.image as mpimg

from pixelpal.utils import build_list_of_images_in_dir, fix_missing_alpha_channel


def load_data(folder, classes=['32x32', '64x64'], shapes=[(32, 32), (64, 64)]):
    files_per_classes = {
        x: list(build_list_of_images_in_dir(os.path.join(folder, x)))
        for x in classes
    }

    num_images = len(files_per_classes[classes[0]])
    X = np.empty((num_images, *shapes[0], 4))
    Y = np.empty((num_images, *shapes[1], 4))

    for i in tqdm(range(num_images)):
        lowres_file, highres_file = files_per_classes[classes[0]][i], files_per_classes[classes[1]][i]

        lowres_image = fix_missing_alpha_channel(mpimg.imread(lowres_file))
        hires_image = fix_missing_alpha_channel(mpimg.imread(highres_file))

        X[i, :, :, :] = lowres_image
        Y[i, :, :, :] = hires_image

    return X, Y
