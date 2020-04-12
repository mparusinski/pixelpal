import os
import numpy as np


def fix_missing_alpha_channel(image):
    if image.shape[-1] == 3:  # missing alpha channel
        new_image = np.empty((*image.shape[0:2], 4))
        new_image[:, :, 0:3] = image
        new_image[:, :, 3] = 1.0
        return new_image
    else:
        return image


def build_list_of_images_in_dir(folder):
    for file_in_folder in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, file_in_folder)
        extension = file_in_folder.split('.')[-1]
        if os.path.isfile(full_path) and extension in ('png', 'jpg'):
            yield full_path
