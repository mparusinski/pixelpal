import numpy as np


def fix_missing_alpha_channel(image):
    if image.shape[-1] == 3:  # missing alpha channel
        new_image = np.empty((*image.shape[0:2], 4))
        new_image[:, :, 0:3] = image
        new_image[:, :, 3] = 1.0
        return new_image
    else:
        return image