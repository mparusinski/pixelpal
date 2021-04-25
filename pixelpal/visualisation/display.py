import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from tqdm.auto import tqdm

from pixelpal.model.base import get_model, load_weights, augment


def display_file_or_dir(image_or_dir, augmentator=None, weights_file=None, horizontal_flip=False, vertical_flip=False, save_file_or_dir=None):
    if augmentator is None and weights_file:
        raise Exception("Option 'weights_file' only supported with a 'module_name' option")
    if augmentator:
        augmentation_model = get_model(augmentator)
        if weights_file:
            load_weights(augmentation_model, weights_file)
    else:
        augmentation_model = None
    
    if horizontal_flip and vertical_flip:
        def im_modifier(im):
            return np.flipud(np.fliplr(im))
    elif horizontal_flip:
        def im_modifier(im):
            return np.fliplr(im)
    elif vertical_flip:
        def im_modifier(im):
            return np.flipud(im)
    else:
        def im_modifier(im):
            return im

    if os.path.isdir(image_or_dir):
        if not os.path.isdir(save_file_or_dir):
            raise Exception(f"{save_file_or_dir} does not exist")
        display_image_folder(image_or_dir, augmentation_model, im_modifier, save_file_or_dir)
    elif os.path.isfile(image_or_dir):
        display_single_file(image_or_dir, augmentation_model, im_modifier, save_file_or_dir)
    else:
        raise Exception("Option 'image_or_dir' must be file or directory")

def display_image_folder(image_dir, augmentation_model=None, im_modifier=None, save_file_dir=None):
    im_files = [os.path.join(image_dir, im_file) for im_file in os.listdir(image_dir)]
    images = [im_modifier(mpimg.imread(im_file)) for im_file in im_files]

    if augmentation_model:
        images = augment(augmentation_model, images)

    for i, im_file in tqdm(enumerate(im_files), total=len(im_files)):
        plt.imshow(images[i])
        if save_file_dir:
            plt.savefig(os.path.join(save_file_dir, os.path.basename(im_file)))
        else:
            plt.show()
        plt.clf()

def display_single_file(image, augmentation_model=None, im_modifier=None, save_file=None):
    image = mpimg.imread(image)
    image = im_modifier(image)

    if augmentation_model:
        image = augment(augmentation_model, image)[0]
    plt.imshow(image)
    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()

