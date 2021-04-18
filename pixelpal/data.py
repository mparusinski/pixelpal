import os
from tqdm import tqdm
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.utils import Sequence

from pixelpal.utils import build_list_of_images_in_dir, fix_missing_alpha_channel


class DataGenerator(Sequence):

    def __init__(self, folder, classes=['32x32', '64x64'], shapes=[(32, 32), (64, 64)], shuffle=True, batch_size=32, augmentations=[]):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.classes = classes
        self.shapes = shapes
        self.augmentations = []
        self.files_per_classes = {
            x: list(build_list_of_images_in_dir(os.path.join(folder, x)))
            for x in self.classes
        }

        self.on_epoch_end()

    def __augment__(self, x_elem, y_elem):
        x_ag_elem, y_ag_elem = x_elem, y_elem
        for ag in self.augmentations:
            x_ag_elem, y_ag_elem = ag(x_ag_elem, y_ag_elem)
        return x_ag_elem, y_ag_elem

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        chosen_indices = self.indexes[index * self.batch_size:(index + 1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in chosen_indices]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        num_images = len(self.files_per_classes[self.classes[0]])
        self.indexes = np.arange(num_images)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.shapes[0], 4))
        Y = np.empty((self.batch_size, *self.shapes[1], 4))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            lowres_file, highres_file = self.files_per_classes[self.classes[0]][ID], self.files_per_classes[self.classes[1]][ID]

            lowres_image = fix_missing_alpha_channel(mpimg.imread(lowres_file))
            hires_image = fix_missing_alpha_channel(mpimg.imread(highres_file))

            lowres_image, hires_image = self.__augment__(lowres_image, hires_image)

            X[i, :, :, :] = lowres_image
            Y[i, :, :, :] = hires_image

        return X, Y


def load_data(folder, classes=['32x32', '64x64'], shapes=[(32, 32), (64, 64)], shuffle=True, batch_size=32, augmentations=[]):
    return DataGenerator(folder, classes, shapes, shuffle, batch_size, augmentations)


# def load_data(folder, classes=['32x32', '64x64'], shapes=[(32, 32), (64, 64)], shuffle=False):
#     files_per_classes = {
#         x: list(build_list_of_images_in_dir(os.path.join(folder, x)))
#         for x in classes
#     }

#     num_images = len(files_per_classes[classes[0]])
#     X = np.empty((num_images, *shapes[0], 4))
#     Y = np.empty((num_images, *shapes[1], 4))

#     for i in tqdm(range(num_images)):
#         lowres_file, highres_file = files_per_classes[classes[0]][i], files_per_classes[classes[1]][i]

#         lowres_image = fix_missing_alpha_channel(mpimg.imread(lowres_file))
#         hires_image = fix_missing_alpha_channel(mpimg.imread(highres_file))

#         X[i, :, :, :] = lowres_image
#         Y[i, :, :, :] = hires_image

#     if shuffle:
#         random_indices = np.random.permutation(X.shape[-1])
#         return X[:, :, :, random_indices], Y[:, :, :, random_indices]
#     else:
#         return X, Y
