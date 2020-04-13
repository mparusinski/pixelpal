from abc import abstractmethod
import os
import importlib
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from pixelpal.utils import fix_missing_alpha_channel


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


class AbstractAugmentor(object):

    def __init__(self, **kwargs):
        self.build_model(**kwargs)
        self.model.compile(
            optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
            metrics=[psnr_metric, ssim_metric]
        )

    @abstractmethod
    def build_model(self, input_shape=(32, 32), channels=4, **kwargs):
        self.model = None

    def learn(self, x_data, y_data, batch_size=32):
        self.model.fit(x_data, y_data, batch_size=batch_size)

    def save_weights(self, weights_file):
        extension = weights_file.split('.')[-1]
        if extension == 'h5':
            self.model.save_weights(weights_file)
        else:
            raise Exception("Unsupported weights format {}".format(extension))

    def load_weights(self, weights_file):
        extension = weights_file.split('.')[-1]
        if extension == 'h5':
            self.model.load_weights(weights_file)
        else:
            raise Exception("Unsupported weights format {}".format(extension))

    def augment(self, images):
        if type(images) == list and len(images) >= 1:
            # Fix alpha channel
            images = [fix_missing_alpha_channel(image) for image in images]
            image_as_batch = np.array(images)
        elif type(images) == list and len(images) == 0:
            raise Exception("Unable to augment empty list")
        elif type(images) == np.ndarray and len(images.shape) == 4:
            if images.shape[-1] == 3:  # Missing alpha channel
                images_as_batch = np.empty(images.shape[0:3] + (4,))
                for i in range(images.shape[0]):
                    images_as_batch[i, :, :, :] = fix_missing_alpha_channel(
                        images[i, :, :, :].reshape(images.shape[1:])
                    )
            else:
                image_as_batch = images
        elif type(images) == np.ndarray and len(images.shape) == 3:
            images = fix_missing_alpha_channel(images)
            images_as_batch = images.reshape((1, *images.shape))
        else:
            raise Exception("Unable to augment {}".format(images))

        prediction_batch = self.model.predict(images_as_batch)
        return [np.array(x) for x in prediction_batch.tolist()]


def __get_model_name__(module_name):
    module_name_tokens = module_name.split('_')
    return ''.join(token.capitalize() for token in module_name_tokens)


def get_model(module_name, **kwargs):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise Exception("Can't import {}\n\t(current working directory is {})".format(module_name, os.getcwd()))
    modules_tokens = module_name.split('.')
    model_class_name = __get_model_name__(modules_tokens[-1])
    model_class = getattr(module, model_class_name)
    return model_class(**kwargs)
