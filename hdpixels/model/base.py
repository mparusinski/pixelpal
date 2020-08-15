import os
import importlib
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

from hdpixels.utils import fix_missing_alpha_channel


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def learn(model, x_data, y_data, batch_size=32, epochs=10, callbacks=[], data_augmentation=None, **kwargs):
    if data_augmentation:      
        x_data, y_data = data_augmentation(x_data, y_data)
    model.fit(
        x_data, y_data, epochs=epochs, callbacks=callbacks, batch_size=batch_size, **kwargs
    )


def save_weights(sk_model, weights_file):
    extension = weights_file.split('.')[-1]
    if extension == 'h5':
        sk_model.save(weights_file)
    else:
        raise Exception("Unsupported weights format {}".format(extension))


def load_weights(sk_model, weights_file):
    extension = weights_file.split('.')[-1]
    if extension == 'h5':
        sk_model.load_weights(weights_file)
    else:
        raise Exception("Unsupported weights format {}".format(extension))


def augment(model, images):
    if type(images) == list and len(images) == 1:
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
            images_as_batch = images
    elif type(images) == np.ndarray and len(images.shape) == 3:
        images = fix_missing_alpha_channel(images)
        images_as_batch = images.reshape((1, *images.shape))
    else:
        raise Exception("Unable to augment {}".format(images))

    prediction_batch = model.predict(images_as_batch)
    return [np.array(x) for x in prediction_batch.tolist()]


def get_model(module_name, **kwargs):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise Exception("Can't import {}\n\t(current working directory is {})".format(module_name, os.getcwd()))
    modules_tokens = module_name.split('.')
    model_creator = getattr(module, 'create_model')
    return model_creator()


def get_data_augmentation(augmentation, **kwargs):
    if augmentation is None:
        return None
    try:
        module = importlib.import_module(augmentation)
    except:
        raise Exception("Can't import {}\n\t(current working directory is {})".format(augmentation, os.getcwd()))
    data_augmentation_producer = getattr(module, 'create_generator')
    return data_augmentation_producer 


def get_callbacks(callbacks, **kwargs):
    if callbacks is None:
        return []
    keras_callbacks = []
    for cb in callbacks:
        try:
            module = importlib.import_module(cb)
        except ModuleNotFoundError:
            raise Exception("Can't import {}\n\t(current working directory is {})".format(cb, os.getcwd()))
        callback_producer = getattr(module, 'produce_callback')
        keras_callbacks.append(callback_producer())
    return keras_callbacks

