import os
import importlib
import numpy as np
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Flatten, Activation, BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam

from pixelpal.utils import fix_missing_alpha_channel


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def build_discriminator(input_shape=(64, 64), channels=4, filters=64, kernel_size=3, dropout=0.5, num_conv_layers=4):
    print('Building discriminator')
    input_layer = Input((*input_shape, channels))

    previous_layer = input_layer
    for i in range(num_conv_layers):
        conv_layer = Conv2D(
            filters, kernel_size, strides=2, padding="same", activation=LeakyReLU(alpha=0.2)
        )(previous_layer)
        norm_layer = BatchNormalization()(conv_layer)
        previous_layer = Dropout(dropout)(norm_layer)

    flat_layer = Flatten()(previous_layer)
    dense_layer = Dense(1)(flat_layer)

    output_layer = Activation('sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


def learn_with_gan(model, train_data_gen, batch_size=32, iterations=4, callbacks=[], **kwargs):
    # Train one epoch in classical way
    model.fit(
        train_data_gen, epochs=1, callbacks=callbacks, batch_size=batch_size, **kwargs
    )

    discriminator = build_discriminator()
    adverserial_model = Sequential()
    adverserial_model.add(model)
    adverserial_model.add(discriminator)
    adverserial_model.layers[-1].trainable = False

    # TODO: This should model loss + gan loss
    adverserial_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    for i in range(iterations):
        for j in range(len(train_data_gen)):
            # Preparing batch data
            x, y_real = train_data_gen[j] # Input to SR algorithm and ground truth
            y_fake = model.predict(x, batch_size=batch_size)
 
            # Preparing discriminator data
            x_disc = np.concatenate((y_real, y_fake))
            y_disc = np.ones([2 * batch_size, 1])
            y_disc[batch_size:, :] = 0

            # Training the discriminator
            d_loss = discriminator.train_on_batch(x_disc, y_disc)
            print(f"Discriminator loss {d_loss}")

            # Training the adverserial model
            y_advs = np.ones([batch_size, 1])
            a_loss = adverserial_model.train_on_batch(x, y_advs)
            print(f"Adverserial loss {a_loss}")


def learn(model, train_data_gen, batch_size=32, epochs=10, callbacks=[], **kwargs):
    model.fit(
        train_data_gen, epochs=epochs, callbacks=callbacks, batch_size=batch_size, **kwargs
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
    if type(images) == list and len(images) >= 1:
        # Fix alpha channel
        images = [fix_missing_alpha_channel(image) for image in images]
        images_as_batch = np.array(images)
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

    prediction_batch = model.predict(images_as_batch, batch_size=128, verbose=1)
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
        return []
    data_augmentation = []
    for aug in augmentation:
        try:
            module = importlib.import_module(aug)
        except:
            raise Exception("Can't import {}\n\t(current working directory is {})".format(augmentation, os.getcwd()))
        data_augmentation_producer = getattr(module, 'create_generator')
        data_augmentation.append(data_augmentation_producer)
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

