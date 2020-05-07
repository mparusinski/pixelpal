import os

from hdpixels.model.base import get_model, get_callbacks, get_data_augmentation
from hdpixels.data import load_data


def train(module, dataset, weights, **kwargs):
    os.makedirs(os.path.dirname(weights), exist_ok=True)
    model = get_model(module)
    callbacks = get_callbacks(kwargs.get('callbacks', []))
    data_augmentation = get_data_augmentation(kwargs.get('data_augmentation', None))
    x_train, y_train = load_data(dataset)
    validation_data = None
    if 'validation_dataset' in kwargs:
        x_valid, y_valid = load_data(kwargs['validation_dataset'])
        validation_data = (x_valid, y_valid)
    model.learn(
        x_train, y_train, validation_data=validation_data, callbacks=callbacks, data_augmentation=data_augmentation
    )
    model.save_weights(weights)


