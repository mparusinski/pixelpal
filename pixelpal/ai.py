import os

from hdpixels.model.base import learn, get_model, save_weights, get_callbacks, get_data_augmentation
from hdpixels.data import load_data


def train(module, dataset, weights, **kwargs):
    os.makedirs(os.path.dirname(weights), exist_ok=True)
    model = get_model(module)
    callbacks = get_callbacks(kwargs.get('callbacks', []))
    data_augmentation = get_data_augmentation(kwargs.get('data_augmentation', []))
    train_data_gen= load_data(dataset, augmentations=data_augmentation)
    valid_data_gen = None
    if 'validation_dataset' in kwargs:
        valid_data_gen = load_data(kwargs['validation_dataset'])
    learn(
        model, train_data_gen, validation_data=valid_data_gen, callbacks=callbacks
    )
    save_weights(model, weights)


