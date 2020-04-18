import os

from pixelpal.model.base import get_model
from pixelpal.data import load_data


def train(module, dataset, weights, **kwargs):
    os.makedirs(os.path.dirname(weights), exist_ok=True)
    model = get_model(module)
    x_train, y_train = load_data(dataset)
    if 'validation_dataset' in kwargs:
        x_valid, y_valid = load_data(kwargs['validation_dataset'])
        model.learn(x_train, y_train, validation_data=(x_valid, y_valid))
    else:
        model.learn(x_train, y_train)
    model.save_weights(weights)


