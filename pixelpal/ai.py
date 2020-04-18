import os

from pixelpal.model.base import get_model
from pixelpal.data import load_data


def train(module, dataset, weights, **kwargs):
    os.makedirs(os.path.dirname(weights), exist_ok=True)
    model = get_model(module)
    x, y = load_data(dataset)
    model.learn(x, y)
    model.save_weights(weights)


