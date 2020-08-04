import numpy as np


def create_generator(X, y):
    x_augmented = []
    y_augmented = []
    for x_elem, y_elem in zip(X, y):
        x_augmented.append(x_elem)
        x_augmented.append(np.flipud(x_elem))
        x_augmented.append(np.fliplr(x_elem))
        x_augmented.append(np.flipud(np.fliplr(x_elem)))

        y_augmented.append(y_elem)
        y_augmented.append(np.flipud(y_elem))
        y_augmented.append(np.fliplr(y_elem))
        y_augmented.append(np.flipud(np.fliplr(y_elem)))
    return (np.array(x_augmented), np.array(y_augmented))
