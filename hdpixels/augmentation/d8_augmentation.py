import numpy as np


def yield_d8(elem):
    elem_id = elem
    elem_r1 = np.rot90(elem)
    elem_r2 = np.rot90(elem, 2)
    elem_r3 = np.rot90(elem, 3)

    elem_d = np.fliplr(np.flipud(elem))
    elem_r1_d = np.rot90(elem_d)
    elem_r2_d = np.rot90(elem_d, 2)
    elem_r3_d = np.rot90(elem_d, 3)

    return [elem_id, elem_r1, elem_r2, elem_r3, elem_d, elem_r1_d, elem_r2_d, elem_r3_d]


def create_generator(X, y):
    x_augmented = []
    y_augmented = []
    for x_elem, y_elem in zip(X, y):
        x_augmented += yield_d8(x_elem)
        y_augmented += yield_d8(y_elem)
    return (np.array(x_augmented), np.array(y_augmented))
