import numpy as np


def yield_rgb_augmentation(elem):
    # elem is of size N x N x C where C is either 4
    # possible permutations are rgb, rbg, grb, gbr, brg, bgr
    elem_rgb = elem
    elem_rbg = elem[:, :, [0,2,1,4]]
    elem_grb = elem[:, :, [1,0,2,4]]
    elem_gbr = elem[:, :, [1,2,0,4]]
    elem_brg = elem[:, :, [2,0,1,4]]
    elem_bgr = elem[:, :, [2,1,0,4]]

    return [elem_rgb, elem_rbg, elem_gbr, elem_gbr, elem_brg, elem_bgr]


def create_generator(X, y):
    x_augmented = []
    y_augmented = []
    for x_elem, y_elem in zip(X, y):
        x_augmented += yield_rgb_augmentation(x_elem)
        y_augmented += yield_rgb_augmentation(y_elem)
    return (np.array(x_augmented), np.array(y_augmented))


__AUGMENTORS__ = {
    0: lambda elem: elem
    1: lambda elem: elem[:, :, [0,2,1,4]]
    2: lambda elem: elem[:, :, [1,0,2,4]]
    3: lambda elem: elem[:, :, [1,2,0,4]]
    4: lambda elem: elem[:, :, [2,0,1,4]]
    5: lambda elem: elem[:, :, [2,1,0,4]]
}

def create_generator(x, y):
    idx = np.random.randint(0, 6)
    return __AUGMENTORS__[idx](x), __AUGMENTORS__[idx](y)
