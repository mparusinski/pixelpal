import numpy as np


__AUGMENTORS__ = {
    0: lambda elem: elem,
    1: lambda elem: np.rot90(elem),
    2: lambda elem: np.rot90(elem, 2),
    3: lambda elem: np.rot90(elem, 3),
    4: lambda elem: np.fliplr(np.flipud(elem)),
    5: lambda elem: np.rot90(np.fliplr(np.flipud(elem))),
    6: lambda elem: np.rot90(np.fliplr(np.flipud(elem)), 2),
    7: lambda elem: np.rot90(np.fliplr(np.flipud(elem)), 3),
}

def create_generator(x, y):
    idx = np.random.randint(0,  8)
    return __AUGMENTORS__[idx](x), __AUGMENTORS__[idx](y)
