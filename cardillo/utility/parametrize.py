import numpy as np


def parametrize(f):
    if callable(f):
        return f
    f = np.asarray(f)
    return lambda xi: np.broadcast_to(f, np.shape(xi) + f.shape)
