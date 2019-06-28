from . import core
import numpy as np

# check shape match with broadcasting aware
# only allow the first dimension to be different
# both x and y are Operators
def check_shape_match(x, y):
    if x.data().shape[1:] == y.data().shape \
       or x.data().shape == y.data().shape[1:] \
       or x.data().shape == y.data().shape:
        pass
    else:
        raise core.ShapeNotMatchError('Shape does not match: {}, {}'.format(x.data().shape,\
                                                                            y.data().shape))

def check_shape_exact_match(x, y):
    if x.data().shape != y.data().shape:
        raise core.ShapeNotMatchError('Not exact match: {}, {}'.format(x.data().shape,\
                                                                       y.data().shape))

def sigmoid(x):
    return 1. / (1. + np.exp(x))

# logits are subtracted by the max element to
# avoid blow-up
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

# assume the first axis of x is batch size.
# is there a more optimized way?
def batch_softmax(x):
    return np.array([softmax(x) for m in x])
