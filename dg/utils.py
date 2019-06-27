from . import core

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


