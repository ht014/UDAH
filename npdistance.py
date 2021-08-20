import numpy as np
import sys
 

def norm(x, keepdims=False):
    '''
    Param:
        x: matrix of shape (n1, n2, ..., nk)
        keepdims: Whether keep dims or not
    Return: norm of matrix of shape (n1, n2, ..., n_{k-1})
    '''
    return np.sqrt(np.sum(np.square(x), axis=-1, keepdims=keepdims))

def normed(x):
    '''
    Param: matrix of shape (n1, n2, ..., nk)
    Return: normed matrix
    '''
    return x / (1e-20 + norm(x, keepdims=True))

def euclidean2(x1, x2):
    return np.sum(np.square(x1 - x2), axis=-1)

def euclidean(x1, x2):
    return np.sqrt(euclidean2(x1, x2))

def averaged_euclidean2(x1, x2):
    return np.mean(np.square(x1 - x2), axis=-1)

def averaged_euclidean(x1, x2):
    return np.sqrt(averaged_euclidean2(x1, x2))

def normed_euclidean2(x1, x2):
    return euclidean2(normed(x1), normed(x2))

def inner_product(x1, x2, pair=False):
    if pair:
        return - np.inner(x1, x2)
    else:
        return - np.sum(x1 * x2, axis=-1)

def cosine(x1, x2):
    return (1 + inner_product(normed(x1), normed(x2))) / 2

def distance(x1, x2=None, pair=True, dist_type="euclidean2", ifsign=False):
    '''
    Param:
        x2: if x2 is None, distance between x1 and x1 will be returned.
        pair: if True, for i, j, x1_i, x2_j will be calculated
              if False, for i, x1_i, x2_i will be calculated, and it requires the dimension of x1 and x2 is same.
        dist_type: distance type, can be euclidean2, normed_euclidean2, inner_product, cosine
    '''
    if x2 is None:
        x2 = x1
    if ifsign:
        x1 = util.sign(x1)
        x2 = util.sign(x2)
    if dist_type == 'inner_product':
        return inner_product(x1, x2, pair)
    if pair:
        x1 = np.expand_dims(x1, 1)
        x2 = np.expand_dims(x2, 0)
    return getattr(sys.modules[__name__], dist_type)(x1, x2)
