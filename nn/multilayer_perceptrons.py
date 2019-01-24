import autograd.numpy as np
from .normalizers import standard_normalize


# an example activation function - tanh
def activation(v):
    return np.tanh(v)
    # return np.maximum(0,v)


# a feature_transforms function for computing
# U_L L layer perceptron units efficiently
def feature_transforms(a, w):
    # loop through each layer matrix
    for W in w:
        # compute inner product with current layer weights
        a = W[0] + np.dot(a.T, W[1:])

        # output of layer activation
        a = standard_normalize(activation(a.T))
    return a
