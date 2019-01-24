import autograd.numpy as np
from .multilayer_perceptrons import feature_transforms

def model(x, w):
    # feature transformation
    f = feature_transforms(x, w[0])

    # compute linear combination and return
    a = w[1][0] + np.dot(f.T, w[1][1:])
    return a.T

def least_squares(x,y,w):
    return np.sum((model(x, w) - y) ** 2) / float(np.size(y))

def twoclass_softmax(x,y,w):
    cost = np.sum(np.log(1 + np.exp(-y * model(x, w))))
    return cost / float(np.size(y))

def multiclass_softmax(X,Y,w,lam):
    # pre-compute predictions on all points
    all_evals = model(X, w)

    # compute softmax across data points
    a = np.log(np.sum(np.exp(all_evals), axis=0))

    # compute cost in compact form using numpy broadcasting
    b = all_evals[Y.astype(int).flatten(), np.arange(np.size(Y))]
    cost = np.sum(a - b)

    # add regularizer
    # cost = cost + lam * np.linalg.norm(w[1:, :], 'fro') ** 2

    # return average
    return cost / float(np.size(Y))

def multiclass_counting_cost(x,y,w):
    # pre-compute predictions on all points
    all_evals = model(x,w)

    # compute predictions of each input point
    y_predict = (np.argmax(all_evals,axis = 0))[np.newaxis,:]

    if np.size(y_predict,1)>np.size(y_predict,0):
        y_predict = y_predict.T
    if np.size(y,1)>np.size(y,0):
        y = y.T
    # compare predicted label to actual label
    count = np.sum(np.abs(np.sign(y - y_predict)))

    # return number of misclassifications
    return count