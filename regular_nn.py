import numpy as np
import nn as DL

datapath = 'datasets/'
data = np.loadtxt(datapath + '3eggs_multiclass.csv', delimiter = ',')
X = DL.normalizers.standard_normalize(data[:-1,:])
Y = data[-1,:][np.newaxis,:].T

# create initial weights for arbitrary feedforward network
def initialize_network_weights(layer_sizes,scale):
    # container for entire weight tensor
    weights = []

    # loop over desired layer sizes and create appropriately sized initial
    # weight matrix for each layer
    for k in range(len(layer_sizes) - 1):
        # get layer sizes for current weight matrix
        U_k = layer_sizes[k]
        U_k_plus_1 = layer_sizes[k + 1]

        # make weight matrix
        weight = scale * np.random.randn(U_k + 1, U_k_plus_1)
        weights.append(weight)

    # re-express weights so that w_init[0] = omega_inner contains all
    # internal weight matrices, and w_init = w contains weights of
    # final linear combination in predict function
    w_init = [weights[:-1], weights[-1]]

    return w_init


def run_gradient(X, Y, w,a,b):

    weight, cost = DL.optimizers.gradient_descent(lambda w: DL.cost_functions.multiclass_softmax(X,Y,w,0.005), w,a,200,b,'')

    return weight, cost


def define_layers(N,M,num_hidden,units):
    return [N] + [units] * num_hidden + [M]


# An example 4 hidden layer network, with 10 units in each layer
N = np.size(X,0)  # dimension of input
M = np.size(np.unique(Y))  # dimension of output
# U_1 = 50; U_2 = 50; U_3 = 50; U_4 = 200; U_5=3 # number of units per hidden layer
layer_sizes = [N,30,20,10,M]  #define_layers(N,M,2,20)
# the list defines our network architecture

w = initialize_network_weights(layer_sizes, scale=0.5)
w,c = run_gradient(X,Y,w,.1,0)
count = DL.history_plotter.plot(X, Y, w, c)
print("The misclassification rate is %f" % (count))
