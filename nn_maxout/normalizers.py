import autograd.numpy as np

# standard normalization function - input data, output standaard normalized version
def standard_normalize(data):
    # compute the mean and standard deviation of the input
    data_means = np.mean(data,axis = 1)[:,np.newaxis]
    data_stds = np.std(data,axis = 1)[:,np.newaxis]

    # check to make sure thta x_stds > small threshold, for those not
    # divide by 1 instead of original standard deviation
    ind = np.argwhere(data_stds < 10**(-2))
    if len(ind) > 0:
        ind = [v[0] for v in ind]
        adjust = np.zeros((data_stds.shape))
        adjust[ind] = 1.0
        data_stds += adjust

    # return standard normalized data
    return (data - data_means)/data_stds