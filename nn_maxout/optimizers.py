import autograd.numpy as np
from autograd import value_and_grad
from autograd import misc


def gradient_descent(g, w, alpha, max_its, beta, version):
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = misc.flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    w_hist = []
    w_hist.append(unflatten(w))
    cost_history = []

    # start gradient descent loop
    z = np.zeros((np.shape(w)))  # momentum term

    # over the line
    for k in range(max_its):
        # plug in value into func and derivative
        cost_eval, grad_eval = grad(w)
        grad_eval.shape = np.shape(w)

        ### normalized or unnormalized descent step? ###
        if version == 'normalized':
            grad_norm = np.linalg.norm(grad_eval)
            if grad_norm == 0:
                grad_norm += 10 ** -6 * np.sign(2 * np.random.rand(1) - 1)
            grad_eval /= grad_norm

        # take descent step with momentum
        z = beta * z + grad_eval
        w = w - alpha * z

        # record weight update
        w_hist.append(unflatten(w))
        cost_history.append(cost_eval)

    return w_hist, cost_history