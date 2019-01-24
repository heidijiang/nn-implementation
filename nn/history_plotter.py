import matplotlib.pyplot as plt
from .cost_functions import multiclass_counting_cost
def plot(X,Y,weight,cost):
    plt.subplot(121)
    plt.plot(cost)
    plt.ylabel("cost")
    plt.title("cost history & misclassification rate")
    plt.subplot(122)
    count = [multiclass_counting_cost(X, Y, i)/len(Y) for i in weight]
    plt.plot(count)
    plt.xlabel("iters")
    plt.ylabel("misclassification rate")
    plt.show()
    return count[-1]