import numpy as np


def sigmoid(x):
    """sigmoid
    TODO: 
    Sigmoid function. Output = 1 / (1 + exp(-x)).
    :param x: input
    """
    #[TODO 1.1]
    return 1/(1+np.exp(-x))


def sigmoid_grad(a):
    """sigmoid_grad
    TODO:
    Compute gradient of sigmoid with respect to input. g'(x) = g(x)*(1-g(x))
    :param a: output of the sigmoid function
    """
    #[TODO 1.1]
    return sigmoid(a)*(1-sigmoid(a))



def reLU(x):
    """reLU
    TODO:
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    #[TODO 1.1]
    return np.maximum(x, 0)


def reLU_grad(a):
    """reLU_grad
    TODO:
    Compute gradient of ReLU with respect to input
    :param x: output of ReLU
    """
    #[TODO 1.1]
    a[a>0]=1
    a[a<=0]=0
    return a


def tanh(x):
    """tanh
    TODO:
    Tanh function.
    :param x: input
    """
    #[TODO 1.1]
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_grad(a):
    """tanh_grad
    TODO:
    Compute gradient for tanh w.r.t input
    :param a: output of tanh
    """
    #[TODO 1.1]
    return 1 - tanh(a)*tanh(a)


def softmax(x):
    """softmax
    TODO:
    Softmax function.
    :param x: input
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def softmax_minus_max(x):
    """softmax_minus_max
    TODO:
    Stable softmax function.
    :param x: input
    """
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=1, keepdims=True) 
