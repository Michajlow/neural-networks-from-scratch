################################################################################
# File contains support classes and functions like optimizer and activation
# functions used in layers.py
################################################################################

import numpy as np
import sys

# progress func helps to control long loops and show the process bar (for
# instance batch processing during epoch)
def progress(count, total, title, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('%s -[%s] %s%s ...%s\r' % (title, bar, percents, '%',
                                                status))
    sys.stdout.flush()

# split examples on batches
def define_minibatches(m, mb_size, seed):
    np.random.seed(seed)
    res = np.array_split(np.random.permutation(m),
                         np.arange(0, m, mb_size).shape[0])
    return res

# generates array of given shape and method
def init_parameters(shape, method, bias = True):
    if method == 'he':
        w = np.random.randn(shape[0], shape[1])*np.sqrt(2/shape[1])
    if method == 'rand':
        w = np.random.randn(shape[0], shape[1])
    b = np.zeros((shape[0], 1)) if bias == True else None
    return (w, b)

# implementation of Adam optimizatin algorithm
class Adam():

    def __init__(self, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = 'adam'
        self.all_vs = []
        self.all_ss = []

    def add_initials(self, params):
        vs = [];ss = []
        for p in params:
            vs.append(np.zeros(p.shape))
            ss.append(np.zeros(p.shape))
        self.all_vs.append(vs)
        self.all_ss.append(ss)

    def update(self, params, grads, lr, layer_index, t):

        assert len(grads) == len(params)
        vs = self.all_vs[layer_index]
        ss = self.all_ss[layer_index]

        for i in range(len(params)):
            vs[i] = (self.beta1*vs[i])+((1-self.beta1)*grads[i])
            vc = vs[i]/(1-(self.beta1**t))
            ss[i] = (self.beta2*ss[i])+(((1-self.beta2)*grads[i]**2))
            sc = ss[i]/(1-(self.beta2**t))
            params[i] = (params[i]-(lr*(vc/(self.epsilon + np.sqrt(sc)))))

        self.all_vs[layer_index] = vs
        self.all_ss[layer_index] = ss

        return params

# cross entopy softmax loss
def cross_entropy_softmax(A,Y):
    assert (A.shape == Y.shape)
    shape = Y.shape
    A2d = A.reshape(shape[0], np.prod(shape[1:]))
    Y2d = Y.reshape(shape[0], np.prod(shape[1:]))
    res = -np.mean(np.sum(Y2d*np.log(A2d), axis=0))
    return res

# next activation functions presented as simple classes which contain forward
# propagation and backward propagation equations
class relu:
    @staticmethod
    def activate(Z):
        s = np.maximum(0,Z)
        return s
    @staticmethod
    def backprop(dA, Z):
        dZ = np.multiply(dA, np.int64(Z > 0))
        return dZ

class softmax:
    @staticmethod
    def activate(Z):
        exp_scores = np.exp(Z - np.max(Z))
        s = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
        return s
    @staticmethod
    def backprop(dA, Z):
        dZ = dA
        return dZ
    @staticmethod
    def compute_dA(A, Y):
        dA = np.copy(A)
        dA[Y.astype(np.bool)] -= 1
        return dA

class tanh:
    @staticmethod
    def activate(Z):
        return np.tanh(Z)
    @staticmethod
    def backprop(dA, Z, *args):
        p2 = (dA+sum(args))
        return (1-(np.tanh(Z)**2))*p2

class sigmoid:
    @staticmethod
    def activate(Z):
        return 1 / (1 + np.exp(-Z))
