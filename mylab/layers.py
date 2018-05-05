################################################################################
# File contains classes which represent layers of neural network as the main
# architecturel units: embedding layer, simple dence layer, RNN etc.
################################################################################

import mylab.utils as utils
import numpy as np

# Embedding layer represents the optimized version of simple dense layer in case
# of one-hot-vectors input. Instead of using high dimensional one-hot-vectors we
# pass to the layer inputs' indeces (for example words' indeces in given
# vocabluary) and the vector representaion of given dimensionality will be
# selected from the matrix hidden units x vocabluary size. The layer accepts
# numpy array of indeces as input with 2D shape (1, 'array length') and
# 3D shape (1, 'barch size', 'array length')

class embedding_layer():

    def __init__(self, hidden_units, vacab_size, init_method = 'he'):
        self.hidden_units = hidden_units
        self.vacab_size = vacab_size
        self.init_method = init_method

    @classmethod
    def check(cls):
        return 'layer'

    # initialize layer instance with parameters which will be trained. Method
    # outputs initialized parameters for Adam optimizer and shape for next layer
    # initialization
    def initialize(self, shape):

        vocabluary_size = shape[0]
        self.W, _ = utils.init_parameters((self.hidden_units,
                                           self.vacab_size),
                                           method = self.init_method,
                                           bias = False)
        output_shape = [self.hidden_units]
        output_shape.extend(shape[1:])
        return tuple(output_shape), (self.W, )

    # Method just performs parameters' update on given new parameters
    def update_parameters(self, new_parameters):
        self.W = new_parameters[0]

    # forward propagation
    def forward(self, X):
        shape = X.shape

        # Before the "selecting" input shape is reduces to 2D array (in case
        # of sequance models and 3D-shape input)
        Xr = np.copy(X).reshape(shape[0], np.prod(shape[1:]))
        A = self.W[:,Xr.T]
        A = A.reshape(np.append(self.hidden_units, shape[1:]))
        return A, Xr

    # backward propagation
    def backward(self, dA, cache):
        Xr = cache
        shape_da = dA.shape
        shape_x = Xr.shape
        dw = np.zeros(self.W.shape)
        dA = dA.reshape(shape_da[0], np.prod(shape_da[1:]))
        np.add.at(dw.T, Xr, dA.T)
        da_prev = None
        assert (dw.shape == self.W.shape)
        return da_prev, [dw], [self.W]

# simple feed-forward layer, requires activation function, accepts
# numpy array of indeces as input with 2D shape (hidden units num, 'array length')
#and 3D shape ('hidden units num', 'batch size', 'array length')
class layer():

    def __init__(self, hidden_units, activation_func, init_method = 'he'):
        self.hidden_units = hidden_units
        self.activation_func = activation_func
        self.init_method = init_method

    @classmethod
    def check(cls):
        return 'layer'

    def initialize(self, shape):

        hidden_units_prev = shape[0]
        self.W, self.b = utils.init_parameters((self.hidden_units,
                                                hidden_units_prev),
                                                method = self.init_method)
        output_shape = [self.hidden_units]
        output_shape.extend(shape[1:])
        return tuple(output_shape), (self.W, self.b)

    def update_parameters(self, new_parameters):
        self.W, self.b = new_parameters

    # forward propagation
    def forward(self, X):
        shape = X.shape

        # Before the dot product input shape is reduces to 2D array (in case
        # of sequance models, and 3D-shape input)
        Z = np.dot(self.W, (np.copy(X).reshape(shape[0],
                            np.prod(shape[1:]))))+self.b
        A = self.activation_func.activate(Z)
        A = A.reshape(np.append(self.hidden_units, shape[1:]))
        return A, (Z, X)

    # backward propagation
    def backward(self, dA, cache):
        Z, X = cache
        shape_da = dA.shape
        shape_x = X.shape
        dA = dA.reshape(shape_da[0], np.prod(shape_da[1:]))
        Xr = X.reshape(shape_x[0], np.prod(shape_x[1:]))

        dZ = self.activation_func.backprop(dA, Z)
        dw = np.dot(dZ, Xr.T)
        db = np.sum(dZ, axis=1, keepdims=True)

        da_prev = (self.W).T.dot(dZ).reshape(shape_x) # dA dimensional is
                                                      # returned to initial one

        assert (da_prev.shape == X.shape)
        assert (dw.shape == self.W.shape)
        assert (db.shape == self.b.shape)

        return da_prev, [dw, db], [self.W, self.b]


# Vanila RNN layer, requires activation function, accepts
# numpy array 3D shape ('hidden units num', 'batch size', 'array length')

class rnn_layer(layer):

    def initialize(self, shape):
        n_a_prev, m, Ta = shape
        hidden_units_prev = n_a_prev

        self.Wx, self.b = utils.init_parameters((self.hidden_units,
                                                 hidden_units_prev),
                                                 method = self.init_method)
        self.Wa, _ = utils.init_parameters((self.hidden_units,
                                            self.hidden_units),
                                            method = self.init_method,
                                            bias = False)
        return (self.hidden_units, m, Ta), (self.Wx, self.Wa, self.b)

    def update_parameters(self, new_parameters):
        self.Wx, self.Wa, self.b = new_parameters

    # forward propagation, iterate over timestamps and returns ALL hidden states
    def forward(self, X):
        assert len(X.shape)==3
        n_x, m, T = X.shape
        cache = []
        a_prev_t = np.zeros((self.hidden_units, m))
        A = np.zeros((self.hidden_units, m, T))
        for t in range(T):
            Zt = np.dot(self.Wx, X[:,:,t]) + np.dot(self.Wa, a_prev_t) + self.b
            a_next = self.activation_func.activate(Zt)
            cache.append((Zt, a_prev_t))
            a_prev_t = a_next
            A[:,:,t] = a_next
        return A, (X, cache)

    # backward propagation
    def backward(self, dA, caches):
        X, cache = caches
        n_a, m, T = dA.shape

        dwx = np.zeros((self.Wx.shape))
        dwa = np.zeros((self.Wa.shape))
        db = np.zeros((self.b.shape))

        dA_next = np.zeros(X.shape)
        dA_next_t = np.zeros((n_a, m))

        for t in reversed(range(T)):

            Zt, a_prev_t = cache[t]
            dZ = self.activation_func.backprop(dA[:,:,t], Zt, dA_next_t)

            dwx += np.dot(dZ, X[:,:,t].T)
            dwa += np.dot(dZ, a_prev_t.T)
            db  += np.sum(dZ, axis=1, keepdims=True)

            dA_next_t = np.dot(self.Wa.T, dZ)
            dA_next[:,:,t] = np.dot(self.Wx.T, dZ)
        return dA_next, [dwx, dwa, db], [self.Wx, self.Wa, self.b]

# Standart LSTM implementation, tanh function is used by default, accepts
# numpy array 3D shape ('hidden units num', 'batch size', 'array length')

class lstm_layer(layer):

    def __init__(self, hidden_units, init_method = 'he'):
        self.hidden_units = hidden_units
        self.init_method = init_method

    def initialize(self, shape, adam = True):
        n_a_prev, m, Ta = shape
        hidden_units_prev = n_a_prev

        self.Wf, self.bf = utils.init_parameters((self.hidden_units,
                                        self.hidden_units+hidden_units_prev),
                                        method = self.init_method)
        self.Wi, self.bi = utils.init_parameters((self.hidden_units,
                                        self.hidden_units+hidden_units_prev),
                                        method = self.init_method)
        self.Wc, self.bc = utils.init_parameters((self.hidden_units,
                                        self.hidden_units+hidden_units_prev),
                                        method = self.init_method)
        self.Wo, self.bo = utils.init_parameters((self.hidden_units,
                                        self.hidden_units+hidden_units_prev),
                                        method = self.init_method)

        # yeah, we need a loot of parameters in LSTM
        return (self.hidden_units, m, Ta), (self.Wf, self.bf, self.Wi, self.bi,
                                            self.Wc, self.bc, self.Wo, self.bo)


    def update_parameters(self, new_parameters):
        (self.Wf, self.bf, self.Wi, self.bi,
        self.Wc, self.bc, self.Wo, self.bo) = new_parameters


    # forward propagation, iterate over timestamps and returns ALL hidden states
    def forward(self, X):

        assert len(X.shape) == 3
        n_x, m, T = X.shape
        cache = []

        a_prev_t = np.random.randn(self.hidden_units, m)
        c_prev_t = np.zeros((self.hidden_units, m))

        A = np.zeros((self.hidden_units, m, T))

        for t in range(T):

            concat = np.concatenate((a_prev_t, X[:,:,t]), axis=0)

            Gft = utils.sigmoid.activate(np.dot(self.Wf, concat)+self.bf)
            Git = utils.sigmoid.activate(np.dot(self.Wi, concat)+self.bi)

            cct = utils.tanh.activate(np.dot(self.Wc, concat)+self.bc)
            c_next = Gft*c_prev_t+Git*cct

            Got = utils.sigmoid.activate(np.dot(self.Wo, concat)+self.bo)
            a_next = Got*utils.tanh.activate(c_next)

            cache.append((a_next, c_next, a_prev_t, c_prev_t,
                          Gft, Git, cct, Got))

            a_prev_t = a_next
            c_prev_t = c_next
            A[:,:,t] = a_next

        return A, (X, cache)


    def backward(self, dA, cache_main):

        X, cache = cache_main

        n_a, m, T = dA.shape

        dWf = np.zeros(self.Wf.shape)
        dbf = np.zeros(self.bf.shape)
        dWi = np.zeros(self.Wi.shape)
        dbi = np.zeros(self.bi.shape)
        dWo = np.zeros(self.Wo.shape)
        dbo = np.zeros(self.bo.shape)
        dWc = np.zeros(self.Wc.shape)
        dbc = np.zeros(self.bc.shape)

        dA_prev_layer = np.zeros(X.shape)

        dA_next_t = np.zeros((n_a, m))
        dC_next_t = np.zeros((n_a, m))

        for t in reversed(range(T)):

            dA_next_total = dA[:,:,t]+dA_next_t

            a_next_t, c_next_t, a_prev_t, c_prev_t, Gft, Git, cct, Got = cache[t]

            concat = np.concatenate((a_prev_t, X[:,:,t]))

            dGot = (dA_next_total*np.tanh(c_next_t)*Got)*(1-Got)

            dcct = ((dC_next_t*Git+Got*(1-(np.tanh(c_next_t)**2))*
                                                Git*dA_next_total)*(1-cct**2))

            dGit = ((dC_next_t*cct+Got*(1-(np.tanh(c_next_t)**2))*
                                                cct*dA_next_total)*Git*(1-Git))

            dGft = ((dC_next_t*c_prev_t+Got*(1-(np.tanh(c_next_t)**2))*
                                            c_prev_t*dA_next_total)*Gft*(1-Gft))

            dWf += np.dot(dGft, concat.T)
            dWi += np.dot(dGit, concat.T)
            dWc += np.dot(dcct, concat.T)
            dWo += np.dot(dGot, concat.T)
            dbf += np.sum(dGft, axis=1, keepdims=True)
            dbi += np.sum(dGit, axis=1, keepdims=True)
            dbc += np.sum(dcct, axis=1, keepdims=True)
            dbo += np.sum(dGot, axis=1, keepdims=True)

            dA_prev_t = (np.dot(self.Wf[:,:n_a].T, dGft)+
                         np.dot(self.Wi[:,:n_a].T, dGit)+
                         np.dot(self.Wc[:,:n_a].T, dcct)+
                         np.dot(self.Wo[:,:n_a].T, dGot))

            dC_prev_t = (dC_next_t*Gft+Got*(1-(np.tanh(c_next_t)**2))*
                                                            Gft*dA_next_total)


            dA_next_t = dA_prev_t
            dC_next_t = dC_prev_t

            dA_prev_layer[:,:,t] = (np.dot(self.Wf[:,n_a:].T, dGft)+
                                    np.dot(self.Wi[:,n_a:].T, dGit)+
                                    np.dot(self.Wc[:,n_a:].T, dcct)+
                                    np.dot(self.Wo[:,n_a:].T, dGot))

        return (dA_prev_layer, [dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo],
                                           [self.Wf, self.bf, self.Wi, self.bi,
                                            self.Wc, self.bc, self.Wo, self.bo])


# implementation of BRNN. Requires so-called 'cell' - in fact layer RNN class,
# which will be doubled as forward pass (takes from the past) and backward pass
# (takes from the future) and both passed are stacked togather. There isnt any
# interactiones between them, so forward and backward propagation are explicitly
# the same as in unidercation approach. The only one thing we are doing is
# manipulation with inputs and outputs of stacked layers.

class brnn():

    def __init__(self, cell):
        self.cf = cell # initialize forward pass cell
        self.cb = cell # initialize backward pass cell
        self.hidden_units = cell.hidden_units

    @classmethod
    def check(cls):
        return 'layer'

    def initialize(self, shape):
        (cf_hidden_units, m, Ta), params_f = self.cf.initialize(shape)
        (cb_hidden_units, m, Ta), params_b = self.cb.initialize(shape)
        self.n_parameters = len(params_f)
        assert len(params_f) == len(params_b)
        result_shape = (cf_hidden_units*2, m, Ta)
        result_params = list(params_f)+list(params_b)
        return result_shape, result_params

    def update_parameters(self, new_parameters):
        self.cf.update_parameters(new_parameters[:self.n_parameters])
        self.cb.update_parameters(new_parameters[self.n_parameters:])

    def forward(self, X):
        A_forward, cashes_forward = self.cf.forward(X)

        # Reverse input for backward pass layer
        X_reversed = np.copy(X[:,:,::-1])
        A_backward, cashes_backward = self.cb.forward(X_reversed)

        # Then concatenate results of forward and backward passes. Moreover,
        # the result from backward pass should be unrevered to represent the
        # same sequance of hidden units as result of forward pass layer

        A = np.concatenate((A_forward, A_backward[:,:,::-1]), axis=0)
        return A, (cashes_forward, cashes_backward)

    def backward(self, dA, caches):
        h = self.hidden_units
        cashes_f, cashes_b = caches

        # As the output of forward propagation was concatenated from outputs of
        # stacked layers, we need to share derivative error for backward
        # propagation between these stacked kayers as weel
        dA_f = np.copy(dA[:h,:,:])

        # pay attention we again reverse part of derivative error for backward
        # pass backpropagation. This is obligatory because we use cashe from
        # forward propagation as a list of cashes per timestamps, in order the
        # forward propagation was performed.
        dA_b = np.copy(dA[h:,:,::-1])
        da_next_f, grads_f, params_f = self.cf.backward(dA_f, cashes_f)
        da_next_b, grads_b, params_b = self.cb.backward(dA_b, cashes_b)

        # we add derivative error from both stacked layers. The output of b
        # backward pass should be reverseed again
        da_next = da_next_f+da_next_b[:,:,::-1]
        grads = grads_f+grads_b
        params = params_f+params_b

        return da_next, grads, params
