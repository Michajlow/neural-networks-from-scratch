################################################################################
# File contains support classes and functions like optimizer and activation
# functions used in layers.py
################################################################################

import mylab.utils as utils
import numpy as np

# class nnet presents the neural network model, which in fact takes layers with
# .add method and performs computation logic - forward propagation and backward
# propagation per layer, compute gradients and update parameters

class nnet():

    def __init__(self):
        self._layers = []
        self.parameters = []
        self.optimiziation = utils.Adam()
        self.report_function = None
        self.report_values = []

    def add(self, layer):
        if layer.check() == 'layer':
            self._layers.append(layer)
        else:
            raise Exception('Unacceptable object')

    # method train returns list with costs per epoch and report values if such
    # are specified by report function
    def train(self, X, Y, lr = 0.01, # learinig rate
                num_epoch = 100, # number of epochs
                mb = 64, # batch size
                plot = True, # plot the lost change if true (matplotlib required)
                epoch_bar = False, # show the epoch progress if true
                clip_value = None, # clip value between -n and +n if n is passed
                save_loss = 1): # append epoch cost to the returned list of
                                # per given number

        self.costs = []

        # temporary solution, to be changed with different optmization
        # algorithms
        if self.optimiziation.name == 'adam':
            adam = True
        else:
            adam = False
        if len(self._layers)==0:
            raise Exception('Layers were not added')

        # initialize parameters and dimensionality checking
        # could be preplaced or expanded with more comprehensive initialization
        # if needed.
        p_shape = X.shape
        print('Dimensional checking...\n')
        print('input shape - {}\n'.format(p_shape))
        for n, l in enumerate(self._layers):
            next_shape, params = l.initialize(p_shape)
            if adam:
                self.optimiziation.add_initials(params)
            p_shape = next_shape
            print('layer {0} output shape - '.format(n), p_shape)
        print('Parameters are initialized\n')
        seed = 10
        t = 0
        for i in range(num_epoch):
            epoch_cost = 0
            seed += 1
            minibatches_indeces = utils.define_minibatches(p_shape[1], mb, seed)
            len_mbs = len(minibatches_indeces)
            for i_mb, indeces in enumerate(minibatches_indeces, 1):
                mb_x = X[:,indeces]
                mb_y = Y[:,indeces]
                caches = []
                A_prev = mb_x
                for n, l in enumerate(self._layers):
                    A, cache = l.forward(A_prev)
                    A_prev = A
                    caches.append(cache)

                # compute dA (takes the activation function of the last
                # layer)
                dA = self._layers[n].activation_func.compute_dA(A, mb_y)

                #find current cost for minibatch
                epoch_cost += utils.cross_entropy_softmax(A, mb_y)/len_mbs

                #backward propagation and parameters update
                t += 1
                for n,l in reversed(list(enumerate(self._layers))):
                    dA_prev, grads, params = l.backward(dA, caches[n])
                    dA = dA_prev

                    #clip gradient before optmization in a given min-max frames
                    ###########################################################
                    if clip_value:
                        for gradient in grads:
                            np.clip(gradient, a_min = -clip_value,
                                    a_max = clip_value, out = gradient)
                    ###########################################################

                    l.update_parameters(self.optimiziation.update(params, grads,
                                                                  lr, n, t))
                if epoch_bar:
                    utils.progress(i_mb, len(minibatches_indeces),
                                    title = 'epoch {}'.format(i),
                                    status = epoch_cost)

            # the report funciton could be specified before training and will
            # be executed every epoch. The function takes 2 arguments:
            # number of epoch and epoch cost. The example of such funciton
            # could be found in jupyther notebook with use cases.
            if self.report_function:
                res = self.report_function(self, i, epoch_cost) #args
                if res:
                    self.report_values.append(res)

            if i % save_loss == 0:
                self.costs.append(epoch_cost)
        if plot:
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                plt.plot(self.costs)
                plt.ylabel('cost')
                plt.xlabel('Iterations (per '+save_loss+')')
                plt.title("Learning rate =" + str(lr))
                plt.show()
            except:
                print('matplotlib wasnt found, plot was skipped')
        return [self.report_values, self.costs]

    def predict(self, X_test):
        A_prev = X_test
        for n, l in enumerate(self._layers):
            A, cache = l.forward(A_prev)
            A_prev = A
        return A
