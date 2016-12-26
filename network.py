import time

import numpy as np
import theano
import theano.tensor as T
import theano.typed_list
import lasagne

import process_data

DEBUG = True
theano.config.optimizer = 'fast_compile'

class Network(object):

    def __init__(self, n_h_layers, n_h_units, vocab_size, seq_length,
                 drop_p=0.5, grad_clip=5):
        # network parameters
        self.n_h_layers = n_h_layers
        self.n_h_units = n_h_units
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.drop_p = drop_p
        self.grad_clip = grad_clip

        self.x = T.tensor3('x')
        self.y = T.tensor3('y')

        # symbolic variables for the outputs of GRU layers
        self.gru_sym_inits = theano.typed_list.TypedListType(
            T.TensorType(theano.config.floatX, (False, False)))()
        [self.gru_sym_inits.append(T.matrix("gru_%d" % i)) for i in xrange(n_h_layers)]

        # ----------------------------------
        if DEBUG:
            theano.config.compute_test_value = 'warn'

            tr_data = process_data.vec_data
            batch_size = 50

            gen_tr_batch = process_data.gen_batches(
                tr_data, batch_size, self.seq_length)

            # initialize gru hidden states to 0
            gru_inits = [np.zeros((batch_size, self.n_h_units))
                         for _ in xrange(self.n_h_layers)]

            x, y = next(gen_tr_batch)

            self.x.tag.test_value = x
            self.y.tag.test_value = y
            self.gru_sym_inits.tag.test_value = gru_inits
        # ----------------------------------

        self.gru_layers, self.network = self._build_network()

    def train(self, data, batch_size, eta, n_epochs, tf):
        r = lasagne.layers.get_output(self.gru_layers + [self.network], self.x)
        gru_outs, pred = r[:-1], r[-1]

        loss = lasagne.objectives.categorical_crossentropy(pred, self.y).mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=eta)

        # compile functions
        train_fn = theano.function([self.x, self.y, self.gru_sym_inits],
                                   [loss] + gru_outs, updates=updates)

        # do the actual training
        tr_data = process_data.vec_data

        print("Training...")
        for epoch_n in xrange(n_epochs):
            start_time = time.time()

            # full pass over the training data
            train_err = 0
            gen_tr_batch = process_data.gen_batches(
                tr_data, batch_size, self.seq_length)

            # initialize gru hidden states to 0
            gru_inits = [np.zeros((batch_size, self.n_h_units))
                         for _ in xrange(self.n_h_layers)]

            for tr_batches_n, (x, y) in enumerate(gen_tr_batch, 1):
                train_fn(x, y, gru_inits)

    def _build_network(self):
        """Input shape should be (batch_size, seq_length, vocab_size)."""
        network = lasagne.layers.InputLayer(
            shape=(None, self.seq_length, self.vocab_size),
            input_var=self.x)

        # build GRU layers
        gru_layers = []
        for i in xrange(self.n_h_layers):
            network = lasagne.layers.GRULayer(
                lasagne.layers.dropout(network, self.drop_p), self.n_h_units,
                hid_init=self.gru_sym_inits[i],
                grad_clipping=self.grad_clip)
            gru_layers.append(network)

        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, self.drop_p), self.vocab_size,
            nonlinearity=lasagne.nonlinearities.softmax)

        return gru_layers, network

