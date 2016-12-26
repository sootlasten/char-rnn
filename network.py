import time

import numpy as np
import theano
import theano.tensor as T
import theano.typed_list
import lasagne

import process_data
from custom_gru import GRULayer

DEBUG = True
theano.config.optimizer = 'fast_compile'


class Network(object):

    def __init__(self, n_h_layers, n_h_units, batch_size, seq_length,
                 vocab_size, drop_p=0.5, grad_clip=5):
        # network parameters
        self.n_h_layers = n_h_layers
        self.n_h_units = n_h_units

        self.batch_size = batch_size
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

            gen_tr_batch = process_data.gen_batches(
                tr_data, self.batch_size, self.seq_length)

            # initialize gru hidden states to 0
            gru_inits = [np.zeros((self.batch_size, self.n_h_units))
                         for _ in xrange(self.n_h_layers)]

            x, y = next(gen_tr_batch)

            self.x.tag.test_value = x
            self.y.tag.test_value = y
            self.gru_sym_inits.tag.test_value = gru_inits
        # ----------------------------------

        self.gru_layers, self.network = self._build_network()

    def train(self, eta, n_epochs, tf):
        # BUILD GRAPH FOR TRAINING ...
        r = lasagne.layers.get_output(self.gru_layers + [self.network], self.x)
        gru_layers_outs_tr, pred = r[:-1], r[-1]

        # we only care about the last outputs in GRU layers
        gru_outs_tr = [gru_layer_outs[:, -1, :]
                       for gru_layer_outs in gru_layers_outs_tr]

        # reshape y to (self.batch_size * self.seq_length, self.vocab_size) so it matches
        # the shape of pred (only 2-D tensors can be used in cross-entropy loss).
        y_reshaped = self.y.reshape((self.batch_size * self.seq_length, -1))

        tr_loss = lasagne.objectives.categorical_crossentropy(pred, y_reshaped).mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(tr_loss, params, learning_rate=eta)

        # BUILD GRAPH FOR VALIDATION ...
        r = lasagne.layers.get_output(self.gru_layers + [self.network], self.x,
                                      deterministic=True)
        gru_layers_outs_val, pred = r[:-1], r[-1]

        # we only care about the last outputs in GRU layers
        gru_outs_val = [gru_layer_outs[:, -1, :]
                       for gru_layer_outs in gru_layers_outs_val]

        val_loss = lasagne.objectives.\
            categorical_crossentropy(pred, y_reshaped).mean()

        # compile functions
        train_fn = theano.function([self.x, self.y, self.gru_sym_inits],
                                   [tr_loss] + gru_outs_tr, updates=updates)
        val_fn = theano.function([self.x, self.y, self.gru_sym_inits],
                                [val_loss] + gru_outs_val)

        # do the actual training
        tr_data, val_data = process_data.split_data(tf)

        print("Training...")
        for epoch_n in xrange(n_epochs):
            start_time = time.time()

            # FULL PASS OVER THE TRAINING SET
            gen_tr_batch = process_data.gen_batches(
                tr_data, self.batch_size, self.seq_length)

            # initialize gru hidden states to 0
            gru_prevs = [np.zeros((self.batch_size, self.n_h_units))
                         for _ in xrange(self.n_h_layers)]

            for tr_batches_n, (x, y) in enumerate(gen_tr_batch, 1):
                r = train_fn(x, y, gru_prevs)
                err, gru_prevs = r[0], r[1:]
                print(err)

            print("Batch completed in %d seconds" % time.time() - start_time)

            # FULL PASS OVER THE VALIDATION SET
            total_val_err = 0
            gen_val_batch = process_data.gen_batches(
                val_data, self.batch_size, self.seq_length)

            # initialize gru hidden states to 0
            gru_prevs = [np.zeros((self.batch_size, self.n_h_units))
                         for _ in xrange(self.n_h_layers)]

            for val_batches_n, (x, y) in enumerate(gen_val_batch, 1):
                r = val_fn(x, y, gru_prevs)
                err, gru_prevs = r[0], r[1:]

    def _build_network(self):
        """Input shape should be (batch_size, seq_length, vocab_size).
        Network output shape is (batch_size * seq_length, vocab_size)."""
        network = lasagne.layers.InputLayer(
            (self.batch_size, self.seq_length, self.vocab_size),
            input_var=self.x)

        # build GRU layers
        gru_layers = []
        for i in xrange(self.n_h_layers):
            network = GRULayer(
                lasagne.layers.dropout(network, self.drop_p), self.n_h_units,
                hid_init=self.gru_sym_inits[i],
                grad_clipping=self.grad_clip)
            gru_layers.append(network)

        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, self.drop_p), self.vocab_size,
            nonlinearity=None,
            num_leading_axes=2)

        network = lasagne.layers.ReshapeLayer(
            network, (self.batch_size * self.seq_length, self.vocab_size))

        network = lasagne.layers.NonlinearityLayer(
            network, nonlinearity=lasagne.nonlinearities.softmax)

        return gru_layers, network
