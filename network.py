import os
import time
import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T
import theano.typed_list
import lasagne

import process_data
from custom_gru import GRULayer


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

        self.gru_layers, self.network = self._build_network()

    def sample(self, n, prime_text, chars, char_to_ix, to_char):
        """Sample from the model."""

        def vectorize(c):
            """Turn a single character into a one-hot vector."""
            c_idx = char_to_ix[c]
            c_vec = np.zeros(self.vocab_size, dtype=theano.config.floatX)
            c_vec[c_idx] = 1
            # need reshaping because network expects 3 dimensional input:
            return c_vec.reshape((1, 1, -1))

        # build graph
        r = lasagne.layers.get_output(self.gru_layers + [self.network], self.x,
                                      deterministic=True)
        gru_layers_outs, pred = r[:-1], r[-1]
        c_pred = pred.argmax()

        gru_outs = [gru_layer_outs[:, -1, :]
                       for gru_layer_outs in gru_layers_outs]

        pred_fn = theano.function([self.x, self.gru_sym_inits],
                                  [c_pred] + gru_outs)

        # SAMPLE ...
        if not prime_text:  # if no prime text, choose a random char
            prime_text = chars[np.random.randint(self.vocab_size)]

        # initialize gru hidden states to 0
        gru_prevs = [np.zeros((self.batch_size, self.n_h_units),
                              dtype=theano.config.floatX)
                     for _ in xrange(self.n_h_layers)]

        # feed in prime text
        for c in prime_text:
            c_vec = vectorize(c)
            r = pred_fn(c_vec, gru_prevs)
            c, gru_prevs = to_char[int(r[0])], r[1:]

        # now start generating custom text
        gen_text = prime_text + c
        for _ in xrange(n):
            c_vec = vectorize(c)
            r = pred_fn(c_vec, gru_prevs)
            c, gru_prevs = to_char[int(r[0])], r[1:]
            gen_text += c

        return gen_text

    def train(self, eta, n_epochs, tf, checkpoint_dir):
        self._print_model_info(eta, n_epochs, tf)

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
        best_val_err, best_epoch_n = float('inf'), 0

        print("Training...")
        for epoch_n in xrange(1, n_epochs + 1):
            start_time = time.time()

            # FULL PASS OVER THE TRAINING SET ...
            gen_tr_batch = process_data.gen_batches(
                tr_data, self.batch_size, self.seq_length)

            # initialize gru hidden states to 0
            gru_prevs = [np.zeros((self.batch_size, self.n_h_units),
                                  dtype=theano.config.floatX)
                         for _ in xrange(self.n_h_layers)]

            total_tr_err = 0
            for tr_batches_n, (x, y) in enumerate(gen_tr_batch, 1):
                r = train_fn(x, y, gru_prevs)
                err, gru_prevs = r[0], r[1:]
                total_tr_err += err

            # FULL PASS OVER THE VALIDATION SET ...
            total_val_err = 0
            gen_val_batch = process_data.gen_batches(
                val_data, self.batch_size, self.seq_length)

            # initialize gru hidden states to 0
            gru_prevs = [np.zeros((self.batch_size, self.n_h_units),
                                  dtype=theano.config.floatX)
                         for _ in xrange(self.n_h_layers)]

            for val_batches_n, (x, y) in enumerate(gen_val_batch, 1):
                r = val_fn(x, y, gru_prevs)
                err, gru_prevs = r[0], r[1:]
                total_val_err += err

            # OUTPUT INFO
            total_tr_err /= tr_batches_n
            total_val_err /= val_batches_n

            if total_val_err < best_val_err:
                best_val_err = total_val_err
                best_epoch_n = epoch_n

            filename = "model_e{}_{:.4f}.pickle".format(epoch_n, total_val_err)
            self._save_weights_and_hyperparams(
                os.path.join(checkpoint_dir, filename))

            print("Epoch %d completed in %d seconds" % (epoch_n, time.time() - start_time))
            print("Training loss:       %f" % total_tr_err)
            print("Validation loss:     %f" % total_val_err)
            print("Best epoch:          %d" % best_epoch_n)
            print("Best val loss:       %f\n" % best_val_err)

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

    def _save_weights_and_hyperparams(self, file_path):
        """Saves the params (weights) and model (hyper)parameters to a file."""
        weights = lasagne.layers.get_all_param_values(self.network)

        model_param_dict = {
            'n_h_layers': self.n_h_layers,
            'n_h_units': self.n_h_units,
            'vocab_size': self.vocab_size,
            'chars': process_data.chars,
            'char_to_ix': process_data.char_to_ix,
            'to_char': process_data.to_char}

        with open(file_path, 'wb') as f:
            pickle.dump((model_param_dict, weights), f)

    def _model_params_size(self):
        """Returns the number of trainable parameters of the model."""
        return sum([param.size for param in
                    lasagne.layers.get_all_param_values(self.network)])

    def _print_model_info(self, eta, n_epochs, tf):
        print("Model hyperparams and info:")
        print("---------------------------")
        print("n_h_layers:              %d" % self.n_h_layers)
        print("n_h_units:               %d" % self.n_h_units)
        print("batch_size:              %d" % self.batch_size)
        print("seq_length:              %d" % self.seq_length)
        print("vocab_size:              %d\n" % self.vocab_size)

        print("drop_p:                  {:.3f}".format(self.drop_p))
        print("grad_clip:               %d" % self.grad_clip)
        print("eta:                     {:.6f}".format(eta))
        print("n_epochs:                %d" % n_epochs)
        print("train_frac:              {:.2f}\n".format(tf))

        print("data_size:               %d" % process_data.data_size)
        print("# of trainabe params:    %d\n" % self._model_params_size())

