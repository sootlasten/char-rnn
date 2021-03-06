import os
import numpy as np
import theano

DATA_FOLDER = 'data'
DATA_FILE = 'input.txt'

with open(os.path.join(DATA_FOLDER, DATA_FILE), 'r') as f:
    data = f.read()

chars = sorted(list(set(data)))
data_size = len(data)
vocab_size = len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
to_char = {i: ch for i, ch in enumerate(chars)}


def _vectorize(data):
    """Vectorize data into a matrix of size (data_size, vocab_size), where each
    row is a one-hot vector, where the index of 1 corresponds to the index of
    the character."""
    data_ix = [char_to_ix[c] for c in data]
    data_vec = np.zeros((len(data), vocab_size),
                        dtype=theano.config.floatX)
    data_vec[np.arange(len(data)), data_ix] = 1
    return data_vec

vec_data = _vectorize(data)


def gen_batches(data, batch_size, seq_length):
    """Generates batches with the shape (batch_size, seq_length, vocab_size).
    'data' is a matrix of size (data_size, vocab_size).

    Also rearranges data so batches process sequential data.

    If we have the dataset:

    x_in = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

    and batch_size is 2 and seq_len is 3. Then the dataset is
    reordered such that:

                   Batch 1    Batch 2
                 ------------------------
    batch pos 1  [1, 2, 3]   [4, 5, 6]
    batch pos 2  [7, 8, 9]   [10, 11, 12]

    This ensures that we use the last hidden state of batch 1 to initialize
    batch 2."""
    # if data doesn't fit evenly into the batches, then cut off the end
    chars_in_batch = batch_size * seq_length
    end_idx = chars_in_batch * (len(data) / chars_in_batch)

    x = data[:end_idx].reshape(batch_size, -1, vocab_size)
    y = np.vstack((data[1: end_idx], data[0])).\
        reshape((batch_size, -1, vocab_size))  # last label is wrong, but who really cares ...

    x_batches = np.split(x, x.shape[1] / seq_length, 1)
    y_batches = np.split(y, y.shape[1] / seq_length, 1)

    for x, y in zip(x_batches, y_batches):
        yield x, y


def split_data(tf):
    """Splits data into train and val set. 'tf' is the fraction of train data."""
    assert 0 < tf <= 1

    train_n = int(data_size * tf)
    tr_data = vec_data[:train_n]
    val_data = vec_data[train_n:]

    return tr_data, val_data
