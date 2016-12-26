import argparse
import cPickle as pickle
import lasagne

from network import Network


def load_model(file_path):
    """Loads the network from file."""
    with open(file_path, 'rb') as f:
        d, params = pickle.load(f)

    net = Network(d['n_h_layers'], d['n_h_units'],
                  d['vocab_size'], d['seq_length'],
                  d['drop_p'], d['grad_clip'])

    lasagne.layers.set_all_param_values(net.network, params)
    return net


def parse_args():
    parser = argparse.ArgumentParser(description='Sample from the network.')
    parser.add_argument('--model_file', help='Path to the file that contains '
                                             'the model information')
    parser.add_argument('--prime_text', help='Text to feed the network with before sampling',
                        default=None)
    parser.add_argument('--length', help='Number of characters to sample',
                        type=int, default=100)
    return parser.parse_args()


args = parse_args()
net = load_model(args.model_file)

with open('sample.txt', 'w') as f:
    sample_text = net.sample(args.length, args.prime_text)
    f.write(sample_text)

