import argparse
import cPickle as pickle
import lasagne

from network import Network


def load_model(file_path):
    """Loads the network from file."""
    with open(file_path, 'rb') as f:
        d, params = pickle.load(f)

    net = Network(d['n_h_layers'], d['n_h_units'],
                  1, 1, d['vocab_size'])

    lasagne.layers.set_all_param_values(net.network, params)
    return net, d['chars'], d['char_to_ix'], d['to_char']


def parse_args():
    parser = argparse.ArgumentParser(description='Sample from the network.')
    parser.add_argument('--model_file', help='Path to the file that contains '
                                             'the model information')
    parser.add_argument('--prime_text', help='Text to feed the network with before sampling',
                        default=None)
    parser.add_argument('--length', help='Number of characters to sample. Defaults to 100.',
                        type=int, default=100)
    parser.add_argument('--out_file', help='Path to output file. Defaults to ./sample.txt',
                        default='sample.txt')
    return parser.parse_args()


args = parse_args()
net, chars, char_to_ix, to_char = load_model(args.model_file)

with open(args.out_file, 'w') as f:
    sample_text = net.sample(args.length, args.prime_text,
                             chars, char_to_ix, to_char)
    f.write(sample_text)
