import argparse

from network import Network
import process_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train the network.')
    parser.add_argument('--checkpoints_dir', help='Path to the directory to save '
                                                  'checkpoints to.',
                        default='checkpoints/')
    parser.add_argument('--params_file', help='Path to the file where to read model '
                                              'params from.',
                        default='params.txt')
    return parser.parse_args()


args = parse_args()

with open(args.params_file, 'r') as f:
    params = f.readlines()

p_dict = dict()
for param in params:
    p = param.rstrip('\n').split('=')
    try:
        p_dict[p[0]] = int(p[1])
    except ValueError:
        p_dict[p[0]] = float(p[1])

data = process_data.vec_data

net = Network(
    n_h_layers=p_dict.get('n_h_layers', 2),
    n_h_units=p_dict.get('n_h_units', 100),
    batch_size=p_dict.get('batch_size', 40),
    seq_length=p_dict.get('seq_length', 25),
    vocab_size=process_data.vocab_size,
    drop_p=p_dict.get('drop_p', 0.5),
    grad_clip=p_dict.get('grad_clip', 5))

net.train(
    eta=p_dict.get('eta', 0.001),
    n_epochs=p_dict.get('n_epochs', 10),
    tf=p_dict.get('train_frac', 0.95),
    checkpoint_dir=args.checkpoints_dir)
