from network import Network
import process_data

import theano
theano.config.optimizer = 'fast_compile'

PARAMS_FILE = 'params.txt'

with open(PARAMS_FILE, 'r') as f:
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
    data=data,
    eta=p_dict.get('eta', 0.001),
    n_epochs=p_dict.get('n_epochs', 10),
    tf=p_dict.get('train_frac', 0.95)
)