# char-rnn

An implementation of Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) in Theano/Lasagne using 
GRU units. The code is developed in Python 2.7 and needs some tweaking if one wants to run it in Python 3.x.

### Training
To train the model, the script `train.py` must be run with the command `python train.py`. The script loads hyperparemeters from the file
`params.txt` and run the model from `network.py` with these parameters set. During training, after each epoch, the code prints some 
information about the performace of the model to standard output. It also saves the checkpoint files (that contain weights and additional 
info needed to reload the model) to `checkpoints/` after each epoch. 

### Sampling
To sample from the model, the script `sample.py` must be run with the command `python sample.py`. This script also takes command line 
arguments (see `python sample.py -h` for additional info). The only mandatory argument, though, is `--model_file`, which should point 
to one of the checkpoint files saved in the training phase.

### TODO
1. learning rate decay
2. output info more often than each epoch