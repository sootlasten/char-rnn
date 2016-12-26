An implementation of [Andrej Karpathy's char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) in Theano/Lasagne using 
GRU units.

# Training
To train the model, the script `train.py` must be run with the command `python train.py`. The script loads hyperparemeters from the file
`params.txt` and run the model from `network.py` with these parameters set. During training, after each epoch, the code prints some 
information about the performace of the model to standard output. It also saves the checkpoint files (that contain weights and additional 
info needed to reload the model) to `checkpoints/` after each epoch. 

# TODO!
1. learning rate decay
2. output info more often than each epoch