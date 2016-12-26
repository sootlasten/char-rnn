# char-rnn

An implementation of Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) in Theano/Lasagne using 
GRU units. The code is developed in Python 2.7 and needs some tweaking if one wants to run it in Python 3.x.

### Training
To train the model, the script `train.py` must be run with the command `python train.py`. The script loads hyperparemeters from the file
`params.txt` and runs the model from `network.py` with these parameters set. The input file is located in `data/input.txt` by default. 
During training, after each epoch, the code prints some information about the performace of the model to standard output. It also saves 
the checkpoint files (that contain weights and additional info needed to reload the model) to `checkpoints/` (if this directory doesn't 
exist, it must be created) after each epoch. Each checkpoint file has a filename of the form `model_e[x]_[y].pickle`, where `x` specifies 
the epoch number and `y` specifies validation loss for this epoch.

### Sampling
To sample from the model, the script `sample.py` must be run. This script also takes command line arguments (see `python sample.py -h` 
for additional info). The only mandatory argument, though, is `--model_file`, which should point to one of the checkpoint files saved in 
the training phase. A minimal example of running the script: `python sample.py --model_file chechpoints/model_e24_1.63.pickle`. By default,
the sampled text is saved into `sample.txt`.

### A fun example
Just after a few minutes of training on the tinyshakespeare dataset, the rather stupid network is a nationalist (the network was primed
with the text `the mearning of is `:
```
the meaning of life is the state,
And then the state and the state and the state,
And then the state and the state and the state,
And then the state and the state and the state,
And then the state and the state and the state,
And then the state and the state and the state,
And then the state and the state and the state
```

However, after it has had a bit more schooling, it has a more profound worldview:
```
the meaning of life is the country,
And then the country of the country's country,
And then the sense of the common than the country,
And there is not the country of the country.
```

### TODO
1. learning rate decay
2. output info more often than each epoch
3. plot losses
4. sample during training?