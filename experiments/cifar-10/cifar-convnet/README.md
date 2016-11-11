# Experiments with TT-conv layer

This folder contains the framework we used to conduct experiments on the CIFAR-10 dataset.

## Training

The following command runs the training procedure:

```bash
python3 train.py --net_module=<net_module> \
                 --log_dir=<log_dir> \
                 --data_dir=<data_dir> \
                 --num_gpus=<num_gpus>
```

where
* ```net_module``` is a path to a python-file with network description (e.g.  ```./nets/conv.py```);


* ```log_dir``` is a path to directory where summaries and checkpoints should be saved (e.g. ```./log/conv```);

* ```data_dir``` is a path to directory with data (e.g. ```../data/```);


* ```num_gpus``` is a number of gpu's that will be used for training.

### Training with pretrained convolutional part initialization

There is auxiliary scipt for training a network with convolutional part initialized with pretrained weights:

```bash
python3 train_with_pretrained_convs.py --net_module=<net_module>\
                                       --log_dir=<log_dir> \
                                       --num_gpus=<num_gpus> \
                                       --data_dir=<data_dir> \
                                       --pretrained_ckpt=<pretrained_ckpt>
```

where ```pretrained_ckpt``` is the path to the checkpoint file with pretrained weights.

## Evaluation

The following command runs the evaluation process of a trained network:

```bash
python3 eval.py --net_module=<net_module> \
                --log_dir=<log_dir> \
                --data_dir=<data_dir>
```
