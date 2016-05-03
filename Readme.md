# TensorNet

This is a TensorFlow implementation of the _Tensor Train layer_ (_TT-layer_) of a neural network. In short, the TT-layer acts as a fully-connected layer but is much more compact and allows to use lots of hidden units without slowing down the learning and inference.   
For the additional information see the following paper:

Tensorizing Neural Networks  
Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, Dmitry Vetrov; In _Advances in Neural Information Processing Systems 28_ (NIPS-2015) [[arXiv](http://arxiv.org/abs/1509.06569)].

Please cite it if you write a scientific paper using this code.  
In BiBTeX format:
```latex
@incollection{novikov15tensornet,
  author    = {Novikov, Alexander and Podoprikhin, Dmitry and Osokin, Anton and Vetrov, Dmitry},
  title     = {Tensorizing Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems 28 (NIPS)},
  year      = {2015},
}
```

# Prerequisites
* [TensorFlow](https://www.tensorflow.org/)

# MATLAB and Theano
We also published a [MATLAB and Theano+Lasagne implementation](https://github.com/Bihaqo/TensorNet) in a separate repository.
