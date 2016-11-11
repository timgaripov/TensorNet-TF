# TensorNet

This is a TensorFlow implementation of the Tensor Train compression method for neural networks. It supports _TT-FC_ layer [1] and _TT-conv_ layer [2], which act as a fully-connected and convolutional layers correspondingly, but are much more compact. The TT-FC layer is also faster than its uncompressed analog and allows to use hundreds of thousands of hidden units. The ```experiments``` folder contains the code to reproduce the experiments from the papers.   


[1] _Tensorizing Neural Networks_  
Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, Dmitry Vetrov; In _Advances in Neural Information Processing Systems 28_ (NIPS-2015) [[arXiv](http://arxiv.org/abs/1509.06569)].

[2] _Ultimate tensorization: compressing convolutional and FC layers alike_   
Timur Garipov, Dmitry Podoprikhin, Alexander Novikov, Dmitry Vetrov; _Learning with Tensors: Why Now and How?_, NIPS-2016 workshop (NIPS-2015) [[arXiv](https://arxiv.org/abs/1611.03214)].


Please cite our work if you write a scientific paper using this code.  
In BiBTeX format:
```latex
@incollection{novikov15tensornet,
  author    = {Novikov, Alexander and Podoprikhin, Dmitry and Osokin, Anton and Vetrov, Dmitry},
  title     = {Tensorizing Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems 28 (NIPS)},
  year      = {2015},
}
@article{garipov16ttconv,
  author    = {Garipov, Timur and Podoprikhin, Dmitry and Novikov, Alexander and Vetrov, Dmitry},
  title     = {Ultimate tensorization: compressing convolutional and {FC} layers alike},
  journal   = {arXiv preprint arXiv:1611.03214},
  year      = {2016}
}
```

# Prerequisites
* [TensorFlow](https://www.tensorflow.org/) (tested with v. 0.7.1)
* [NumPy](http://www.numpy.org/)

# MATLAB and Theano
We also published a [MATLAB and Theano+Lasagne implementation](https://github.com/Bihaqo/TensorNet) in a separate repository.
