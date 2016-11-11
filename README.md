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

# FAQ
### What is _tensor_ anyway?
Its just a synonym for a multidimensional array. For example a matrix is a 2-dimensional tensor.

### But in the fully-connected case you work with matrices, why do you need tensor decompositions?
Good point. Actually, the Tensor Train format  coincides the matrix low-rank format when applied to matrices. For this reason, there is a special _matrix Tensor Train format_, which basically does two things: reshapes the matrix into a tensor (say 10-dimensional) and permutes its dimensions in a special way; uses tensor decomposition on the resulting tensor. This way proved to be more efficient than the matrix low-rank format for the matrix of the fully-connected layer.

### Where I can read more about this _Tensor Train_ format?
Look at the original paper: Ivan Oseledets, Tensor-Train decomposition, 2011 [[pdf](http://spring.inm.ras.ru/osel/wp-content/plugins/wp-publications-archive/openfile.php?action=open&file=28)]. You can also check out my (Alexander Novikov's) [slides](http://www.slideshare.net/AlexanderNovikov8/tensor-train-decomposition-in-machine-learning), from slide 3 to 14.

By the way, **train** means like actual train, with wheels. The name comes from the pictures like the one below that illustrate the Tensor Train format and naturally look like a train (at least they say so).

<img src="TT.png" alt="Tensor Train format" width="280"/>

<!-- ### I have a matrix from a fully-connected layer. How do I convert it into the TT-FC layer?
Actually, the idea was to train a network with the TT-FC layers from scratch as opposed to firstly learning a network with regular fully-connected layers and then converting them into TT-FC layers.

But if you absolutely want to convert a matrix into the TT-FC layer (e.g. for debugging), you can do the following:
# TODO -->

### Are TensorFlow, MATLAB, and Theano implementations compatible?
Unfortunately not (at least not yet).


### I want to implement this in Caffe (or other library without autodiff). Any tips on doing the backward pass?
Great! Write us when you're done or if you have questions along the way.  
The MATLAB version of the code has the [backward pass implementation](https://github.com/Bihaqo/TensorNet/blob/master/src/matlab/vl_nntt_backward.m) for TT-FC layer. But note that the forward pass in MATLAB and TensorFlow versions is implemented differently.

### Have you tried other tensor decompositions, like CP-decomposition?
We haven't, but this paper uses CP-decomposition to compress the kernel of the convolutional layer: Lebedev V., Ganin Y. et al., Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition [[arXiv](https://arxiv.org/abs/1412.6553)] [[code](https://github.com/vadim-v-lebedev/cp-decomposition)]. They got nice compression results, but was not able to train CP-conv layers from scratch, only to train a network with regular convolutional layers, represent them in the CP-format, and when finetune the rest of the network. Even _finetuning_ an CP-conv layer often diverges.
