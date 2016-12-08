# Ultimate tensorization: compressing convolutional and FC layers alike  
Links: [[arXiv](https://arxiv.org/abs/1611.03214)] [[poster pdf](https://github.com/timgaripov/TensorNet-TF/raw/master/ultimate_tensorization_poster.pdf)]


Convolutional neural networks excel in image recognition tasks, but this comes at the cost of high computational and memory complexity. To tackle this problem, [1] developed a tensor factorization framework to compress fully-connected layers. In this paper, we focus on compressing convolutional layers. We show that while the direct application of the tensor framework [1] to the 4-dimensional kernel of convolution does compress the layer, we can do better. We reshape the convolutional kernel into a tensor of higher order and factorize it. We combine the proposed approach with the previous work to compress both convolutional and fully-connected layers of a network and achieve 80x network compression rate with 1.1% accuracy drop on the CIFAR-10 dataset
