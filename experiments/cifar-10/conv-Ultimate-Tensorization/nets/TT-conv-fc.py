import tensorflow as tf
import math
import numpy as np
import sys


sys.path.append('../../../')
import tensornet

NUM_CLASSES = 10

opts = {}
opts['use_dropout'] = True
opts['initial_learning_rate'] = 0.1
opts['num_epochs_per_decay'] = 30.0
opts['learning_rate_decay_factor'] = 0.1

def aug_train(image, aux):
    aug_image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
    aug_image = tf.random_crop(aug_image, [32, 32, 3])
    aug_image = tf.image.random_flip_left_right(aug_image)
    aug_image = tf.image.random_contrast(aug_image, 0.75, 1.25)
    aug_image = (aug_image - aux['mean']) / aux['std']
    return aug_image

def aug_eval(image, aux):
    aug_image = (image - aux['mean']) / aux['std']
    return aug_image


def inference(images, train_phase, cpu_variables=False):
    """Build the model up to where it may be used for inference.
    Args:
        images: Images placeholder.
        train_phase: Train phase placeholder
    Returns:
        logits: Output tensor with the computed logits.
    """
    tn_init = lambda dev: lambda shape: tf.truncated_normal(shape, stddev=dev)
    tu_init = lambda bound: lambda shape: tf.random_uniform(shape, minval = -bound, maxval = bound)

    dropout_rate = lambda p: (opts['use_dropout'] * (p - 1.0)) * tf.to_float(train_phase) + 1.0

    layers = []
    layers.append(images)

    layers.append(tensornet.layers.conv(layers[-1],
                                        64,
                                        [3, 3],
                                        cpu_variables=cpu_variables,
                                        biases_initializer=None,
                                        scope='conv1.1'))

    layers.append(tensornet.layers.batch_normalization(layers[-1],
                                                       train_phase,
                                                       cpu_variables=cpu_variables,
                                                       scope='bn1.1'))

    layers.append(tf.nn.relu(layers[-1],
                             name='relu1.1'))

    layers.append(tf.nn.dropout(layers[-1],
                                dropout_rate(0.7)))


    layers.append(tensornet.layers.tt_conv_full(layers[-1],
                                                [3, 3],
                                                np.array([4,4,4],dtype=np.int32),
                                                np.array([4,4,4],dtype=np.int32),
                                                np.array([16,16,16,1],dtype=np.int32),
                                                [1, 1],
                                                cpu_variables=cpu_variables,
                                                biases_initializer=None,
                                                scope='tt_conv1.2'))

    layers.append(tensornet.layers.batch_normalization(layers[-1],
                                                       train_phase,
                                                       cpu_variables=cpu_variables,
                                                       scope='bn1.2'))

    layers.append(tf.nn.relu(layers[-1],
                             name='relu1.2'))



    layers.append(tf.nn.max_pool(layers[-1],
                                 [1, 3, 3, 1],
                                 [1, 2, 2, 1],
                                 'SAME',
                                 name='max_pool1'))


    layers.append(tensornet.layers.tt_conv_full(layers[-1],
                                                [3, 3],
                                                np.array([4,4,4],dtype=np.int32),
                                                np.array([4,8,4],dtype=np.int32),
                                                np.array([16,16,16,1],dtype=np.int32),
                                                [1, 1],
                                                cpu_variables=cpu_variables,
                                                biases_initializer=None,
                                                scope='tt_conv2.1'))



    layers.append(tensornet.layers.batch_normalization(layers[-1],
                                                       train_phase,
                                                       cpu_variables=cpu_variables,
                                                       scope='bn2.1'))

    layers.append(tf.nn.relu(layers[-1],
                             name='relu2.1'))


    layers.append(tensornet.layers.tt_conv_full(layers[-1],
                                                [3, 3],
                                                np.array([4,8,4],dtype=np.int32),
                                                np.array([4,8,4],dtype=np.int32),
                                                np.array([16,16,16,1],dtype=np.int32),
                                                [1, 1],
                                                cpu_variables=cpu_variables,
                                                biases_initializer=None,
                                                scope='tt_conv2.2'))



    layers.append(tensornet.layers.batch_normalization(layers[-1],
                                                       train_phase,
                                                       cpu_variables=cpu_variables,
                                                       scope='bn2.2'))

    layers.append(tf.nn.relu(layers[-1],
                             name='relu2.2'))



    layers.append(tf.nn.max_pool(layers[-1],
                                 [1, 3, 3, 1],
                                 [1, 2, 2, 1],
                                 'SAME',
                                 name='max_pool2'))

    layers.append(tensornet.layers.tt_conv_full(layers[-1],
                                                [3, 3],
                                                np.array([4,8,4],dtype=np.int32),
                                                np.array([4,8,4],dtype=np.int32),
                                                np.array([16,16,16,1],dtype=np.int32),
                                                [1, 1],
                                                cpu_variables=cpu_variables,
                                                biases_initializer=None,
                                                scope='tt_conv3.1'))



    layers.append(tensornet.layers.batch_normalization(layers[-1],
                                                       train_phase,
                                                       cpu_variables=cpu_variables,
                                                       scope='bn3.1'))

    layers.append(tf.nn.relu(layers[-1],
                             name='relu3.1'))


    layers.append(tensornet.layers.tt_conv_full(layers[-1],
                                                [3, 3],
                                                np.array([4,8,4],dtype=np.int32),
                                                np.array([4,8,4],dtype=np.int32),
                                                np.array([16,16,16,1],dtype=np.int32),
                                                [1, 1],
                                                cpu_variables=cpu_variables,
                                                biases_initializer=None,
                                                scope='tt_conv3.2'))


    layers.append(tensornet.layers.batch_normalization(layers[-1],
                                                       train_phase,
                                                       cpu_variables=cpu_variables,
                                                       scope='bn3.2'))

    layers.append(tf.nn.relu(layers[-1],
                             name='relu3.2'))

    sz = np.prod(layers[-1].get_shape().as_list()[1:])

    layers.append(tensornet.layers.linear(tf.reshape(layers[-1], [-1, sz]),
                                          1024 + 512,
                                          cpu_variables=cpu_variables,
                                          biases_initializer=None,
                                          scope='linear4.1'))

    layers.append(tensornet.layers.batch_normalization(layers[-1],
                                                       train_phase,
                                                       cpu_variables=cpu_variables,
                                                       scope='bn4.1'))

    layers.append(tf.nn.relu(layers[-1],
                             name='relu4.1'))


    layers.append(tensornet.layers.linear(layers[-1],
                                          512,
                                          cpu_variables=cpu_variables,
                                          biases_initializer=None,
                                          scope='linear4.2'))

    layers.append(tensornet.layers.batch_normalization(layers[-1],
                                                       train_phase,
                                                       cpu_variables=cpu_variables,
                                                       scope='bn4.2'))

    layers.append(tf.nn.relu(layers[-1],
                             name='relu4.2'))

    layers.append(tensornet.layers.linear(layers[-1],
                                          NUM_CLASSES,
                                          cpu_variables=cpu_variables,
                                          scope='linear4.3'))

    return layers[-1]

def losses(logits, labels):
    """Calculates losses from the logits and the labels.
    Args:
        logits: input tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
    Returns:
        losses: list of loss tensors of type float.
    """
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(xentropy, name='loss')
    return [loss]

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    correct_flags = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.cast(correct_flags, tf.int32)
