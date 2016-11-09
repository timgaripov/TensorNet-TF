import tensorflow as tf
import numpy as np
import math
from .aux import get_var_wrap

def tt_conv(inp,         
            window,
            inp_ch_modes,              
            out_ch_modes,
            ranks,
            strides=[1, 1],
            padding='SAME',
            filters_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            filters_regularizer=None,
            cores_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            cores_regularizer=None,
            biases_initializer=tf.zeros_initializer,
            biases_regularizer=None,
            trainable=True,
            cpu_variables=False,        
            scope=None):
    """ tt-conv-layer (convolution of full input tensor with tt-filters (core by core))
    Args:
        inp: input tensor, float - [batch_size, H, W, C]
        window: convolution window size, list [wH, wW]
        inp_ch_modes: input channels modes, np.array (int32) of size d
        out_ch_modes: output channels modes, np.array (int32) of size d
        ranks: tt-filters ranks, np.array (int32) of size (d + 1)        
        strides: strides, list of 2 ints - [sx, sy] 
        padding: 'SAME' or 'VALID', string
        filters_initializer: filters init function
        filters_regularizer: filters regularizer function
        cores_initializer: cores init function, could be a list of functions for specifying different function for each core
        cores_regularizer: cores regularizer function, could be a list of functions for specifying different function for each core
        biases_initializer: biases init function (if None then no biases will be used)
        biases_regularizer: biases regularizer function        
        trainable: trainable variables flag, bool
        cpu_variables: cpu variables flag, bool
        scope: layer variable scope name, string
    Returns:
        out: output tensor, float - [batch_size, prod(out_modes)]
    """                

    with tf.variable_scope(scope):
        inp_shape = inp.get_shape().as_list()[1:]
        inp_h, inp_w, inp_ch = inp_shape[0:3]
        tmp = tf.reshape(inp, [-1, inp_h, inp_w, inp_ch])
        tmp = tf.transpose(tmp, [0, 3, 1, 2])
        tmp = tf.reshape(tmp, [-1, inp_h, inp_w, 1]) 
        
        filters_shape = [window[0], window[1], 1, ranks[0]]        
        if (window[0] * window[1] * 1 * ranks[0] == 1):
            filters = get_var_wrap('filters',
                                   shape=filters_shape,
                                   initializer=tf.ones_initializer,
                                   regularizer=None,
                                   trainable=False,
                                   cpu_variable=cpu_variables)
        else: 
            filters = get_var_wrap('filters',
                                   shape=filters_shape,
                                   initializer=filters_initializer,
                                   regularizer=filters_regularizer,
                                   trainable=trainable,
                                   cpu_variable=cpu_variables)

        tmp = tf.nn.conv2d(tmp, filters, [1] + strides + [1], padding)    
        
        #tmp shape = [batch_size * inp_ch, h, w, r]
        h, w = tmp.get_shape().as_list()[1:3]
        tmp = tf.reshape(tmp, [-1, inp_ch, h, w, ranks[0]])
        tmp = tf.transpose(tmp, [4, 1, 0, 2, 3])        
        #tmp shape = [r, c, b, h, w]
        
        d = inp_ch_modes.size
        
        cores = []
        for i in range(d):
            
            if type(cores_initializer) == list:
                cinit = cores_initializer[i]
            else:
                cinit = cores_initializer
            
            if type(cores_regularizer) == list:
                creg = cores_regularizer[i]
            else:
                creg = cores_regularizer
                
            cores.append(get_var_wrap('core_%d' % (i + 1),
                                      shape=[out_ch_modes[i] * ranks[i + 1], ranks[i] * inp_ch_modes[i]],
                                      initializer=cinit,
                                      regularizer=creg,
                                      trainable=trainable,
                                      cpu_variable=cpu_variables))                                                    
        
        for i in range(d):            
            tmp = tf.reshape(tmp, [ranks[i] * inp_ch_modes[i], -1])
            tmp = tf.matmul(cores[i], tmp)
            tmp = tf.reshape(tmp, [out_ch_modes[i], -1])
            tmp = tf.transpose(tmp, [1, 0])
        out_ch = np.prod(out_ch_modes)
        
        if biases_initializer is not None:
            biases = get_var_wrap('biases',
                                  shape=[out_ch],
                                  initializer=biases_initializer,
                                  regularizer=biases_regularizer,
                                  trainable=trainable,
                                  cpu_variable=cpu_variables)

            out = tf.reshape(tmp, [-1, h, w, out_ch])
            out = tf.add(out, biases, name='out')
        else:
            out = tf.reshape(tmp, [-1, h, w, out_ch], name='out')

    return out
