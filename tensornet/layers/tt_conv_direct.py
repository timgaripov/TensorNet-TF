import tensorflow as tf
import numpy as np
import math
from .aux import get_var_wrap

def tt_conv_direct(inp,         
                   window,
                   out_ch,
                   ranks,
                   strides=[1, 1],
                   padding='SAME',
                   cores_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                   cores_regularizer=None,
                   biases_initializer=tf.zeros_initializer,
                   biases_regularizer=None,
                   trainable=True,
                   cpu_variables=False,        
                   scope=None):
    """ tt-conv-layer (convolution of full input tensor with straightforward decomposed tt-filters (make tt full then use conv2d))
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
        
        modes = np.array([window[0], window[1], inp_ch, out_ch])
        
        cores = []
        for i in range(4):

            sz = modes[i] * ranks[i] * ranks[i + 1]
            if (sz == 1):
                cinit = tf.ones_initializer
            elif type(cores_initializer) == list:
                cinit = cores_initializer[i]
            else:
                cinit = cores_initializer
            
            if type(cores_regularizer) == list:
                creg = cores_regularizer[i]
            else:
                creg = cores_regularizer
                
            cores.append(get_var_wrap('core_%d' % (i + 1),
                                      shape=[ranks[i], modes[i] * ranks[i + 1]],
                                      initializer=cinit,
                                      regularizer=creg,
                                      trainable=trainable and (sz > 1),
                                      cpu_variable=cpu_variables))                                                    
        
        full = cores[0]
        
        for i in range(1, 4):            
            full = tf.reshape(full, [-1, ranks[i]])
            full = tf.matmul(full, cores[i])
            
        full = tf.reshape(full, [window[0], window[1], inp_ch, out_ch])
        
        
        tmp = tf.nn.conv2d(tmp,
                           full,
                           [1] + strides + [1],
                           padding,
                           name='conv2d')
        
        if biases_initializer is not None:
            biases = get_var_wrap('biases',
                                  shape=[out_ch],
                                  initializer=biases_initializer,
                                  regularizer=biases_regularizer,
                                  trainable=trainable,
                                  cpu_variable=cpu_variables)
            
            out = tf.add(tmp, biases, name='out')
        else:
            out = tf.identity(tmp, name='out')

    return out
