import tensorflow as tf
import numpy as np
import math
from .aux import get_var_wrap
import tt_conv_full

def tt_conv1d_full(inp,         
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
    """ 
    conv1d wrapper for conv2d function. Internally tensorflow does a conv2d for its vanilla
    conv1d. Similarly, this process is applied here. Input is expanded by dim 1 and then output is simply squeezed. 

    Note: window should be [1, w] where you insert your width
          strides should be [1, stride_width]


    tt-conv-layer (convolution of full input tensor with tt-filters (make tt full then use conv2d))
    Args:
        inp: input tensor, float - [batch_size, W, C]
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
        out: output tensor, float - [batch_size, W,  prod(out_modes)]
    """                
    inp_expanded = tf.expand_dims(inp, dim = 1) # expand on height dim

    conv2d_output = tt_conv_full(inp,         
                 window,
                 inp_ch_modes,              
                 out_ch_modes,
                 ranks,
                 strides=strides,
                 padding=padding,
                 filters_initializer=filters_initializer,
                 filters_regularizer=filters_regularizer,
                 cores_initializer=cores_initializer,
                 cores_regularizer=cores_regularizer,
                 biases_initializer=biases_initializer,
                 biases_regularizer=biases_regularizer,
                 trainable=trainable,
                 cpu_variables=cpu_variables,        
                 scope=scope)

    return tf.squeeze(conv2d_output) # get rid of height dimension
