import tensorflow as tf
from .aux import get_var_wrap

def conv(inp,
         out_ch,
         window_size,         
         strides=[1, 1],
         padding='SAME',
         filters_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
         filters_regularizer=None,         
         biases_initializer=tf.zeros_initializer,
         biases_regularizer=None,
         trainable=True,
         cpu_variables=False,
         scope=None):
    """ convolutional layer
    Args:
        inp: input tensor, float - [batch_size, H, W, C]        
        out_ch: output channels count count, int
        window_size: convolution window size, list [wH, wW]
        strides: strides, list [sx, sy]
        padding: 'SAME' or 'VALID', string
        filters_initializer: filters init function
        filters_regularizer: filters regularizer function
        biases_initializer: biases init function (if None then no biases will be used)
        biases_regularizer: biases regularizer function        
        trainable: trainable variables flag, bool
        cpu_variables: cpu variables flag, bool
        scope: layer variable scope name, string
    Returns:
        out: output tensor, float - [batch_size, H', W', out_ch]
    """            

    with tf.variable_scope(scope):
        shape = inp.get_shape().as_list()
        assert len(shape) == 4, "Not 4D input tensor"
        in_ch = shape[-1]

        filters = get_var_wrap('filters',
                               shape=window_size + [in_ch, out_ch],
                               initializer=filters_initializer,
                               regularizer=filters_regularizer,
                               trainable=trainable,
                               cpu_variable=cpu_variables)
                                   
        out = tf.nn.conv2d(inp, filters, [1] + strides + [1], padding, name='conv2d')
        
        if biases_initializer is not None:            
            biases = get_var_wrap('biases',
                                  shape=[out_ch],
                                  initializer=biases_initializer,
                                  regularizer=biases_regularizer,
                                  trainable=trainable,
                                  cpu_variable=cpu_variables)
            out = tf.add(out, biases, name='out')
        else:
            out = tf.identity(out, name='out')
        return out
