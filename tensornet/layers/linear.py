import tensorflow as tf
from .aux import get_var_wrap

def linear(inp,
           out_size,
           weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
           weights_regularizer=None,
           biases_initializer=tf.zeros_initializer,
           biases_regularizer=None,
           trainable=True,
           cpu_variables=False,
           scope=None):
    """ linear layer
    Args:
        inp: input tensor, float - [batch_size, inp_size]        
        out_size: layer units count, int
        weights_initializer: weights init function
        weights_regularizer: weights regularizer function
        biases_initializer: biases init function (if None then no biases will be used)
        biases_regularizer: biases regularizer function        
        trainable: trainable variables flag, bool
        cpu_variables: cpu variables flag, bool
        scope: layer variable scope name, string
    Returns:
        out: output tensor, float - [batch_size, out_size]
    """
    with tf.variable_scope(scope):
        shape = inp.get_shape().as_list()
        assert len(shape) == 2, 'Not 2D input tensor'
        inp_size = shape[-1]
                                        
        weights = get_var_wrap('weights',
                               shape=[inp_size, out_size],
                               initializer=weights_initializer,
                               regularizer=weights_regularizer,
                               trainable=trainable,
                               cpu_variable=cpu_variables)                                    

        if biases_initializer is not None:            
            biases = get_var_wrap('biases',
                                  shape=[out_size],
                                  initializer=biases_initializer,
                                  regularizer=biases_regularizer,
                                  trainable=trainable,
                                  cpu_variable=cpu_variables)

            out = tf.add(tf.matmul(inp, weights, name='matmul'), biases, name='out')
        else:
            out = tf.matmul(inp, weights, name='out')        
    return out
