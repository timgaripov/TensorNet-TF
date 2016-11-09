import tensorflow as tf
from .aux import get_var_wrap

def batch_normalization(inp,
                        train_phase,                        
                        ema_decay=0.9,
                        eps=1e-3,
                        use_scale=True,
                        use_shift=True,
                        trainable=True,
                        cpu_variables=False,
                        scope=None):
    """Batch normalization layer
        Args:
            inp: input tensor [batch_el, ...]
            shape: input tensor shape
            train_phase: tensor [1] of bool, train pahse indicator
            scope: string, layer scope name
            ema_decay: moving average decay
            eps: number added to variance, to exclude zero division
        Reutrns:
            out: normalizaed tensor of the same shape as inp
    """
    with tf.variable_scope(scope):        
        shape = inp.get_shape().as_list()
        assert len(shape) in [2, 4]
        n_out = shape[-1]
        
        if len(shape) == 2:
            batch_mean, batch_variance = tf.nn.moments(inp, [0], name='moments')
        else:
            batch_mean, batch_variance = tf.nn.moments(inp, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        def mean_variane_with_update():
            with tf.control_dependencies([ema.apply([batch_mean, batch_variance])]):
                return (tf.identity(batch_mean),
                        tf.identity(batch_variance))
        mean, variance = tf.cond(train_phase,
                                 mean_variane_with_update,
                                 lambda: (ema.average(batch_mean),
                                          ema.average(batch_variance)))
                
        std = tf.sqrt(variance + eps, name='std')
        out = (inp - mean) / std
        if use_scale:
            weights = get_var_wrap('weights',
                                   shape=[n_out],
                                   initializer=tf.ones_initializer,
                                   trainable=trainable,
                                   regularizer=None,
                                   cpu_variable=cpu_variables)                   
                                      
            out = tf.mul(out, weights)
        if use_shift:
            biases = get_var_wrap('biases',
                                  shape=[n_out],
                                  initializer=tf.zeros_initializer,
                                  trainable=trainable,
                                  regularizer=None,
                                  cpu_variable=cpu_variables)

            out = tf.add(out, biases)
    return out
