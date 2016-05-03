import tensorflow as tf

def batch_normalization(inp, shape, train_phase, scope, ema_decay=0.9, eps=1e-3):
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
    with tf.name_scope(scope):
        batch_mean, batch_variance = tf.nn.moments(inp, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        def mean_variane_with_update():
            with tf.control_dependencies([ema.apply([batch_mean, batch_variance])]):
                return (tf.identity(batch_mean),
                        tf.identity(batch_variance))
        mean, variance = tf.cond(train_phase,
                                 mean_variane_with_update,
                                 lambda: (ema.average(batch_mean),
                                          ema.average(batch_variance)))
                
        weights = tf.Variable(tf.ones(shape), name='weigths')
        biases = tf.Variable(tf.zeros(shape), name='biases')

        std = tf.sqrt(variance + eps, name='std')
        out = (inp - tf.expand_dims(mean, 0)) / tf.expand_dims(std, 0)
        out = tf.mul(out, tf.expand_dims(weights, 0))
        out = tf.add(out, tf.expand_dims(biases, 0), name='out')
    return out
