import tensorflow as tf
from .aux import get_var_wrap

def batch_normalization(inp,
                        train_phase,
                        ema_decay=0.99,
                        eps=1e-3,
                        use_scale=True,
                        use_shift=True,
                        trainable=True,
                        cpu_variables=False,
                        scope=None):
    """Batch normalization layer
        Args:
            inp: input tensor [batch_el, ...] with 2 or 4 dimensions
            train_phase: tensor [1] of bool, train pahse indicator
            ema_decay: moving average decay
            eps: number added to variance, to exclude division by zero
            use_scale: bool flag of scale transform applying
            use_shift: bool flag of shift transform applying
            trainable: trainable variables flag, bool
            cpu_variables: cpu variables flag, bool
            scope: layer variable scope name, string
        Reutrns:
            out: normalizaed tensor of the same shape as inp
    """

    reuse = tf.get_variable_scope().reuse
    with tf.variable_scope(scope):

        shape = inp.get_shape().as_list()
        assert len(shape) in [2, 4]
        n_out = shape[-1]

        if len(shape) == 2:
            batch_mean, batch_variance = tf.nn.moments(inp, [0], name='moments')
        else:
            batch_mean, batch_variance = tf.nn.moments(inp, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay, zero_debias=True)
        if not reuse:
            def mean_variance_with_update():
                with tf.control_dependencies([ema.apply([batch_mean, batch_variance])]):
                    return (tf.identity(batch_mean),
                            tf.identity(batch_variance))

            mean, variance = tf.cond(train_phase,
                                     mean_variance_with_update,
                                     lambda: (ema.average(batch_mean),
                                              ema.average(batch_variance)))
        else:
            print("At scope %s reuse is truned on! Using previously created ema variables." % tf.get_variable_scope().name)

            #It's a kind of workaround
            vars = tf.get_variable_scope().global_variables()
            transform = lambda s: '/'.join(s.split('/')[-5:])

            mean_name = transform(ema.average_name(batch_mean))
            variance_name = transform(ema.average_name(batch_variance))

            existed = {}
            for v in vars:
                if (transform(v.op.name) == mean_name):
                    existed['mean'] = v
                if (transform(v.op.name) == variance_name):
                    existed['variance'] = v

            print('Using:')
            print('\t' + existed['mean'].op.name)
            print('\t' + existed['variance'].op.name)


            mean, variance = tf.cond(train_phase,
                                     lambda: (batch_mean,
                                              batch_variance),
                                     lambda: (existed['mean'],
                                              existed['variance']))

        std = tf.sqrt(variance + eps, name='std')
        out = (inp - mean) / std
        if use_scale:
            weights = get_var_wrap('weights',
                                   shape=[n_out],
                                   initializer=tf.ones_initializer,
                                   trainable=trainable,
                                   regularizer=None,
                                   cpu_variable=cpu_variables)

            out = tf.multiply(out, weights)
        if use_shift:
            biases = get_var_wrap('biases',
                                  shape=[n_out],
                                  initializer=tf.zeros_initializer,
                                  trainable=trainable,
                                  regularizer=None,
                                  cpu_variable=cpu_variables)

            out = tf.add(out, biases)
    return out
