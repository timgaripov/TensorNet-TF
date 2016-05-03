import tensorflow as tf
import numpy as np

def tt(inp, inp_modes, out_modes, mat_ranks, init=2.0, scope="tt", use_biases=True, init_params=None):
    """ tt-layer ('old' tt-linear layer, tt-matrix by full tensor product)
    Args:
        inp: input tensor, float - [batch_size, prod(inp_modes)]
        inp_modes: input tensor modes
        out_modes: output tensor modes
        mat_ranks: tt-matrix ranks
        init: lambda function (shape) used for weights initialization
        scope: layer scope name, string
        use_biases: biases using flag, bool
    Returns:
        out: output tensor, float - [batch_size, prod(out_modes)]
    """
    with tf.name_scope(scope):
        dim = inp_modes.size
        mat_ps = np.cumsum(np.concatenate(([0], mat_ranks[:-1] * inp_modes * out_modes * mat_ranks[1:])))

        mat_size = mat_ps[-1]
        if type(init) == float:
            for i in range(dim):
                n_in = mat_ranks[i] * inp_modes[i]
                mat_core = tf.truncated_normal([mat_ps[i + 1] - mat_ps[i]],
                                               0.0,
                                               init / n_in,
                                               tf.float32)
                if (i == 0):
                    mat = mat_core
                else:
                    mat = tf.concat(0, [mat, mat_core])
        else:
            init_params['inp_modes'] = inp_modes
            init_params['out_modes'] = out_modes
            init_params['ranks'] = mat_ranks
            mat = init(init_params)
        mat = tf.Variable(mat, name="weights")
        out = tf.reshape(inp, [-1, np.prod(inp_modes)])
        out = tf.transpose(out, [1, 0])
        
        for i in range(dim):
            out = tf.reshape(out, [mat_ranks[i] * inp_modes[i], -1])
            
            mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
            mat_core = tf.reshape(mat_core, [mat_ranks[i] * inp_modes[i], out_modes[i] * mat_ranks[i + 1]])
            mat_core = tf.transpose(mat_core, [1, 0])

            out = tf.matmul(mat_core, out)
            out = tf.reshape(out, [out_modes[i], -1])
            out = tf.transpose(out, [1, 0])
        if use_biases:
            biases = tf.Variable(tf.zeros([np.prod(out_modes)]), name="biases")
            out = tf.add(tf.reshape(out, [-1, np.prod(out_modes)]), biases, name="out")
        else:
            out = tf.reshape(out, [-1, np.prod(out_modes)], name="out")
    return out
