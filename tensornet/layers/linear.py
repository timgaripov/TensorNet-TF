import tensorflow as tf

def linear(inp, inp_size, out_size, init, scope, use_biases=True):
    """ linear_layer
    Args:
        inp: input tensor, float - [batch_size, inp_size]
        inp_size: input size, int
        out_size: layer units count, int
        init: lambda function (shape) used for weights initialization
        scope: layer scope name, string
        use_biases: biases using flag, bool
    Returns:
        out: output tensor, float - [batch_size, out_size]
    """
    with tf.name_scope(scope):
        weights = tf.Variable(init([inp_size, out_size]), name="weights")
        if use_biases:
            biases = tf.Variable(tf.zeros([out_size]), name="biases")
            out = tf.add(tf.matmul(inp, weights, name="matmul"), biases, name="out")
        else:
            out = tf.matmul(inp, weights, name="out")        
    return out
