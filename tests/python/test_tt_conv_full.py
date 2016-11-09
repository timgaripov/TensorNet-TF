import numpy as np
import time
import tensorflow as tf
import sys

sys.path.append('../../')

import tensorflow as tf
import tensornet

def run_test(batch_size=30, test_num=10, tol=1e-5):
    print('*' * 80)
    print('*' + ' ' * 29 + 'Testing tt conv full' + ' ' * 29 + '*')
    print('*' * 80)

    in_h = 32
    in_w = 32
    
    padding = 'SAME'


    inp_ch_modes = np.array([4, 4, 4, 3], dtype=np.int32)    
    in_c = np.prod(inp_ch_modes)
    out_ch_modes = np.array([5, 2, 5, 5], dtype=np.int32)    
    out_c = np.prod(out_ch_modes)
    ranks = np.array([3, 2, 2, 3, 1], dtype=np.int32)


    inp = tf.placeholder(tf.float32, [None, in_h, in_w, in_c])

    
    wh = 5
    ww = 5
    

    w_ph = tf.placeholder(tf.float32, [wh, ww, in_c, out_c])

    s = [1, 1]

    corr = tf.nn.conv2d(inp, w_ph, [1] + s + [1], padding)

    
    out = tensornet.layers.tt_conv_full(inp,                                   
                                        [wh, ww],
                                        inp_ch_modes,                                     
                                        out_ch_modes,
                                        ranks,
                                        s,
                                        padding,
                                        biases_initializer=None,
                                        scope='tt_conv')
             
    sess = tf.Session()
    graph = tf.get_default_graph()
    init_op = tf.initialize_all_variables()

    d = inp_ch_modes.size
    
    filters_t = graph.get_tensor_by_name('tt_conv/filters:0')
    
    cores_t = []
    for i in range(d):
        cores_t.append(graph.get_tensor_by_name('tt_conv/core_%d:0' % (i + 1)))
    
    for test in range(test_num):        
        sess.run(init_op)

             
        filters = sess.run([filters_t])
        cores = sess.run(cores_t)
        
        w = np.reshape(filters.copy(), [wh, ww, ranks[0]])


        
        #mat = np.reshape(inp_cores[inp_ps[0]:inp_ps[1]], [inp_ch_ranks[0], inp_ch_modes[0], inp_ch_ranks[1]])
        
        for i in range(0, d):
            core = cores[i].copy()
            #[out_ch_modes[i] * ranks[i + 1], ranks[i] * inp_ch_modes[i]]
            core = np.transpose(core, [1, 0])            
            core = np.reshape(core, [ranks[i], inp_ch_modes[i] * out_ch_modes[i] * ranks[i + 1]])

            w = np.reshape(w, [-1, ranks[i]])
            w = np.dot(w, core)

        #w = np.dot(w, np.reshape(mat, [inp_ch_ranks[0], -1]))
        
        L = []        
        for i in range(d):
            L.append(inp_ch_modes[i])
            L.append(out_ch_modes[i])
            
        w = np.reshape(w, [-1] + L)        
        w = np.transpose(w, [0] + list(range(1, 2 * d + 1, 2)) + list(range(2, 2 * d + 1, 2)))
                
        w = np.reshape(w, [wh, ww, in_c, out_c])

        X = np.random.normal(0.0, 0.2, size=(batch_size, in_h, in_w, in_c))

        t1 = time.clock()
        correct = sess.run(corr, feed_dict={w_ph: w, inp: X})
        t2 = time.clock()
        y = sess.run(out, feed_dict={w_ph: w, inp: X})
        t3 = time.clock()



        err = np.max(np.abs(correct - y))        
        print('Test #{0:02d}. Error: {1:0.2g}'.format(test + 1, err))
        print('TT-conv time: {0:.2f} sec. conv time: {1:.2f} sec.'.format(t3 - t2, t2 - t1))
        assert err <= tol, 'Error = {0:0.2g} is bigger than tol = {1:0.2g}'.format(err, tol)

    
if __name__ == '__main__':
    run_test()
                         
