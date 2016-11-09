import numpy as np
import tensorflow as tf
import sys

sys.path.append('../../')

import tensornet

def run_test(batch_size=100, test_num=10, 
             inp_modes=np.array([3, 8, 9, 5], dtype='int32'),
             out_modes=np.array([5, 6, 10, 6], dtype='int32'),
             mat_ranks=np.array([1, 3, 6, 4, 1], dtype='int32'),
             tol=1e-5):
    print('*' * 80)
    print('*' + ' ' * 31 + 'Testing tt layer' + ' ' * 31 + '*')
    print('*' * 80)
    
    graph = tf.Graph()
    with graph.as_default(): 

        d = inp_modes.size

        INP_SIZE = np.prod(inp_modes)
        OUT_SIZE = np.prod(out_modes)

            

        inp = tf.placeholder('float', shape=[None, INP_SIZE])
        out = tensornet.layers.tt(inp,
                                  inp_modes,
                                  out_modes,
                                  mat_ranks,
                                  biases_initializer=None,
                                  scope='tt')


        sess = tf.Session()
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        for test in range(test_num):
            mat_cores = []
            for i in range(d):
                mat_cores.append(graph.get_tensor_by_name('tt/mat_core_%d:0' % (i + 1)))
                mat_cores[-1] = sess.run(mat_cores[-1])
            
            
            w = np.reshape(mat_cores[0], [out_modes[0] * mat_ranks[1], mat_ranks[0] * inp_modes[0]])        
            w = np.transpose(w, [1, 0])
            w = np.reshape(w, [-1, mat_ranks[1]])
            for i in range(1, d):
                core = np.reshape(mat_cores[i], [out_modes[i] * mat_ranks[i + 1], mat_ranks[i] * inp_modes[i]])
                core = np.transpose(core, [1, 0])
                core = np.reshape(core, [mat_ranks[i], -1])
                w = np.dot(w, core)
                w = np.reshape(w, [-1, mat_ranks[i + 1]])
            w = np.reshape(w, w.shape[:-1])
            shape = np.hstack((inp_modes.reshape([-1, 1]), out_modes.reshape([-1, 1]))).ravel()        
            w = np.reshape(w, shape)
            order = np.concatenate((np.arange(0, 2 * d, 2), np.arange(1, 2 * d, 2)))
            w = np.reshape(np.transpose(w, axes=order), [INP_SIZE, OUT_SIZE])        

            
                    
            X = np.random.normal(0.0, 0.2, size=(batch_size, np.prod(inp_modes)))
            feed_dict = {inp: X}
            y = sess.run(out, feed_dict=feed_dict)        
            Y = np.dot(X, w)
            result = np.max(np.abs(Y - y))
            print('Test #{0:02d}. Error: {1:0.2g}'.format(test + 1, result))
            assert result <= tol, 'Error = {0:0.2g} is bigger than tol = {1:0.2g}'.format(result, tol)
        sess.close()
    
if __name__ == '__main__':
	run_test()

