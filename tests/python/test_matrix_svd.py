import numpy as np
import sys
sys.path.append('../../')
import tensornet

 
def run_test(left_modes = np.array([4, 6, 8, 3], dtype=np.int32),
             right_modes = np.array([5, 2, 7, 4], dtype=np.int32),             
             test_num=10,
             tol=1e-5):
    print('*' * 80)
    print('*' + ' ' * 28 + 'Testing matrix TT-SVD' + ' ' * 29 + '*')
    print('*' * 80)	
    d = left_modes.size
    L = np.prod(left_modes)
    R = np.prod(right_modes)
    ranks = tensornet.tt.max_ranks(left_modes * right_modes)
    ps = np.cumsum(np.concatenate(([0], ranks[:-1] * left_modes * right_modes * ranks[1:])))
    for test in range(test_num):
        W = np.random.normal(0.0, 1.0, size=(L, R))
        T = tensornet.tt.matrix_svd(W, left_modes, right_modes, ranks)
        w = np.reshape(T[ps[0]:ps[1]], [left_modes[0] * right_modes[0], ranks[1]])        
        for i in range(1, d):
            core = np.reshape(T[ps[i]:ps[i + 1]], [ranks[i], left_modes[i] * right_modes[i] * ranks[i + 1]])
            w = np.dot(w, core)
            w = np.reshape(w, [-1, ranks[i + 1]])
        w = np.reshape(w, w.shape[:-1])
        shape = np.hstack((left_modes.reshape([-1, 1]), right_modes.reshape([-1, 1]))).ravel()        
        w = np.reshape(w, shape)
        order = np.concatenate((np.arange(0, 2 * d, 2), np.arange(1, 2 * d, 2)))
        w = np.reshape(np.transpose(w, axes=order), [L, R])
        result = np.max(np.abs(W - w))
        print('Test #{0:02d}. Error: {1:0.2g}'.format(test + 1, result))
        assert result <= tol, 'Error = {0:0.2g} is bigger than tol = {1:0.2g}'.format(result, tol)


if __name__ == '__main__':
	run_test()
