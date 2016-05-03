import numpy as np
from .svd import svd

def matrix_svd(X, left_modes, right_modes, ranks):
    """ TT-SVD for matrix
        Args:
            X: input matrix, numpy array float32
            left_modes: tt-left-modes, numpy array int32
            right_modes: tt-right-modes, numpy array int32
            ranks: tt-ranks, numpy array int32
        Returns:
            core: tt-cores array, numpy 1D array float32
    """
    c = X.copy()
    d = left_modes.size
    c = np.reshape(c, np.concatenate((left_modes, right_modes)))
    order = np.repeat(np.arange(0, d), 2) + np.tile([0, d], d)
    c = np.transpose(c, axes=order)    
    c = np.reshape(c, left_modes * right_modes)    
    return svd(c, left_modes * right_modes, ranks)
