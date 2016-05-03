import numpy as np

def svd(X, modes, ranks):
	""" TT-SVD
	Args:
		X: input array, numpy array float32
		modes: tt-modes, numpy array int32
        ranks: tt-ranks, numpy array int32
    Returns:
		core: tt-cores array, numpy 1D array float32
	"""
	c = X.copy()
	d = modes.size
	core = np.zeros(np.sum(ranks[:-1] * modes * ranks[1:]), dtype='float32')
	pos = 0
	for i in range(0, d-1):
		m = ranks[i] * modes[i]
		c = np.reshape(c, [m, -1])
		u, s, v = np.linalg.svd(c, full_matrices=False)		
		u = u[:, 0:ranks[i + 1]]		
		s = s[0:ranks[i + 1]]
		v = v[0:ranks[i + 1], :]
		core[pos:pos + ranks[i] * modes[i] * ranks[i + 1]] = u.ravel()
		pos += ranks[i] * modes[i] * ranks[i + 1]		
		c = np.dot(np.diag(s), v)
	core[pos:pos + ranks[d - 1] * modes[d - 1] * ranks[d]] = c.ravel()
	return core
