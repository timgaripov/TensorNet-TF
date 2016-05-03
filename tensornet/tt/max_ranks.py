import numpy as np

def max_ranks(modes):
	""" Computation of maximal ranks for TT-SVD
	Args:
		modes: tt-modes, numpy array int32
	Returns:
		ranks: maximal tt-ranks, numpy array int32
	"""
	d = modes.size
	ranks = np.zeros(d + 1, dtype='int32')
	ranks[0] = 1
	prod = np.prod(modes)
	for i in range(d):
		m = ranks[i] * modes[i]				
		ranks[i + 1] = min(m, prod // m);		
		prod = prod // m * ranks[i + 1]		
	ranks[d] = 1
	return ranks
